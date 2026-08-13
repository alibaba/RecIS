"""Build a startup estimate for profiler-countable FLOPS and MFU.

The numerator only includes events for which ``torch.profiler`` exposes a
non-zero ``flops`` value. Fused/custom/sparse/embedding operators may therefore
be absent. The cached MFU basis is a lower bound for that incomplete operator
population and must not be compared directly with a paper's model-wide MFU.
"""

import json
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import torch

from recis.monitor.gpuinfo_inquirer import GpuVendor, Inquirer, Precision
from recis.utils.logger import Logger


logger = Logger(__name__)


class StepType(str, Enum):
    TRAIN = "train"
    EVAL = "eval"

    @classmethod
    def from_is_train(cls, is_train: bool):
        return cls.TRAIN if is_train else cls.EVAL


class InputDType(str, Enum):
    FP64 = "fp64"
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProfileFlopsResult:
    """FLOPS grouped by profiler-reported operator input dtype.

    Args:
        flops_by_input_dtype (Mapping[InputDType, float]): Profiler-countable
            FLOPS assigned to each input dtype without double counting events.
        input_dtype_source (str): Profiler metadata path used for dtype
            classification.
        tf32_unaccounted (bool, optional): Whether FP32-input events may have
            used TF32 while remaining classified as FP32. Defaults to False.
        primary_exact_input_dtype_coverage (float, optional): Exact FLOPS
            coverage before a low-coverage metadata fallback. Defaults to 1.
        exact_input_dtype_coverage (float, optional): Exact FLOPS coverage
            after choosing the best profiler metadata source, before any
            precision-hint imputation. Defaults to 1.
        resolved_input_dtype_coverage (float, optional): FLOPS coverage after
            imputing only events that could not be joined to profiler input
            metadata. Defaults to 1.
        unmatched_flops (float, optional): FLOPS whose event id was absent
            from the selected input-dtype metadata source. Defaults to 0.
        explicit_unknown_flops (float, optional): FLOPS whose metadata was
            present but mixed or unsupported. These are never imputed.
            Defaults to 0.
        imputed_flops (float, optional): Unmatched FLOPS assigned from the
            Trainer precision hint after exact fallbacks failed. Defaults to 0.
        imputed_dtype (InputDType, optional): Precision hint used for
            imputed_flops. Defaults to None.
    """

    flops_by_input_dtype: Mapping[InputDType, float]
    input_dtype_source: str
    tf32_unaccounted: bool = False
    primary_exact_input_dtype_coverage: float = 1.0
    exact_input_dtype_coverage: float = 1.0
    resolved_input_dtype_coverage: float = 1.0
    unmatched_flops: float = 0.0
    explicit_unknown_flops: float = 0.0
    imputed_flops: float = 0.0
    imputed_dtype: Optional[InputDType] = None


@dataclass(frozen=True)
class CompilerFlopsExample:
    """Aggregated diagnostic for FLOPS kept by the compiler-scope filter."""

    event_name: str
    ancestor_name: Optional[str]
    root_name: Optional[str]
    profiler_step: Optional[str]
    input_shapes: str
    reason: str
    calls: int
    total_flops: float


@dataclass(frozen=True)
class CompilerFlopsFilterResult:
    """One-shot compiler-only FLOPS filtering result for a profiler trace."""

    raw_total_flops: float
    filtered_total_flops: float
    excluded_compile_flops: float
    ambiguous_kept_flops: float
    parent_unknown_kept_flops: float
    excluded_by_scope: Mapping[str, float]
    ambiguous_examples: Tuple[CompilerFlopsExample, ...]
    parent_unknown_examples: Tuple[CompilerFlopsExample, ...]
    excluded_event_object_ids: FrozenSet[int] = field(repr=False)

    @classmethod
    def unfiltered(cls, total_flops: float):
        total_flops = float(total_flops)
        return cls(
            raw_total_flops=total_flops,
            filtered_total_flops=total_flops,
            excluded_compile_flops=0.0,
            ambiguous_kept_flops=0.0,
            parent_unknown_kept_flops=0.0,
            excluded_by_scope=MappingProxyType({}),
            ambiguous_examples=(),
            parent_unknown_examples=(),
            excluded_event_object_ids=frozenset(),
        )


@dataclass(frozen=True)
class StartupFlopsEstimate:
    """Immutable startup estimate consumed by periodic monitor reporting.

    Args:
        sample_step_type (StepType): Train or eval type of the sampled steps.
        sample_steps (int): Number of complete active steps in the estimate.
        train_equiv_flops_per_step (float): Mean profiler-countable FLOPS
            normalized to one train step.
        train_equiv_ideal_seconds_per_step (float, optional): Cached ideal
            device time for one train-equivalent step. None disables MFU.
        mixed_tflops_peak (float, optional): Effective peak TFLOPS implied by
            the sampled dtype mixture. None when no valid MFU peak is available.
        train_equiv_flops_by_input_dtype (Mapping[InputDType, float]):
            Per-dtype FLOPS normalized to one train step.
        input_dtype_coverage (float): Fraction of sampled FLOPS classified to a
            supported input dtype.
        peak_coverage (float): Fraction of sampled FLOPS with a known device
            peak for its classified dtype.
        mfu_invalid_reason (str, optional): Reason MFU cannot be reported, or
            None when the estimate is valid.
        tf32_unaccounted (bool, optional): Whether possible TF32 execution is
            intentionally not reflected in the input-dtype peak. Defaults to
            False.
        source (str, optional): Estimate provenance written to metric tags.
            Defaults to "torch_profiler_startup_estimate".
        precision_basis (str, optional): Precision basis used for the MFU peak.
            Defaults to "operator_input_dtype".
    """

    sample_step_type: StepType
    sample_steps: int
    train_equiv_flops_per_step: float
    train_equiv_ideal_seconds_per_step: Optional[float]
    mixed_tflops_peak: Optional[float]
    train_equiv_flops_by_input_dtype: Mapping[InputDType, float]
    input_dtype_coverage: float
    peak_coverage: float
    mfu_invalid_reason: Optional[str]
    tf32_unaccounted: bool = False
    source: str = "torch_profiler_startup_estimate"
    precision_basis: str = "operator_input_dtype"


@dataclass
class StartupFlopsState:
    """Mutable handoff state shared by startup profiling and metric reporting.

    Args:
        estimate (StartupFlopsEstimate, optional): Published startup estimate.
            Defaults to None until profiling completes successfully.
        invalid_reason (str, optional): Reason no estimate can be published.
            Defaults to None.
    """

    estimate: Optional[StartupFlopsEstimate] = None
    invalid_reason: Optional[str] = None

    def reset(self):
        self.estimate = None
        self.invalid_reason = None

    def set_estimate(self, estimate: StartupFlopsEstimate):
        self.estimate = estimate
        self.invalid_reason = None

    def invalidate(self, reason: str):
        self.estimate = None
        self.invalid_reason = reason


_TRACE_TYPE_TO_DTYPE = {
    "double": InputDType.FP64,
    "float64": InputDType.FP64,
    "torch.float64": InputDType.FP64,
    "float": InputDType.FP32,
    "float32": InputDType.FP32,
    "torch.float32": InputDType.FP32,
    "half": InputDType.FP16,
    "float16": InputDType.FP16,
    "torch.float16": InputDType.FP16,
    "bfloat16": InputDType.BF16,
    "torch.bfloat16": InputDType.BF16,
    "char": InputDType.INT8,
    "int8": InputDType.INT8,
    "torch.int8": InputDType.INT8,
    "float8_e4m3fn": InputDType.FP8_E4M3,
    "float8_e4m3fnuz": InputDType.FP8_E4M3,
    "torch.float8_e4m3fn": InputDType.FP8_E4M3,
    "torch.float8_e4m3fnuz": InputDType.FP8_E4M3,
    "float8_e5m2": InputDType.FP8_E5M2,
    "float8_e5m2fnuz": InputDType.FP8_E5M2,
    "torch.float8_e5m2": InputDType.FP8_E5M2,
    "torch.float8_e5m2fnuz": InputDType.FP8_E5M2,
}
_NON_TENSOR_TRACE_TYPES = {
    "",
    "none",
    "scalar",
    "scalarlist",
    "tensor",
}
_INPUT_DTYPE_TO_PRECISION = {
    InputDType.FP64: Precision.fp64,
    InputDType.FP32: Precision.fp32,
    InputDType.FP16: Precision.fp16,
    InputDType.BF16: Precision.bf16,
    InputDType.INT8: Precision.int8,
}
_TF32_MATMUL_OPS = {"aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm"}
_TF32_CONV_OPS = {"aten::conv2d"}

# _COMPILER_ONLY_SCOPE_NAMES&_COMPILED_RUNTIME_SCOPE_NAMES are dynamic sets
#   They can be updated if the filter works abnormal for some specific opts.
_COMPILER_ONLY_SCOPE_NAMES = frozenset(
    {
        "InductorBenchmarker.benchmark_gpu (dynamo_timed)",
        "CachingAutotuner.benchmark_all_configs (dynamo_timed)",
        "pad_mm_benchmark (dynamo_timed)",
        "pad_mm_benchmark_get_do_bench (dynamo_timed)",
        "async_compile.precompile (dynamo_timed)",
        "StaticAutotunerFuture.warm_precompile (dynamo_timed)",
        # AOTAutograd executes the joint forward/backward graph while tracing
        # it for compilation. torch.profiler can attach algorithmic FLOPS to
        # those tracing events even though they are not a model runtime step.
        "aot_trace_joint_graph (dynamo_timed)",
    }
)
_COMPILED_RUNTIME_SCOPE_NAMES = frozenset(
    {
        "CompiledFunction",
        "CompiledFunctionBackward",
        "autograd::engine::evaluate_function: CompiledFunctionBackward",
        "TorchDynamo Cache Lookup",
        "AOTDispatcher Runtime Wrapper Prologue",
        "Pregraph bytecode",
    }
)
_COMPILED_RUNTIME_SCOPE_PREFIXES = ("Torch-Compiled Region:",)
_COMPILER_RELATED_SCOPE = re.compile(
    r"compile|compiled|dynamo|inductor|aot|autotun|benchmark|do_bench",
    re.IGNORECASE,
)
_MAX_COMPILER_FILTER_EXAMPLES = 10


@dataclass(frozen=True)
class _EventAncestry:
    compiler_only_scope: Optional[str]
    ambiguous_scope: Optional[str]
    root_name: Optional[str]
    profiler_step: Optional[str]
    parent_issue: Optional[str]


def _is_compiled_runtime_scope(name: str) -> bool:
    return name in _COMPILED_RUNTIME_SCOPE_NAMES or name.startswith(
        _COMPILED_RUNTIME_SCOPE_PREFIXES
    )


def _classify_event_ancestry(event) -> _EventAncestry:
    current = getattr(event, "cpu_parent", None)
    if current is None:
        return _EventAncestry(None, None, None, None, "cpu_parent_unavailable")

    seen = set()
    compiler_only_scope = None
    ambiguous_scope = None
    root_name = None
    profiler_step = None
    parent_issue = None
    while current is not None:
        current_object_id = id(current)
        if current_object_id in seen:
            parent_issue = "cpu_parent_cycle"
            break
        seen.add(current_object_id)

        name = str(getattr(current, "name", ""))
        root_name = name or root_name
        if name.startswith("ProfilerStep"):
            profiler_step = name
        if name in _COMPILER_ONLY_SCOPE_NAMES and compiler_only_scope is None:
            compiler_only_scope = name
        elif (
            ambiguous_scope is None
            and not _is_compiled_runtime_scope(name)
            and _COMPILER_RELATED_SCOPE.search(name)
        ):
            ambiguous_scope = name
        current = getattr(current, "cpu_parent", None)

    if parent_issue is None and profiler_step is None:
        parent_issue = "no_profiler_step_ancestor"
    return _EventAncestry(
        compiler_only_scope,
        ambiguous_scope,
        root_name,
        profiler_step,
        parent_issue,
    )


def _compact_input_shapes(event, limit: int = 200) -> str:
    shapes = repr(getattr(event, "input_shapes", None)).replace("\n", "")
    return shapes if len(shapes) <= limit else shapes[: limit - 3] + "..."


def _add_filter_example(aggregates, event, ancestry, ancestor_name, reason, flops):
    key = (
        str(getattr(event, "name", "unknown")),
        ancestor_name,
        ancestry.root_name,
        ancestry.profiler_step,
        _compact_input_shapes(event),
        reason,
    )
    calls, total_flops = aggregates.get(key, (0, 0.0))
    aggregates[key] = (calls + 1, total_flops + flops)


def _top_filter_examples(aggregates) -> Tuple[CompilerFlopsExample, ...]:
    ordered = sorted(
        aggregates.items(),
        key=lambda item: item[1][1],
        reverse=True,
    )[:_MAX_COMPILER_FILTER_EXAMPLES]
    return tuple(
        CompilerFlopsExample(
            event_name=key[0],
            ancestor_name=key[1],
            root_name=key[2],
            profiler_step=key[3],
            input_shapes=key[4],
            reason=key[5],
            calls=value[0],
            total_flops=value[1],
        )
        for key, value in ordered
    )


def filter_compiler_only_flops(
    events: Iterable,
    raw_total_flops: float,
) -> CompilerFlopsFilterResult:
    """Remove FLOPS only from events under confirmed compiler-only scopes.

    ``key_averages`` remains the production raw-total source. Before any
    subtraction, its total must reconcile with the raw event population so a
    vendor profiler cannot make RecIS subtract FLOPS from a different basis.
    Unknown compiler-like scopes and incomplete parent chains are retained and
    summarized for diagnosis.
    """

    events = tuple(events)
    raw_total_flops = float(raw_total_flops)
    event_total_flops = sum(
        float(getattr(event, "flops", None) or 0.0) for event in events
    )
    join_tolerance = max(1.0, abs(raw_total_flops) * 1e-9)
    if abs(event_total_flops - raw_total_flops) > join_tolerance:
        raise ValueError(
            "profiler events/key_averages FLOPS mismatch: "
            f"events_total={event_total_flops}, raw_total={raw_total_flops}"
        )

    excluded_flops = 0.0
    ambiguous_kept_flops = 0.0
    parent_unknown_kept_flops = 0.0
    excluded_by_scope = defaultdict(float)
    ambiguous_examples = {}
    parent_unknown_examples = {}
    excluded_event_object_ids: Set[int] = set()

    for event in events:
        flops = float(getattr(event, "flops", None) or 0.0)
        if flops == 0:
            continue
        ancestry = _classify_event_ancestry(event)
        if ancestry.compiler_only_scope is not None:
            excluded_flops += flops
            excluded_by_scope[ancestry.compiler_only_scope] += flops
            excluded_event_object_ids.add(id(event))
        elif ancestry.ambiguous_scope is not None:
            ambiguous_kept_flops += flops
            _add_filter_example(
                ambiguous_examples,
                event,
                ancestry,
                ancestry.ambiguous_scope,
                "unknown_compiler_related_scope",
                flops,
            )
        elif ancestry.parent_issue is not None:
            parent_unknown_kept_flops += flops
            _add_filter_example(
                parent_unknown_examples,
                event,
                ancestry,
                None,
                ancestry.parent_issue,
                flops,
            )

    filtered_total_flops = raw_total_flops - excluded_flops
    if filtered_total_flops < -join_tolerance:
        raise ValueError(
            "compiler-only FLOPS exceeds raw total: "
            f"excluded={excluded_flops}, raw_total={raw_total_flops}"
        )
    if filtered_total_flops < 0:
        filtered_total_flops = 0.0

    ordered_excluded_scopes = dict(
        sorted(
            excluded_by_scope.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:_MAX_COMPILER_FILTER_EXAMPLES]
    )
    return CompilerFlopsFilterResult(
        raw_total_flops=raw_total_flops,
        filtered_total_flops=filtered_total_flops,
        excluded_compile_flops=excluded_flops,
        ambiguous_kept_flops=ambiguous_kept_flops,
        parent_unknown_kept_flops=parent_unknown_kept_flops,
        excluded_by_scope=MappingProxyType(ordered_excluded_scopes),
        ambiguous_examples=_top_filter_examples(ambiguous_examples),
        parent_unknown_examples=_top_filter_examples(parent_unknown_examples),
        excluded_event_object_ids=frozenset(excluded_event_object_ids),
    )


def _normalize_trace_type(value) -> str:
    normalized = str(value).strip().lower()
    if normalized.startswith("c10::"):
        normalized = normalized[len("c10::") :]
    return normalized


def classify_input_dtype(input_types) -> InputDType:
    """Classify one profiler event without duplicating its FLOPS.

    Classification uses the operator input dtype reported by torch.profiler,
    not the backend math mode. FP32 inputs may execute through TF32, BF16
    decomposition, or another lower-precision algorithm; RecIS deliberately
    does not reclassify those events because the actual algorithm is not
    reliably observable across supported torch versions and accelerators.
    """

    if not isinstance(input_types, (list, tuple)):
        input_types = [input_types]

    dtypes = set()
    has_unknown_type = False
    for input_type in input_types:
        normalized = _normalize_trace_type(input_type)
        if normalized in _NON_TENSOR_TRACE_TYPES:
            continue
        dtype = _TRACE_TYPE_TO_DTYPE.get(normalized)
        if dtype is None:
            has_unknown_type = True
        else:
            dtypes.add(dtype)

    if len(dtypes) == 1 and not has_unknown_type:
        return next(iter(dtypes))
    return InputDType.UNKNOWN


def _normalize_external_id(value) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _merge_event_dtype(
    input_dtypes: Dict[str, InputDType],
    event_id: str,
    dtype: InputDType,
):
    previous = input_dtypes.get(event_id)
    if previous is None or previous == dtype:
        input_dtypes[event_id] = dtype
    elif previous == InputDType.UNKNOWN:
        input_dtypes[event_id] = dtype
    elif dtype != InputDType.UNKNOWN:
        input_dtypes[event_id] = InputDType.UNKNOWN


def _walk_event_tree(nodes):
    for node in nodes:
        yield node
        yield from _walk_event_tree(getattr(node, "children", ()))


def _event_tree_input_dtypes(profiler) -> Optional[Dict[str, InputDType]]:
    """Read input dtypes in memory, or return None when the API is unavailable.

    Torch 2.4 and 2.10 expose ``extra_fields.inputs`` as a list of
    ``TensorMetadata``/scalar objects (not an object with a ``dtypes`` field).
    ``correlation_id`` matches ``FunctionEvent.id`` on both versions.
    """

    torch_profiler = getattr(profiler, "profiler", None)
    kineto_results = getattr(torch_profiler, "kineto_results", None)
    event_tree = getattr(kineto_results, "experimental_event_tree", None)
    if not callable(event_tree):
        return None

    try:
        input_dtypes: Dict[str, InputDType] = {}
        for node in _walk_event_tree(event_tree()):
            correlation_id = getattr(node, "correlation_id", None)
            extra_fields = getattr(node, "extra_fields", None)
            inputs = getattr(extra_fields, "inputs", None)
            if correlation_id is None or inputs is None:
                continue
            input_types = [
                getattr(input_value, "dtype", "Scalar") for input_value in inputs
            ]
            _merge_event_dtype(
                input_dtypes,
                _normalize_external_id(correlation_id),
                classify_input_dtype(input_types),
            )
        return input_dtypes
    except Exception as error:
        logger.warning(
            "Profiler event-tree dtype extraction failed; falling back to a "
            "temporary Chrome trace (%s)",
            error,
        )
        return None


def _preferred_trace_directory() -> Optional[str]:
    standard_log_dir = os.environ.get("STD_LOG_DIR")
    if not standard_log_dir:
        return None
    trace_directory = os.path.join(
        standard_log_dir,
        os.environ.get("RANK", "0"),
    )
    try:
        os.makedirs(trace_directory, exist_ok=True)
    except OSError as error:
        logger.warning(
            "Cannot create profiler dtype trace directory %s; using the system "
            "temporary directory instead (%s)",
            trace_directory,
            error,
        )
        return None
    return trace_directory


def _chrome_trace_input_dtypes(profiler) -> Dict[str, InputDType]:
    descriptor, trace_path = tempfile.mkstemp(
        prefix="recis-mfu-input-dtype-",
        suffix=".json",
        dir=_preferred_trace_directory(),
    )
    os.close(descriptor)
    exported = False
    keep_for_debugging = False
    try:
        profiler.export_chrome_trace(trace_path)
        exported = True
        with open(trace_path, encoding="utf-8") as trace_file:
            trace = json.load(trace_file)
    except Exception:
        keep_for_debugging = exported
        if keep_for_debugging:
            logger.warning(
                "Profiler dtype trace parsing failed; retaining trace for "
                "debugging at %s",
                trace_path,
            )
        raise
    finally:
        if not keep_for_debugging:
            try:
                os.unlink(trace_path)
            except OSError as error:
                logger.debug(
                    "Failed to clean up profiler dtype trace file %s: %s",
                    trace_path,
                    error,
                )

    input_dtypes: Dict[str, InputDType] = {}
    for trace_event in trace.get("traceEvents", []):
        args = trace_event.get("args") or {}
        external_id = args.get("External id")
        if external_id is None or "Input type" not in args:
            continue
        _merge_event_dtype(
            input_dtypes,
            _normalize_external_id(external_id),
            classify_input_dtype(args["Input type"]),
        )
    return input_dtypes


def _new_tf32_setting(*owners) -> Optional[bool]:
    """Resolve the first explicit 2.9+ precision setting by specificity.

    Torch 2.10 returns ``"none"`` when a layer inherits from its parent; it
    does not mean IEEE FP32. If every new-API layer is unset, callers may safely
    inspect the legacy flag instead.
    """

    for owner in owners:
        if owner is None:
            continue
        try:
            setting = str(owner.fp32_precision).lower()
        except (AttributeError, RuntimeError):
            continue
        if setting != "none":
            return setting == "tf32"
    return None


def _matmul_tf32_enabled() -> bool:
    new_setting = _new_tf32_setting(
        torch.backends.cuda.matmul,
        torch.backends.cuda,
        torch.backends,
    )
    if new_setting is not None:
        return new_setting
    return bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False))


def _conv_tf32_enabled() -> bool:
    new_setting = _new_tf32_setting(
        getattr(torch.backends.cudnn, "conv", None),
        torch.backends.cudnn,
        torch.backends,
    )
    if new_setting is not None:
        return new_setting
    return bool(getattr(torch.backends.cudnn, "allow_tf32", False))


def _nvidia_tf32_supported() -> bool:
    if Inquirer.vendor != GpuVendor.NVIDIA:
        return False
    is_supported = getattr(torch.cuda, "is_tf32_supported", None)
    if callable(is_supported):
        try:
            return bool(is_supported())
        except Exception:
            pass
    try:
        return torch.cuda.get_device_capability(0)[0] >= 8
    except Exception:
        return False


def _event_may_use_unaccounted_tf32(event_name: str, dtype: InputDType) -> bool:
    if dtype != InputDType.FP32 or not _nvidia_tf32_supported():
        return False
    # This is intentionally warning-only. In particular, "high"/"medium"
    # float32 matmul policies may choose TF32 or BF16 decomposition depending on
    # hardware and heuristics, so they cannot select an exact peak denominator.
    if event_name in _TF32_MATMUL_OPS:
        return _matmul_tf32_enabled()
    if event_name in _TF32_CONV_OPS:
        return _conv_tf32_enabled()
    return False


@dataclass(frozen=True)
class _InputDTypeJoinResult:
    flops_by_input_dtype: Mapping[InputDType, float]
    unmatched_flops: float
    explicit_unknown_flops: float
    tf32_unaccounted: bool

    @property
    def total_flops(self) -> float:
        return sum(self.flops_by_input_dtype.values())

    @property
    def exact_coverage(self) -> float:
        known_flops = self.total_flops - self.flops_by_input_dtype.get(
            InputDType.UNKNOWN,
            0.0,
        )
        return _bounded_coverage(known_flops, self.total_flops)


def _join_flops_with_input_dtypes(
    events: Sequence,
    input_dtypes: Mapping[str, InputDType],
    excluded_event_object_ids: FrozenSet[int],
) -> _InputDTypeJoinResult:
    flops_by_dtype: Dict[InputDType, float] = {}
    unmatched_flops = 0.0
    explicit_unknown_flops = 0.0
    tf32_unaccounted = False
    for event in events:
        if id(event) in excluded_event_object_ids:
            continue
        flops = getattr(event, "flops", None)
        if flops is None or flops == 0:
            continue
        flops = float(flops)
        event_id = _normalize_external_id(getattr(event, "id", ""))
        if event_id not in input_dtypes:
            dtype = InputDType.UNKNOWN
            unmatched_flops += flops
        else:
            dtype = input_dtypes[event_id]
            if dtype == InputDType.UNKNOWN:
                explicit_unknown_flops += flops
        flops_by_dtype[dtype] = flops_by_dtype.get(dtype, 0.0) + flops
        tf32_unaccounted = tf32_unaccounted or _event_may_use_unaccounted_tf32(
            getattr(event, "name", ""),
            dtype,
        )
    return _InputDTypeJoinResult(
        flops_by_input_dtype=MappingProxyType(flops_by_dtype),
        unmatched_flops=unmatched_flops,
        explicit_unknown_flops=explicit_unknown_flops,
        tf32_unaccounted=tf32_unaccounted,
    )


def _impute_unmatched_flops(
    join_result: _InputDTypeJoinResult,
    fallback_input_dtype: Optional[InputDType],
) -> Tuple[Mapping[InputDType, float], float]:
    if fallback_input_dtype is None or join_result.unmatched_flops <= 0:
        return join_result.flops_by_input_dtype, 0.0
    if fallback_input_dtype == InputDType.UNKNOWN:
        raise ValueError("fallback_input_dtype cannot be UNKNOWN")

    flops_by_dtype = dict(join_result.flops_by_input_dtype)
    remaining_unknown = (
        flops_by_dtype.get(InputDType.UNKNOWN, 0.0)
        - join_result.unmatched_flops
    )
    tolerance = max(1.0, abs(join_result.total_flops) * 1e-9)
    if remaining_unknown < -tolerance:
        raise ValueError(
            "unmatched FLOPS exceeds UNKNOWN bucket: "
            f"unmatched={join_result.unmatched_flops}, "
            f"unknown={flops_by_dtype.get(InputDType.UNKNOWN, 0.0)}"
        )
    if remaining_unknown > tolerance:
        flops_by_dtype[InputDType.UNKNOWN] = remaining_unknown
    else:
        flops_by_dtype.pop(InputDType.UNKNOWN, None)
    flops_by_dtype[fallback_input_dtype] = (
        flops_by_dtype.get(fallback_input_dtype, 0.0)
        + join_result.unmatched_flops
    )
    return MappingProxyType(flops_by_dtype), join_result.unmatched_flops


def profile_flops_by_input_dtype(
    profiler,
    *,
    events: Optional[Sequence] = None,
    excluded_event_object_ids: Optional[FrozenSet[int]] = None,
    min_exact_coverage: float = 0.99,
    fallback_input_dtype: Optional[InputDType] = None,
) -> ProfileFlopsResult:
    """Group FLOPS by input dtype with exact and estimated fallbacks.

    The in-memory event tree is the low-overhead primary source. Some Torch
    builds expose the API while only partially joining its correlation ids to
    ``FunctionEvent.id``. When unjoined FLOPS prevent the requested exact
    coverage, the same profiler window is exported to a temporary Chrome trace
    and joined by its stable ``External id`` metadata. If that exact fallback
    fails, only still-unjoined FLOPS may use the Trainer precision hint;
    explicitly mixed or unsupported input metadata always remains UNKNOWN.
    """

    if not 0 < min_exact_coverage <= 1:
        raise ValueError("min_exact_coverage must be in (0, 1]")
    if events is None:
        events = tuple(profiler.events())
    else:
        events = tuple(events)
    if excluded_event_object_ids is None:
        excluded_event_object_ids = frozenset()

    input_dtypes = _event_tree_input_dtypes(profiler)
    if input_dtypes is None:
        input_dtypes = _chrome_trace_input_dtypes(profiler)
        input_dtype_source = "chrome_trace_fallback"
    else:
        input_dtype_source = "experimental_event_tree"

    join_result = _join_flops_with_input_dtypes(
        events,
        input_dtypes,
        excluded_event_object_ids,
    )
    primary_exact_coverage = join_result.exact_coverage
    join_tolerance = max(1.0, abs(join_result.total_flops) * 1e-9)

    if (
        input_dtype_source == "experimental_event_tree"
        and primary_exact_coverage < min_exact_coverage
        and join_result.unmatched_flops > join_tolerance
    ):
        try:
            chrome_input_dtypes = _chrome_trace_input_dtypes(profiler)
            chrome_join_result = _join_flops_with_input_dtypes(
                events,
                chrome_input_dtypes,
                excluded_event_object_ids,
            )
        except Exception as error:
            logger.warning(
                "Profiler event-tree input-dtype coverage is incomplete and "
                "Chrome trace recovery failed; keeping the event-tree result "
                "(exact_coverage=%.5f, unmatched_flops=%.0f, error=%s)",
                primary_exact_coverage,
                join_result.unmatched_flops,
                error,
            )
        else:
            if chrome_join_result.exact_coverage > join_result.exact_coverage:
                logger.info(
                    "Profiler input-dtype coverage recovered by Chrome trace "
                    "(event_tree_coverage=%.5f, chrome_trace_coverage=%.5f, "
                    "event_tree_unmatched_flops=%.0f)",
                    primary_exact_coverage,
                    chrome_join_result.exact_coverage,
                    join_result.unmatched_flops,
                )
                join_result = chrome_join_result
                input_dtype_source = "chrome_trace_coverage_fallback"
            else:
                logger.warning(
                    "Profiler Chrome trace did not improve input-dtype coverage; "
                    "keeping the event-tree result (event_tree_coverage=%.5f, "
                    "chrome_trace_coverage=%.5f)",
                    join_result.exact_coverage,
                    chrome_join_result.exact_coverage,
                )

    flops_by_dtype, imputed_flops = _impute_unmatched_flops(
        join_result,
        fallback_input_dtype,
    )
    if imputed_flops > 0:
        input_dtype_source = (
            f"{input_dtype_source}+missing_join_imputed_"
            f"{fallback_input_dtype.value}"
        )
        logger.warning(
            "Profiler input-dtype metadata remained incomplete; assigning only "
            "unjoined FLOPS to the Trainer precision hint "
            "(imputed_dtype=%s, imputed_flops=%.0f, exact_coverage=%.5f)",
            fallback_input_dtype.value,
            imputed_flops,
            join_result.exact_coverage,
        )

    known_flops = sum(
        flops
        for dtype, flops in flops_by_dtype.items()
        if dtype != InputDType.UNKNOWN
    )
    resolved_coverage = _bounded_coverage(known_flops, join_result.total_flops)
    return ProfileFlopsResult(
        flops_by_input_dtype=flops_by_dtype,
        input_dtype_source=input_dtype_source,
        tf32_unaccounted=join_result.tf32_unaccounted,
        primary_exact_input_dtype_coverage=primary_exact_coverage,
        exact_input_dtype_coverage=join_result.exact_coverage,
        resolved_input_dtype_coverage=resolved_coverage,
        unmatched_flops=join_result.unmatched_flops,
        explicit_unknown_flops=join_result.explicit_unknown_flops,
        imputed_flops=imputed_flops,
        imputed_dtype=(fallback_input_dtype if imputed_flops > 0 else None),
    )


def get_input_dtype_peak_tflops(
    input_dtype: InputDType,
    device_index: int = 0,
) -> Optional[float]:
    # Peak selection intentionally follows the profiler-reported input dtype.
    # An FP32-input event may execute as TF32 internally; for now that case is
    # warning-only via ``tf32_unaccounted`` because correcting it requires
    # complete TF32 peak datasheets for every supported device family.
    precision = _INPUT_DTYPE_TO_PRECISION.get(input_dtype)
    if precision is None:
        return None
    return Inquirer.get_peak_tflops(
        device_index=device_index,
        precision=precision,
    )


def _bounded_coverage(covered_flops: float, total_flops: float) -> float:
    if total_flops <= 0:
        return 0.0
    return min(max(covered_flops / total_flops, 0.0), 1.0)


def build_startup_flops_estimate(
    *,
    total_flops: float,
    sample_step_type: StepType,
    sample_steps: int,
    eval_flops_ratio: float,
    flops_by_input_dtype: Optional[Mapping[InputDType, float]] = None,
    scalar_tflops_peak: Optional[float] = None,
    scalar_precision_basis: str = "explicit_tflops_peak",
    peak_lookup: Callable[[InputDType], Optional[float]] = (
        get_input_dtype_peak_tflops
    ),
    min_peak_coverage: float = 0.99,
    tf32_unaccounted: bool = False,
) -> StartupFlopsEstimate:
    """Cache the O(dtype-count) mixed-peak calculation for O(1) reporting.

    Unknown input dtypes and missing peak-table entries contribute zero ideal
    time. When ``peak_coverage >= min_peak_coverage``, the resulting MFU is
    emitted as a lower bound. Coverage stays in the cached estimate and startup
    logs for operational diagnosis; it is not added to monitor metric tags.
    """

    if sample_steps <= 0:
        raise ValueError("sample_steps must be positive")
    if eval_flops_ratio <= 0:
        raise ValueError("eval_flops_ratio must be positive")
    if not 0 < min_peak_coverage <= 1:
        raise ValueError("min_peak_coverage must be in (0, 1]")

    sample_ratio = 1.0 if sample_step_type == StepType.TRAIN else eval_flops_ratio
    train_equiv_total = float(total_flops) / sample_steps / sample_ratio

    if flops_by_input_dtype is None:
        valid_scalar_peak = scalar_tflops_peak is not None and scalar_tflops_peak > 0
        invalid_reason = None
        ideal_seconds = None
        if train_equiv_total <= 0:
            invalid_reason = "no_countable_flops"
        elif not valid_scalar_peak:
            invalid_reason = "unknown_peak"
        else:
            ideal_seconds = train_equiv_total / (scalar_tflops_peak * 1e12)
        return StartupFlopsEstimate(
            sample_step_type=sample_step_type,
            sample_steps=sample_steps,
            train_equiv_flops_per_step=train_equiv_total,
            train_equiv_ideal_seconds_per_step=ideal_seconds,
            mixed_tflops_peak=(
                scalar_tflops_peak
                if valid_scalar_peak and train_equiv_total > 0
                else None
            ),
            train_equiv_flops_by_input_dtype=MappingProxyType({}),
            input_dtype_coverage=(1.0 if train_equiv_total > 0 else 0.0),
            peak_coverage=(1.0 if valid_scalar_peak and train_equiv_total > 0 else 0.0),
            mfu_invalid_reason=invalid_reason,
            tf32_unaccounted=tf32_unaccounted,
            precision_basis=scalar_precision_basis,
        )

    train_equiv_by_dtype = {
        dtype: float(flops) / sample_steps / sample_ratio
        for dtype, flops in flops_by_input_dtype.items()
    }
    dtype_total = sum(train_equiv_by_dtype.values())
    join_tolerance = max(1.0, abs(train_equiv_total) * 1e-9)
    trace_join_mismatch = abs(dtype_total - train_equiv_total) > join_tolerance
    if trace_join_mismatch:
        relative_deviation = abs(dtype_total - train_equiv_total) / max(
            abs(train_equiv_total),
            1.0,
        )
        logger.warning(
            "Profiler trace/events FLOPS join mismatch: dtype_total=%s, "
            "train_equiv_total=%s, relative_deviation=%.6g",
            dtype_total,
            train_equiv_total,
            relative_deviation,
        )
        # Keep production FLOPS, but never manufacture a partial peak from an
        # event join that changed across torch/Kineto versions.
        train_equiv_by_dtype = {InputDType.UNKNOWN: train_equiv_total}

    classified_flops = sum(
        flops
        for dtype, flops in train_equiv_by_dtype.items()
        if dtype != InputDType.UNKNOWN
    )
    input_dtype_coverage = _bounded_coverage(classified_flops, train_equiv_total)

    ideal_seconds = 0.0
    peak_covered_flops = 0.0
    for dtype, flops in train_equiv_by_dtype.items():
        if flops == 0:
            continue
        peak_tflops = peak_lookup(dtype)
        if peak_tflops is None or peak_tflops <= 0:
            continue
        peak_covered_flops += flops
        ideal_seconds += flops / (peak_tflops * 1e12)

    peak_coverage = _bounded_coverage(peak_covered_flops, train_equiv_total)
    invalid_reason = None
    mixed_tflops_peak = None
    if train_equiv_total <= 0:
        ideal_seconds = None
        invalid_reason = "no_countable_flops"
    elif trace_join_mismatch:
        ideal_seconds = None
        invalid_reason = "trace_join_mismatch"
    elif peak_coverage < min_peak_coverage:
        ideal_seconds = None
        invalid_reason = (
            "unknown_input_dtype"
            if input_dtype_coverage < min_peak_coverage
            else "missing_dtype_peak"
        )
    elif ideal_seconds > 0:
        # Only the peak-covered numerator belongs in this harmonic mean.
        # Using the full numerator would overstate the effective peak whenever
        # coverage is accepted below 100%.
        mixed_tflops_peak = peak_covered_flops / ideal_seconds / 1e12
    else:
        ideal_seconds = None
        invalid_reason = "invalid_dtype_peak"

    return StartupFlopsEstimate(
        sample_step_type=sample_step_type,
        sample_steps=sample_steps,
        train_equiv_flops_per_step=train_equiv_total,
        train_equiv_ideal_seconds_per_step=ideal_seconds,
        mixed_tflops_peak=mixed_tflops_peak,
        train_equiv_flops_by_input_dtype=MappingProxyType(dict(train_equiv_by_dtype)),
        input_dtype_coverage=input_dtype_coverage,
        peak_coverage=peak_coverage,
        mfu_invalid_reason=invalid_reason,
        tf32_unaccounted=tf32_unaccounted,
    )
