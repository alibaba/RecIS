import os
from typing import Callable, Optional

from torch.profiler import ProfilerActivity, profile, schedule

from recis.hooks.hook import Hook
from recis.monitor.flops_estimator import (
    CompilerFlopsFilterResult,
    InputDType,
    ProfileFlopsResult,
    StartupFlopsState,
    StepType,
    build_startup_flops_estimate,
    filter_compiler_only_flops,
    get_input_dtype_peak_tflops,
    profile_flops_by_input_dtype,
)
from recis.utils.logger import Logger


INITIAL_PROFILE_SKIP_STEPS = 5
INITIAL_PROFILE_WARMUP_STEPS = 1
INITIAL_PROFILE_ACTIVE_STEPS = 3
INITIAL_PROFILE_FIRST_ACTIVE_STEP = (
    INITIAL_PROFILE_SKIP_STEPS + INITIAL_PROFILE_WARMUP_STEPS + 1
)
INITIAL_PROFILE_LAST_STEP = (
    INITIAL_PROFILE_SKIP_STEPS
    + INITIAL_PROFILE_WARMUP_STEPS
    + INITIAL_PROFILE_ACTIVE_STEPS
)
USER_PROFILER_CREATE_STEP = INITIAL_PROFILE_LAST_STEP + 1


class _InitialProfilerHook(Hook):
    """Internal hook that estimates startup FLOPS.

    The _InitialProfilerHook is automatically used by the built-in data collection module of `recis`.
    Note that _InitialProfilerHook CANNOT BE USED EXTERNALLY!!!

    Args:
        flops_state (StartupFlopsState, optional): State shared with
            MetricReportHook for publishing the startup FLOPS estimate. A new
            state is created when omitted.
        eval_flops_ratio (float, optional): Eval-to-train FLOPS ratio used to
            normalize an eval startup sample to train-equivalent FLOPS.
            Defaults to 1/3.
        collect_input_dtypes (bool, optional): Whether to record operator input
            shapes and classify profiler-countable FLOPS by input dtype for
            mixed-precision peak estimation. Defaults to True.
        mixed_precision (str, optional): Trainer autocast precision hint. It is
            used only for FLOPS whose events cannot be joined to profiler dtype
            metadata after exact fallbacks fail; explicit mixed or unsupported
            dtype metadata is never overwritten. Supports "bf16" and "fp16".
            Defaults to None.
        scalar_tflops_peak (float, optional): Explicit scalar TFLOPS peak used
            instead of input-dtype mixed peak estimation. Defaults to None.
        scalar_precision_basis (str, optional): Precision label associated with
            scalar_tflops_peak in logs and metric tags. Defaults to
            "explicit_tflops_peak".
        min_peak_coverage (float, optional): Minimum fraction of classified
            FLOPS with a known device peak required to produce MFU. Defaults to
            0.99.
        peak_lookup (Callable, optional): Function mapping an InputDType to the
            device peak TFLOPS used by mixed-precision estimation. Defaults to
            get_input_dtype_peak_tflops.

    Startup steps 1 through 5 are skipped, step 6 warms up the profiler, and
    steps 7 through 9 are sampled. The profiler is fully closed after step 9
    before the public ProfilerHook can be created at step 10.

    The three-step mean is cached for the whole run. Starting the active window
    at step 7 gives lazy compiler work more time to finish; exact compiler-only
    event filtering remains the correctness boundary if compilation still
    overlaps the window. The shorter window trades some shape smoothing for
    that startup separation and for a deterministic step-10 Kineto handoff.
    """

    def __init__(
        self,
        flops_state: Optional[StartupFlopsState] = None,
        eval_flops_ratio: float = 1.0 / 3.0,
        collect_input_dtypes: bool = True,
        mixed_precision: Optional[str] = None,
        scalar_tflops_peak: Optional[float] = None,
        scalar_precision_basis: str = "explicit_tflops_peak",
        min_peak_coverage: float = 0.99,
        peak_lookup: Callable[[InputDType], Optional[float]] = (
            get_input_dtype_peak_tflops
        ),
    ):
        self.logger = Logger("_InitialProfilerHook")
        self.prof_step_count = 0
        self.prof = None
        self.sample_step_type = None
        self.flops_state = flops_state or StartupFlopsState()
        self.flops_state.reset()
        self.eval_flops_ratio = eval_flops_ratio
        self.collect_input_dtypes = collect_input_dtypes
        normalized_mixed_precision = (
            mixed_precision.lower() if mixed_precision is not None else None
        )
        precision_hints = {
            "bf16": InputDType.BF16,
            "fp16": InputDType.FP16,
        }
        if (
            normalized_mixed_precision is not None
            and normalized_mixed_precision not in precision_hints
        ):
            raise ValueError(
                "mixed_precision must be 'bf16', 'fp16', or None"
            )
        self.fallback_input_dtype = precision_hints.get(
            normalized_mixed_precision
        )
        self.scalar_tflops_peak = scalar_tflops_peak
        self.scalar_precision_basis = scalar_precision_basis
        self.min_peak_coverage = min_peak_coverage
        self.peak_lookup = peak_lookup
        if eval_flops_ratio <= 0:
            raise ValueError("eval_flops_ratio must be positive")
        if not 0 < min_peak_coverage <= 1:
            raise ValueError("min_peak_coverage must be in (0, 1]")
        if os.environ.get("RECIS_MONITOR_ON", "1") != "1":
            return
        self.prof = profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            schedule=schedule(
                wait=0,
                warmup=INITIAL_PROFILE_WARMUP_STEPS,
                active=INITIAL_PROFILE_ACTIVE_STEPS,
                repeat=1,
                skip_first=INITIAL_PROFILE_SKIP_STEPS,
            ),
            record_shapes=collect_input_dtypes,
            profile_memory=False,
            with_stack=False,
            with_flops=True,
            with_modules=False,
        )
        self.prof.__enter__()

    def after_step(self, is_train=True, *args, **kwargs):
        if self.prof is None:
            return

        next_step_count = self.prof_step_count + 1
        if INITIAL_PROFILE_FIRST_ACTIVE_STEP <= next_step_count:
            step_type = StepType.from_is_train(is_train)
            if self.sample_step_type is None:
                self.sample_step_type = step_type
            elif self.sample_step_type != step_type:
                self.prof_step_count = next_step_count
                self.flops_state.invalidate("mixed_startup_step_types")
                self._close_profiler()
                self.logger.warning(
                    "FLOPS startup estimate unavailable: "
                    "mixed_startup_step_types (first=%s, current=%s, step=%s)",
                    self.sample_step_type.value,
                    step_type.value,
                    next_step_count,
                )
                return

        # torch.profiler actions describe the next real iteration. Advancing
        # only here maps warmup to step 6 and recording to steps 7-9.
        # A Trainer step may be a gradient-accumulation micro-step; the report
        # hook uses the same step definition, keeping the FLOPS/QPS ratio aligned.
        self.prof_step_count = next_step_count
        self.prof.step()
        if self.prof_step_count == INITIAL_PROFILE_LAST_STEP:
            try:
                self._publish_estimate()
            except Exception as error:
                self.flops_state.invalidate("profiler_estimate_error")
                self.logger.error(
                    "FLOPS startup estimate unavailable: profiler_estimate_error (%s)",
                    error,
                )
            finally:
                self._close_profiler()

    def _publish_estimate(self):
        raw_total_flops = sum(
            float(event.flops)
            for event in self.prof.key_averages()
            if getattr(event, "flops", None) is not None
        )
        events = None
        compile_filter = CompilerFlopsFilterResult.unfiltered(raw_total_flops)
        try:
            events = tuple(self.prof.events())
            compile_filter = filter_compiler_only_flops(
                events,
                raw_total_flops,
            )
        except Exception as error:
            self.logger.warning(
                "Compiler-only FLOPS filter failed; keeping the unfiltered "
                "startup total (compile_filter_failed=%s)",
                error,
            )
        self._log_compile_filter(compile_filter)
        total_flops = compile_filter.filtered_total_flops

        default_coverage = 1.0 if total_flops > 0 else 0.0
        profile_result = ProfileFlopsResult(
            flops_by_input_dtype={},
            input_dtype_source="not_collected",
            primary_exact_input_dtype_coverage=default_coverage,
            exact_input_dtype_coverage=default_coverage,
            resolved_input_dtype_coverage=default_coverage,
        )
        if self.collect_input_dtypes:
            try:
                profile_result = profile_flops_by_input_dtype(
                    self.prof,
                    events=events,
                    excluded_event_object_ids=(
                        compile_filter.excluded_event_object_ids
                    ),
                    min_exact_coverage=self.min_peak_coverage,
                    fallback_input_dtype=self.fallback_input_dtype,
                )
            except Exception as error:
                profile_result = ProfileFlopsResult(
                    flops_by_input_dtype={InputDType.UNKNOWN: total_flops},
                    input_dtype_source="classification_failed",
                    primary_exact_input_dtype_coverage=0.0,
                    exact_input_dtype_coverage=0.0,
                    resolved_input_dtype_coverage=0.0,
                    unmatched_flops=total_flops,
                )
                self.logger.warning(
                    "FLOPS input-dtype classification failed; MFU disabled "
                    "for this startup estimate (%s)",
                    error,
                )

        estimate = build_startup_flops_estimate(
            total_flops=total_flops,
            sample_step_type=self.sample_step_type,
            sample_steps=INITIAL_PROFILE_ACTIVE_STEPS,
            eval_flops_ratio=self.eval_flops_ratio,
            flops_by_input_dtype=(
                profile_result.flops_by_input_dtype
                if self.collect_input_dtypes
                else None
            ),
            scalar_tflops_peak=self.scalar_tflops_peak,
            scalar_precision_basis=self.scalar_precision_basis,
            peak_lookup=self.peak_lookup,
            min_peak_coverage=self.min_peak_coverage,
            tf32_unaccounted=profile_result.tf32_unaccounted,
        )
        self.flops_state.set_estimate(estimate)
        self.logger.info(
            "FLOPS startup estimate=%s train-equivalent FLOPS/step "
            "(sample_type=%s, sample_steps=%s-%s, precision_basis=%s, "
            "input_dtype_source=%s, input_dtype_coverage=%.5f, "
            "peak_coverage=%.5f, min_peak_coverage=%.5f, "
            "mixed_tflops_peak=%s, mfu_invalid_reason=%s, "
            "dtype_primary_exact_coverage=%.5f, "
            "dtype_exact_coverage=%.5f, "
            "dtype_resolved_coverage=%.5f, "
            "dtype_unmatched_flops=%.0f, "
            "dtype_explicit_unknown_flops=%.0f, "
            "dtype_imputed_flops=%.0f, dtype_imputed_as=%s, "
            "compile_filtered_flops=%.0f, "
            "compile_ambiguous_kept_flops=%.0f, "
            "compile_parent_unknown_kept_flops=%.0f)",
            estimate.train_equiv_flops_per_step,
            estimate.sample_step_type.value,
            INITIAL_PROFILE_FIRST_ACTIVE_STEP,
            INITIAL_PROFILE_LAST_STEP,
            estimate.precision_basis,
            profile_result.input_dtype_source,
            estimate.input_dtype_coverage,
            estimate.peak_coverage,
            self.min_peak_coverage,
            estimate.mixed_tflops_peak,
            estimate.mfu_invalid_reason,
            profile_result.primary_exact_input_dtype_coverage,
            profile_result.exact_input_dtype_coverage,
            profile_result.resolved_input_dtype_coverage,
            profile_result.unmatched_flops,
            profile_result.explicit_unknown_flops,
            profile_result.imputed_flops,
            (
                profile_result.imputed_dtype.value
                if profile_result.imputed_dtype is not None
                else None
            ),
            compile_filter.excluded_compile_flops,
            compile_filter.ambiguous_kept_flops,
            compile_filter.parent_unknown_kept_flops,
        )
        if self.collect_input_dtypes:
            self._log_dtype_distribution(estimate, profile_result)
        if estimate.tf32_unaccounted:
            self.logger.warning(
                "recis_tf32_unaccounted=true: NVIDIA TF32 is enabled for at "
                "least one profiler-countable FP32 matmul/conv event, but the "
                "MFU denominator remains the literal FP32 input-dtype peak; "
                "MFU may be overestimated."
            )

    def _log_dtype_distribution(self, estimate, profile_result):
        dtype_flops = {
            dtype.value: float(
                estimate.train_equiv_flops_by_input_dtype.get(dtype, 0.0)
            )
            for dtype in InputDType
        }
        total_flops = sum(dtype_flops.values())
        dtype_ratios = {
            dtype: (
                f"{flops / total_flops:.8%}"
                if total_flops > 0
                else "0.00000000%"
            )
            for dtype, flops in dtype_flops.items()
        }
        sample_ratio = (
            1.0
            if estimate.sample_step_type == StepType.TRAIN
            else self.eval_flops_ratio
        )
        fallback_applied_flops = (
            profile_result.imputed_flops
            / INITIAL_PROFILE_ACTIVE_STEPS
            / sample_ratio
        )
        self.logger.info(
            "MFU dtype distribution: "
            "dtype_flops_per_train_equiv_step=%s, dtype_flops_ratio=%s, "
            "input_dtype_source=%s, fallback_dtype=%s, "
            "fallback_applied_flops=%s",
            dtype_flops,
            dtype_ratios,
            profile_result.input_dtype_source,
            (
                self.fallback_input_dtype.value
                if self.fallback_input_dtype is not None
                else None
            ),
            fallback_applied_flops,
        )

    def _log_compile_filter(self, result: CompilerFlopsFilterResult):
        if result.excluded_compile_flops > 0:
            excluded_ratio = (
                result.excluded_compile_flops / result.raw_total_flops
                if result.raw_total_flops > 0
                else 0.0
            )
            self.logger.info(
                "Compiler-only FLOPS filtered: raw_total_flops=%.0f, "
                "excluded_compile_flops=%.0f, excluded_ratio=%.6f, "
                "filtered_total_flops=%.0f, excluded_by_scope=%s",
                result.raw_total_flops,
                result.excluded_compile_flops,
                excluded_ratio,
                result.filtered_total_flops,
                list(result.excluded_by_scope.items()),
            )
        if result.ambiguous_kept_flops > 0:
            self.logger.warning(
                "Compiler-related FLOPS kept without deduction: "
                "decision=kept_not_deducted, "
                "reason=unknown_compiler_related_scope, total_flops=%.0f, "
                "examples=%s",
                result.ambiguous_kept_flops,
                list(result.ambiguous_examples),
            )
        if result.parent_unknown_kept_flops > 0:
            self.logger.info(
                "Parent-unknown FLOPS kept: decision=kept_not_deducted, "
                "window_flops=%.0f, mean_per_active_step=%.0f, examples=%s",
                result.parent_unknown_kept_flops,
                result.parent_unknown_kept_flops / INITIAL_PROFILE_ACTIVE_STEPS,
                list(result.parent_unknown_examples),
            )

    def _close_profiler(self):
        if self.prof is None:
            return
        self.prof.__exit__(None, None, None)
        self.prof = None

    def _close_incomplete_profiler(self):
        if self.prof is None:
            return
        self._close_profiler()
        self.flops_state.invalidate("insufficient_profile_steps")
        self.logger.warning(
            "FLOPS startup estimate unavailable: insufficient_profile_steps "
            "(completed_steps=%s, required_steps=%s)",
            self.prof_step_count,
            INITIAL_PROFILE_LAST_STEP,
        )

    def after_train(self, *args, **kwargs):
        self._close_incomplete_profiler()

    def after_eval(self, *args, **kwargs):
        self._close_incomplete_profiler()

    def out_off_data(self, *args, **kwargs):
        self._close_incomplete_profiler()

    def end(self, *args, **kwargs):
        self._close_incomplete_profiler()
