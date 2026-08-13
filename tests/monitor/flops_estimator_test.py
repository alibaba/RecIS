import json
import os
import tempfile
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from recis.monitor.flops_estimator import (
    InputDType,
    StepType,
    _new_tf32_setting,
    build_startup_flops_estimate,
    classify_input_dtype,
    filter_compiler_only_flops,
    profile_flops_by_input_dtype,
)


class _Event:
    def __init__(
        self,
        event_id,
        flops,
        name="",
        cpu_parent=None,
        input_shapes=None,
    ):
        self.id = event_id
        self.flops = flops
        self.name = name
        self.cpu_parent = cpu_parent
        self.input_shapes = input_shapes


class _Profiler:
    def __init__(self):
        self.trace_path = None

    def events(self):
        return [_Event(10, 100), _Event(11, 300), _Event(12, 50)]

    def export_chrome_trace(self, path):
        self.trace_path = path
        with open(path, "w", encoding="utf-8") as trace_file:
            json.dump(
                {
                    "traceEvents": [
                        {
                            "args": {
                                "External id": 10,
                                "Input type": ["float", "float"],
                            }
                        },
                        {
                            "args": {
                                "External id": "11",
                                "Input type": ["c10::Half", "c10::Half"],
                            }
                        },
                    ]
                },
                trace_file,
            )


class _TensorMetadata:
    def __init__(self, dtype):
        self.dtype = dtype


class _TreeNode:
    def __init__(self, correlation_id, *input_dtypes):
        self.correlation_id = correlation_id
        self.children = []
        self.extra_fields = SimpleNamespace(
            inputs=[_TensorMetadata(dtype) for dtype in input_dtypes]
        )


class _EventTreeProfiler:
    def __init__(self):
        tree = [
            _TreeNode(10, torch.float32, torch.float32),
            _TreeNode(11, torch.float32, torch.int64),
        ]
        self.profiler = SimpleNamespace(
            kineto_results=SimpleNamespace(experimental_event_tree=lambda: tree)
        )

    def events(self):
        return [
            _Event(10, 524288, "aten::mm"),
            _Event(11, 4096, "aten::add"),
        ]

    def export_chrome_trace(self, path):
        raise AssertionError("event-tree path must not export a Chrome trace")


class _PartialEventTreeProfiler:
    def __init__(self, chrome_trace_error=None):
        self.trace_path = None
        self.chrome_trace_error = chrome_trace_error
        self.profiler = SimpleNamespace(
            kineto_results=SimpleNamespace(
                experimental_event_tree=lambda: [
                    _TreeNode(10, torch.float32, torch.float32)
                ]
            )
        )

    def events(self):
        return [
            _Event(10, 100, "aten::mm"),
            _Event(11, 300, "aten::bmm"),
        ]

    def export_chrome_trace(self, path):
        self.trace_path = path
        if self.chrome_trace_error is not None:
            raise self.chrome_trace_error
        with open(path, "w", encoding="utf-8") as trace_file:
            json.dump(
                {
                    "traceEvents": [
                        {
                            "args": {
                                "External id": 10,
                                "Input type": ["float", "float"],
                            }
                        },
                        {
                            "args": {
                                "External id": 11,
                                "Input type": [
                                    "c10::BFloat16",
                                    "c10::BFloat16",
                                ],
                            }
                        },
                    ]
                },
                trace_file,
            )


class _ExplicitUnknownEventTreeProfiler:
    def __init__(self):
        self.profiler = SimpleNamespace(
            kineto_results=SimpleNamespace(
                experimental_event_tree=lambda: [
                    _TreeNode(10, torch.float32, torch.float32),
                    _TreeNode(11, torch.float32, torch.int64),
                ]
            )
        )

    def events(self):
        return [
            _Event(10, 100, "aten::mm"),
            _Event(11, 300, "aten::mm"),
        ]

    def export_chrome_trace(self, path):
        raise AssertionError("explicit UNKNOWN must not trigger trace fallback")


def _scope(name, parent=None):
    return _Event(None, 0, name=name, cpu_parent=parent)


def test_compiler_benchmark_descendant_is_excluded():
    profiler_step = _scope("ProfilerStep#7")
    benchmark = _scope(
        "InductorBenchmarker.benchmark_gpu (dynamo_timed)",
        profiler_step,
    )
    event = _Event(10, 120, "aten::mm", benchmark, [[2, 3], [3, 4]])

    result = filter_compiler_only_flops([event], raw_total_flops=120)

    assert result.filtered_total_flops == 0
    assert result.excluded_compile_flops == 120
    assert result.excluded_by_scope == {benchmark.name: 120}
    assert result.excluded_event_object_ids == {id(event)}


def test_aot_joint_graph_trace_descendant_is_excluded():
    profiler_step = _scope("ProfilerStep#7")
    aot_trace = _scope("aot_trace_joint_graph (dynamo_timed)", profiler_step)
    event = _Event(10, 120, "aten::bmm", aot_trace)

    result = filter_compiler_only_flops([event], raw_total_flops=120)

    assert result.filtered_total_flops == 0
    assert result.excluded_compile_flops == 120
    assert result.ambiguous_kept_flops == 0
    assert result.excluded_by_scope == {aot_trace.name: 120}


def test_compiled_runtime_descendant_is_kept_without_ambiguity():
    profiler_step = _scope("ProfilerStep#7")
    compiled_region = _scope("Torch-Compiled Region: 0/0", profiler_step)
    compiled_function = _scope("CompiledFunction", compiled_region)
    event = _Event(10, 120, "aten::mm", compiled_function)

    result = filter_compiler_only_flops([event], raw_total_flops=120)

    assert result.filtered_total_flops == 120
    assert result.excluded_compile_flops == 0
    assert result.ambiguous_kept_flops == 0
    assert result.parent_unknown_kept_flops == 0


def test_unknown_compiler_scope_is_kept_and_reported_as_ambiguous():
    profiler_step = _scope("ProfilerStep#7")
    unknown_scope = _scope("SomeNewInductorScope", profiler_step)
    event = _Event(10, 120, "aten::mm", unknown_scope)

    result = filter_compiler_only_flops([event], raw_total_flops=120)

    assert result.filtered_total_flops == 120
    assert result.ambiguous_kept_flops == 120
    assert result.ambiguous_examples[0].ancestor_name == unknown_scope.name
    assert result.ambiguous_examples[0].reason == "unknown_compiler_related_scope"


def test_parent_unknown_event_is_kept():
    event = _Event(10, 120, "aten::mm")

    result = filter_compiler_only_flops([event], raw_total_flops=120)

    assert result.filtered_total_flops == 120
    assert result.parent_unknown_kept_flops == 120
    assert result.parent_unknown_examples[0].reason == "cpu_parent_unavailable"


def test_outer_compiler_only_scope_overrides_compiled_runtime_scope():
    profiler_step = _scope("ProfilerStep#7")
    benchmark = _scope(
        "CachingAutotuner.benchmark_all_configs (dynamo_timed)",
        profiler_step,
    )
    compiled_function = _scope("CompiledFunction", benchmark)
    event = _Event(10, 120, "aten::mm", compiled_function)

    result = filter_compiler_only_flops([event], raw_total_flops=120)

    assert result.filtered_total_flops == 0
    assert result.excluded_by_scope == {benchmark.name: 120}


def test_parent_cycle_keeps_flops_and_records_quality_reason():
    parent_a = _scope("ordinary_parent_a")
    parent_b = _scope("ordinary_parent_b", parent_a)
    parent_a.cpu_parent = parent_b
    event = _Event(10, 120, "aten::mm", parent_a)

    result = filter_compiler_only_flops([event], raw_total_flops=120)

    assert result.filtered_total_flops == 120
    assert result.parent_unknown_kept_flops == 120
    assert result.parent_unknown_examples[0].reason == "cpu_parent_cycle"


def test_filter_examples_are_aggregated_and_limited():
    profiler_step = _scope("ProfilerStep#7")
    events = [
        _Event(
            index,
            index + 1,
            "aten::mm",
            _scope(f"SomeInductorScope{index}", profiler_step),
        )
        for index in range(12)
    ]
    total = sum(event.flops for event in events)

    result = filter_compiler_only_flops(events, raw_total_flops=total)

    assert len(result.ambiguous_examples) == 10
    assert result.ambiguous_kept_flops == total
    assert result.ambiguous_examples[0].total_flops == 12


def test_filtered_dtype_buckets_use_the_same_event_population():
    profiler = _EventTreeProfiler()
    profiler_step = _scope("ProfilerStep#7")
    benchmark = _scope(
        "InductorBenchmarker.benchmark_gpu (dynamo_timed)",
        profiler_step,
    )
    compiled_function = _scope("CompiledFunction", profiler_step)
    events = [
        _Event(10, 524288, "aten::mm", benchmark),
        _Event(11, 4096, "aten::add", compiled_function),
    ]
    filter_result = filter_compiler_only_flops(
        events,
        raw_total_flops=528384,
    )

    dtype_result = profile_flops_by_input_dtype(
        profiler,
        events=events,
        excluded_event_object_ids=filter_result.excluded_event_object_ids,
    )

    assert filter_result.filtered_total_flops == 4096
    assert dtype_result.flops_by_input_dtype == {InputDType.UNKNOWN: 4096}
    assert sum(dtype_result.flops_by_input_dtype.values()) == (
        filter_result.filtered_total_flops
    )


def test_filter_refuses_mismatched_raw_event_population():
    event = _Event(10, 120, "aten::mm")

    with pytest.raises(ValueError, match="events/key_averages FLOPS mismatch"):
        filter_compiler_only_flops([event], raw_total_flops=1210)


def test_real_profiler_compiler_scope_filtering_keeps_runtime_and_unknown():
    value = torch.ones((4, 4))
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True,
    ) as profiler:
        with torch.profiler.record_function(
            "InductorBenchmarker.benchmark_gpu (dynamo_timed)"
        ):
            torch.mm(value, value)
        with torch.profiler.record_function(
            "aot_trace_joint_graph (dynamo_timed)"
        ):
            torch.mm(value, value)
        with torch.profiler.record_function("CompiledFunction"):
            torch.mm(value, value)
        with torch.profiler.record_function("SomeNewInductorScope"):
            torch.mm(value, value)

    raw_total = sum(event.flops or 0 for event in profiler.key_averages())
    result = filter_compiler_only_flops(profiler.events(), raw_total)

    assert raw_total == 4 * 2 * 4**3
    assert result.excluded_compile_flops == 2 * 2 * 4**3
    assert result.filtered_total_flops == 2 * 2 * 4**3
    assert result.ambiguous_kept_flops == 2 * 4**3


def test_input_dtype_classification_is_conservative():
    assert classify_input_dtype(["float", "float"]) == InputDType.FP32
    assert classify_input_dtype(["c10::Half", "c10::Half", "Scalar"]) == InputDType.FP16
    assert classify_input_dtype(["float", "c10::Half"]) == InputDType.UNKNOWN
    assert classify_input_dtype(["c10::ComplexFloat"]) == InputDType.UNKNOWN


def test_profile_events_join_trace_input_types_by_external_id(monkeypatch, tmp_path):
    monkeypatch.delenv("STD_LOG_DIR", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    profiler = _Profiler()

    result = profile_flops_by_input_dtype(profiler)

    assert result.flops_by_input_dtype == {
        InputDType.FP32: 100,
        InputDType.FP16: 300,
        InputDType.UNKNOWN: 50,
    }
    assert result.input_dtype_source == "chrome_trace_fallback"
    assert os.path.dirname(profiler.trace_path) == str(tmp_path)
    assert not os.path.exists(profiler.trace_path)


def test_trace_cleanup_failure_does_not_discard_dtype_result(monkeypatch, tmp_path):
    monkeypatch.delenv("STD_LOG_DIR", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    profiler = _Profiler()

    with patch(
        "recis.monitor.flops_estimator.os.unlink",
        side_effect=PermissionError("cleanup denied"),
    ):
        result = profile_flops_by_input_dtype(profiler)

    assert result.flops_by_input_dtype[InputDType.FP32] == 100


def test_partial_event_tree_coverage_recovers_from_chrome_trace(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv("STD_LOG_DIR", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    profiler = _PartialEventTreeProfiler()

    result = profile_flops_by_input_dtype(profiler)

    assert result.flops_by_input_dtype == {
        InputDType.FP32: 100,
        InputDType.BF16: 300,
    }
    assert result.input_dtype_source == "chrome_trace_coverage_fallback"
    assert result.primary_exact_input_dtype_coverage == pytest.approx(0.25)
    assert result.exact_input_dtype_coverage == 1.0
    assert result.resolved_input_dtype_coverage == 1.0
    assert result.unmatched_flops == 0
    assert result.imputed_flops == 0
    assert not os.path.exists(profiler.trace_path)


def test_real_profiler_low_coverage_recovers_from_chrome_trace(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv("STD_LOG_DIR", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    value = torch.ones((4, 4))
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True,
    ) as profiler:
        torch.mm(value, value)
    events = tuple(profiler.events())
    expected_flops = sum(float(event.flops or 0) for event in events)

    with patch(
        "recis.monitor.flops_estimator._event_tree_input_dtypes",
        return_value={},
    ):
        result = profile_flops_by_input_dtype(profiler, events=events)

    assert expected_flops == 2 * 4**3
    assert result.flops_by_input_dtype == {InputDType.FP32: expected_flops}
    assert result.input_dtype_source == "chrome_trace_coverage_fallback"
    assert result.primary_exact_input_dtype_coverage == 0.0
    assert result.exact_input_dtype_coverage == 1.0
    assert result.imputed_flops == 0


def test_failed_exact_fallback_imputes_only_missing_join_from_trainer_hint(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv("STD_LOG_DIR", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    profiler = _PartialEventTreeProfiler(
        chrome_trace_error=RuntimeError("trace export failed")
    )

    result = profile_flops_by_input_dtype(
        profiler,
        fallback_input_dtype=InputDType.BF16,
    )

    assert result.flops_by_input_dtype == {
        InputDType.FP32: 100,
        InputDType.BF16: 300,
    }
    assert result.input_dtype_source == (
        "experimental_event_tree+missing_join_imputed_bf16"
    )
    assert result.primary_exact_input_dtype_coverage == pytest.approx(0.25)
    assert result.exact_input_dtype_coverage == pytest.approx(0.25)
    assert result.resolved_input_dtype_coverage == 1.0
    assert result.unmatched_flops == 300
    assert result.explicit_unknown_flops == 0
    assert result.imputed_flops == 300
    assert result.imputed_dtype == InputDType.BF16


def test_precision_hint_never_overwrites_explicit_unknown_dtype():
    profiler = _ExplicitUnknownEventTreeProfiler()

    result = profile_flops_by_input_dtype(
        profiler,
        fallback_input_dtype=InputDType.BF16,
    )

    assert result.flops_by_input_dtype == {
        InputDType.FP32: 100,
        InputDType.UNKNOWN: 300,
    }
    assert result.input_dtype_source == "experimental_event_tree"
    assert result.exact_input_dtype_coverage == pytest.approx(0.25)
    assert result.resolved_input_dtype_coverage == pytest.approx(0.25)
    assert result.unmatched_flops == 0
    assert result.explicit_unknown_flops == 300
    assert result.imputed_flops == 0
    assert result.imputed_dtype is None


def test_chrome_trace_fallback_uses_ranked_standard_log_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("STD_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "7")
    profiler = _Profiler()

    profile_flops_by_input_dtype(profiler)

    assert os.path.dirname(profiler.trace_path) == str(tmp_path / "7")
    assert not os.path.exists(profiler.trace_path)


def test_mixed_peak_is_flop_weighted_harmonic_mean():
    peak_by_dtype = {
        InputDType.FP32: 100.0,
        InputDType.FP16: 200.0,
    }

    estimate = build_startup_flops_estimate(
        total_flops=400e12,
        sample_step_type=StepType.TRAIN,
        sample_steps=1,
        eval_flops_ratio=1.0 / 3.0,
        flops_by_input_dtype={
            InputDType.FP32: 100e12,
            InputDType.FP16: 300e12,
        },
        peak_lookup=peak_by_dtype.get,
    )

    assert estimate.train_equiv_flops_per_step == 400e12
    assert estimate.train_equiv_ideal_seconds_per_step == pytest.approx(2.5)
    assert estimate.mixed_tflops_peak == pytest.approx(160.0)
    assert estimate.input_dtype_coverage == 1.0
    assert estimate.peak_coverage == 1.0
    assert estimate.mfu_invalid_reason is None


def test_eval_sample_is_normalized_to_train_equivalent_once():
    estimate = build_startup_flops_estimate(
        total_flops=100e12,
        sample_step_type=StepType.EVAL,
        sample_steps=1,
        eval_flops_ratio=1.0 / 3.0,
        scalar_tflops_peak=100.0,
    )

    assert estimate.train_equiv_flops_per_step == pytest.approx(300e12)
    assert estimate.train_equiv_ideal_seconds_per_step == pytest.approx(3.0)
    assert estimate.mixed_tflops_peak == 100.0


def test_unknown_dtype_preserves_flops_but_disables_mfu():
    estimate = build_startup_flops_estimate(
        total_flops=100e12,
        sample_step_type=StepType.TRAIN,
        sample_steps=1,
        eval_flops_ratio=1.0 / 3.0,
        flops_by_input_dtype={InputDType.UNKNOWN: 100e12},
        peak_lookup=lambda dtype: None,
    )

    assert estimate.train_equiv_flops_per_step == 100e12
    assert estimate.train_equiv_ideal_seconds_per_step is None
    assert estimate.mixed_tflops_peak is None
    assert estimate.input_dtype_coverage == 0.0
    assert estimate.peak_coverage == 0.0
    assert estimate.mfu_invalid_reason == "unknown_input_dtype"


def test_small_uncovered_fraction_reports_lower_bound_mfu():
    peak_by_dtype = {InputDType.FP32: 100.0}
    estimate = build_startup_flops_estimate(
        total_flops=100e12,
        sample_step_type=StepType.TRAIN,
        sample_steps=1,
        eval_flops_ratio=1.0 / 3.0,
        flops_by_input_dtype={
            InputDType.FP32: 99e12,
            InputDType.UNKNOWN: 1e12,
        },
        peak_lookup=peak_by_dtype.get,
        min_peak_coverage=0.99,
    )

    assert estimate.train_equiv_ideal_seconds_per_step == pytest.approx(0.99)
    assert estimate.mixed_tflops_peak == pytest.approx(100.0)
    assert estimate.input_dtype_coverage == pytest.approx(0.99)
    assert estimate.peak_coverage == pytest.approx(0.99)
    assert estimate.mfu_invalid_reason is None


def test_peak_coverage_below_threshold_disables_mfu():
    estimate = build_startup_flops_estimate(
        total_flops=100e12,
        sample_step_type=StepType.TRAIN,
        sample_steps=1,
        eval_flops_ratio=1.0 / 3.0,
        flops_by_input_dtype={
            InputDType.FP32: 98e12,
            InputDType.UNKNOWN: 2e12,
        },
        peak_lookup=lambda dtype: 100.0 if dtype == InputDType.FP32 else None,
    )

    assert estimate.train_equiv_ideal_seconds_per_step is None
    assert estimate.mixed_tflops_peak is None
    assert estimate.mfu_invalid_reason == "unknown_input_dtype"


def test_trace_join_mismatch_has_distinct_reason_and_warning():
    with patch("recis.monitor.flops_estimator.logger.warning") as warning_mock:
        estimate = build_startup_flops_estimate(
            total_flops=100e12,
            sample_step_type=StepType.TRAIN,
            sample_steps=1,
            eval_flops_ratio=1.0 / 3.0,
            flops_by_input_dtype={InputDType.FP32: 80e12},
            peak_lookup=lambda dtype: 100.0,
        )

    assert estimate.mfu_invalid_reason == "trace_join_mismatch"
    assert estimate.train_equiv_ideal_seconds_per_step is None
    assert warning_mock.call_count == 1
    assert "dtype_total=%s" in warning_mock.call_args.args[0]
    assert warning_mock.call_args.args[1:3] == (80e12, 100e12)


def test_no_countable_flops_is_not_reported_as_zero_mfu():
    estimate = build_startup_flops_estimate(
        total_flops=0,
        sample_step_type=StepType.TRAIN,
        sample_steps=6,
        eval_flops_ratio=1.0 / 3.0,
        flops_by_input_dtype={},
        peak_lookup=lambda dtype: None,
    )

    assert estimate.train_equiv_flops_per_step == 0
    assert estimate.train_equiv_ideal_seconds_per_step is None
    assert estimate.mixed_tflops_peak is None
    assert estimate.input_dtype_coverage == 0.0
    assert estimate.peak_coverage == 0.0
    assert estimate.mfu_invalid_reason == "no_countable_flops"

    scalar_estimate = build_startup_flops_estimate(
        total_flops=0,
        sample_step_type=StepType.TRAIN,
        sample_steps=6,
        eval_flops_ratio=1.0 / 3.0,
        scalar_tflops_peak=None,
    )
    assert scalar_estimate.train_equiv_ideal_seconds_per_step is None
    assert scalar_estimate.peak_coverage == 0.0
    assert scalar_estimate.mfu_invalid_reason == "no_countable_flops"


@pytest.mark.parametrize("tf32_enabled", [False, True])
def test_tf32_warning_flag_tracks_fp32_matmul_policy(tf32_enabled):
    profiler = _Profiler()
    profiler.events = lambda: [_Event(10, 100, "aten::mm")]

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "recis.monitor.flops_estimator._nvidia_tf32_supported",
                return_value=True,
            )
        )
        stack.enter_context(
            patch(
                "recis.monitor.flops_estimator._matmul_tf32_enabled",
                return_value=tf32_enabled,
            )
        )
        result = profile_flops_by_input_dtype(profiler)

    assert result.tf32_unaccounted is tf32_enabled


def test_new_tf32_setting_skips_inherited_none_values():
    assert _new_tf32_setting(
        SimpleNamespace(fp32_precision="none"),
        SimpleNamespace(fp32_precision="tf32"),
    )
    assert not _new_tf32_setting(
        SimpleNamespace(fp32_precision="ieee"),
        SimpleNamespace(fp32_precision="tf32"),
    )
    assert _new_tf32_setting(SimpleNamespace(fp32_precision="none")) is None


def test_event_tree_handles_small_mixed_int_add_as_lower_bound():
    profiler = _EventTreeProfiler()
    result = profile_flops_by_input_dtype(profiler)
    estimate = build_startup_flops_estimate(
        total_flops=528384,
        sample_step_type=StepType.TRAIN,
        sample_steps=1,
        eval_flops_ratio=1.0 / 3.0,
        flops_by_input_dtype=result.flops_by_input_dtype,
        peak_lookup=lambda dtype: 100.0 if dtype == InputDType.FP32 else None,
    )

    assert result.input_dtype_source == "experimental_event_tree"
    assert InputDType.UNKNOWN in result.flops_by_input_dtype
    assert 0.99 <= estimate.peak_coverage < 1.0
    assert estimate.train_equiv_ideal_seconds_per_step is not None
    assert estimate.mfu_invalid_reason is None
