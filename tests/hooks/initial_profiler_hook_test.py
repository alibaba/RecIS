import os
import sys
import types
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch


os.environ.setdefault("BUILD_DOCUMENT", "1")
if "recis.info" not in sys.modules:
    info_module = types.ModuleType("recis.info")
    info_module.is_internal_enabled = lambda: False
    sys.modules["recis.info"] = info_module
if "recis.framework" not in sys.modules:
    framework_package = types.ModuleType("recis.framework")
    framework_package.__path__ = [
        str(Path(__file__).resolve().parents[2] / "recis" / "framework")
    ]
    sys.modules["recis.framework"] = framework_package

from recis.hooks.initial_profiler_hook import (  # noqa: E402
    INITIAL_PROFILE_ACTIVE_STEPS,
    INITIAL_PROFILE_FIRST_ACTIVE_STEP,
    INITIAL_PROFILE_LAST_STEP,
    INITIAL_PROFILE_SKIP_STEPS,
    INITIAL_PROFILE_WARMUP_STEPS,
    USER_PROFILER_CREATE_STEP,
    _InitialProfilerHook,
)
from recis.monitor.flops_estimator import (  # noqa: E402
    InputDType,
    ProfileFlopsResult,
    StartupFlopsState,
    StepType,
)


class _FakeEvent:
    def __init__(self, flops, event_id=1, name="aten::mm", cpu_parent=None):
        self.flops = flops
        self.id = event_id
        self.name = name
        self.cpu_parent = cpu_parent
        self.input_shapes = None


class _FakeProfiler:
    def __init__(self, total_flops, events=None):
        self.total_flops = total_flops
        self.raw_events = events
        self.enter_count = 0
        self.exit_count = 0
        self.step_count = 0

    def __enter__(self):
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit_count += 1

    def step(self):
        self.step_count += 1

    def key_averages(self):
        return [_FakeEvent(self.total_flops)]

    def events(self):
        return self.raw_events or [_FakeEvent(self.total_flops)]


def test_initial_profiler_schedule_and_average():
    fake_profiler = _FakeProfiler(total_flops=600)
    flops_state = StartupFlopsState()

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"RECIS_MONITOR_ON": "1"}))
        schedule_mock = stack.enter_context(
            patch(
                "recis.hooks.initial_profiler_hook.schedule",
                return_value=MagicMock(),
            )
        )
        profile_mock = stack.enter_context(
            patch(
                "recis.hooks.initial_profiler_hook.profile",
                return_value=fake_profiler,
            )
        )
        hook = _InitialProfilerHook(
            flops_state=flops_state,
            collect_input_dtypes=False,
            scalar_tflops_peak=100.0,
        )
        for _ in range(INITIAL_PROFILE_LAST_STEP):
            hook.after_step(is_train=True)

    schedule_mock.assert_called_once_with(
        wait=0,
        warmup=INITIAL_PROFILE_WARMUP_STEPS,
        active=INITIAL_PROFILE_ACTIVE_STEPS,
        repeat=1,
        skip_first=INITIAL_PROFILE_SKIP_STEPS,
    )
    assert fake_profiler.enter_count == 1
    assert fake_profiler.step_count == INITIAL_PROFILE_LAST_STEP
    assert fake_profiler.exit_count == 1
    assert hook.prof is None
    assert profile_mock.call_args.kwargs["record_shapes"] is False
    assert flops_state.estimate.train_equiv_flops_per_step == (
        600 / INITIAL_PROFILE_ACTIVE_STEPS
    )
    assert flops_state.estimate.sample_step_type == StepType.TRAIN
    assert flops_state.estimate.mixed_tflops_peak == 100.0


def test_initial_profiler_preserves_fractional_flops():
    fake_profiler = _FakeProfiler(total_flops=600.5)
    flops_state = StartupFlopsState()

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"RECIS_MONITOR_ON": "1"}))
        stack.enter_context(
            patch(
                "recis.hooks.initial_profiler_hook.schedule",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "recis.hooks.initial_profiler_hook.profile",
                return_value=fake_profiler,
            )
        )
        hook = _InitialProfilerHook(
            flops_state=flops_state,
            collect_input_dtypes=False,
            scalar_tflops_peak=100.0,
        )
        for _ in range(INITIAL_PROFILE_LAST_STEP):
            hook.after_step(is_train=True)

    assert flops_state.estimate.train_equiv_flops_per_step == (
        600.5 / INITIAL_PROFILE_ACTIVE_STEPS
    )


def test_initial_profiler_logs_dtype_distribution_without_fp32_basis_message():
    fake_profiler = _FakeProfiler(total_flops=600)
    flops_state = StartupFlopsState()
    profile_result = ProfileFlopsResult(
        flops_by_input_dtype={
            InputDType.BF16: 450,
            InputDType.FP32: 150,
        },
        input_dtype_source="experimental_event_tree",
    )

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"RECIS_MONITOR_ON": "1"}))
        stack.enter_context(
            patch(
                "recis.hooks.initial_profiler_hook.schedule",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "recis.hooks.initial_profiler_hook.profile",
                return_value=fake_profiler,
            )
        )
        stack.enter_context(
            patch(
                "recis.hooks.initial_profiler_hook.profile_flops_by_input_dtype",
                return_value=profile_result,
            )
        )
        info_mock = stack.enter_context(
            patch("recis.hooks.initial_profiler_hook.Logger.info")
        )
        hook = _InitialProfilerHook(
            flops_state=flops_state,
            mixed_precision="bf16",
            peak_lookup=lambda dtype: {
                InputDType.BF16: 200.0,
                InputDType.FP32: 100.0,
            }.get(dtype),
        )
        for _ in range(INITIAL_PROFILE_LAST_STEP):
            hook.after_step(is_train=True)

    distribution_call = next(
        call
        for call in info_mock.call_args_list
        if call.args[0].startswith("MFU dtype distribution:")
    )
    assert distribution_call.args[1] == {
        "fp64": 0.0,
        "fp32": 50.0,
        "fp16": 0.0,
        "bf16": 150.0,
        "fp8_e4m3": 0.0,
        "fp8_e5m2": 0.0,
        "int8": 0.0,
        "unknown": 0.0,
    }
    assert distribution_call.args[2] == {
        "fp64": "0.00000000%",
        "fp32": "25.00000000%",
        "fp16": "0.00000000%",
        "bf16": "75.00000000%",
        "fp8_e4m3": "0.00000000%",
        "fp8_e5m2": "0.00000000%",
        "int8": "0.00000000%",
        "unknown": "0.00000000%",
    }
    assert distribution_call.args[3:] == (
        "experimental_event_tree",
        "bf16",
        0.0,
    )
    assert not any(
        "MFU dtype basis uses profiler-reported FP32 input dtype" in call.args[0]
        for call in info_mock.call_args_list
    )


def test_initial_profiler_filters_only_confirmed_compiler_scope_flops():
    profiler_step = _FakeEvent(0, name="ProfilerStep#7")
    benchmark = _FakeEvent(
        0,
        name="InductorBenchmarker.benchmark_gpu (dynamo_timed)",
        cpu_parent=profiler_step,
    )
    compiled_function = _FakeEvent(
        0,
        name="CompiledFunction",
        cpu_parent=profiler_step,
    )
    fake_profiler = _FakeProfiler(
        total_flops=600,
        events=[
            _FakeEvent(400, event_id=10, cpu_parent=benchmark),
            _FakeEvent(200, event_id=11, cpu_parent=compiled_function),
        ],
    )
    flops_state = StartupFlopsState()

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"RECIS_MONITOR_ON": "1"}))
        stack.enter_context(
            patch(
                "recis.hooks.initial_profiler_hook.schedule",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "recis.hooks.initial_profiler_hook.profile",
                return_value=fake_profiler,
            )
        )
        info_mock = stack.enter_context(
            patch("recis.hooks.initial_profiler_hook.Logger.info")
        )
        hook = _InitialProfilerHook(
            flops_state=flops_state,
            collect_input_dtypes=False,
            scalar_tflops_peak=100.0,
        )
        for _ in range(INITIAL_PROFILE_LAST_STEP):
            hook.after_step(is_train=True)

    assert flops_state.estimate.train_equiv_flops_per_step == (
        200 / INITIAL_PROFILE_ACTIVE_STEPS
    )
    assert any(
        "Compiler-only FLOPS filtered" in call.args[0]
        for call in info_mock.call_args_list
    )


def test_compile_filter_failure_falls_back_to_unfiltered_total():
    fake_profiler = _FakeProfiler(total_flops=600)
    fake_profiler.events = MagicMock(side_effect=RuntimeError("event read failed"))
    flops_state = StartupFlopsState()

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"RECIS_MONITOR_ON": "1"}))
        stack.enter_context(
            patch(
                "recis.hooks.initial_profiler_hook.schedule",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "recis.hooks.initial_profiler_hook.profile",
                return_value=fake_profiler,
            )
        )
        warning_mock = stack.enter_context(
            patch("recis.hooks.initial_profiler_hook.Logger.warning")
        )
        hook = _InitialProfilerHook(
            flops_state=flops_state,
            collect_input_dtypes=False,
            scalar_tflops_peak=100.0,
        )
        for _ in range(INITIAL_PROFILE_LAST_STEP):
            hook.after_step(is_train=True)

    assert flops_state.estimate.train_equiv_flops_per_step == (
        600 / INITIAL_PROFILE_ACTIVE_STEPS
    )
    assert any(
        "compile_filter_failed=%s" in call.args[0]
        for call in warning_mock.call_args_list
    )


def test_tf32_unaccounted_is_warned_once_when_estimate_is_published():
    fake_profiler = _FakeProfiler(total_flops=600)

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"RECIS_MONITOR_ON": "1"}))
        stack.enter_context(
            patch(
                "recis.hooks.initial_profiler_hook.schedule",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "recis.hooks.initial_profiler_hook.profile",
                return_value=fake_profiler,
            )
        )
        profile_flops_mock = stack.enter_context(
            patch(
                "recis.hooks.initial_profiler_hook.profile_flops_by_input_dtype",
                return_value=ProfileFlopsResult(
                    flops_by_input_dtype={InputDType.FP32: 600},
                    input_dtype_source="test",
                    tf32_unaccounted=True,
                ),
            )
        )
        warning_mock = stack.enter_context(
            patch("recis.hooks.initial_profiler_hook.Logger.warning")
        )
        hook = _InitialProfilerHook(
            collect_input_dtypes=True,
            mixed_precision="bf16",
            peak_lookup=lambda dtype: 100.0,
        )
        for _ in range(INITIAL_PROFILE_LAST_STEP):
            hook.after_step(is_train=True)

    warning_messages = [call.args[0] for call in warning_mock.call_args_list]
    assert sum("recis_tf32_unaccounted=true" in msg for msg in warning_messages) == 1
    assert profile_flops_mock.call_args.kwargs["min_exact_coverage"] == 0.99
    assert (
        profile_flops_mock.call_args.kwargs["fallback_input_dtype"]
        == InputDType.BF16
    )


def test_eval_only_startup_is_normalized_to_train_equivalent():
    fake_profiler = _FakeProfiler(total_flops=600)
    flops_state = StartupFlopsState()

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"RECIS_MONITOR_ON": "1"}))
        stack.enter_context(
            patch(
                "recis.hooks.initial_profiler_hook.schedule",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "recis.hooks.initial_profiler_hook.profile",
                return_value=fake_profiler,
            )
        )
        hook = _InitialProfilerHook(
            flops_state=flops_state,
            collect_input_dtypes=False,
            scalar_tflops_peak=100.0,
        )
        for _ in range(INITIAL_PROFILE_LAST_STEP):
            hook.after_step(is_train=False)

    assert fake_profiler.step_count == INITIAL_PROFILE_LAST_STEP
    assert fake_profiler.exit_count == 1
    assert flops_state.estimate.sample_step_type == StepType.EVAL
    assert flops_state.estimate.train_equiv_flops_per_step == (
        600 / INITIAL_PROFILE_ACTIVE_STEPS / (1.0 / 3.0)
    )


def test_short_phase_closes_incomplete_window():
    fake_profiler = _FakeProfiler(total_flops=600)
    flops_state = StartupFlopsState()

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"RECIS_MONITOR_ON": "1"}))
        stack.enter_context(
            patch(
                "recis.hooks.initial_profiler_hook.schedule",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "recis.hooks.initial_profiler_hook.profile",
                return_value=fake_profiler,
            )
        )
        hook = _InitialProfilerHook(flops_state=flops_state)
        hook.after_step(is_train=False)
        hook.after_eval()
        hook.after_eval()

    assert fake_profiler.step_count == 1
    assert fake_profiler.exit_count == 1
    assert flops_state.estimate is None
    assert flops_state.invalid_reason == "insufficient_profile_steps"


def test_monitor_off_does_not_construct_schedule_or_profiler():
    flops_state = StartupFlopsState()

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"RECIS_MONITOR_ON": "0"}))
        schedule_mock = stack.enter_context(
            patch("recis.hooks.initial_profiler_hook.schedule")
        )
        profile_mock = stack.enter_context(
            patch("recis.hooks.initial_profiler_hook.profile")
        )
        hook = _InitialProfilerHook(flops_state=flops_state)
        hook.after_step(is_train=True)

    schedule_mock.assert_not_called()
    profile_mock.assert_not_called()
    assert hook.prof is None
    assert flops_state.estimate is None
    assert flops_state.invalid_reason is None


def test_auto_mixed_peak_mode_enables_record_shapes():
    fake_profiler = _FakeProfiler(total_flops=600)

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"RECIS_MONITOR_ON": "1"}))
        stack.enter_context(
            patch(
                "recis.hooks.initial_profiler_hook.schedule",
                return_value=MagicMock(),
            )
        )
        profile_mock = stack.enter_context(
            patch(
                "recis.hooks.initial_profiler_hook.profile",
                return_value=fake_profiler,
            )
        )
        hook = _InitialProfilerHook(collect_input_dtypes=True)
        hook.end()

    assert profile_mock.call_args.kwargs["record_shapes"] is True


def test_active_window_rejects_mixed_step_types():
    fake_profiler = _FakeProfiler(total_flops=600)
    flops_state = StartupFlopsState()

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"RECIS_MONITOR_ON": "1"}))
        stack.enter_context(
            patch(
                "recis.hooks.initial_profiler_hook.schedule",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "recis.hooks.initial_profiler_hook.profile",
                return_value=fake_profiler,
            )
        )
        hook = _InitialProfilerHook(flops_state=flops_state)
        for _ in range(INITIAL_PROFILE_FIRST_ACTIVE_STEP):
            hook.after_step(is_train=True)
        hook.after_step(is_train=False)

    assert fake_profiler.exit_count == 1
    assert hook.prof is None
    assert flops_state.estimate is None
    assert flops_state.invalid_reason == "mixed_startup_step_types"


def test_profiler_boundaries_match_the_sampling_contract():
    assert (
        INITIAL_PROFILE_SKIP_STEPS,
        INITIAL_PROFILE_WARMUP_STEPS,
        INITIAL_PROFILE_ACTIVE_STEPS,
        INITIAL_PROFILE_FIRST_ACTIVE_STEP,
        INITIAL_PROFILE_LAST_STEP,
        USER_PROFILER_CREATE_STEP,
    ) == (5, 1, 3, 7, 9, 10)


def test_real_profiler_excludes_warmup_and_averages_steps_seven_to_nine():
    flops_state = StartupFlopsState()

    with patch.dict(os.environ, {"RECIS_MONITOR_ON": "1"}):
        hook = _InitialProfilerHook(
            flops_state=flops_state,
            collect_input_dtypes=False,
            scalar_tflops_peak=100.0,
        )
        for size in range(1, INITIAL_PROFILE_LAST_STEP + 1):
            value = torch.ones((size, size))
            torch.mm(value, value)
            hook.after_step(is_train=True)

    expected_total = sum(
        2 * size**3
        for size in range(
            INITIAL_PROFILE_FIRST_ACTIVE_STEP,
            INITIAL_PROFILE_LAST_STEP + 1,
        )
    )
    assert flops_state.estimate.train_equiv_flops_per_step == (
        expected_total / INITIAL_PROFILE_ACTIVE_STEPS
    )


def test_auto_peak_mode_records_shapes_and_classifies_real_fp32_events():
    flops_state = StartupFlopsState()

    with patch.dict(os.environ, {"RECIS_MONITOR_ON": "1"}):
        hook = _InitialProfilerHook(
            flops_state=flops_state,
            collect_input_dtypes=True,
            peak_lookup=lambda dtype: (100.0 if dtype == InputDType.FP32 else None),
        )
        for size in range(1, INITIAL_PROFILE_LAST_STEP + 1):
            value = torch.ones((size, size))
            torch.mm(value, value)
            hook.after_step(is_train=True)

    estimate = flops_state.estimate
    assert estimate is not None
    assert estimate.precision_basis == "operator_input_dtype"
    assert estimate.input_dtype_coverage == 1.0
    assert estimate.peak_coverage == 1.0
    assert estimate.mixed_tflops_peak == 100.0
    assert set(estimate.train_equiv_flops_by_input_dtype) == {InputDType.FP32}
