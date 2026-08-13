import os
import sys
import types
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest
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

from recis.hooks.monitor_report_hook import (  # noqa: E402
    MetricReportHook,
    ReportArguments,
)
from recis.monitor.flops_estimator import (  # noqa: E402
    InputDType,
    StepType,
    build_startup_flops_estimate,
)
from recis.monitor.gpuinfo_inquirer import Precision  # noqa: E402
from recis.monitor.monitor_reporter import (  # noqa: E402
    FLOPS_NAME,
    FLOPS_PEAK,
    MFU_NAME,
    QPS_NAME,
)


def _make_hook(report_args, mixed_precision=None, detected_peak=100.0):
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "recis.hooks.monitor_report_hook.filter_out_sparse_param",
                return_value={},
            )
        )
        stack.enter_context(
            patch(
                "recis.hooks.monitor_report_hook.Inquirer.get_peak_tflops",
                return_value=detected_peak,
            )
        )
        return MetricReportHook(
            model=torch.nn.Linear(2, 2),
            report_args=report_args,
            mixed_precision=mixed_precision,
        )


def _metric_calls(report_mock, metric_name):
    return [
        recorded_call
        for recorded_call in report_mock.call_args_list
        if recorded_call.args[0] == metric_name
    ]


def _set_scalar_estimate(
    hook,
    train_equiv_flops_per_step=100e12,
    peak_tflops=100.0,
    sample_step_type=StepType.TRAIN,
):
    sample_ratio = (
        1.0 if sample_step_type == StepType.TRAIN else hook.args.eval_flops_ratio
    )
    hook.flops_state.set_estimate(
        build_startup_flops_estimate(
            total_flops=train_equiv_flops_per_step * sample_ratio,
            sample_step_type=sample_step_type,
            sample_steps=1,
            eval_flops_ratio=hook.args.eval_flops_ratio,
            scalar_tflops_peak=peak_tflops,
        )
    )


def test_auto_mode_uses_input_dtype_mix_without_scalar_peak_lookup():
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "recis.hooks.monitor_report_hook.filter_out_sparse_param",
                return_value={},
            )
        )
        peak_mock = stack.enter_context(
            patch(
                "recis.hooks.monitor_report_hook.Inquirer.get_peak_tflops",
                return_value=123.0,
            )
        )
        hook = MetricReportHook(
            model=torch.nn.Linear(2, 2),
            report_args=ReportArguments(),
            mixed_precision="bf16",
        )

    peak_mock.assert_not_called()
    assert hook.precision == Precision.bf16
    assert hook.precision_source == "accelerator"
    assert hook.args.tflops_peak is None
    assert hook.scalar_tflops_peak is None
    assert hook.collect_input_dtypes


def test_explicit_precision_takes_precedence_over_mixed_precision():
    hook = _make_hook(
        ReportArguments(compute_precision=Precision.fp16),
        mixed_precision="bf16",
    )

    assert hook.precision == Precision.fp16
    assert hook.precision_source == "explicit"
    assert hook.scalar_tflops_peak == 100.0
    assert not hook.collect_input_dtypes


def test_phase_aware_flops_uses_eval_ratio_and_real_step_counts():
    hook = _make_hook(ReportArguments(tflops_peak=100.0))
    hook.interval_time = 0.0
    hook.train_steps = 2
    hook.eval_steps = 3
    _set_scalar_estimate(hook, train_equiv_flops_per_step=300e12)

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "recis.hooks.monitor_report_hook.time.monotonic",
                return_value=10.0,
            )
        )
        report_mock = stack.enter_context(
            patch("recis.hooks.monitor_report_hook.MonitorReporter.report")
        )
        hook._report_metrics()

    flops_calls = _metric_calls(report_mock, FLOPS_NAME)
    actual_flops_call = next(
        recorded_call
        for recorded_call in flops_calls
        if recorded_call.args[2]["recis_flops_type"] == FLOPS_NAME
    )
    peak_call = next(
        recorded_call
        for recorded_call in flops_calls
        if recorded_call.args[2]["recis_flops_type"] == FLOPS_PEAK
    )
    assert actual_flops_call.args[1] == 90e12
    assert actual_flops_call.args[2] == {"recis_flops_type": FLOPS_NAME}
    assert peak_call.args[1] == 100e12
    assert peak_call.args[2] == {"recis_flops_type": FLOPS_PEAK}

    mfu_call = _metric_calls(report_mock, MFU_NAME)[0]
    assert mfu_call.args[1] == 0.9
    assert mfu_call.args[2] == {"recis_mfu_type": MFU_NAME}


def test_first_report_happens_after_interval_completed():
    hook = _make_hook(ReportArguments(interval_step=2, tflops_peak=100.0))
    _set_scalar_estimate(hook)

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "recis.hooks.monitor_report_hook.time.monotonic",
                side_effect=[0.0, 1.0, 2.0],
            )
        )
        report_mock = stack.enter_context(
            patch("recis.hooks.monitor_report_hook.MonitorReporter.report")
        )
        stack.enter_context(
            patch("recis.hooks.monitor_report_hook.MonitorReporter.set_reportable")
        )
        hook.before_step()
        hook.after_step()
        assert report_mock.call_count == 0

        hook.before_step()
        hook.after_step()

    qps_call = _metric_calls(report_mock, QPS_NAME)[0]
    assert qps_call.args[1] == 1.0
    assert hook.train_steps == 0
    assert hook.eval_steps == 0


def test_partial_window_accumulates_across_loop_boundaries_until_end():
    hook = _make_hook(ReportArguments(interval_step=100, tflops_peak=100.0))
    _set_scalar_estimate(hook)

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "recis.hooks.monitor_report_hook.time.monotonic",
                side_effect=[0.0, 2.0],
            )
        )
        report_mock = stack.enter_context(
            patch("recis.hooks.monitor_report_hook.MonitorReporter.report")
        )
        stack.enter_context(
            patch("recis.hooks.monitor_report_hook.MonitorReporter.set_reportable")
        )
        hook.before_step()
        hook.after_step()
        hook.after_train()
        hook.before_step(is_train=False)
        hook.after_step(is_train=False)
        hook.after_eval()
        assert report_mock.call_count == 0
        hook.end()

    qps_call = _metric_calls(report_mock, QPS_NAME)[0]
    assert qps_call.args[1] == 1.0
    flops_call = next(
        recorded_call
        for recorded_call in _metric_calls(report_mock, FLOPS_NAME)
        if recorded_call.args[2]["recis_flops_type"] == FLOPS_NAME
    )
    assert flops_call.args[2] == {"recis_flops_type": FLOPS_NAME}


def test_unknown_peak_reports_flops_but_not_peak_or_mfu():
    hook = _make_hook(
        ReportArguments(compute_precision=Precision.fp32),
        detected_peak=None,
    )
    hook.interval_time = 0.0
    hook.train_steps = 1
    _set_scalar_estimate(hook, peak_tflops=None)

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "recis.hooks.monitor_report_hook.time.monotonic",
                return_value=1.0,
            )
        )
        report_mock = stack.enter_context(
            patch("recis.hooks.monitor_report_hook.MonitorReporter.report")
        )
        hook._report_metrics()

    flops_calls = _metric_calls(report_mock, FLOPS_NAME)
    assert len(flops_calls) == 1
    assert flops_calls[0].args[2] == {"recis_flops_type": FLOPS_NAME}
    assert _metric_calls(report_mock, MFU_NAME) == []


def test_no_profiler_sample_reports_qps_only():
    hook = _make_hook(ReportArguments(tflops_peak=100.0))
    hook.interval_time = 0.0
    hook.train_steps = 1

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "recis.hooks.monitor_report_hook.time.monotonic",
                return_value=1.0,
            )
        )
        report_mock = stack.enter_context(
            patch("recis.hooks.monitor_report_hook.MonitorReporter.report")
        )
        hook._report_metrics()

    assert len(_metric_calls(report_mock, QPS_NAME)) == 3
    assert _metric_calls(report_mock, FLOPS_NAME) == []
    assert _metric_calls(report_mock, MFU_NAME) == []


def test_no_countable_flops_reports_zero_flops_but_not_mfu():
    hook = _make_hook(ReportArguments(tflops_peak=100.0))
    hook.interval_time = 0.0
    hook.train_steps = 1
    _set_scalar_estimate(hook, train_equiv_flops_per_step=0)

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "recis.hooks.monitor_report_hook.time.monotonic",
                return_value=1.0,
            )
        )
        report_mock = stack.enter_context(
            patch("recis.hooks.monitor_report_hook.MonitorReporter.report")
        )
        hook._report_metrics()

    actual_flops_call = next(
        recorded_call
        for recorded_call in _metric_calls(report_mock, FLOPS_NAME)
        if recorded_call.args[2]["recis_flops_type"] == FLOPS_NAME
    )
    assert actual_flops_call.args[1] == 0
    assert actual_flops_call.args[2] == {"recis_flops_type": FLOPS_NAME}
    assert _metric_calls(report_mock, MFU_NAME) == []


def test_report_arguments_do_not_autodetect_fp32_peak():
    with patch("recis.hooks.monitor_report_hook.Inquirer.get_peak_tflops") as peak_mock:
        args = ReportArguments(interval_step=10)

    assert args.tflops_peak is None
    assert peak_mock.mock_calls == []


def test_report_arguments_validate_peak_coverage_threshold():
    with pytest.raises(ValueError, match="min_peak_coverage"):
        ReportArguments(min_peak_coverage=0)


@pytest.mark.parametrize(
    ("sample_step_type", "report_is_train", "expected_flops"),
    [
        (StepType.TRAIN, True, 300e12),
        (StepType.TRAIN, False, 100e12),
        (StepType.EVAL, True, 300e12),
        (StepType.EVAL, False, 100e12),
    ],
)
def test_sample_and_report_phase_combinations(
    sample_step_type,
    report_is_train,
    expected_flops,
):
    hook = _make_hook(ReportArguments(tflops_peak=100.0))
    hook.interval_time = 0.0
    hook.train_steps = int(report_is_train)
    hook.eval_steps = int(not report_is_train)
    _set_scalar_estimate(
        hook,
        train_equiv_flops_per_step=300e12,
        sample_step_type=sample_step_type,
    )

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "recis.hooks.monitor_report_hook.time.monotonic",
                return_value=1.0,
            )
        )
        report_mock = stack.enter_context(
            patch("recis.hooks.monitor_report_hook.MonitorReporter.report")
        )
        hook._report_metrics()

    actual_flops_call = next(
        recorded_call
        for recorded_call in _metric_calls(report_mock, FLOPS_NAME)
        if recorded_call.args[2]["recis_flops_type"] == FLOPS_NAME
    )
    assert actual_flops_call.args[1] == pytest.approx(expected_flops)


def test_cached_mixed_ideal_seconds_drive_mfu_without_dtype_recalculation():
    hook = _make_hook(ReportArguments())
    hook.interval_time = 0.0
    hook.train_steps = 2
    peak_by_dtype = {
        InputDType.FP32: 100.0,
        InputDType.FP16: 200.0,
    }
    hook.flops_state.set_estimate(
        build_startup_flops_estimate(
            total_flops=400e12,
            sample_step_type=StepType.TRAIN,
            sample_steps=1,
            eval_flops_ratio=hook.args.eval_flops_ratio,
            flops_by_input_dtype={
                InputDType.FP32: 100e12,
                InputDType.FP16: 300e12,
            },
            peak_lookup=peak_by_dtype.get,
        )
    )

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "recis.hooks.monitor_report_hook.time.monotonic",
                return_value=10.0,
            )
        )
        report_mock = stack.enter_context(
            patch("recis.hooks.monitor_report_hook.MonitorReporter.report")
        )
        hook._report_metrics()

    peak_call = next(
        recorded_call
        for recorded_call in _metric_calls(report_mock, FLOPS_NAME)
        if recorded_call.args[2]["recis_flops_type"] == FLOPS_PEAK
    )
    assert peak_call.args[1] == pytest.approx(160e12)
    assert _metric_calls(report_mock, MFU_NAME)[0].args[1] == 0.5


def test_quality_metadata_does_not_change_legacy_metric_tags():
    hook = _make_hook(ReportArguments())
    hook.interval_time = 0.0
    hook.train_steps = 1
    hook.flops_state.set_estimate(
        build_startup_flops_estimate(
            total_flops=100e12,
            sample_step_type=StepType.TRAIN,
            sample_steps=1,
            eval_flops_ratio=hook.args.eval_flops_ratio,
            flops_by_input_dtype={
                InputDType.FP32: 99e12,
                InputDType.UNKNOWN: 1e12,
            },
            peak_lookup=lambda dtype: (100.0 if dtype == InputDType.FP32 else None),
            tf32_unaccounted=True,
        )
    )

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "recis.hooks.monitor_report_hook.time.monotonic",
                return_value=1.0,
            )
        )
        report_mock = stack.enter_context(
            patch("recis.hooks.monitor_report_hook.MonitorReporter.report")
        )
        hook._report_metrics()

    flops_calls = _metric_calls(report_mock, FLOPS_NAME)
    actual_flops_call = next(
        recorded_call
        for recorded_call in flops_calls
        if recorded_call.args[2]["recis_flops_type"] == FLOPS_NAME
    )
    peak_call = next(
        recorded_call
        for recorded_call in flops_calls
        if recorded_call.args[2]["recis_flops_type"] == FLOPS_PEAK
    )
    mfu_call = _metric_calls(report_mock, MFU_NAME)[0]
    assert mfu_call.args[1] == 0.99
    assert actual_flops_call.args[2] == {"recis_flops_type": FLOPS_NAME}
    assert peak_call.args[2] == {"recis_flops_type": FLOPS_PEAK}
    assert mfu_call.args[2] == {"recis_mfu_type": MFU_NAME}
