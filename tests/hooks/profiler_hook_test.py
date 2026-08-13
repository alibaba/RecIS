import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.profiler import ProfilerAction


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
    INITIAL_PROFILE_LAST_STEP,
    USER_PROFILER_CREATE_STEP,
    _InitialProfilerHook,
)
from recis.hooks.profiler_hook import ProfilerHook  # noqa: E402
from recis.monitor.flops_estimator import StartupFlopsState, StepType  # noqa: E402


def test_public_profiler_is_created_at_step_ten():
    hook = ProfilerHook(wait=1, warmup=1, active=1, repeat=1)
    fake_profiler = MagicMock()

    with patch.object(hook, "get_prof_schedule", return_value=fake_profiler):
        for _ in range(1, USER_PROFILER_CREATE_STEP):
            hook.before_step()
            hook.after_step()
            assert hook.prof is None

        hook.before_step()
        assert hook.prof is fake_profiler
        hook.after_step()

    fake_profiler.step.assert_called_once_with()


def test_public_profiler_requires_positive_wait_for_safe_handoff():
    with pytest.raises(
        ValueError,
        match="ProfilerHook wait must be greater than 0",
    ):
        ProfilerHook(wait=0)


def test_public_profiler_schedule_uses_only_user_configuration():
    hook = ProfilerHook(wait=3, warmup=4, active=2, repeat=1)
    scheduler = MagicMock()

    with patch(
        "recis.hooks.profiler_hook.schedule",
        return_value=scheduler,
    ) as schedule_mock:
        with patch(
            "recis.hooks.profiler_hook.profile",
            return_value=MagicMock(),
        ):
            hook.get_prof_schedule()

    schedule_mock.assert_called_once_with(
        wait=3,
        warmup=4,
        active=2,
        repeat=1,
    )


def test_real_profilers_handoff_without_kineto_overlap():
    with patch.dict(os.environ, {"RECIS_MONITOR_ON": "1"}):
        initial_hook = _InitialProfilerHook(
            collect_input_dtypes=False,
            scalar_tflops_peak=100.0,
        )
        public_hook = ProfilerHook(wait=1, warmup=1, active=1, repeat=1)
        trace_records = []

        def record_trace(prof):
            total_flops = sum(event.flops or 0 for event in prof.key_averages())
            trace_records.append((prof.step_num, total_flops))

        with patch.object(
            public_hook,
            "get_trace_handler",
            return_value=record_trace,
        ):
            for real_step in range(1, 19):
                initial_hook.before_step(is_train=True)
                public_hook.before_step(is_train=True)

                if real_step == USER_PROFILER_CREATE_STEP:
                    assert initial_hook.prof is None
                    assert public_hook.prof is not None
                    assert public_hook.prof.current_action == ProfilerAction.NONE

                value = torch.ones((real_step, real_step))
                torch.mm(value, value)

                initial_hook.after_step(is_train=True)
                public_hook.after_step(is_train=True)

                if real_step < USER_PROFILER_CREATE_STEP:
                    assert public_hook.prof is None
                if real_step == INITIAL_PROFILE_LAST_STEP:
                    assert initial_hook.prof is None
                    assert public_hook.prof is None
                elif real_step == USER_PROFILER_CREATE_STEP:
                    assert public_hook.prof.current_action == ProfilerAction.WARMUP

        # With wait=1/warmup=1/active=1, the local schedule is:
        # global step 10 wait, step 11 warmup, step 12 active.
        # The trace closes at local step_num=3 and contains only step 12.
        assert trace_records == [(3, 2 * 12**3)]


def test_real_handoff_after_short_train_window():
    """A short train block closes Initial before eval can reach the public gate."""
    with patch.dict(os.environ, {"RECIS_MONITOR_ON": "1"}):
        initial_hook = _InitialProfilerHook(
            collect_input_dtypes=False,
            scalar_tflops_peak=100.0,
        )
        public_hook = ProfilerHook(wait=1, warmup=1, active=1, repeat=1)

        with patch.object(
            public_hook,
            "get_trace_handler",
            return_value=lambda prof: None,
        ):
            for real_step in range(1, 19):
                is_train = real_step <= 3
                initial_hook.before_step(is_train=is_train)
                public_hook.before_step(is_train=is_train)

                value = torch.ones((real_step, real_step))
                torch.mm(value, value)

                initial_hook.after_step(is_train=is_train)
                public_hook.after_step(is_train=is_train)

                if real_step == 3:
                    initial_hook.after_train()
                    assert initial_hook.prof is None

                if real_step < USER_PROFILER_CREATE_STEP:
                    assert public_hook.prof is None
                elif real_step == USER_PROFILER_CREATE_STEP:
                    assert initial_hook.prof is None
                    assert public_hook.prof is not None


def test_eval_only_startup_estimates_flops_without_kineto_overlap():
    flops_state = StartupFlopsState()
    with patch.dict(os.environ, {"RECIS_MONITOR_ON": "1"}):
        initial_hook = _InitialProfilerHook(
            flops_state=flops_state,
            collect_input_dtypes=False,
            scalar_tflops_peak=100.0,
        )
        public_hook = ProfilerHook(wait=1, warmup=1, active=1, repeat=1)

        with patch.object(
            public_hook,
            "get_trace_handler",
            return_value=lambda prof: None,
        ):
            for real_step in range(1, USER_PROFILER_CREATE_STEP + 1):
                public_hook.before_step(is_train=False)

                value = torch.ones((real_step, real_step))
                torch.mm(value, value)

                initial_hook.after_step(is_train=False)
                public_hook.after_step(is_train=False)

            initial_hook.after_eval()

    assert initial_hook.prof is None
    assert public_hook.prof is not None
    assert flops_state.estimate is not None
    assert flops_state.estimate.sample_step_type == StepType.EVAL
