import os
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from recis.framework.trainer import Trainer
from recis.hooks.auto_profiler_hook import AutoProfilerArguments


class _FakeProfilerHook:
    def __init__(
        self,
        wait=1,
        warmup=48,
        active=1,
        repeat=4,
        output_dir=None,
    ):
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.output_dir = output_dir


class _FakeAutoProfilerHook(_FakeProfilerHook):
    def __init__(self, rank, **kwargs):
        self.rank = rank
        super().__init__(**kwargs)


def _fake_hook(*args, **kwargs):
    # MetricReportHook instances expose flops_state / args / scalar_tflops_peak
    # to _InitialProfilerHook, so the fake must support attribute access.
    return MagicMock()


class TestTrainerProfilerHook(unittest.TestCase):
    def _make_trainer(self, hooks=None, process_index=0, num_processes=2):
        trainer = Trainer.__new__(Trainer)
        trainer.hooks = [] if hooks is None else hooks
        trainer._auto_profiler_hook = None
        trainer.args = SimpleNamespace(
            log_steps=100,
            ckpt_save_arg=None,
            save_steps=1000,
            save_every_n_windows=1,
            save_every_n_epochs=None,
            save_end=True,
            ckpt_load_arg=None,
            load_update_steps=None,
            load_update_windows=1,
            load_update_epochs=None,
        )
        trainer.saver = SimpleNamespace()
        trainer._auto_profiler_args = None
        trainer._global_step = object()
        trainer._epoch = object()
        trainer.model = object()
        trainer.mixed_precision = None
        trainer._monitor_report_args = None
        trainer.accelerator = SimpleNamespace(
            process_index=process_index,
            num_processes=num_processes,
        )
        return trainer

    def _init_hooks(self, trainer, profiler_env, build_side_effect=None):
        hook_patches = {
            "LoggerHook": _fake_hook,
            "CheckpointSaveHook": _fake_hook,
            "CheckpointLoadHook": _fake_hook,
            "_InitialProfilerHook": _fake_hook,
            "MetricReportHook": _fake_hook,
            "ProfilerHook": _FakeProfilerHook,
        }
        build_context = (
            patch(
                "recis.framework.trainer._build_auto_profiler",
                side_effect=build_side_effect,
            )
            if build_side_effect
            else nullcontext()
        )
        with patch.multiple("recis.framework.trainer", **hook_patches):
            with patch(
                "recis.hooks.auto_profiler_hook._LocalTimelineProfilerHook",
                _FakeAutoProfilerHook,
            ):
                with build_context:
                    with patch.dict(os.environ, profiler_env, clear=True):
                        Trainer.init_hooks(trainer)

    def test_profiler_is_enabled_by_default_on_main_process(self):
        trainer = self._make_trainer()

        self._init_hooks(trainer, {})

        profilers = [
            hook for hook in trainer.hooks if isinstance(hook, _FakeProfilerHook)
        ]
        self.assertEqual(len(profilers), 1)
        self.assertEqual(profilers[0].rank, 0)
        self.assertEqual(profilers[0].wait, 100)
        self.assertEqual(profilers[0].warmup, 1)
        self.assertEqual(profilers[0].active, 1)
        self.assertEqual(profilers[0].repeat, 1)
        self.assertIs(trainer._auto_profiler_hook, profilers[0])

    def test_profiler_can_be_disabled(self):
        trainer = self._make_trainer()

        self._init_hooks(trainer, {"RECIS_PROFILER_ON": "0"})

        self.assertFalse(
            any(isinstance(hook, _FakeProfilerHook) for hook in trainer.hooks)
        )

    def test_profiler_is_not_registered_on_unselected_global_rank(self):
        trainer = self._make_trainer(process_index=1)

        self._init_hooks(trainer, {"RECIS_PROFILER_ON": "1"})

        self.assertFalse(
            any(isinstance(hook, _FakeProfilerHook) for hook in trainer.hooks)
        )

    def test_profiler_can_select_nonzero_global_rank(self):
        trainer = self._make_trainer(process_index=1)
        trainer._auto_profiler_args = AutoProfilerArguments(
            ranks=[1], wait=7, warmup=2, active=3, repeat=4
        )

        self._init_hooks(trainer, {"RECIS_PROFILER_ON": "1"})

        profilers = [
            hook for hook in trainer.hooks if isinstance(hook, _FakeProfilerHook)
        ]
        self.assertEqual(len(profilers), 1)
        self.assertEqual(profilers[0].wait, 7)
        self.assertEqual(profilers[0].warmup, 2)
        self.assertEqual(profilers[0].active, 3)
        self.assertEqual(profilers[0].repeat, 4)

    def test_build_failure_does_not_break_init_hooks(self):
        trainer = self._make_trainer()

        self._init_hooks(
            trainer,
            {"RECIS_PROFILER_ON": "1"},
            build_side_effect=RuntimeError("build failed"),
        )

        self.assertIsNone(trainer._auto_profiler_hook)
        self.assertFalse(
            any(isinstance(hook, _FakeProfilerHook) for hook in trainer.hooks)
        )

    def test_constructor_profiler_prevents_auto_registration(self):
        manual_profiler = _FakeProfilerHook(output_dir="/manual")
        trainer = self._make_trainer(hooks=[manual_profiler])

        self._init_hooks(trainer, {"RECIS_PROFILER_ON": "1"})

        profilers = [
            hook for hook in trainer.hooks if isinstance(hook, _FakeProfilerHook)
        ]
        self.assertEqual(profilers, [manual_profiler])
        self.assertIsNone(trainer._auto_profiler_hook)

    def test_add_hook_replaces_auto_profiler(self):
        auto_profiler = _FakeProfilerHook(output_dir="/auto")
        manual_profiler = _FakeProfilerHook(output_dir="/manual")
        trainer = self._make_trainer(hooks=[auto_profiler])
        trainer._auto_profiler_hook = auto_profiler

        with patch("recis.framework.trainer.ProfilerHook", _FakeProfilerHook):
            Trainer.add_hook(trainer, manual_profiler)

        self.assertEqual(trainer.hooks, [manual_profiler])
        self.assertIsNone(trainer._auto_profiler_hook)


if __name__ == "__main__":
    unittest.main()
