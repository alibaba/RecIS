import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from recis.hooks.auto_profiler_hook import (
    AutoProfilerArguments,
    _build_auto_profiler,
    _LocalTimelineProfilerHook,
)


class TestAutoProfilerArguments(unittest.TestCase):
    def test_defaults_match_current_auto_profiler_schedule(self):
        args = AutoProfilerArguments()

        self.assertTrue(args.enabled)
        self.assertEqual(args.ranks, [0])
        self.assertEqual(
            (args.wait, args.warmup, args.active, args.repeat), (100, 1, 1, 1)
        )

    def test_invalid_schedule_raises_value_error(self):
        for kwargs in (
            {"wait": 0},
            {"warmup": -1},
            {"active": 0},
            {"repeat": 0},
            {"ranks": [False]},
            {"ranks": ["0"]},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    AutoProfilerArguments(**kwargs)


class TestBuildAutoProfiler(unittest.TestCase):
    def test_env_off_returns_none(self):
        with patch.dict(os.environ, {"RECIS_PROFILER_ON": "0"}, clear=True):
            hook = _build_auto_profiler(
                AutoProfilerArguments(), rank=0, world_size=2
            )

        self.assertIsNone(hook)

    def test_disabled_returns_none(self):
        with patch.dict(os.environ, {}, clear=True):
            hook = _build_auto_profiler(
                AutoProfilerArguments(enabled=False), rank=0, world_size=2
            )

        self.assertIsNone(hook)

    def test_unselected_rank_returns_none(self):
        with patch.dict(os.environ, {}, clear=True):
            hook = _build_auto_profiler(
                AutoProfilerArguments(ranks=[0]), rank=1, world_size=2
            )

        self.assertIsNone(hook)

    def test_selected_rank_uses_local_timeline_hook(self):
        sentinel = object()
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "recis.hooks.auto_profiler_hook._LocalTimelineProfilerHook",
                return_value=sentinel,
            ) as hook_cls:
                hook = _build_auto_profiler(
                    AutoProfilerArguments(
                        ranks=[1], wait=7, warmup=2, active=3, repeat=4
                    ),
                    rank=1,
                    world_size=2,
                )

        self.assertIs(hook, sentinel)
        hook_cls.assert_called_once_with(
            rank=1,
            wait=7,
            warmup=2,
            active=3,
            repeat=4,
        )


class TestLocalTimelineProfilerHook(unittest.TestCase):
    def _make_hook(self, tmp_dir, rank=3, run_datetime="20260818T102030Z"):
        # Redirect the fixed output path into the test sandbox.
        with patch.dict(os.environ, {"APP_ID": "xdl-app"}, clear=True):
            with patch(
                "recis.hooks.auto_profiler_hook._AUTO_PROFILER_OUTPUT_DIR",
                os.path.join(tmp_dir, "recis_auto_profiler_traces"),
            ):
                with patch("recis.hooks.auto_profiler_hook.datetime") as mock_datetime:
                    mock_datetime.now.return_value.strftime.return_value = run_datetime
                    return _LocalTimelineProfilerHook(
                        rank=rank,
                        wait=100,
                        warmup=1,
                        active=1,
                        repeat=1,
                    )

    def test_output_dir_is_fixed_to_var_log(self):
        with patch.dict(os.environ, {"APP_ID": "xdl-app"}, clear=True):
            with patch("recis.hooks.auto_profiler_hook.datetime") as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = (
                    "20260818T102030Z"
                )
                hook = _LocalTimelineProfilerHook(rank=0)

        self.assertEqual(hook.output_dir, "/var/log/recis_auto_profiler_traces")

    def test_std_log_dir_is_ignored(self):
        with patch.dict(
            os.environ,
            {"APP_ID": "xdl-app", "STD_LOG_DIR": "/tmp/user/anywhere"},
            clear=True,
        ):
            with patch("recis.hooks.auto_profiler_hook.datetime") as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = (
                    "20260818T102030Z"
                )
                hook = _LocalTimelineProfilerHook(rank=0)

        self.assertEqual(hook.output_dir, "/var/log/recis_auto_profiler_traces")

    def test_file_name_follows_log_contract(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            hook = self._make_hook(tmp_dir, rank=7)

            self.assertEqual(
                hook._build_file_name(123),
                "xdl-app-007-timeline-123-20260818T102030Z.json.gz",
            )

    def test_callback_exports_directly_to_log_dir_and_keeps_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            hook = self._make_hook(tmp_dir)
            prof = MagicMock(step_num=123)

            def export(path):
                Path(path).write_bytes(b"trace")

            prof.export_chrome_trace.side_effect = export
            hook.get_trace_handler()(prof)

            saved_file = os.path.join(
                hook.output_dir,
                "xdl-app-003-timeline-123-20260818T102030Z.json.gz",
            )
            prof.export_chrome_trace.assert_called_once_with(saved_file)
            self.assertTrue(os.path.exists(saved_file))

    def test_failover_run_datetime_prevents_filename_collision(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            first_hook = self._make_hook(tmp_dir, run_datetime="20260818T102030Z")
            second_hook = self._make_hook(tmp_dir, run_datetime="20260818T102031Z")

        self.assertNotEqual(
            first_hook._build_file_name(123), second_hook._build_file_name(123)
        )

    def test_directory_creation_failure_does_not_escape_handler(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            hook = self._make_hook(tmp_dir)
            prof = MagicMock(step_num=123)
            with patch(
                "recis.hooks.auto_profiler_hook.os.makedirs",
                side_effect=PermissionError("not writable"),
            ):
                # Logger sets propagate=False, so assertLogs must target the
                # named logger instead of the root logger.
                with self.assertLogs("ProfilerHook", level="ERROR") as logs:
                    hook.get_trace_handler()(prof)

        prof.export_chrome_trace.assert_not_called()
        self.assertTrue(any("failed" in line.lower() for line in logs.output))

    def test_stuck_export_times_out_without_blocking_training(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            hook = self._make_hook(tmp_dir)
            prof = MagicMock(step_num=123)
            prof.export_chrome_trace.side_effect = lambda path: time.sleep(5)

            with patch(
                "recis.hooks.auto_profiler_hook._TRACE_SAVE_TIMEOUT", 0.2
            ):
                with self.assertLogs("ProfilerHook", level="ERROR") as logs:
                    started = time.monotonic()
                    hook.get_trace_handler()(prof)
                    elapsed = time.monotonic() - started

        self.assertLess(elapsed, 4)
        self.assertTrue(any("timed out" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
