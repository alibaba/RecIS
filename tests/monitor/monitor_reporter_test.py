import os
import unittest
from unittest.mock import MagicMock, patch


os.environ.setdefault("BUILD_DOCUMENT", "1")

from recis.monitor.monitor_reporter import MonitorReporter  # noqa: E402


class TestMonitorReporterDeviceSafety(unittest.TestCase):
    """Tests CUDA synchronization used by timing reports."""

    @patch.object(MonitorReporter, "report")
    @patch("recis.monitor.monitor_reporter.torch.cuda.current_stream")
    @patch("recis.monitor.monitor_reporter.torch.cuda.is_initialized")
    def test_report_time_skips_cuda_before_initialization(
        self, is_initialized, current_stream, report
    ):
        is_initialized.return_value = False

        with MonitorReporter.report_time("load", force=True):
            pass

        current_stream.assert_not_called()
        report.assert_called_once()

    @patch.object(MonitorReporter, "report")
    @patch("recis.monitor.monitor_reporter.torch.cuda.current_stream")
    @patch("recis.monitor.monitor_reporter.torch.cuda.is_initialized")
    def test_report_time_synchronizes_current_stream(
        self, is_initialized, current_stream, report
    ):
        is_initialized.return_value = True
        stream = MagicMock()
        current_stream.return_value = stream

        with MonitorReporter.report_time("load", force=True):
            pass

        stream.synchronize.assert_called_once_with()
        report.assert_called_once()


if __name__ == "__main__":
    unittest.main()
