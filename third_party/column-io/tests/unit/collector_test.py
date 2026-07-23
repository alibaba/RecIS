"""Unit tests for collector.py diff: static_cast + field count guard in GPUInfo.QueryGPUInfo."""

import sys
from unittest.mock import MagicMock, patch

# Mock external deps before import
sys.modules.setdefault("column_io.lib.py_interface", MagicMock())
sys.modules.setdefault("column_io.lib.interface", MagicMock())
sys.modules.setdefault("recis", MagicMock())
sys.modules.setdefault("recis.info", MagicMock())

from column_io.dataset import collector
from column_io.dataset.collector import GPUInfo

collector.SetupLogger()


class TestGPUInfoQueryParsing:
    """Tests covering the diff changes in GPUInfo.QueryGPUInfo (static_cast + field parsing)."""

    @patch("subprocess.check_output")
    def test_query_gpu_info_normal(self, mock_subprocess):
        GPUInfo._nvidia_smi_exist = True
        mock_subprocess.return_value = "0, NVIDIA A100, 85, 70, 45, 81920, 32000, 65, 250.5\n"
        result = GPUInfo.QueryGPUInfo()
        assert len(result) == 1
        assert result[0].id == "0"
        assert result[0].gpu_util == "0.8500"
        assert result[0].mem_total_MB == "81920"

    @patch("subprocess.check_output")
    def test_query_gpu_info_multi_gpu(self, mock_subprocess):
        GPUInfo._nvidia_smi_exist = True
        mock_subprocess.return_value = (
            "0, A100, 50, 40, 30, 81920, 10000, 55, 200.0\n"
            "1, A100, 60, 50, 35, 81920, 20000, 60, 220.0\n"
        )
        result = GPUInfo.QueryGPUInfo()
        assert len(result) == 2
        assert result[1].id == "1"

    @patch("subprocess.check_output")
    def test_query_gpu_info_static_cast_invalid_values(self, mock_subprocess):
        """static_cast should handle non-numeric values gracefully."""
        GPUInfo._nvidia_smi_exist = True
        mock_subprocess.return_value = "N/A, GPU, N/A, N/A, N/A, N/A, N/A, N/A, N/A\n"
        result = GPUInfo.QueryGPUInfo()
        assert len(result) == 1
        # static_cast(int, "N/A") -> int() -> 0
        assert result[0].id == "0"
        # static_cast(float, "N/A") / 100 -> 0.0 / 100 -> 0.0
        assert result[0].gpu_util == "0.0000"

    @patch("subprocess.check_output")
    def test_query_gpu_info_fewer_fields_skipped(self, mock_subprocess):
        """Lines with fewer than 9 fields should be skipped."""
        GPUInfo._nvidia_smi_exist = True
        mock_subprocess.return_value = "0, GPU, 50\n"
        result = GPUInfo.QueryGPUInfo()
        assert len(result) == 0

    @patch("subprocess.check_output")
    def test_query_gpu_info_mixed_valid_invalid_lines(self, mock_subprocess):
        GPUInfo._nvidia_smi_exist = True
        mock_subprocess.return_value = (
            "bad, line\n"
            "0, A100, 80, 70, 60, 40960, 20000, 72, 300.0\n"
        )
        result = GPUInfo.QueryGPUInfo()
        # first line skipped (fewer fields), second parsed
        assert len(result) == 1
        assert result[0].id == "0"

    def test_query_gpu_info_no_nvidia(self):
        GPUInfo._nvidia_smi_exist = False
        result = GPUInfo.QueryGPUInfo()
        assert result == []
