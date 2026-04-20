import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# Handle the recis initialization process, avoid importing recis.so. The module does not depend on the interfaces provided by recis.so.
for _pkg in ("recis", "recis.monitor", "recis.utils"):
    if _pkg not in sys.modules:
        _m = types.ModuleType(_pkg)
        _m.__path__ = []
        _m.__package__ = _pkg
        sys.modules[_pkg] = _m

_spec = importlib.util.spec_from_file_location(
    "recis.monitor.gpuinfo_inquirer",
    "recis/monitor/gpuinfo_inquirer.py",
    submodule_search_locations=[],
)
_module = importlib.util.module_from_spec(_spec)
with patch.dict(sys.modules, {"recis.utils.logger": MagicMock()}):
    _spec.loader.exec_module(_module)
sys.modules["recis.monitor.gpuinfo_inquirer"] = _module

GpuDevice = _module.GpuDevice
GpuInfoInquirer = _module.GpuInfoInquirer
GpuVendor = _module.GpuVendor
Precision = _module.Precision
_normalize_gpu_name = _module._normalize_gpu_name
_query_devices_via_torch = _module._query_devices_via_torch
# Handle done


class NormalizeGpuNameTest(unittest.TestCase):
    def test_normalize_all_vendors(self):
        cases = {
            "NVIDIA H20": "H20",
            "Tesla T4": "T4",
            "GeForce RTX 5090": "RTX 5090",
            "AMD Instinct MI308X": "MI308X",
            "PPU-ZW810E": "ZW810E",
            "A100": "A100",
        }
        for raw, expected in cases.items():
            self.assertEqual(_normalize_gpu_name(raw), expected, f"Failed for: {raw}")


class QueryDevicesViaTorchTest(unittest.TestCase):
    def test_no_cuda_returns_empty(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.object(_module, "torch", mock_torch):
            self.assertEqual(_query_devices_via_torch(), [])

    def test_detects_nvidia_device(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_name.return_value = "NVIDIA H20"
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=96 * 1024 * 1024 * 1024
        )
        with patch.object(_module, "torch", mock_torch):
            devices = _query_devices_via_torch()
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].vendor, GpuVendor.NVIDIA)
        self.assertEqual(devices[0].memory_total_mb, 96 * 1024)


class GpuInfoInquirerTest(unittest.TestCase):
    def _make_inquirer(self, devices):
        with patch.object(_module, "_query_devices_via_torch", return_value=devices):
            return GpuInfoInquirer()

    def test_peak_tflops_nvidia(self):
        inquirer = self._make_inquirer(
            [
                GpuDevice(index=0, name="NVIDIA H20", vendor=GpuVendor.NVIDIA),
            ]
        )
        self.assertEqual(inquirer.get_peak_tflops(0, Precision.fp32), 44.0)

    def test_peak_tflops_amd(self):
        inquirer = self._make_inquirer(
            [
                GpuDevice(index=0, name="AMD Instinct MI308X", vendor=GpuVendor.AMD),
            ]
        )
        self.assertEqual(inquirer.get_peak_tflops(0, Precision.fp16), 653.7)

    def test_peak_tflops_ppu(self):
        inquirer = self._make_inquirer(
            [
                GpuDevice(index=0, name="PPU-ZW810E", vendor=GpuVendor.PPU),
            ]
        )
        self.assertEqual(inquirer.get_peak_tflops(0, Precision.bf16), 148.0)

    def test_peak_tflops_fuzzy_match(self):
        inquirer = self._make_inquirer(
            [
                GpuDevice(index=0, name="NVIDIA H100 SXM5", vendor=GpuVendor.NVIDIA),
            ]
        )
        self.assertEqual(inquirer.get_peak_tflops(0, Precision.fp32), 67.0)

    def test_unknown_model_returns_none(self):
        inquirer = self._make_inquirer(
            [
                GpuDevice(index=0, name="NVIDIA V100", vendor=GpuVendor.NVIDIA),
            ]
        )
        self.assertIsNone(inquirer.get_peak_tflops(0, Precision.fp32))

    def test_no_device_returns_none(self):
        inquirer = self._make_inquirer([])
        self.assertIsNone(inquirer.get_peak_tflops(0, Precision.fp32))
        self.assertEqual(inquirer.vendor, GpuVendor.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
