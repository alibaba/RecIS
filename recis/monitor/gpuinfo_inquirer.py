"""GPU information inquirer module.

Uses torch.cuda as the primary interface to detect GPUs (NVIDIA / AMD / PPU),
retrieves GPU name and model, and provides peak FLOPS lookup based on GPU
model and data precision.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch


try:
    from recis.utils.logger import Logger

    logger = Logger("GPUInfoInquirer")
except ImportError:
    # Allows running in informal environments such as unittest and notebook
    import logging

    logger = logging.getLogger(__name__)


class GpuVendor(Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    PPU = "ppu"
    UNKNOWN = "unknown"


class Precision(Enum):
    fp64 = "fp64"
    fp32 = "fp32"
    tf32 = "tf32"
    fp16 = "fp16"
    bf16 = "bf16"
    int8 = "int8"


@dataclass
class GpuDevice:
    index: int
    name: str
    vendor: GpuVendor
    memory_total_mb: int = 0


def _ppu_fp32_tensor_override() -> bool:
    """Check whether PPU fp32 tensor-core override is enabled.

    When the environment variable ``PPU_FP32_TENSOR_OVERRIDE`` is set to
    ``"1"`` (default), the PPU device reports the higher fp32 peak FLOPS
    achieved via tensor-core mode.  Otherwise it falls back to the
    non-tensor-core baseline.
    """
    return os.environ.get("PPU_FP32_TENSOR_OVERRIDE", "1") == "1"


# ---------------------------------------------------------------------------
# Peak FLOPS in TFLOPS for each GPU model and precision.
# Sources: official NVIDIA / AMD spec sheets.
# ---------------------------------------------------------------------------
# Vendor -> peak table mapping
_PEAK_TFLOP_DATASHEET: dict[GpuVendor, dict[str, dict[Precision, float]]] = {}
_PEAK_TFLOP_DATASHEET[GpuVendor.NVIDIA] = {
    "H100 SXM": {
        Precision.fp32: 60.0,
        Precision.fp16: 990.0,
        Precision.bf16: 990.0,
        Precision.int8: 1980.0,
    },
    "H20": {
        Precision.fp32: 44.0,
        Precision.fp16: 148.0,
        Precision.bf16: 148.0,
        Precision.int8: 296.0,
    },
    "A100": {
        Precision.fp32: 19.5,
        Precision.fp16: 312.0,
        Precision.bf16: 312.0,
        Precision.int8: 624.0,
    },
    "A10": {
        Precision.fp32: 31.2,
        Precision.fp16: 125.0,
        Precision.bf16: 125.0,
        Precision.int8: 250.0,
    },
    "L40S": {
        Precision.fp32: 91.6,
        Precision.fp16: 362.0,
        Precision.bf16: 362.0,
        Precision.int8: 724.0,
    },
    "L20": {
        Precision.fp32: 59.8,
        Precision.fp16: 119.5,
        Precision.bf16: 119.5,
        Precision.int8: 239.0,
    },
    # TODO: support L20[A-Z]? seems too many for a small sheet
    "T4": {
        Precision.fp32: 8.1,
        Precision.fp16: 65.0,
        Precision.bf16: 65.0,
        Precision.int8: 130.0,
    },
    "RTX 5090": {
        Precision.fp32: 104.9,
        Precision.fp16: 839,
        Precision.bf16: 209.0,
        Precision.int8: 1677.0,
    },
}
_PEAK_TFLOP_DATASHEET[GpuVendor.PPU] = {
    "ZW810E": {
        Precision.fp32: 61.5 if _ppu_fp32_tensor_override() else 25.0,
        Precision.fp16: 123.0,
        Precision.bf16: 123.0,
        Precision.int8: 246.0,
    },
}
_PEAK_TFLOP_DATASHEET[GpuVendor.AMD] = {
    "MI308X": {
        Precision.fp32: 26.0,
        Precision.fp16: 204.0,
        Precision.bf16: 204.0,
        Precision.int8: 408.0,
    },
}

# Vendor keyword -> GpuVendor mapping (order matters: first match wins)
_VENDOR_KEYWORDS: list[tuple[str, GpuVendor]] = [
    ("NVIDIA", GpuVendor.NVIDIA),
    ("GeForce", GpuVendor.NVIDIA),
    ("Tesla", GpuVendor.NVIDIA),
    ("Quadro", GpuVendor.NVIDIA),
    ("AMD", GpuVendor.AMD),
    ("Radeon", GpuVendor.AMD),
    ("Instinct", GpuVendor.AMD),
    ("PPU", GpuVendor.PPU),
]


def _normalize_gpu_name(raw_name: str) -> str:
    """Strip common vendor prefixes to get a clean GPU model name.

    Applies prefix stripping iteratively so multi-prefix names like
    ``"AMD Instinct MI308X"`` are fully normalized to ``"MI308X"``.

    Examples::
        "NVIDIA H20"           -> "H20"
        "AMD Instinct MI308X"  -> "MI308X"
        "PPU-ZW810E"           -> "ZW810E"
    """
    name = raw_name.strip()
    vendor_prefixes = (
        # Nvidia prefixes
        "NVIDIA ",
        "Tesla ",
        "GeForce ",
        # Ali Pingtouge prefixes
        "PPU-",
        "PPU ",
        # Amd prefixes
        "AMD Instinct ",
        "Instinct ",
        "AMD ",
    )
    for prefix in vendor_prefixes:
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name.strip()


def _query_devices_via_torch() -> list[GpuDevice]:
    """Query all GPU devices using torch.cuda (works for NVIDIA, AMD, PPU)."""
    if not torch.cuda.is_available():
        logger.info("torch.cuda unavailable. Cannot detect GPU peak FLOPS.")
        return []

    device_count = torch.cuda.device_count()
    devices: list[GpuDevice] = []

    def _infer_vendor(device_name: str) -> GpuVendor:
        """Infer GPU vendor from the device name string."""
        upper_name = device_name.upper()
        for keyword, vendor in _VENDOR_KEYWORDS:
            if keyword.upper() in upper_name:
                return vendor
        return GpuVendor.UNKNOWN

    for idx in range(device_count):
        name = torch.cuda.get_device_name(idx)
        vendor = _infer_vendor(name)
        memory_total_mb = 0
        try:
            props = torch.cuda.get_device_properties(idx)
            memory_total_mb = props.total_memory // (1024 * 1024)
        except Exception:
            pass
        devices.append(
            GpuDevice(
                index=idx,
                name=name,
                vendor=vendor,
                memory_total_mb=memory_total_mb,
            )
        )
    return devices


def _query_devices_via_smi() -> list[GpuDevice]:
    pass  # unimplemented yet


@dataclass
class GpuInfoInquirer:
    """Detects local GPUs via torch.cuda and provides peak FLOPS lookup.

    Usage::

        inquirer = GpuInfoInquirer()
        # Get all detected GPU devices
        for device in inquirer.devices:
            print(device.name, device.vendor)

        # Get peak TFLOPS for the first GPU at bf16 precision
        peak = inquirer.get_peak_tflops(device_index=0, precision=DataPrecision.BF16)
        print(f"Peak BF16: {peak} TFLOPS")
    """

    devices: list[GpuDevice] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.devices = _query_devices_via_torch()
        if self.devices:
            logger.debug(
                f"Detected {len(self.devices)} GPU(s): , ".join(
                    f"{d.name} ({d.vendor.value})" for d in self.devices
                )
            )
        else:
            logger.warning("No GPU detected on this machine.")

    @property
    def vendor(self) -> GpuVendor:
        """Return the vendor of the first GPU, or UNKNOWN if none detected."""
        if self.devices:
            return self.devices[0].vendor
        return GpuVendor.UNKNOWN

    def get_device(self, device_index: int = 0) -> Optional[GpuDevice]:
        """Get a GPU device by its index."""
        if 0 <= device_index < len(self.devices):
            return self.devices[device_index]
        return None

    def get_gpu_name(self, device_index: int = 0) -> Optional[str]:
        """Get the raw GPU name string for a given device index."""
        device = self.get_device(device_index)
        return device.name if device else None

    def get_peak_tflops(
        self,
        device_index: int = 0,
        precision: Precision = Precision.fp32,
    ) -> Optional[float]:
        """Get peak TFLOPS for a GPU at the specified data precision.

        Args:
            device_index: GPU device index (default 0).
            precision: Data precision enum (default FP32).

        Returns:
            Peak TFLOPS as float, or None if GPU model is not found
            in the built-in table or the precision is not available.
        """
        device = self.get_device(device_index)
        if device is None:
            logger.warning("Device index %d not found.", device_index)
            return None

        normalized = _normalize_gpu_name(device.name)
        peak_table = _PEAK_TFLOP_DATASHEET.get(device.vendor)
        if peak_table is None:
            logger.warning(
                "No peak FLOPS table for vendor '%s' (GPU: '%s').",
                device.vendor.value,
                device.name,
            )
            return None

        def _match_gpu_model(
            normalized: str, peak_table: dict[str, dict[str, float]]
        ) -> Optional[str]:
            if normalized in peak_table:
                return normalized
            best_match: Optional[str] = None
            best_length = 0
            for model_key in peak_table:
                if model_key in normalized and len(model_key) > best_length:
                    best_match = model_key
                    best_length = len(model_key)
            return best_match

        matched_model = _match_gpu_model(normalized, peak_table)
        if matched_model is None:
            logger.warning(
                "GPU model '%s' (normalized: '%s') not found in peak FLOPS table.",
                device.name,
                normalized,
            )
            return None

        model_specs = peak_table[matched_model]
        if precision not in model_specs:
            logger.warning(
                "Precision '%s' not available for GPU model '%s'. Available: %s",
                precision.value,
                matched_model,
                [p.value for p in model_specs.keys()],
            )
            return None

        return model_specs.get(precision, None)


Inquirer = GpuInfoInquirer()

if __name__ == "__main__":
    inquirer = GpuInfoInquirer()
    print(inquirer.devices[0])
    print(f"device len: {len(inquirer.devices)}")
    print(inquirer.get_peak_tflops(device_index=0, precision=Precision.fp32))
