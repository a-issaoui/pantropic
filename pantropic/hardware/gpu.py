"""Pantropic - GPU Monitoring.

Uses pynvml (nvidia-ml-py3) for reliable GPU detection and VRAM monitoring.
Falls back to psutil for system memory if no GPU available.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

from pantropic.observability.logging import get_logger

log = get_logger("gpu")

# Try to import pynvml
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    log.warning("pynvml not available - GPU monitoring disabled")


@dataclass
class GPUInfo:
    """GPU hardware information."""
    index: int
    name: str
    total_vram_gb: float
    driver_version: str
    compute_capability: str | None = None


@dataclass
class GPUStatus:
    """Current GPU status."""
    index: int
    total_vram_gb: float
    used_vram_gb: float
    free_vram_gb: float
    temperature_c: int | None = None
    utilization_percent: int | None = None

    @property
    def available_vram_gb(self) -> float:
        """Get available VRAM with safety margin."""
        return max(0, self.free_vram_gb - 0.5)  # 0.5GB safety margin


class GPUMonitor:
    """Monitor GPU status and VRAM usage using pynvml."""

    def __init__(self) -> None:
        """Initialize GPU monitor."""
        self._gpus: list[GPUInfo] = []
        self._has_gpu = False
        self._nvml_initialized = False
        self._init_nvml()

    def _init_nvml(self) -> None:
        """Initialize NVML library."""
        if not PYNVML_AVAILABLE:
            log.warning("No GPU support - pynvml not installed")
            return

        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            self._detect_gpus()
        except pynvml.NVMLError as e:
            log.warning(f"Failed to initialize NVML: {e}")

    def _detect_gpus(self) -> None:
        """Detect available GPUs using pynvml."""
        if not self._nvml_initialized:
            return

        try:
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")

                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_vram_gb = memory_info.total / (1024 ** 3)

                gpu = GPUInfo(
                    index=i,
                    name=name,
                    total_vram_gb=total_vram_gb,
                    driver_version=driver_version,
                    compute_capability=None,
                )
                self._gpus.append(gpu)
                log.info(f"Detected GPU {i}: {name} ({total_vram_gb:.1f}GB)")

            self._has_gpu = len(self._gpus) > 0

        except pynvml.NVMLError as e:
            log.warning(f"Failed to detect GPUs: {e}")

    @property
    def has_gpu(self) -> bool:
        """Check if GPU is available."""
        return self._has_gpu

    @property
    def gpus(self) -> list[GPUInfo]:
        """Get list of detected GPUs."""
        return self._gpus

    @property
    def primary_gpu(self) -> GPUInfo | None:
        """Get primary GPU (index 0)."""
        return self._gpus[0] if self._gpus else None

    @property
    def total_vram_gb(self) -> float:
        """Get total VRAM across all GPUs."""
        return sum(gpu.total_vram_gb for gpu in self._gpus)

    def get_status(self, gpu_index: int = 0) -> GPUStatus | None:
        """Get current GPU status with real-time VRAM info.

        Args:
            gpu_index: GPU index to query

        Returns:
            GPU status or None if unavailable
        """
        if not self._nvml_initialized or gpu_index >= len(self._gpus):
            return None

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # Get temperature
            temperature = None
            with contextlib.suppress(pynvml.NVMLError):
                temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )

            # Get utilization
            utilization = None
            try:
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = util_rates.gpu
            except pynvml.NVMLError:
                pass

            return GPUStatus(
                index=gpu_index,
                total_vram_gb=memory_info.total / (1024 ** 3),
                used_vram_gb=memory_info.used / (1024 ** 3),
                free_vram_gb=memory_info.free / (1024 ** 3),
                temperature_c=temperature,
                utilization_percent=utilization,
            )

        except pynvml.NVMLError as e:
            log.warning(f"Failed to get GPU status: {e}")
            # Fallback: return static info
            gpu = self._gpus[gpu_index]
            return GPUStatus(
                index=gpu_index,
                total_vram_gb=gpu.total_vram_gb,
                used_vram_gb=0,
                free_vram_gb=gpu.total_vram_gb,
            )

    @property
    def available_vram_gb(self) -> float:
        """Get available VRAM on primary GPU (real-time)."""
        status = self.get_status(0)
        return status.available_vram_gb if status else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        status = self.get_status(0)
        return {
            "has_gpu": self._has_gpu,
            "gpu_count": len(self._gpus),
            "gpus": [
                {
                    "index": gpu.index,
                    "name": gpu.name,
                    "total_vram_gb": gpu.total_vram_gb,
                    "compute_capability": gpu.compute_capability,
                }
                for gpu in self._gpus
            ],
            "current_status": {
                "total_vram_gb": status.total_vram_gb if status else 0,
                "used_vram_gb": status.used_vram_gb if status else 0,
                "free_vram_gb": status.free_vram_gb if status else 0,
                "available_vram_gb": status.available_vram_gb if status else 0,
                "temperature_c": status.temperature_c if status else None,
                "utilization_percent": status.utilization_percent if status else None,
            } if status else None,
        }

    def __del__(self) -> None:
        """Cleanup NVML on destruction."""
        if self._nvml_initialized:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
