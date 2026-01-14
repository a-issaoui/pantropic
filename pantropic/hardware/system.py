"""Pantropic - System Monitoring.

Uses psutil for system resource monitoring (RAM, CPU, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psutil

from pantropic.observability.logging import get_logger

log = get_logger("system")


@dataclass
class SystemStatus:
    """Current system status."""
    cpu_percent: float
    cpu_count: int
    ram_total_gb: float
    ram_available_gb: float
    ram_used_percent: float
    swap_total_gb: float
    swap_used_gb: float


class SystemMonitor:
    """Monitor system resources using psutil."""

    def __init__(self) -> None:
        """Initialize system monitor."""
        self._cpu_count = psutil.cpu_count(logical=True)

    @property
    def cpu_count(self) -> int:
        """Get logical CPU count."""
        return self._cpu_count

    def get_status(self) -> SystemStatus:
        """Get current system status.

        Returns:
            Current system resource status
        """
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return SystemStatus(
            cpu_percent=cpu_percent,
            cpu_count=self._cpu_count,
            ram_total_gb=memory.total / (1024 ** 3),
            ram_available_gb=memory.available / (1024 ** 3),
            ram_used_percent=memory.percent,
            swap_total_gb=swap.total / (1024 ** 3),
            swap_used_gb=swap.used / (1024 ** 3),
        )

    @property
    def available_ram_gb(self) -> float:
        """Get available RAM in GB."""
        return psutil.virtual_memory().available / (1024 ** 3)

    @property
    def cpu_utilization(self) -> float:
        """Get current CPU utilization percentage."""
        return psutil.cpu_percent(interval=0.1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        status = self.get_status()
        return {
            "cpu_count": status.cpu_count,
            "cpu_percent": status.cpu_percent,
            "ram_total_gb": round(status.ram_total_gb, 2),
            "ram_available_gb": round(status.ram_available_gb, 2),
            "ram_used_percent": status.ram_used_percent,
        }
