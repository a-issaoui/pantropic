"""Pantropic - VRAM Allocator.

Two-phase allocation pattern from Flux:
1. Reserve - Hold VRAM space before loading
2. Commit - Confirm allocation after successful load
3. Cancel - Release reservation if load fails

Prevents race conditions when multiple models are loading.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pantropic.observability.logging import get_logger

if TYPE_CHECKING:
    from pantropic.hardware.gpu import GPUMonitor

log = get_logger("allocator")


@dataclass
class VRAMReservation:
    """A pending VRAM reservation."""
    model_id: str
    size_gb: float
    created_at: float = field(default_factory=time.monotonic)
    committed: bool = False


class VRAMAllocator:
    """Two-phase VRAM allocation manager.

    Pattern:
        reservation = allocator.reserve(model_id, size_gb)
        try:
            # ... load model ...
            allocator.commit(model_id)
        except:
            allocator.cancel(model_id)
    """

    def __init__(
        self,
        gpu_monitor: GPUMonitor,
        safety_margin_gb: float = 0.5,
        reservation_timeout_sec: float = 60.0,
    ) -> None:
        """Initialize allocator.

        Args:
            gpu_monitor: GPU monitor for VRAM checks
            safety_margin_gb: VRAM headroom to maintain
            reservation_timeout_sec: Auto-expire stale reservations
        """
        self.gpu_monitor = gpu_monitor
        self.safety_margin_gb = safety_margin_gb
        self.reservation_timeout = reservation_timeout_sec

        self._reservations: dict[str, VRAMReservation] = {}
        self._allocations: dict[str, float] = {}  # Committed allocations
        self._lock = threading.Lock()

    @property
    def available_gb(self) -> float:
        """Get available VRAM minus reservations and allocations."""
        status = self.gpu_monitor.get_status(0)
        if not status:
            return 0.0

        physical_free = status.free_vram_gb
        reserved = sum(r.size_gb for r in self._reservations.values() if not r.committed)

        return max(0, physical_free - reserved - self.safety_margin_gb)

    @property
    def total_reserved_gb(self) -> float:
        """Get total reserved VRAM."""
        return sum(r.size_gb for r in self._reservations.values())

    @property
    def total_allocated_gb(self) -> float:
        """Get total committed allocations."""
        return sum(self._allocations.values())

    def can_allocate(self, size_gb: float) -> bool:
        """Check if we can reserve the given amount."""
        return self.available_gb >= size_gb

    def reserve(self, model_id: str, size_gb: float) -> VRAMReservation | None:
        """Reserve VRAM for a model (phase 1).

        Returns:
            Reservation on success, None if insufficient VRAM
        """
        with self._lock:
            # Clean up expired reservations first
            self._cleanup_expired()

            # Check if already reserved/allocated
            if model_id in self._reservations:
                existing = self._reservations[model_id]
                log.debug(f"Reusing existing reservation for {model_id}")
                return existing

            if model_id in self._allocations:
                log.debug(f"{model_id} already allocated")
                return VRAMReservation(model_id, self._allocations[model_id], committed=True)

            # Check available VRAM
            if not self.can_allocate(size_gb):
                log.warning(f"Cannot reserve {size_gb:.2f}GB for {model_id}, only {self.available_gb:.2f}GB available")
                return None

            # Create reservation
            reservation = VRAMReservation(model_id=model_id, size_gb=size_gb)
            self._reservations[model_id] = reservation
            log.debug(f"Reserved {size_gb:.2f}GB for {model_id}")

            return reservation

    def commit(self, model_id: str) -> bool:
        """Commit a reservation (phase 2 - after successful load).

        Returns:
            True if committed successfully
        """
        with self._lock:
            if model_id not in self._reservations:
                log.warning(f"No reservation to commit for {model_id}")
                return False

            reservation = self._reservations.pop(model_id)
            reservation.committed = True
            self._allocations[model_id] = reservation.size_gb

            log.debug(f"Committed {reservation.size_gb:.2f}GB for {model_id}")
            return True

    def cancel(self, model_id: str) -> bool:
        """Cancel a reservation (if load fails).

        Returns:
            True if cancelled
        """
        with self._lock:
            if model_id in self._reservations:
                reservation = self._reservations.pop(model_id)
                log.debug(f"Cancelled reservation for {model_id} ({reservation.size_gb:.2f}GB)")
                return True
            return False

    def free(self, model_id: str) -> bool:
        """Free a committed allocation (when model unloads).

        Returns:
            True if freed
        """
        with self._lock:
            if model_id in self._allocations:
                size = self._allocations.pop(model_id)
                log.debug(f"Freed {size:.2f}GB from {model_id}")
                return True

            # Also try reservations
            if model_id in self._reservations:
                self._reservations.pop(model_id)
                return True

            return False

    def _cleanup_expired(self) -> None:
        """Remove stale reservations."""
        now = time.monotonic()
        expired = [
            mid for mid, r in self._reservations.items()
            if not r.committed and (now - r.created_at) > self.reservation_timeout
        ]
        for mid in expired:
            log.warning(f"Expiring stale reservation for {mid}")
            self._reservations.pop(mid)

    def get_stats(self) -> dict:
        """Get allocator statistics."""
        status = self.gpu_monitor.get_status(0)
        return {
            "total_vram_gb": status.total_vram_gb if status else 0,
            "free_vram_gb": status.free_vram_gb if status else 0,
            "available_gb": self.available_gb,
            "reserved_gb": self.total_reserved_gb,
            "allocated_gb": self.total_allocated_gb,
            "safety_margin_gb": self.safety_margin_gb,
            "active_reservations": len(self._reservations),
            "active_allocations": len(self._allocations),
        }
