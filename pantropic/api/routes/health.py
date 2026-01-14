"""Health check routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from pantropic.api.app import get_container
from pantropic.core.container import Container
from pantropic.hardware.system import SystemMonitor

router = APIRouter()

# System monitor singleton
_system_monitor: SystemMonitor | None = None


def get_system_monitor() -> SystemMonitor:
    """Get or create system monitor."""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
    return _system_monitor


@router.get("/health")
async def health_check(
    request: Request,
    container: Container = Depends(get_container),
):
    """Health check endpoint."""
    models = container.model_registry.list_models()
    gpu_info = container.gpu_monitor.to_dict()
    sys_monitor = get_system_monitor()

    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_available": len(models),
        "gpu": {
            "available": gpu_info["has_gpu"],
            "count": gpu_info["gpu_count"],
        },
        "system": {
            "cpu_count": sys_monitor.cpu_count,
            "ram_available_gb": round(sys_monitor.available_ram_gb, 1),
        },
    }


@router.get("/metrics")
async def metrics(
    request: Request,
    container: Container = Depends(get_container),
):
    """System metrics endpoint."""
    models = container.model_registry.list_models()
    gpu_info = container.gpu_monitor.to_dict()
    sys_monitor = get_system_monitor()
    sys_status = sys_monitor.get_status()

    # Count models by capability
    by_capability = {
        "chat": 0,
        "embed": 0,
        "vision": 0,
        "audio": 0,
        "reasoning": 0,
        "tools": 0,
    }
    for model in models:
        caps = model.capabilities
        if caps.chat:
            by_capability["chat"] += 1
        if caps.embed:
            by_capability["embed"] += 1
        if caps.vision:
            by_capability["vision"] += 1
        if caps.audio:
            by_capability["audio"] += 1
        if caps.reasoning:
            by_capability["reasoning"] += 1
        if caps.tools:
            by_capability["tools"] += 1

    return {
        "system": {
            "cpu": {
                "count": sys_status.cpu_count,
                "percent": sys_status.cpu_percent,
            },
            "ram": {
                "total_gb": round(sys_status.ram_total_gb, 1),
                "available_gb": round(sys_status.ram_available_gb, 1),
                "used_percent": sys_status.ram_used_percent,
            },
            "gpu": gpu_info,
        },
        "models": {
            "total": len(models),
            "by_capability": by_capability,
        },
    }

