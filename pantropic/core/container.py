"""Pantropic - Dependency Injection Container.

Provides lazy-initialized services with proper lifecycle management.
"""

from __future__ import annotations

import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pantropic.core.config import Config
from pantropic.observability.logging import get_logger, setup_logging

if TYPE_CHECKING:
    from pantropic.hardware.gpu import GPUMonitor
    from pantropic.hardware.vram import VRAMProfiler
    from pantropic.inference.engine import InferenceEngine
    from pantropic.model_manager.registry import ModelRegistry

log = get_logger("container")


class Container:
    """Dependency injection container for Pantropic services.

    All services are lazily initialized on first access.
    Use `async with Container.create(config)` for proper lifecycle.
    """

    def __init__(self, config: Config) -> None:
        """Initialize container with configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.shutdown_event = asyncio.Event()

        # Lazy-initialized services
        self._gpu_monitor: GPUMonitor | None = None
        self._vram_profiler: VRAMProfiler | None = None
        self._model_registry: ModelRegistry | None = None
        self._inference_engine: InferenceEngine | None = None

        # Background tasks
        self._background_tasks: list[asyncio.Task[Any]] = []

    @classmethod
    async def create(cls, config: Config | None = None) -> Container:
        """Create and initialize container.

        Args:
            config: Configuration (loads from file if None)

        Returns:
            Initialized container
        """
        if config is None:
            config = Config.load()

        # Setup logging based on config
        setup_logging(
            level=config.server.log_level,
            json_output=False,
            log_file=Path("logs/pantropic.log") if config.server.access_log else None,
        )

        container = cls(config)
        await container.initialize()
        return container

    async def initialize(self) -> None:
        """Initialize core services."""
        log.info("Initializing Pantropic services...")

        # Initialize GPU monitoring first
        _ = self.gpu_monitor

        # Discover models
        if self._model_registry is None:
            self._model_registry = await self._create_model_registry()

        await self._model_registry.discover()

        log.info(f"Discovered {len(self._model_registry.list_models())} models")

    @property
    def gpu_monitor(self) -> GPUMonitor:
        """Get GPU monitor (lazy init)."""
        if self._gpu_monitor is None:
            from pantropic.hardware.gpu import GPUMonitor
            self._gpu_monitor = GPUMonitor()
        return self._gpu_monitor

    @property
    def vram_profiler(self) -> VRAMProfiler:
        """Get VRAM profiler (lazy init)."""
        if self._vram_profiler is None:
            from pantropic.hardware.vram import VRAMProfiler
            cache_dir = Path(".cache/vram_profiles")
            self._vram_profiler = VRAMProfiler(cache_dir)
        return self._vram_profiler

    @property
    def model_registry(self) -> ModelRegistry:
        """Get model registry."""
        if self._model_registry is None:
            msg = "Container not initialized. Call initialize() first."
            raise RuntimeError(msg)
        return self._model_registry

    @property
    def inference_engine(self) -> InferenceEngine:
        """Get inference engine (lazy init)."""
        if self._inference_engine is None:
            from pantropic.inference.engine import InferenceEngine
            self._inference_engine = InferenceEngine(
                container=self,
                config=self.config.inference,
            )
        return self._inference_engine

    async def _create_model_registry(self) -> ModelRegistry:
        """Create model registry with scanner integration."""
        from pantropic.model_manager.registry import ModelRegistry
        return ModelRegistry(
            models_dir=self.config.models.directory,
            vram_profiler=self.vram_profiler,
            gpu_monitor=self.gpu_monitor,
        )

    def start_background_tasks(self) -> None:
        """Start background workers."""
        log.info("Starting background tasks...")

        # Auto-unload worker
        task = asyncio.create_task(self._auto_unload_worker())
        self._background_tasks.append(task)

    async def _auto_unload_worker(self) -> None:
        """Background worker to unload idle models."""
        timeout = self.config.models.auto_unload_timeout

        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                # Check for idle models if inference engine exists
                if self._inference_engine:
                    await self._inference_engine.unload_idle_models(timeout)

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.exception(f"Auto-unload worker error: {e}")

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        log.info("Shutting down Pantropic...")

        # Signal shutdown
        self.shutdown_event.set()

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Unload all models
        if self._inference_engine:
            await self._inference_engine.unload_all()

        # Shutdown executor
        self.executor.shutdown(wait=True, cancel_futures=True)

        log.info("Pantropic shutdown complete")

    async def __aenter__(self) -> Container:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.shutdown()
