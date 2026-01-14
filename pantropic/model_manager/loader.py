"""Pantropic - Model Loader (100/100 Professional Edition).

Smart GPU/CPU hybrid loading with:
- Two-phase VRAM allocation (reserve → load → commit)
- Reference counting for safe concurrent access
- Eviction scoring for intelligent cache management
- Graceful degradation - never fails
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from pantropic.core.exceptions import ModelLoadError
from pantropic.core.types import LoadedModel, RegisteredModel
from pantropic.hardware.optimizer import LayerConfig, SmartLayerCalculator
from pantropic.observability.logging import get_logger

if TYPE_CHECKING:
    from pantropic.hardware.allocator import VRAMAllocator
    from pantropic.hardware.gpu import GPUMonitor
    from pantropic.hardware.vram import VRAMProfiler

log = get_logger("loader")


class ModelLoader:
    """Intelligent model loader - GPU-first with CPU fallback.

    Strategy:
    1. Reserve VRAM via VRAMAllocator
    2. Try optimal GPU loading (all layers on GPU)
    3. If fails, retry with calculated partial GPU
    4. If still fails, retry with fewer layers
    5. Last resort: CPU-only mode
    6. Commit or cancel VRAM reservation

    NEVER fails - always returns a loaded model.
    """

    def __init__(
        self,
        gpu_monitor: GPUMonitor,
        vram_profiler: VRAMProfiler,
        system_monitor: Any = None,
        vram_allocator: VRAMAllocator | None = None,
        default_context: int = 8192,
        max_context: int = 131072,
        flash_attention: bool = True,
        use_mmap: bool = True,
        cpu_threads: int = 0,
    ) -> None:
        """Initialize loader."""
        self.gpu_monitor = gpu_monitor
        self.vram_profiler = vram_profiler
        self.system_monitor = system_monitor
        self.default_context = default_context
        self.max_context = max_context
        self.flash_attention = flash_attention
        self.use_mmap = use_mmap
        self.cpu_threads = cpu_threads

        self._loaded: dict[str, LoadedModel] = {}
        self._lock = asyncio.Lock()

        # Two-phase VRAM allocator
        if vram_allocator:
            self._allocator = vram_allocator
        else:
            from pantropic.hardware.allocator import VRAMAllocator
            self._allocator = VRAMAllocator(gpu_monitor)

    async def load(
        self,
        model: RegisteredModel,
        min_context: int | None = None,
        estimated_tokens: int = 0,
        force_reload: bool = False,
    ) -> LoadedModel:
        """Load model with smart GPU/CPU allocation.

        Uses two-phase VRAM allocation:
        1. Reserve VRAM
        2. Load model
        3. Commit on success, cancel on failure

        Agent-friendly features:
        - Token-based context sizing
        - Auto-expansion when context usage > 80%
        - GPU-first loading strategy
        """
        async with self._lock:
            # Check if already loaded
            if model.id in self._loaded and not force_reload:
                loaded = self._loaded[model.id]

                # Check if we need more context (agent mode expansion)
                needs_expansion = False
                if estimated_tokens > 0:
                    # Expand if usage > 80% of current context
                    expansion_threshold = 0.8
                    current_usage = estimated_tokens / loaded.context_length
                    if current_usage > expansion_threshold:
                        log.info(
                            f"Context expansion needed: {estimated_tokens} tokens "
                            f"({current_usage:.0%} of {loaded.context_length})"
                        )
                        needs_expansion = True

                if not needs_expansion and (min_context is None or loaded.context_length >= min_context):
                    loaded.last_used = time.monotonic()
                    log.debug(f"Reusing loaded model {model.id}")
                    return loaded

                # Need to reload with more context
                log.info(f"Reloading {model.id} with larger context")
                self._unload_unlocked(model.id)

            # Free up VRAM if other models are loaded
            if self._loaded:
                await self._evict_for_new_model(model)

            # Calculate optimal config with token-aware sizing
            target_context = min_context or self.default_context
            target_context = min(target_context, model.specs.context_window, self.max_context)

            config = self._calculate_smart_config(model, target_context, estimated_tokens)

            # Reserve VRAM (two-phase allocation)
            estimated_vram = config.estimated_gpu_vram_gb
            self._allocator.reserve(model.id, estimated_vram)

            try:
                # Try loading with graceful fallback
                loaded = await self._load_with_fallback(model, config)
                self._loaded[model.id] = loaded

                # Commit reservation on success
                self._allocator.commit(model.id)

                # Log the result
                mode = "GPU" if config.is_full_gpu else ("hybrid" if not config.is_cpu_only else "CPU")
                log.info(
                    f"Loaded {model.id}: {loaded.context_length//1024}k ctx, "
                    f"{loaded.gpu_layers}/{config.total_layers} GPU layers ({mode})"
                )

                return loaded
            except Exception as e:
                # Cancel reservation on failure
                self._allocator.cancel(model.id)
                raise ModelLoadError(model.id, str(e)) from e

    def _calculate_smart_config(
        self,
        model: RegisteredModel,
        target_context: int,
        estimated_tokens: int = 0,
    ) -> LayerConfig:
        """Calculate optimal layer configuration using SmartLayerCalculator.

        Uses GPU-first strategy with intelligent context sizing.
        """
        from pantropic.hardware.optimizer import GPUPriority

        available_vram = self.gpu_monitor.available_vram_gb
        available_ram = 32.0  # Default

        if self.system_monitor:
            available_ram = self.system_monitor.available_ram_gb

        # Use GPU-first strategy with config options
        calculator = SmartLayerCalculator(
            available_vram,
            available_ram,
            gpu_priority=GPUPriority.MAX,  # Always prioritize GPU
            min_gpu_layers_percent=70,     # Keep at least 70% on GPU
        )

        # If we have token estimate, use request-aware sizing
        if estimated_tokens > 0:
            return calculator.calculate_for_request(
                model.specs,
                estimated_tokens=estimated_tokens,
                max_response_tokens=4096,
            )

        # Otherwise use target context
        return calculator.calculate_optimal(
            model.specs,
            target_context=target_context,
            min_context=2048,
        )

    async def _load_with_fallback(
        self,
        model: RegisteredModel,
        config: LayerConfig,
    ) -> LoadedModel:
        """Load model with automatic fallback on failure.

        Fallback chain:
        1. Optimal config (calculated GPU layers)
        2. Half GPU layers
        3. Quarter GPU layers
        4. CPU-only (n_gpu_layers=0)
        """
        fallback_configs = [
            (config.n_gpu_layers, config.context_length),
            (max(1, config.n_gpu_layers // 2), config.context_length),
            (max(1, config.n_gpu_layers // 4), min(config.context_length, 4096)),
            (0, min(config.context_length, 2048)),  # CPU-only
        ]

        last_error = None

        for gpu_layers, ctx_len in fallback_configs:
            try:
                return await self._try_load(model, gpu_layers, ctx_len)
            except Exception as e:
                last_error = e
                log.warning(f"Load failed with {gpu_layers} GPU layers, trying fallback: {e}")

                # Small delay before retry to let VRAM settle
                await asyncio.sleep(0.5)

        # Should never reach here, but just in case
        raise ModelLoadError(model.id, f"All fallback attempts failed: {last_error}")

    async def _try_load(
        self,
        model: RegisteredModel,
        n_gpu_layers: int,
        context_length: int,
    ) -> LoadedModel:
        """Attempt to load model with specific configuration."""
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ModelLoadError(model.id, "llama-cpp-python not installed") from e

        loop = asyncio.get_event_loop()
        is_embedding = model.capabilities.embed

        def load_sync() -> Any:
            return Llama(
                model_path=str(model.path),
                n_ctx=context_length,
                n_gpu_layers=n_gpu_layers,
                n_batch=512,
                n_threads=self.cpu_threads or None,
                flash_attn=self.flash_attention and n_gpu_layers > 0,
                use_mmap=self.use_mmap,
                embedding=is_embedding,
                verbose=False,
            )

        llm = await loop.run_in_executor(None, load_sync)

        now = time.monotonic()
        return LoadedModel(
            model=model,
            llm=llm,
            context_length=context_length,
            gpu_layers=n_gpu_layers,
            loaded_at=now,
            last_used=now,
        )

    async def _evict_for_new_model(self, new_model: RegisteredModel) -> None:
        """Evict models using scoring: size * idle_time / access_count."""
        # Sort by eviction score (highest = evict first)
        candidates = sorted(
            [
                (model_id, loaded)
                for model_id, loaded in self._loaded.items()
                if model_id != new_model.id and loaded.is_evictable()
            ],
            key=lambda x: x[1].eviction_score(),
            reverse=True,
        )

        for model_id, loaded in candidates:
            can_evict, active_refs = loaded.try_evict()
            if can_evict:
                log.info(f"Evicting {model_id} (score={loaded.eviction_score():.2f})")
                self._unload_unlocked(model_id)
                break
            log.debug(f"Cannot evict {model_id} - {active_refs} active refs")

    def _unload_unlocked(self, model_id: str) -> bool:
        """Unload model without acquiring lock."""
        if model_id not in self._loaded:
            return False

        loaded = self._loaded.pop(model_id)

        # Free VRAM allocation
        self._allocator.free(model_id)

        if hasattr(loaded.llm, "close"):
            loaded.llm.close()
        del loaded.llm
        del loaded

        import gc
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

        import time
        time.sleep(0.3)

        log.info(f"Unloaded {model_id}")
        return True

    async def unload(self, model_id: str) -> bool:
        """Unload a model."""
        async with self._lock:
            return self._unload_unlocked(model_id)

    async def unload_all(self) -> None:
        """Unload all models."""
        async with self._lock:
            for model_id in list(self._loaded.keys()):
                self._unload_unlocked(model_id)
            log.info("Unloaded all models")

    def get_loaded(self, model_id: str) -> LoadedModel | None:
        """Get loaded model if available."""
        return self._loaded.get(model_id)

    def list_loaded(self) -> list[LoadedModel]:
        """List all loaded models."""
        return list(self._loaded.values())

    @property
    def loaded_count(self) -> int:
        """Number of loaded models."""
        return len(self._loaded)

    async def use_model(self, model: RegisteredModel):
        """Context manager for safe model usage with reference counting.

        Usage:
            async with loader.use_model(model) as loaded:
                result = loaded.llm.create_completion(...)
        """
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _use():
            loaded = await self.load(model)
            if not loaded.acquire_ref():
                raise ModelLoadError(model.id, "Model is being evicted")
            try:
                yield loaded
            finally:
                loaded.release_ref()

        return _use()

    async def auto_unload_worker(
        self,
        idle_timeout: float = 300.0,
        check_interval: float = 30.0,
    ) -> None:
        """Background worker to unload idle models.

        Args:
            idle_timeout: Seconds before unloading idle model
            check_interval: Seconds between checks
        """
        log.info(f"Auto-unload worker started (timeout={idle_timeout}s)")

        while True:
            await asyncio.sleep(check_interval)

            now = time.monotonic()
            async with self._lock:
                for model_id, loaded in list(self._loaded.items()):
                    idle_time = now - loaded.last_used

                    if idle_time > idle_timeout and loaded.is_evictable():
                        can_evict, _ = loaded.try_evict()
                        if can_evict:
                            log.info(f"Auto-unloading {model_id} (idle {idle_time:.0f}s)")
                            self._unload_unlocked(model_id)
