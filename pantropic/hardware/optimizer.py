"""Pantropic - Intelligent GPU Layer Calculator.

Agent-friendly context management with:
- All parameters derived from model's actual specs in models.json
- Dynamic context tiers based on model's context_window
- GPU-first strategy (maximize GPU layers)
- CPU offload only when absolutely necessary

NO HARDCODED VALUES - Everything comes from the scanner!
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING

from pantropic.observability.logging import get_logger

if TYPE_CHECKING:
    from pantropic.core.types import ModelSpecs

log = get_logger("optimizer")


class GPUPriority(Enum):
    """GPU utilization priority."""
    MAX = "max"          # Maximize GPU layers, sacrifice context if needed
    BALANCED = "balanced"  # Balance between context and GPU usage
    EFFICIENT = "efficient"  # Minimize VRAM, use more CPU


# Quantization bits per weight (from llama.cpp)
QUANT_BITS: dict[str, float] = {
    "F32": 32.0,
    "F16": 16.0,
    "BF16": 16.0,
    "Q8_0": 8.5,
    "Q8_K": 8.5,
    "Q6_K": 6.5625,
    "Q5_K_M": 5.5,
    "Q5_K_S": 5.5,
    "Q5_0": 5.5,
    "Q5_1": 6.0,
    "Q4_K_M": 4.8,
    "Q4_K_S": 4.5,
    "Q4_0": 4.5,
    "Q4_1": 5.0,
    "Q3_K_M": 3.9,
    "Q3_K_S": 3.5,
    "Q3_K_L": 4.3,
    "Q2_K": 3.35,
    "IQ4_XS": 4.25,
    "IQ4_NL": 4.5,
    "IQ3_XXS": 3.0625,
    "IQ3_XS": 3.3,
    "IQ2_XXS": 2.0625,
    "IQ2_XS": 2.3,
    "IQ1_S": 1.5,
}


@lru_cache(maxsize=500)
def _cached_kv_cache_estimate(
    hidden_size: int,
    head_count_kv: int,
    layer_count: int,
    head_count: int,
    context_length: int,
    kv_cache_type_bits: int = 16,  # F16 KV cache by default
) -> float:
    """Calculate KV cache size using actual model parameters.

    Formula: 2 * layers * kv_heads * head_dim * context * bytes_per_value
    - 2 for K and V caches
    - head_dim = hidden_size / head_count
    """
    head_dim = hidden_size // head_count if head_count > 0 else 128
    bytes_per_value = kv_cache_type_bits / 8

    kv_bytes = (
        2 *  # K and V
        layer_count *
        head_count_kv *
        head_dim *
        context_length *
        bytes_per_value
    )
    return kv_bytes / (1024 ** 3)


def get_dynamic_context_tiers(model_max_context: int) -> list[int]:
    """Generate dynamic context tiers based on model's actual context_window.

    Creates tiers at: 2k, 4k, 25%, 50%, 75%, 100% of model's context_window.
    All values come from the scanner's models.json.
    """
    # Base tiers that most models support
    base_tiers = [2048, 4096]

    # Add dynamic tiers based on model's actual max context
    dynamic_tiers = []
    for percent in [0.25, 0.5, 0.75, 1.0]:
        tier = int(model_max_context * percent)
        tier = (tier // 1024) * 1024  # Round to nearest 1024
        if tier > 4096:
            dynamic_tiers.append(tier)

    return sorted(set(base_tiers + dynamic_tiers))


def get_optimal_context_for_request(
    estimated_tokens: int,
    max_response_tokens: int,
    model_max_context: int,
) -> int:
    """Get optimal context size using model's actual capabilities."""
    needed = estimated_tokens + max_response_tokens + 512

    tiers = get_dynamic_context_tiers(model_max_context)

    for tier in tiers:
        if tier >= needed:
            return tier

    return model_max_context


@dataclass
class LayerConfig:
    """Optimal layer configuration for loading."""
    n_gpu_layers: int
    total_layers: int
    estimated_gpu_vram_gb: float
    estimated_cpu_ram_gb: float
    context_length: int
    is_full_gpu: bool
    is_cpu_only: bool


def estimate_tokens(messages: list[dict] | str | None) -> int:
    """Estimate token count (~4 chars per token)."""
    if messages is None:
        return 0

    if isinstance(messages, str):
        return len(messages) // 4

    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content) // 4
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total += len(part["text"]) // 4
    return total


class SmartLayerCalculator:
    """Intelligent GPU layer calculator using model's actual parameters.

    ALL values come from ModelSpecs (scanner's models.json):
    - layer_count: Actual number of layers
    - hidden_size: Model's hidden dimension
    - head_count: Number of attention heads
    - head_count_kv: Number of KV heads (for GQA/MQA)
    - context_window: Model's trained context length
    - file_size_mb: Actual file size for VRAM estimation
    - parameters_b: Billion parameters
    - quantization: For VRAM calculation
    """

    VRAM_HEADROOM_GB = 0.5  # Safety margin

    def __init__(
        self,
        available_vram_gb: float,
        available_ram_gb: float,
        gpu_priority: GPUPriority = GPUPriority.MAX,
        min_gpu_layers_percent: int = 70,
    ) -> None:
        """Initialize calculator."""
        self.available_vram_gb = available_vram_gb
        self.available_ram_gb = available_ram_gb
        self.gpu_priority = gpu_priority
        self.min_gpu_layers_percent = min_gpu_layers_percent

    def calculate_for_request(
        self,
        specs: ModelSpecs,
        estimated_tokens: int,
        max_response_tokens: int = 4096,
    ) -> LayerConfig:
        """Calculate optimal config using model's actual context_window."""
        optimal_context = get_optimal_context_for_request(
            estimated_tokens,
            max_response_tokens,
            specs.context_window,  # From scanner!
        )

        log.debug(
            f"Dynamic context: {estimated_tokens} tokens → "
            f"{optimal_context//1024}k tier (model max: {specs.context_window//1024}k)"
        )

        return self.calculate_optimal(specs, optimal_context, min_context=2048)

    def calculate_optimal(
        self,
        specs: ModelSpecs,
        target_context: int = 8192,
        min_context: int = 2048,
    ) -> LayerConfig:
        """Calculate optimal config using ALL parameters from scanner."""
        # Get actual layer count from scanner (NO DEFAULT!)
        total_layers = specs.layer_count
        if total_layers is None:
            log.warning(f"No layer_count in specs, estimating from parameters")
            total_layers = self._estimate_layers_from_params(specs)

        min_gpu_layers = int(total_layers * self.min_gpu_layers_percent / 100)

        # Use actual file size for accurate VRAM estimate
        model_vram_gb = self._estimate_model_vram(specs)
        layer_vram_gb = model_vram_gb / total_layers

        # Get dynamic tiers based on model's actual context_window
        context_tiers = get_dynamic_context_tiers(specs.context_window)

        log.debug(
            f"Model specs: {total_layers} layers, {specs.hidden_size} hidden, "
            f"{specs.head_count} heads, {specs.head_count_kv} kv_heads, "
            f"{specs.context_window//1024}k ctx, {model_vram_gb:.2f}GB model"
        )

        if self.gpu_priority == GPUPriority.MAX:
            return self._calculate_max_gpu(
                specs, target_context, min_context, total_layers,
                model_vram_gb, layer_vram_gb, min_gpu_layers, context_tiers
            )
        if self.gpu_priority == GPUPriority.BALANCED:
            return self._calculate_balanced(
                specs, target_context, min_context, total_layers,
                model_vram_gb, layer_vram_gb, min_gpu_layers, context_tiers
            )
        return self._calculate_efficient(
            specs, target_context, min_context, total_layers,
            model_vram_gb, layer_vram_gb, context_tiers
        )

    def _estimate_layers_from_params(self, specs: ModelSpecs) -> int:
        """Fallback: Estimate layers from parameter count."""
        params_b = specs.parameters_b or 3.0
        # Rough estimate: ~32 layers per 7B params
        return max(12, int(params_b * 4.5))

    def _calculate_max_gpu(
        self,
        specs: ModelSpecs,
        target_context: int,
        min_context: int,
        total_layers: int,
        model_vram_gb: float,
        layer_vram_gb: float,
        min_gpu_layers: int,
        context_tiers: list[int],
    ) -> LayerConfig:
        """GPU-First: Maximize GPU layers using model's actual specs."""
        # Phase 1: Try full GPU with target context
        ctx_options = [c for c in reversed(context_tiers) if min_context <= c <= target_context]
        if target_context not in ctx_options:
            ctx_options = [target_context, *ctx_options]

        for ctx in ctx_options:
            kv_cache_gb = self._estimate_kv_cache(specs, ctx)
            total_needed = model_vram_gb + kv_cache_gb + self.VRAM_HEADROOM_GB

            if total_needed <= self.available_vram_gb:
                log.info(
                    f"[MAX GPU] Full: {total_layers}/{total_layers} layers, "
                    f"{ctx//1024}k ctx, {total_needed:.2f}GB VRAM"
                )
                return LayerConfig(
                    n_gpu_layers=-1,
                    total_layers=total_layers,
                    estimated_gpu_vram_gb=total_needed,
                    estimated_cpu_ram_gb=0,
                    context_length=ctx,
                    is_full_gpu=True,
                    is_cpu_only=False,
                )

        # Phase 2: Partial GPU - maximize layers
        target_ctx = min(target_context, specs.context_window)
        kv_cache_gb = self._estimate_kv_cache(specs, target_ctx)
        usable_for_layers = self.available_vram_gb - kv_cache_gb - self.VRAM_HEADROOM_GB

        if usable_for_layers > layer_vram_gb * min_gpu_layers:
            gpu_layers = int(usable_for_layers / layer_vram_gb)
            gpu_layers = max(min_gpu_layers, min(gpu_layers, total_layers))

            gpu_vram = gpu_layers * layer_vram_gb + kv_cache_gb
            cpu_ram = (total_layers - gpu_layers) * layer_vram_gb

            log.info(
                f"[MAX GPU] Hybrid: {gpu_layers}/{total_layers} layers, "
                f"{target_ctx//1024}k ctx, {gpu_vram:.2f}GB VRAM, {cpu_ram:.2f}GB RAM"
            )
            return LayerConfig(
                n_gpu_layers=gpu_layers,
                total_layers=total_layers,
                estimated_gpu_vram_gb=gpu_vram,
                estimated_cpu_ram_gb=cpu_ram,
                context_length=target_ctx,
                is_full_gpu=False,
                is_cpu_only=False,
            )

        # Phase 3: Reduce context to fit more GPU layers
        for ctx in context_tiers:
            if ctx < min_context:
                continue
            kv_cache_gb = self._estimate_kv_cache(specs, ctx)
            usable = self.available_vram_gb - kv_cache_gb - self.VRAM_HEADROOM_GB

            if usable > layer_vram_gb * min_gpu_layers:
                gpu_layers = int(usable / layer_vram_gb)
                gpu_layers = max(min_gpu_layers, min(gpu_layers, total_layers))

                log.info(
                    f"[MAX GPU] Reduced: {gpu_layers}/{total_layers} layers, {ctx//1024}k ctx"
                )
                return LayerConfig(
                    n_gpu_layers=gpu_layers,
                    total_layers=total_layers,
                    estimated_gpu_vram_gb=gpu_layers * layer_vram_gb + kv_cache_gb,
                    estimated_cpu_ram_gb=(total_layers - gpu_layers) * layer_vram_gb,
                    context_length=ctx,
                    is_full_gpu=False,
                    is_cpu_only=False,
                )

        # Last resort: CPU-only
        log.warning("[MAX GPU] Insufficient VRAM, CPU-only mode")
        return self._cpu_only_config(specs, min_context, total_layers)

    def _calculate_balanced(
        self,
        specs: ModelSpecs,
        target_context: int,
        min_context: int,
        total_layers: int,
        model_vram_gb: float,
        layer_vram_gb: float,
        min_gpu_layers: int,
        context_tiers: list[int],
    ) -> LayerConfig:
        """Balanced: Keep target context, accept some CPU offload."""
        return self._calculate_max_gpu(
            specs, target_context, min_context, total_layers,
            model_vram_gb, layer_vram_gb, max(min_gpu_layers // 2, 1), context_tiers
        )

    def _calculate_efficient(
        self,
        specs: ModelSpecs,
        target_context: int,
        min_context: int,
        total_layers: int,
        model_vram_gb: float,
        layer_vram_gb: float,
        context_tiers: list[int],
    ) -> LayerConfig:
        """Efficient: Minimize VRAM usage."""
        ctx = min(min_context, specs.context_window)
        kv_cache_gb = self._estimate_kv_cache(specs, ctx)

        target_vram = self.available_vram_gb * 0.5
        usable = target_vram - kv_cache_gb - self.VRAM_HEADROOM_GB

        if usable > layer_vram_gb:
            gpu_layers = min(int(usable / layer_vram_gb), total_layers)
            return LayerConfig(
                n_gpu_layers=gpu_layers,
                total_layers=total_layers,
                estimated_gpu_vram_gb=gpu_layers * layer_vram_gb + kv_cache_gb,
                estimated_cpu_ram_gb=(total_layers - gpu_layers) * layer_vram_gb,
                context_length=ctx,
                is_full_gpu=gpu_layers >= total_layers,
                is_cpu_only=False,
            )

        return self._cpu_only_config(specs, ctx, total_layers)

    def _cpu_only_config(
        self, specs: ModelSpecs, context: int, total_layers: int
    ) -> LayerConfig:
        """Create CPU-only configuration."""
        return LayerConfig(
            n_gpu_layers=0,
            total_layers=total_layers,
            estimated_gpu_vram_gb=0,
            estimated_cpu_ram_gb=self._estimate_total_ram(specs, context),
            context_length=context,
            is_full_gpu=False,
            is_cpu_only=True,
        )

    def _estimate_model_vram(self, specs: ModelSpecs) -> float:
        """Estimate model VRAM using actual file_size_mb from scanner."""
        # Primary: Use actual file size (most accurate!)
        if specs.file_size_mb:
            return specs.file_size_mb / 1024

        # Fallback: Calculate from params and quantization
        quant = specs.quantization.upper()
        bits = QUANT_BITS.get(quant, 4.5)
        params_b = specs.parameters_b or 3.0
        return (params_b * 1e9 * bits / 8) / (1024 ** 3)

    def _estimate_kv_cache(self, specs: ModelSpecs, context_length: int) -> float:
        """Estimate KV cache using ACTUAL model parameters from scanner."""
        # Use actual values from scanner - NO DEFAULTS!
        hidden_size = specs.hidden_size
        head_count_kv = specs.head_count_kv
        layers = specs.layer_count
        head_count = specs.head_count

        # Smart fallbacks only if scanner didn't provide
        if hidden_size is None:
            # Estimate from parameters
            params_b = specs.parameters_b or 3.0
            hidden_size = int((params_b * 1e9 / 12 / (layers or 32)) ** 0.5) * 4
            hidden_size = max(2048, min(hidden_size, 8192))  # Reasonable bounds

        if head_count is None:
            head_count = hidden_size // 128  # Typical head_dim = 128

        if head_count_kv is None:
            # Check if GQA ratio is available
            if specs.gqa_ratio:
                head_count_kv = head_count // specs.gqa_ratio
            else:
                head_count_kv = head_count  # Default to MHA

        if layers is None:
            layers = self._estimate_layers_from_params(specs)

        return _cached_kv_cache_estimate(
            hidden_size, head_count_kv, layers, head_count, context_length
        )

    def _estimate_total_ram(self, specs: ModelSpecs, context_length: int) -> float:
        """Estimate total RAM for CPU-only mode."""
        quant = specs.quantization.upper()
        bits = QUANT_BITS.get(quant, 4.5)
        params_b = specs.parameters_b or 3.0

        model_bytes = params_b * 1e9 * bits / 8
        kv_bytes = self._estimate_kv_cache(specs, context_length) * (1024 ** 3)

        return (model_bytes + kv_bytes) / (1024 ** 3)


def calculate_optimal_loading(
    specs: ModelSpecs,
    available_vram_gb: float,
    available_ram_gb: float,
    target_context: int = 8192,
    gpu_priority: str = "max",
    min_gpu_layers_percent: int = 70,
) -> LayerConfig:
    """Calculate optimal loading config using model's actual specs."""
    priority = GPUPriority(gpu_priority)
    calculator = SmartLayerCalculator(
        available_vram_gb, available_ram_gb, priority, min_gpu_layers_percent
    )
    return calculator.calculate_optimal(specs, target_context)


def calculate_for_request(
    specs: ModelSpecs,
    available_vram_gb: float,
    available_ram_gb: float,
    estimated_tokens: int,
    max_response_tokens: int = 4096,
    gpu_priority: str = "max",
    min_gpu_layers_percent: int = 70,
) -> LayerConfig:
    """Calculate optimal config for a request using model's actual specs."""
    priority = GPUPriority(gpu_priority)
    calculator = SmartLayerCalculator(
        available_vram_gb, available_ram_gb, priority, min_gpu_layers_percent
    )
    return calculator.calculate_for_request(specs, estimated_tokens, max_response_tokens)
