"""Pantropic - VRAM Profiling.

Provides VRAM estimation and optimal configuration calculation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pantropic.core.types import ModelSpecs, VRAMConfig
from pantropic.observability.logging import get_logger

log = get_logger("vram")


# Quantization bits per weight
QUANT_BITS: dict[str, float] = {
    # K-quants
    "Q2_K": 2.5,
    "Q3_K_S": 3.0,
    "Q3_K_M": 3.5,
    "Q3_K_L": 3.75,
    "Q4_K_S": 4.0,
    "Q4_K_M": 4.5,
    "Q4_K": 4.5,
    "Q5_K_S": 5.0,
    "Q5_K_M": 5.5,
    "Q6_K": 6.0,
    "Q8_K": 8.0,
    # Classic quants
    "Q2_0": 2.0,
    "Q4_0": 4.0,
    "Q4_1": 4.5,
    "Q5_0": 5.0,
    "Q5_1": 5.5,
    "Q8_0": 8.0,
    "Q8_1": 8.5,
    # IQ quants
    "IQ1_S": 1.5,
    "IQ2_XXS": 2.0,
    "IQ2_XS": 2.3,
    "IQ2_S": 2.5,
    "IQ3_XXS": 3.0,
    "IQ3_S": 3.5,
    "IQ4_NL": 4.0,
    "IQ4_XS": 4.25,
    # Float
    "F16": 16.0,
    "F32": 32.0,
    "BF16": 16.0,
}


@dataclass
class VRAMProfile:
    """Measured VRAM profile for a model."""
    model_id: str
    base_vram_gb: float
    kv_bytes_per_token: float
    measurements: list[tuple[int, float, int]] = field(default_factory=list)
    # measurements: [(context_length, vram_gb, gpu_layers), ...]

    def estimate_vram(self, context_length: int) -> float:
        """Estimate VRAM for given context length."""
        # Base + KV cache
        kv_cache_gb = (context_length * self.kv_bytes_per_token) / (1024**3)
        return self.base_vram_gb + kv_cache_gb

    def find_optimal_config(
        self,
        context_needed: int,
        available_vram_gb: float,
        max_context: int = 131072,
    ) -> VRAMConfig:
        """Find optimal configuration for available VRAM.

        Args:
            context_needed: Minimum required context
            available_vram_gb: Available VRAM
            max_context: Maximum context to consider

        Returns:
            Optimal VRAM configuration
        """
        # Binary search for largest fitting context
        low, high = context_needed, min(max_context, context_needed * 4)
        best_context = context_needed

        while low <= high:
            mid = (low + high) // 2
            estimated = self.estimate_vram(mid)

            if estimated <= available_vram_gb:
                best_context = mid
                low = mid + 1
            else:
                high = mid - 1

        # Get GPU layers from measurements or use max
        gpu_layers = self._interpolate_gpu_layers(best_context)

        return VRAMConfig(
            gpu_layers=gpu_layers,
            context_length=best_context,
            estimated_vram_gb=self.estimate_vram(best_context),
        )

    def _interpolate_gpu_layers(self, context: int) -> int:
        """Interpolate GPU layers from measurements."""
        if not self.measurements:
            return -1  # All layers

        # Find closest measurement
        closest = min(self.measurements, key=lambda m: abs(m[0] - context))
        return closest[2]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "model_id": self.model_id,
            "base_vram_gb": self.base_vram_gb,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "measurements": self.measurements,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VRAMProfile:
        """Deserialize from dict."""
        return cls(
            model_id=data["model_id"],
            base_vram_gb=data["base_vram_gb"],
            kv_bytes_per_token=data["kv_bytes_per_token"],
            measurements=[tuple(m) for m in data.get("measurements", [])],
        )


class VRAMProfiler:
    """VRAM profiler with estimation and caching."""

    def __init__(self, cache_dir: Path) -> None:
        """Initialize profiler.

        Args:
            cache_dir: Directory for profile cache
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._profiles: dict[str, VRAMProfile] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached profiles."""
        cache_file = self.cache_dir / "profiles.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    for model_id, profile_data in data.items():
                        self._profiles[model_id] = VRAMProfile.from_dict(profile_data)
                log.info(f"Loaded {len(self._profiles)} VRAM profiles from cache")
            except Exception as e:
                log.warning(f"Failed to load VRAM cache: {e}")

    def _save_cache(self) -> None:
        """Save profiles to cache."""
        cache_file = self.cache_dir / "profiles.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(
                    {k: v.to_dict() for k, v in self._profiles.items()},
                    f,
                    indent=2
                )
        except Exception as e:
            log.warning(f"Failed to save VRAM cache: {e}")

    def get_profile(self, model_id: str) -> VRAMProfile | None:
        """Get profile for model."""
        return self._profiles.get(model_id)

    def has_profile(self, model_id: str) -> bool:
        """Check if profile exists."""
        return model_id in self._profiles

    @staticmethod
    def estimate_model_vram(specs: ModelSpecs, context_length: int) -> float:
        """Estimate VRAM for model without measured profile.

        Args:
            specs: Model specifications
            context_length: Target context length

        Returns:
            Estimated VRAM in GB
        """
        # Get bits per weight
        quant = specs.quantization.upper()
        bits = QUANT_BITS.get(quant, 4.5)  # Default to Q4_K_M

        # Estimate parameters if not known
        params_b = specs.parameters_b or 3.0

        # Model weights: params * bits / 8
        weights_gb = (params_b * 1e9 * bits / 8) / (1024**3)

        # KV cache estimation
        hidden_size = specs.hidden_size or 4096
        head_count_kv = specs.head_count_kv or 32
        layers = specs.layer_count or 32

        # KV cache per token: 2 * layers * head_kv * (hidden_size / head_count) * 2 bytes
        kv_per_token = 2 * layers * head_count_kv * (hidden_size // (specs.head_count or 32)) * 2
        kv_cache_gb = (context_length * kv_per_token) / (1024**3)

        # Add overhead (10%)
        total = (weights_gb + kv_cache_gb) * 1.1

        return round(total, 2)

    @staticmethod
    def calculate_optimal_config(
        specs: ModelSpecs,
        available_vram_gb: float,
        target_context: int = 32768,
        max_context: int = 131072,
    ) -> VRAMConfig:
        """Calculate optimal config without measured profile.

        Args:
            specs: Model specifications
            available_vram_gb: Available VRAM
            target_context: Target context length
            max_context: Maximum context

        Returns:
            Optimal configuration
        """
        # Binary search for largest fitting context
        low, high = 2048, min(max_context, specs.context_window)
        best_context = low

        while low <= high:
            mid = (low + high) // 2
            estimated = VRAMProfiler.estimate_model_vram(specs, mid)

            if estimated <= available_vram_gb * 0.95:  # 95% threshold
                best_context = mid
                low = mid + 1
            else:
                high = mid - 1

        # Clamp to target
        best_context = max(best_context, min(target_context, specs.context_window))

        return VRAMConfig(
            gpu_layers=-1,  # All layers
            context_length=best_context,
            estimated_vram_gb=VRAMProfiler.estimate_model_vram(specs, best_context),
        )

    def add_profile(self, profile: VRAMProfile) -> None:
        """Add measured profile."""
        self._profiles[profile.model_id] = profile
        self._save_cache()
        log.info(f"Added VRAM profile for {profile.model_id}")
