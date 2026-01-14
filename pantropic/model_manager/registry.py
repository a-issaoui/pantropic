"""Pantropic - Model Registry (100/100 Edition).

Loads model metadata from pre-scanned models.json for instant startup.
No runtime GGUF parsing - uses cached scanner output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pantropic.core.exceptions import ModelNotFoundError
from pantropic.core.types import (
    ModelCapabilities,
    ModelSpecs,
    RegisteredModel,
    VisionAdapter,
)
from pantropic.observability.logging import get_logger

if TYPE_CHECKING:
    from pantropic.hardware.gpu import GPUMonitor
    from pantropic.hardware.vram import VRAMProfiler

log = get_logger("registry")


class ModelRegistry:
    """Model registry using pre-scanned models.json.

    Loads instantly from JSON instead of parsing GGUF files at startup.
    Falls back to runtime scanning if models.json is missing.
    """

    def __init__(
        self,
        models_dir: Path,
        vram_profiler: VRAMProfiler,
        gpu_monitor: GPUMonitor,
    ) -> None:
        """Initialize registry."""
        self.models_dir = Path(models_dir)
        self.vram_profiler = vram_profiler
        self.gpu_monitor = gpu_monitor
        self._models: dict[str, RegisteredModel] = {}

    async def discover(self) -> None:
        """Discover models from models.json or fallback to scanning."""
        models_json = self.models_dir.parent / "models.json"

        if models_json.exists():
            # Fast path: Load from pre-scanned JSON
            log.info(f"Loading models from {models_json}")
            await self._load_from_json(models_json)
        else:
            # Fallback: Runtime scanning
            log.info("models.json not found, scanning directory...")
            await self._scan_directory()

        log.info(f"Registered {len(self._models)} models")

    async def _load_from_json(self, json_path: Path) -> None:
        """Load models from models.json (instant startup)."""
        try:
            with open(json_path) as f:
                data = json.load(f)

            for model_id, entry in data.items():
                try:
                    model = self._json_to_model(model_id, entry)
                    self._models[model_id] = model

                    # Calculate optimal config
                    available_vram = self.gpu_monitor.available_vram_gb
                    config = self.vram_profiler.calculate_optimal_config(
                        specs=model.specs,
                        available_vram_gb=available_vram,
                    )
                    model.optimal_config = {
                        "gpu_layers": config.gpu_layers,
                        "context_length": config.context_length,
                    }
                except Exception as e:
                    log.warning(f"Failed to load {model_id}: {e}")

            log.info(f"Loaded {len(self._models)} models from JSON")

        except json.JSONDecodeError as e:
            log.exception(f"Invalid models.json: {e}")
            await self._scan_directory()

    async def _scan_directory(self) -> None:
        """Fallback: Scan directory for GGUF files."""
        from pantropic.model_manager.scanner import PerfectGGUFScanner

        if not self.models_dir.exists():
            log.warning(f"Models directory does not exist: {self.models_dir}")
            return

        scanner = PerfectGGUFScanner()
        scanner.scan_directory(str(self.models_dir), output_file=None)

        for model_id, entry in scanner.results.items():
            try:
                model = self._entry_to_model(model_id, entry)
                self._models[model_id] = model
            except Exception as e:
                log.exception(f"Failed to register {model_id}: {e}")

    def _json_to_model(self, model_id: str, entry: dict[str, Any]) -> RegisteredModel:
        """Convert JSON entry to RegisteredModel."""
        specs_data = entry.get("specs", {})
        caps_data = entry.get("capabilities", {})
        prompt_data = entry.get("prompt", {})
        mmproj_data = entry.get("mmproj")

        # Build ModelSpecs from JSON
        specs = ModelSpecs(
            architecture=specs_data.get("architecture", "unknown"),
            quantization=specs_data.get("quantization", "Q4_K_M"),
            parameters_b=specs_data.get("parameters_b", 0),
            layer_count=specs_data.get("layer_count", 32),
            context_window=specs_data.get("context_window", 4096),
            file_size_mb=specs_data.get("file_size_mb", 0),
            hidden_size=specs_data.get("hidden_size", 4096),
            head_count=specs_data.get("head_count", 32),
            head_count_kv=specs_data.get("head_count_kv"),  # Key for KV cache!
            vocab_size=specs_data.get("vocab_size", 32000),
            gqa_ratio=specs_data.get("gqa_ratio", 1),
        )

        # Build capabilities
        capabilities = ModelCapabilities(
            chat=caps_data.get("chat", False),
            embed=caps_data.get("embed", False),
            vision=caps_data.get("vision", False),
            audio=caps_data.get("audio_in", False),
            reasoning=caps_data.get("reasoning", False),
            tools=caps_data.get("tools", False),
        )

        # Vision adapter
        vision_adapter = None
        if mmproj_data:
            vision_adapter = VisionAdapter(
                path=self.models_dir / mmproj_data.get("filename", ""),
                file_size_mb=mmproj_data.get("file_size_mb", 0),
                quantization=mmproj_data.get("quantization", "F16"),
            )

        # Chat template from JSON
        chat_template = prompt_data.get("template") if prompt_data else None

        model_path = self.models_dir / model_id

        return RegisteredModel(
            id=model_id,
            path=model_path,
            specs=specs,
            capabilities=capabilities,
            chat_template=chat_template,
            vision_adapter=vision_adapter,
        )

    def _entry_to_model(self, model_id: str, entry: object) -> RegisteredModel:
        """Convert scanner entry to RegisteredModel (fallback)."""
        specs_data = entry.specs  # type: ignore[attr-defined]
        caps_data = entry.capabilities  # type: ignore[attr-defined]
        prompt_data = entry.prompt  # type: ignore[attr-defined]
        mmproj_data = entry.mmproj  # type: ignore[attr-defined]

        specs = ModelSpecs(
            architecture=specs_data.architecture,
            quantization=specs_data.quantization,
            parameters_b=specs_data.parameters_b,
            layer_count=specs_data.layer_count,
            context_window=specs_data.context_window,
            file_size_mb=specs_data.file_size_mb,
            hidden_size=specs_data.hidden_size,
            head_count=specs_data.head_count,
            head_count_kv=specs_data.head_count_kv,
            vocab_size=specs_data.vocab_size,
            gqa_ratio=specs_data.gqa_ratio,
        )

        capabilities = ModelCapabilities(
            chat=caps_data.chat,
            embed=caps_data.embed,
            vision=caps_data.vision,
            audio=caps_data.audio_in,
            reasoning=caps_data.reasoning,
            tools=caps_data.tools,
        )

        vision_adapter = None
        if mmproj_data:
            vision_adapter = VisionAdapter(
                path=Path(self.models_dir / mmproj_data.get("filename", "")),
                file_size_mb=mmproj_data.get("file_size_mb", 0),
                quantization=mmproj_data.get("quantization", "F16"),
            )

        chat_template = None
        if prompt_data and hasattr(prompt_data, "template"):
            chat_template = prompt_data.template
        elif isinstance(prompt_data, dict):
            chat_template = prompt_data.get("template")

        return RegisteredModel(
            id=model_id,
            path=self.models_dir / model_id,
            specs=specs,
            capabilities=capabilities,
            chat_template=chat_template,
            vision_adapter=vision_adapter,
        )

    def get_model(self, model_id: str) -> RegisteredModel:
        """Get model by ID."""
        if model_id in self._models:
            return self._models[model_id]

        # Fuzzy match
        normalized = model_id.replace("-", "_").lower()
        for mid, model in self._models.items():
            if mid.replace("-", "_").lower() == normalized:
                return model

        raise ModelNotFoundError(model_id)

    def list_models(self) -> list[RegisteredModel]:
        """List all registered models."""
        return list(self._models.values())

    def get_by_capability(self, capability: str) -> list[RegisteredModel]:
        """Get models with specific capability."""
        return [
            model for model in self._models.values()
            if getattr(model.capabilities, capability, False)
        ]

    def get_embedding_models(self) -> list[RegisteredModel]:
        """Get all embedding models."""
        return self.get_by_capability("embed")

    def get_chat_models(self) -> list[RegisteredModel]:
        """Get all chat models."""
        return self.get_by_capability("chat")

    def get_vision_models(self) -> list[RegisteredModel]:
        """Get all vision-capable models."""
        return [m for m in self._models.values() if m.has_vision]

    def get_tools_models(self) -> list[RegisteredModel]:
        """Get all tool-calling capable models."""
        return self.get_by_capability("tools")

    def get_reasoning_models(self) -> list[RegisteredModel]:
        """Get all reasoning models (e.g., DeepSeek-R1)."""
        return self.get_by_capability("reasoning")
