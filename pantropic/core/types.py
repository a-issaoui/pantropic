"""Pantropic - Shared Type Definitions.

Contains dataclasses and TypedDicts used across the application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict


class InferenceMode(str, Enum):
    """Inference optimization mode."""
    SPEED = "speed"       # Minimize latency
    BALANCED = "balanced" # Balance speed and context
    CONTEXT = "context"   # Maximize context window


class Capability(str, Enum):
    """Model capabilities."""
    CHAT = "chat"
    EMBED = "embed"
    VISION = "vision"
    AUDIO = "audio"
    REASONING = "reasoning"
    TOOLS = "tools"


@dataclass(frozen=True)
class ModelCapabilities:
    """Immutable model capabilities."""
    chat: bool = False
    embed: bool = False
    vision: bool = False
    audio: bool = False
    reasoning: bool = False
    tools: bool = False

    def as_list(self) -> list[str]:
        """Return list of active capabilities."""
        caps = []
        if self.chat:
            caps.append("chat")
        if self.embed:
            caps.append("embed")
        if self.vision:
            caps.append("vision")
        if self.audio:
            caps.append("audio")
        if self.reasoning:
            caps.append("reasoning")
        if self.tools:
            caps.append("tools")
        return caps


@dataclass
class ModelSpecs:
    """Technical specifications for a model."""
    architecture: str
    quantization: str
    parameters_b: float | None
    layer_count: int | None
    context_window: int
    file_size_mb: int
    hidden_size: int | None = None
    head_count: int | None = None
    head_count_kv: int | None = None
    vocab_size: int | None = None
    gqa_ratio: int | None = None


@dataclass
class VisionAdapter:
    """Vision adapter (mmproj) information."""
    path: Path
    file_size_mb: int
    quantization: str
    architecture: str | None = None


@dataclass
class RegisteredModel:
    """A model registered in the system."""
    id: str
    path: Path
    specs: ModelSpecs
    capabilities: ModelCapabilities
    chat_template: str | None = None
    vision_adapter: VisionAdapter | None = None
    optimal_config: dict[str, int] = field(default_factory=dict)

    @property
    def has_vision(self) -> bool:
        """Check if model has vision capability."""
        return self.capabilities.vision or self.vision_adapter is not None


@dataclass
class LoadedModel:
    """A model currently loaded in memory with reference counting.

    Features:
    - Reference counting for safe concurrent access
    - Eviction scoring for intelligent cache management
    - Never evict models with active requests
    """
    model: RegisteredModel
    llm: Any  # llama_cpp.Llama instance
    context_length: int
    gpu_layers: int
    loaded_at: float
    last_used: float
    active_requests: int = 0
    _ref_count: int = 0
    _eviction_pending: bool = False
    access_count: int = 0

    @property
    def is_idle(self) -> bool:
        """Check if model has no active requests."""
        return self.active_requests == 0 and self._ref_count == 0

    @property
    def size_gb(self) -> float:
        """Estimate model size in GB."""
        if self.model.specs.file_size_mb:
            return self.model.specs.file_size_mb / 1024
        return 0.0

    def acquire_ref(self) -> bool:
        """Acquire a reference (for starting inference).

        Returns False if eviction is pending.
        """
        if self._eviction_pending:
            return False
        self._ref_count += 1
        self.access_count += 1
        return True

    def release_ref(self) -> None:
        """Release a reference (after inference complete)."""
        if self._ref_count > 0:
            self._ref_count -= 1

    def try_evict(self) -> tuple[bool, int]:
        """Try to mark for eviction.

        Returns (success, active_refs).
        """
        if self._ref_count > 0:
            return (False, self._ref_count)
        self._eviction_pending = True
        return (True, 0)

    def cancel_eviction(self) -> None:
        """Cancel pending eviction."""
        self._eviction_pending = False

    def is_evictable(self) -> bool:
        """Check if model can be evicted."""
        return self._ref_count == 0 and not self._eviction_pending

    def touch(self) -> None:
        """Update last used time."""
        import time
        self.last_used = time.monotonic()
        self.access_count += 1

    def eviction_score(self) -> float:
        """Calculate eviction score (higher = evict first).

        Formula: size_gb * idle_time / (access_count + 1)
        """
        import time
        idle_time = time.monotonic() - self.last_used
        return self.size_gb * idle_time / (self.access_count + 1)



@dataclass
class VRAMConfig:
    """VRAM allocation configuration."""
    gpu_layers: int
    context_length: int
    estimated_vram_gb: float
    batch_size: int = 512


class ToolFunction(TypedDict):
    """Tool function definition."""
    name: str
    description: str
    parameters: dict[str, Any]


class ToolDefinition(TypedDict):
    """Complete tool definition."""
    type: str  # "function"
    function: ToolFunction


class ToolCall(TypedDict):
    """A tool call made by the model."""
    id: str
    type: str  # "function"
    function: dict[str, Any]


@dataclass
class InferenceResult:
    """Result of an inference operation."""
    content: str | None
    tool_calls: list[ToolCall] | None
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
