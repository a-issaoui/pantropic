"""Pantropic - Custom Exceptions.

Hierarchical exception system for clean error handling.
"""

from __future__ import annotations


class PantropicError(Exception):
    """Base exception for all Pantropic errors."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(PantropicError):
    """Configuration-related errors."""


class ConfigNotFoundError(ConfigurationError):
    """Configuration file not found."""


class ConfigValidationError(ConfigurationError):
    """Configuration validation failed."""


# =============================================================================
# Model Errors
# =============================================================================

class ModelError(PantropicError):
    """Model-related errors."""


class ModelNotFoundError(ModelError):
    """Model not found in registry."""

    def __init__(self, model_id: str) -> None:
        super().__init__(f"Model not found: {model_id}", code="MODEL_NOT_FOUND")
        self.model_id = model_id


class ModelLoadError(ModelError):
    """Failed to load model."""

    def __init__(self, model_id: str, reason: str) -> None:
        super().__init__(f"Failed to load {model_id}: {reason}", code="MODEL_LOAD_FAILED")
        self.model_id = model_id
        self.reason = reason


class InsufficientVRAMError(ModelError):
    """Not enough VRAM to load model."""

    def __init__(
        self, model_id: str, required_gb: float, available_gb: float
    ) -> None:
        super().__init__(
            f"Insufficient VRAM for {model_id}: need {required_gb:.1f}GB, "
            f"have {available_gb:.1f}GB",
            code="INSUFFICIENT_VRAM"
        )
        self.model_id = model_id
        self.required_gb = required_gb
        self.available_gb = available_gb


class GGUFParseError(ModelError):
    """Failed to parse GGUF file."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"Failed to parse GGUF {path}: {reason}", code="GGUF_PARSE_ERROR")
        self.path = path
        self.reason = reason


# =============================================================================
# Inference Errors
# =============================================================================

class InferenceError(PantropicError):
    """Inference-related errors."""


class ContextOverflowError(InferenceError):
    """Request exceeds model context window."""

    def __init__(self, required: int, available: int) -> None:
        super().__init__(
            f"Context overflow: {required} tokens required, {available} available",
            code="CONTEXT_OVERFLOW"
        )
        self.required = required
        self.available = available


class InferenceTimeoutError(InferenceError):
    """Inference timed out."""


class ToolCallError(InferenceError):
    """Tool calling error."""


class ToolValidationError(ToolCallError):
    """Tool call validation failed."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__(f"Tool validation failed: {errors}", code="TOOL_VALIDATION_FAILED")
        self.errors = errors


# =============================================================================
# API Errors
# =============================================================================

class APIError(PantropicError):
    """API-related errors."""

    def __init__(
        self, message: str, *, status_code: int = 500, code: str | None = None
    ) -> None:
        super().__init__(message, code=code)
        self.status_code = status_code


class AuthenticationError(APIError):
    """Authentication failed."""

    def __init__(self, message: str = "Invalid API key") -> None:
        super().__init__(message, status_code=401, code="AUTHENTICATION_FAILED")


class RateLimitError(APIError):
    """Rate limit exceeded."""

    def __init__(self, retry_after: int = 60) -> None:
        super().__init__(
            f"Rate limit exceeded. Retry after {retry_after}s",
            status_code=429,
            code="RATE_LIMIT_EXCEEDED"
        )
        self.retry_after = retry_after


class ValidationError(APIError):
    """Request validation failed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400, code="VALIDATION_ERROR")


# =============================================================================
# Hardware Errors
# =============================================================================

class HardwareError(PantropicError):
    """Hardware-related errors."""


class GPUNotAvailableError(HardwareError):
    """No GPU available."""


class CUDAError(HardwareError):
    """CUDA error occurred."""
