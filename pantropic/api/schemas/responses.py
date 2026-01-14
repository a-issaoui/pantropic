"""Pantropic - API Response Schemas.

OpenAI compatible response schemas for chat and embeddings.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class ToolCallFunction(BaseModel):
    """Tool call function details."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call."""

    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")
    type: str = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    """Chat completion message."""

    role: str = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    """Completion choice."""

    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    """Token usage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Chat completion response (OpenAI compatible)."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


class StreamChoice(BaseModel):
    """Streaming choice."""

    index: int = 0
    delta: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """Chat completion chunk (streaming)."""

    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[StreamChoice]


class EmbeddingData(BaseModel):
    """Single embedding."""

    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    """Embedding response (OpenAI compatible)."""

    object: str = "list"
    model: str
    data: list[EmbeddingData]
    usage: Usage = Field(default_factory=Usage)
