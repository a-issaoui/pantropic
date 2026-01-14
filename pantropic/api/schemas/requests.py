"""Pantropic - API Request Schemas.

OpenAI compatible request schemas for chat and embeddings.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message (OpenAI compatible)."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ToolFunction(BaseModel):
    """Tool function definition."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class Tool(BaseModel):
    """Tool definition."""

    type: Literal["function"] = "function"
    function: ToolFunction


class ChatCompletionRequest(BaseModel):
    """Chat completion request (OpenAI compatible)."""

    model: str
    messages: list[Message]
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False
    stop: str | list[str] | None = None
    tools: list[Tool] | None = None
    tool_choice: str | dict[str, Any] | None = None

    # Extended parameters
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    seed: int | None = None


class EmbeddingRequest(BaseModel):
    """Embedding request (OpenAI compatible)."""

    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"
