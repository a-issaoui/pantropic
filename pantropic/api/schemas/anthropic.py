"""Anthropic Claude API-compatible schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Request Schemas
# =============================================================================


class AnthropicContentBlock(BaseModel):
    """Content block in a message."""
    type: Literal["text", "image"] = "text"
    text: str | None = None
    # For images
    source: dict[str, Any] | None = None


class AnthropicMessage(BaseModel):
    """A message in the conversation."""
    role: Literal["user", "assistant"]
    content: str | list[AnthropicContentBlock]


class AnthropicTool(BaseModel):
    """Tool definition for Anthropic API."""
    name: str
    description: str
    input_schema: dict[str, Any]


class AnthropicMessagesRequest(BaseModel):
    """Anthropic /v1/messages request format."""
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = Field(default=4096, ge=1)
    system: str | None = None
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    tools: list[AnthropicTool] | None = None
    tool_choice: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


# =============================================================================
# Response Schemas
# =============================================================================


class AnthropicTextContent(BaseModel):
    """Text content block in response."""
    type: Literal["text"] = "text"
    text: str


class AnthropicToolUseContent(BaseModel):
    """Tool use content block in response."""
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class AnthropicUsage(BaseModel):
    """Token usage information."""
    input_tokens: int
    output_tokens: int


class AnthropicMessagesResponse(BaseModel):
    """Anthropic /v1/messages response format."""
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[AnthropicTextContent | AnthropicToolUseContent]
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    stop_sequence: str | None = None
    usage: AnthropicUsage


class AnthropicErrorResponse(BaseModel):
    """Anthropic error response format."""
    type: Literal["error"] = "error"
    error: dict[str, str]
