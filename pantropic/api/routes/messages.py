"""Anthropic Claude API-compatible messages endpoint.

Provides /v1/messages endpoint matching Anthropic's API format.
Supports both non-streaming and streaming responses.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, Request
from starlette.responses import JSONResponse, StreamingResponse

from pantropic.api.app import get_container
from pantropic.api.schemas.anthropic import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicTextContent,
    AnthropicToolUseContent,
    AnthropicUsage,
)
from pantropic.core.container import Container
from pantropic.core.exceptions import ContextOverflowError, ModelNotFoundError
from pantropic.observability.logging import get_logger

router = APIRouter()
log = get_logger("anthropic")


def _convert_anthropic_to_openai_messages(
    messages: list[dict[str, Any]],
    system: str | None = None,
) -> list[dict[str, Any]]:
    """Convert Anthropic message format to OpenAI format."""
    openai_messages = []

    # Add system message if provided
    if system:
        openai_messages.append({"role": "system", "content": system})

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # Handle content blocks
        if isinstance(content, list):
            # Extract text from content blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)

        openai_messages.append({"role": role, "content": content})

    return openai_messages


def _convert_anthropic_tools_to_openai(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Convert Anthropic tool format to OpenAI format."""
    if not tools:
        return None

    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            }
        }
        for tool in tools
    ]


async def _stream_anthropic_response(
    container: Container,
    request: AnthropicMessagesRequest,
    openai_messages: list[dict[str, Any]],
    openai_tools: list[dict[str, Any]] | None,
) -> AsyncGenerator[str, None]:
    """Generate Anthropic-format SSE stream.

    Anthropic streaming uses these event types:
    - message_start: Initial message metadata
    - content_block_start: Start of content block
    - content_block_delta: Text delta
    - content_block_stop: End of content block
    - message_delta: Final message stats (stop_reason, usage)
    - message_stop: End of stream
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Event: message_start
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'model': request.model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

    # Event: content_block_start
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

    # Stream content deltas
    output_tokens = 0
    input_tokens = 0

    async for chunk in container.inference_engine.chat_completion(
        model_id=request.model,
        messages=openai_messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        stop=request.stop_sequences,
        stream=True,
        tools=openai_tools,
    ):
        if hasattr(chunk, 'content') and chunk.content:
            # Event: content_block_delta
            delta_event = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": chunk.content}
            }
            yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
            output_tokens += 1  # Approximate

        # Capture usage if available
        if hasattr(chunk, 'prompt_tokens') and chunk.prompt_tokens:
            input_tokens = chunk.prompt_tokens
        if hasattr(chunk, 'completion_tokens') and chunk.completion_tokens:
            output_tokens = chunk.completion_tokens

    # Event: content_block_stop
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

    # Event: message_delta
    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": output_tokens}
    }
    yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"

    # Event: message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


@router.post("/messages", response_model=None)
async def messages(
    request: AnthropicMessagesRequest,
    req: Request,
    container: Container = Depends(get_container),
):
    """Anthropic-compatible messages endpoint.

    Converts Anthropic format to internal format, processes,
    then converts response back to Anthropic format.
    Supports both streaming and non-streaming responses.
    """
    try:
        # Validate messages not empty
        if not request.messages:
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "messages: at least 1 message is required",
                    }
                }
            )

        # Convert to OpenAI format
        messages_dicts = [msg.model_dump() for msg in request.messages]
        openai_messages = _convert_anthropic_to_openai_messages(
            messages_dicts,
            system=request.system,
        )

        # Convert tools
        tools_dicts = [t.model_dump() for t in request.tools] if request.tools else None
        openai_tools = _convert_anthropic_tools_to_openai(tools_dicts)

        # Handle streaming
        if request.stream:
            return StreamingResponse(
                _stream_anthropic_response(container, request, openai_messages, openai_tools),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )

        # Non-streaming response
        result = await container.inference_engine.chat_completion(
            model_id=request.model,
            messages=openai_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stop=request.stop_sequences,
            stream=False,
            tools=openai_tools,
        )

        # Build Anthropic response
        content: list[AnthropicTextContent | AnthropicToolUseContent] = []
        stop_reason = "end_turn"

        # Check for tool calls
        if result.tool_calls:
            for tc in result.tool_calls:
                arguments = tc.get("function", {}).get("arguments", {})
                # Parse string arguments to dict
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments}

                content.append(AnthropicToolUseContent(
                    id=tc.get("id", f"toolu_{uuid.uuid4().hex[:8]}"),
                    name=tc.get("function", {}).get("name", ""),
                    input=arguments,
                ))
            stop_reason = "tool_use"
        elif result.content:
            content.append(AnthropicTextContent(text=result.content))

        # Map finish reason
        if result.finish_reason == "length":
            stop_reason = "max_tokens"
        elif result.finish_reason == "stop":
            stop_reason = "end_turn"
        elif result.finish_reason == "tool_calls":
            stop_reason = "tool_use"

        return AnthropicMessagesResponse(
            id=f"msg_{uuid.uuid4().hex[:24]}",
            model=request.model,
            content=content,
            stop_reason=stop_reason,
            usage=AnthropicUsage(
                input_tokens=result.prompt_tokens,
                output_tokens=result.completion_tokens,
            ),
        )

    except ModelNotFoundError:
        raise
    except ContextOverflowError:
        raise
    except Exception as e:
        log.exception(f"Messages error: {e}")
        raise
