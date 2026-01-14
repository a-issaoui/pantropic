"""Chat completions routes - OpenAI compatible."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Request
from starlette.responses import JSONResponse, StreamingResponse

from pantropic.api.app import get_container
from pantropic.api.schemas.requests import ChatCompletionRequest
from pantropic.api.schemas.responses import (
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ToolCall,
    ToolCallFunction,
    Usage,
)
from pantropic.core.container import Container
from pantropic.core.exceptions import ContextOverflowError, ModelNotFoundError
from pantropic.observability.logging import get_logger

router = APIRouter()
log = get_logger("api.chat")


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    req: Request,
    container: Container = Depends(get_container),
) -> dict[str, Any] | StreamingResponse | ChatCompletionResponse:
    """OpenAI-compatible chat completions.

    Supports:
    - All standard OpenAI parameters
    - Tool/function calling
    - Streaming via SSE
    """
    try:
        # Validate request
        if not request.messages:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "messages cannot be empty", "type": "invalid_request", "code": 400}}
            )

        # Convert request to engine format
        messages = [msg.model_dump(exclude_none=True) for msg in request.messages]
        tools = [t.model_dump() for t in request.tools] if request.tools else None
        stop = [request.stop] if isinstance(request.stop, str) else request.stop

        result = await container.inference_engine.chat_completion(
            model_id=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stop=stop,
            stream=request.stream,
            tools=tools,
            tool_choice=request.tool_choice if isinstance(request.tool_choice, str) else None,
        )

        if request.stream:
            # Return streaming response
            async def generate():
                async for chunk in result:
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        # Build response
        tool_calls = None
        if result.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.get("id", f"call_{i}"),
                    function=ToolCallFunction(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for i, tc in enumerate(result.tool_calls)
            ]

        return ChatCompletionResponse(
            model=request.model,
            choices=[
                Choice(
                    message=ChatMessage(
                        content=result.content,
                        tool_calls=tool_calls,
                    ),
                    finish_reason=result.finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
            ),
        )

    except ModelNotFoundError:
        raise  # Let global handler format the error
    except ContextOverflowError:
        raise  # Let global handler format the error
    except Exception as e:
        log.exception(f"Chat completion error: {e}")
        raise  # Let global handler format the error
