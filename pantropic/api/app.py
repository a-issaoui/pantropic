"""Pantropic - FastAPI Application Factory.

Creates and configures the FastAPI application.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from pantropic.observability.logging import get_logger, request_id_var

if TYPE_CHECKING:
    from pantropic.core.container import Container

log = get_logger("api")


def create_app(container: Container) -> FastAPI:
    """Create FastAPI application.

    Args:
        container: Dependency injection container

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Pantropic",
        description="Intelligent Local LLM Server",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Store container in app state (state is a Starlette State object)
    app.state.container = container  # type: ignore[attr-defined]

    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=container.config.api.cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Request ID middleware
    @app.middleware("http")
    async def add_request_id(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request_id_var.set(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # Include routers
    from pantropic.api.routes import chat, embeddings, health, messages, models, sessions

    # OpenAI-compatible API
    app.include_router(chat.router, prefix="/v1", tags=["Chat (OpenAI)"])
    app.include_router(embeddings.router, prefix="/v1", tags=["Embeddings"])
    app.include_router(models.router, prefix="/v1", tags=["Models"])
    app.include_router(sessions.router, prefix="/v1/sessions", tags=["Sessions"])

    # Anthropic-compatible API
    app.include_router(messages.router, prefix="/v1", tags=["Messages (Anthropic)"])

    app.include_router(health.router, tags=["Health"])

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "service": "Pantropic",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs",
        }

    # Exception handlers - OpenAI compatible error format
    # Format: {"error": {"message": str, "type": str, "param": str|null, "code": str|null}}
    from pantropic.core.exceptions import (
        APIError,
        PantropicError,
        ModelNotFoundError,
        ModelLoadError,
        ContextOverflowError,
        InsufficientVRAMError,
        InferenceError,
        ValidationError,
    )

    def _is_anthropic_request(request: Request) -> bool:
        """Check if request is for Anthropic API endpoint."""
        return "/messages" in request.url.path

    def _format_error(request: Request, status_code: int, message: str, error_type: str, param: str | None = None, code: str | None = None) -> JSONResponse:
        """Format error based on API type (OpenAI vs Anthropic)."""
        if _is_anthropic_request(request):
            # Anthropic error format
            return JSONResponse(
                status_code=status_code,
                content={
                    "type": "error",
                    "error": {
                        "type": error_type,
                        "message": message,
                    }
                }
            )
        else:
            # OpenAI error format
            return JSONResponse(
                status_code=status_code,
                content={
                    "error": {
                        "message": message,
                        "type": error_type,
                        "param": param,
                        "code": code or error_type,
                    }
                }
            )

    @app.exception_handler(ModelNotFoundError)
    async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
        return _format_error(request, 404, exc.message, "invalid_request_error", "model", "model_not_found")

    @app.exception_handler(ContextOverflowError)
    async def context_overflow_handler(request: Request, exc: ContextOverflowError):
        return _format_error(request, 400, exc.message, "invalid_request_error", "messages")

    @app.exception_handler(InsufficientVRAMError)
    async def insufficient_vram_handler(request: Request, exc: InsufficientVRAMError):
        return _format_error(request, 503, exc.message, "overloaded_error")

    @app.exception_handler(ModelLoadError)
    async def model_load_handler(request: Request, exc: ModelLoadError):
        log.error(f"Model load error: {exc.message}")
        return _format_error(request, 503, exc.message, "overloaded_error", "model")

    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError):
        return _format_error(request, 400, exc.message, "invalid_request_error")

    @app.exception_handler(InferenceError)
    async def inference_error_handler(request: Request, exc: InferenceError):
        log.error(f"Inference error: {exc.message}")
        return _format_error(request, 500, exc.message, "api_error")

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        return _format_error(request, exc.status_code, exc.message, "api_error")

    @app.exception_handler(PantropicError)
    async def pantropic_error_handler(request: Request, exc: PantropicError):
        log.error(f"Pantropic error: {exc.message}")
        return _format_error(request, 500, exc.message, "api_error")

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        log.exception(f"Unhandled error: {exc}")
        return _format_error(request, 500, "An unexpected error occurred", "api_error")

    return app


def get_container(request: Request) -> Container:
    """Get container from request state (dependency injection)."""
    return request.app.state.container  # type: ignore[attr-defined]
