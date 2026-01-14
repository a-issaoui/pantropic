"""Embeddings routes - OpenAI compatible."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from pantropic.api.app import get_container
from pantropic.api.schemas.requests import EmbeddingRequest
from pantropic.api.schemas.responses import EmbeddingData, EmbeddingResponse, Usage
from pantropic.core.container import Container
from pantropic.core.exceptions import ModelNotFoundError
from pantropic.observability.logging import get_logger

router = APIRouter()
log = get_logger("api.embeddings")


def _flatten_embedding(embedding):
    """Ensure embedding is a flat list of floats.
    
    Some models (like bge-m3) return nested lists for multi-vector embeddings.
    We take the first/main embedding if nested.
    """
    if not embedding:
        return []

    # Check if it's a nested list (list of lists)
    if isinstance(embedding[0], list):
        # Take the first embedding vector (dense embedding for multi-vector models)
        return embedding[0]

    return embedding


@router.post("/embeddings")
async def create_embeddings(
    request: EmbeddingRequest,
    req: Request,
    container: Container = Depends(get_container),
) -> EmbeddingResponse:
    """OpenAI-compatible embeddings endpoint.

    Supports single text or batch embedding.
    """
    try:
        result = await container.inference_engine.create_embedding(
            model_id=request.model,
            input_text=request.input,
        )

        return EmbeddingResponse(
            model=request.model,
            data=[
                EmbeddingData(
                    index=d["index"],
                    embedding=_flatten_embedding(d["embedding"])
                )
                for d in result["data"]
            ],
            usage=Usage(
                prompt_tokens=result["usage"]["prompt_tokens"],
                total_tokens=result["usage"]["total_tokens"],
            ),
        )

    except ModelNotFoundError as e:
        return {"error": {"message": str(e), "type": "model_not_found"}}
    except Exception as e:
        log.exception(f"Embedding error: {e}")
        return {"error": {"message": str(e), "type": "internal_error"}}

