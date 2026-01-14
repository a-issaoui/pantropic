"""Models routes - List and get model information."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from pantropic.api.app import get_container
from pantropic.core.container import Container
from pantropic.core.exceptions import ModelNotFoundError

router = APIRouter()


@router.get("/models")
async def list_models(
    request: Request,
    container: Container = Depends(get_container),
):
    """List available models (OpenAI-compatible)."""
    models = container.model_registry.list_models()

    return {
        "object": "list",
        "data": [
            {
                "id": model.id,
                "object": "model",
                "created": 0,
                "owned_by": "pantropic",
                "permission": [],
                "root": model.id,
                "parent": None,
                # Extended info
                "architecture": model.specs.architecture,
                "parameters_b": model.specs.parameters_b,
                "quantization": model.specs.quantization,
                "context_window": model.specs.context_window,
                "capabilities": model.capabilities.as_list(),
            }
            for model in models
        ],
    }


# NOTE: This route MUST come before /models/{model_id} to avoid matching "loaded"
@router.get("/models/loaded")
async def list_loaded_models(
    request: Request,
    container: Container = Depends(get_container),
):
    """List currently loaded models."""
    engine = container.inference_engine
    loader = engine._get_loader()

    loaded = loader.list_loaded()

    return {
        "count": len(loaded),
        "models": [
            {
                "id": m.model.id,
                "context_length": m.context_length,
                "gpu_layers": m.gpu_layers,
                "active_requests": m.active_requests,
            }
            for m in loaded
        ],
    }


@router.get("/models/{model_id}")
async def get_model(
    model_id: str,
    request: Request,
    container: Container = Depends(get_container),
):
    """Get model details."""
    try:
        model = container.model_registry.get_model(model_id)
    except ModelNotFoundError:
        return {
            "error": {
                "message": f"Model '{model_id}' not found",
                "type": "model_not_found",
            }
        }

    return {
        "id": model.id,
        "object": "model",
        "created": 0,
        "owned_by": "pantropic",
        "specs": {
            "architecture": model.specs.architecture,
            "quantization": model.specs.quantization,
            "parameters_b": model.specs.parameters_b,
            "layer_count": model.specs.layer_count,
            "context_window": model.specs.context_window,
            "file_size_mb": model.specs.file_size_mb,
            "hidden_size": model.specs.hidden_size,
            "head_count": model.specs.head_count,
            "head_count_kv": model.specs.head_count_kv,
            "vocab_size": model.specs.vocab_size,
        },
        "capabilities": model.capabilities.as_list(),
        "has_vision_adapter": model.vision_adapter is not None,
        "optimal_config": model.optimal_config,
    }


@router.post("/models/{model_id}/load")
async def load_model(
    model_id: str,
    request: Request,
    container: Container = Depends(get_container),
):
    """Explicitly load a model into memory.

    Useful for preloading before inference.
    """
    try:
        model = container.model_registry.get_model(model_id)
    except ModelNotFoundError:
        return {"error": {"message": f"Model '{model_id}' not found", "type": "model_not_found"}}

    # Get or create inference engine to load the model
    engine = container.inference_engine
    loader = engine._get_loader()

    try:
        loaded = await loader.load(model)
        return {
            "status": "loaded",
            "model_id": model.id,
            "context_length": loaded.context_length,
            "gpu_layers": loaded.gpu_layers,
        }
    except Exception as e:
        return {"error": {"message": str(e), "type": "load_error"}}


@router.delete("/models/{model_id}/unload")
async def unload_model(
    model_id: str,
    request: Request,
    container: Container = Depends(get_container),
):
    """Unload a model from memory."""
    engine = container.inference_engine
    loader = engine._get_loader()

    success = await loader.unload(model_id)

    if success:
        return {"status": "unloaded", "model_id": model_id}
    return {"status": "not_loaded", "model_id": model_id}
