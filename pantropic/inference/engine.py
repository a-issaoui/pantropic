"""Pantropic - Inference Engine.

High-performance inference engine with:
- Request queuing with priority scheduling
- Batch embedding support
- Concurrency control
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from pantropic.core.exceptions import ContextOverflowError
from pantropic.core.types import InferenceResult, LoadedModel
from pantropic.inference.queue import RequestQueue
from pantropic.observability.logging import get_logger
from pantropic.tools.extractor import ToolExtractor
from pantropic.tools.injector import ToolInjector

if TYPE_CHECKING:
    from pantropic.core.container import Container

log = get_logger("inference")


def _build_prompt(messages: list[dict[str, Any]], loaded: LoadedModel) -> str:
    """Build prompt from messages using model's chat template.

    Uses Jinja2 template from models.json if available, else ChatML.
    """
    # Try to use the model's native chat template
    template = loaded.model.chat_template

    if template:
        try:
            from jinja2 import Template

            # Render the Jinja2 template
            # Note: Don't add bos_token here - llama-cpp adds it automatically
            jinja = Template(template)
            return jinja.render(
                messages=messages,
                add_generation_prompt=True,
                bos_token="",  # Empty - llama-cpp adds it
                eos_token="<|end_of_text|>",
            )
        except Exception as e:
            log.debug(f"Template rendering failed: {e}, using ChatML")

    # Fallback to ChatML format
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


class InferenceEngine:
    """High-performance inference engine with request queuing."""

    def __init__(self, container: Container, config: Any) -> None:
        self.container = container
        self.config = config
        self._loader: Any = None
        self._queue: RequestQueue | None = None
        self._queue_enabled: bool = True  # Can be disabled for simple use

    @property
    def loader(self) -> Any:
        """Get the model loader instance."""
        return self._get_loader()

    def _get_queue(self) -> RequestQueue:
        """Get or create request queue."""
        if self._queue is None:
            self._queue = RequestQueue(
                max_concurrent=1,  # CUDA cannot handle concurrent inference
                max_queue_size=100,
                default_timeout=300.0,
            )
            # Start queue worker
            asyncio.create_task(self._queue.start())
        return self._queue

    async def chat_completion(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> InferenceResult | AsyncIterator[dict[str, Any]]:
        """Generate chat completion."""
        start_time = time.monotonic()
        model = self.container.model_registry.get_model(model_id)

        if tools:
            messages = ToolInjector.inject(messages, tools, model.specs.architecture)

        prompt_tokens = self._estimate_tokens(messages)
        max_new = max_tokens or 4096
        total_needed = prompt_tokens + max_new + 512

        # Load model with intelligent context sizing (agent-friendly)
        loader = self._get_loader()
        loaded = await loader.load(
            model,
            min_context=total_needed,
            estimated_tokens=prompt_tokens,  # For smart context expansion
        )

        if total_needed > loaded.context_length:
            raise ContextOverflowError(total_needed, loaded.context_length)

        prompt = _build_prompt(messages, loaded)
        stop_seqs = self._build_stop_sequences(stop, model.specs.architecture)

        if stream:
            # Streaming bypasses queue (single concurrent stream is OK)
            return self._stream_completion(loaded, prompt, temperature, max_new, top_p, stop_seqs, start_time)

        # Use queue for non-streaming to prevent CUDA concurrent inference crashes
        queue = self._get_queue()
        return await queue.submit(
            model_id=model_id,
            coro=self._complete(loaded, prompt, temperature, max_new, top_p, stop_seqs, start_time),
        )

    @staticmethod
    async def _complete(
            loaded: LoadedModel,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop: list[str],
        start_time: float,
    ) -> InferenceResult:
        loop = asyncio.get_event_loop()
        loaded.active_requests += 1

        try:
            def generate() -> dict[str, Any]:
                return loaded.llm(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop or None, echo=False)

            response = await loop.run_in_executor(None, generate)
            content = response["choices"][0]["text"]
            finish_reason = response["choices"][0].get("finish_reason", "stop")
            tool_calls = ToolExtractor.extract(content, loaded.model.specs.architecture)

            if tool_calls:
                content = None
                finish_reason = "tool_calls"

            return InferenceResult(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                prompt_tokens=response.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=response.get("usage", {}).get("completion_tokens", 0),
                latency_ms=(time.monotonic() - start_time) * 1000,
            )
        finally:
            loaded.active_requests -= 1
            loaded.last_used = time.monotonic()

    @staticmethod
    async def _stream_completion(
            loaded: LoadedModel,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop: list[str],
        _start_time: float,  # Unused but kept for API consistency
    ) -> AsyncIterator[dict[str, Any]]:
        loaded.active_requests += 1
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        try:
            stream = loaded.llm(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop or None, stream=True)
            for chunk in stream:
                text = chunk["choices"][0].get("text", "")
                finish = chunk["choices"][0].get("finish_reason")
                yield {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": loaded.model.id,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text} if text else {},
                        "finish_reason": finish,
                    }],
                }
                if finish:
                    break
        finally:
            loaded.active_requests -= 1
            loaded.last_used = time.monotonic()

    async def create_embedding(self, model_id: str, input_text: str | list[str]) -> dict[str, Any]:
        """Create embeddings for input text."""
        model = self.container.model_registry.get_model(model_id)
        loader = self._get_loader()
        loaded = await loader.load(model)
        texts = [input_text] if isinstance(input_text, str) else input_text
        loop = asyncio.get_event_loop()

        def embed() -> list[list[float]]:
            return [loaded.llm.embed(t) for t in texts]

        embeddings = await loop.run_in_executor(None, embed)
        return {
            "object": "list",
            "model": model_id,
            "data": [{"object": "embedding", "index": i, "embedding": e} for i, e in enumerate(embeddings)],
            "usage": {
                "prompt_tokens": sum(len(t.split()) for t in texts),
                "total_tokens": sum(len(t.split()) for t in texts),
            },
        }

    async def unload_idle_models(self, timeout_seconds: int) -> None:
        """Unload models that have been idle for longer than timeout."""
        if self._loader:
            now = time.monotonic()
            for loaded in self._loader.list_loaded():
                if loaded.is_idle and (now - loaded.last_used) > timeout_seconds:
                    await self._loader.unload(loaded.model.id)

    async def unload_all(self) -> None:
        """Unload all loaded models."""
        if self._loader:
            await self._loader.unload_all()

    def get_queue_stats(self) -> dict:
        """Get request queue statistics."""
        if self._queue:
            return self._queue.get_stats()
        return {"queue_enabled": False, "message": "Queue not initialized"}

    def _get_loader(self) -> Any:
        """Get or create model loader."""
        if self._loader is None:
            from pantropic.model_manager.loader import ModelLoader
            self._loader = ModelLoader(
                gpu_monitor=self.container.gpu_monitor,
                vram_profiler=self.container.vram_profiler,
                default_context=self.container.config.models.default_context,
                max_context=self.container.config.models.max_context,
                flash_attention=self.config.flash_attention,
                use_mmap=self.config.mmap,
                cpu_threads=self.config.cpu_threads,
            )
        return self._loader

    @staticmethod
    def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
        """Estimate token count for messages (rough approximation)."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content) // 4
            elif isinstance(content, list):
                for p in content:
                    if p.get("type") == "text":
                        total += len(p.get("text", "")) // 4
        return total + len(messages) * 10

    @staticmethod
    def _build_stop_sequences(custom: list[str] | None, arch: str) -> list[str]:
        """Build stop sequences based on architecture and custom stops."""
        stops = list(custom or [])
        stops.append("<|im_end|>")

        arch_stops: dict[str, list[str]] = {
            "llama": ["<|eot_id|>", "<|end_of_text|>"],
            "qwen2": ["<|endoftext|>"],
            "gemma": ["<|end_of_text|>"],
            "phi3": ["<|end_of_text|>"],
            "mistral": ["[/INST]", "</s>"],
        }

        arch_key = arch.lower().split("-")[0] if arch else ""
        if arch_key in arch_stops:
            stops.extend(arch_stops[arch_key])

        return list(set(stops))
