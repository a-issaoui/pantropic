"""Pantropic - Request Queue.

Async request queue with:
- Priority queuing
- Concurrency limit per model
- Timeout handling
- Batch grouping
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from pantropic.observability.logging import get_logger

log = get_logger("queue")


class Priority(IntEnum):
    """Request priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2


@dataclass
class QueuedRequest:
    """A request in the queue."""
    id: str
    model_id: str
    priority: Priority
    coro: Coroutine[Any, Any, Any]
    future: asyncio.Future = field(default_factory=asyncio.Future)
    created_at: float = field(default_factory=time.monotonic)
    timeout: float = 300.0

    def __lt__(self, other: QueuedRequest) -> bool:
        # Higher priority first, then older requests
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at


class RequestQueue:
    """Async request queue with concurrency control.

    Features:
    - Priority-based scheduling
    - Per-model concurrency limits
    - Timeout handling
    - Request batching (for embeddings)
    """

    def __init__(
        self,
        max_concurrent: int = 1,
        max_queue_size: int = 100,
        default_timeout: float = 300.0,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout

        self._queue: asyncio.PriorityQueue[QueuedRequest] = asyncio.PriorityQueue()
        self._active: dict[str, int] = {}  # model_id -> active count
        self._lock = asyncio.Lock()
        self._running = False
        self._worker_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the queue worker."""
        if self._running:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        log.info(f"Request queue started (max_concurrent={self.max_concurrent})")

    async def stop(self) -> None:
        """Stop the queue worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
        log.info("Request queue stopped")

    async def submit(
        self,
        model_id: str,
        coro: Coroutine[Any, Any, Any],
        priority: Priority = Priority.NORMAL,
        timeout: float | None = None,
    ) -> Any:
        """Submit a request to the queue.

        Args:
            model_id: Model the request uses
            coro: Coroutine to execute
            priority: Request priority
            timeout: Request timeout (None = default)

        Returns:
            Result of the coroutine
        """
        if self._queue.qsize() >= self.max_queue_size:
            msg = f"Queue full (max={self.max_queue_size})"
            raise QueueFullError(msg)

        request = QueuedRequest(
            id=str(uuid.uuid4())[:8],
            model_id=model_id,
            priority=priority,
            coro=coro,
            timeout=timeout or self.default_timeout,
        )

        await self._queue.put(request)
        log.debug(f"Queued request {request.id} for {model_id}")

        try:
            return await asyncio.wait_for(request.future, timeout=request.timeout)
        except asyncio.TimeoutError:
            log.warning(f"Request {request.id} timed out")
            msg = f"Request timed out after {request.timeout}s"
            raise RequestTimeoutError(msg)

    async def _worker(self) -> None:
        """Background worker that processes requests."""
        while self._running:
            try:
                request = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # Check concurrency limit
            async with self._lock:
                active = self._active.get(request.model_id, 0)
                if active >= self.max_concurrent:
                    # Re-queue and wait
                    await self._queue.put(request)
                    await asyncio.sleep(0.1)
                    continue

                self._active[request.model_id] = active + 1

            # Execute request
            asyncio.create_task(self._execute(request))

    async def _execute(self, request: QueuedRequest) -> None:
        """Execute a queued request."""
        try:
            result = await request.coro
            request.future.set_result(result)
        except Exception as e:
            request.future.set_exception(e)
        finally:
            async with self._lock:
                self._active[request.model_id] = self._active.get(request.model_id, 1) - 1

    def get_stats(self) -> dict:
        """Get queue statistics."""
        return {
            "queue_size": self._queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "max_concurrent": self.max_concurrent,
            "active_requests": dict(self._active),
            "running": self._running,
        }


class QueueFullError(Exception):
    """Queue is at capacity."""


class RequestTimeoutError(Exception):
    """Request timed out."""


class BatchProcessor:
    """Batch processor for embedding requests.

    Groups multiple embedding requests and processes them together.
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_ms: int = 50,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        self._pending: dict[str, list[tuple[str, asyncio.Future]]] = {}  # model_id -> [(text, future)]
        self._lock = asyncio.Lock()
        self._batch_task: asyncio.Task | None = None
        self._process_fn: Callable | None = None

    def set_processor(self, fn: Callable[[str, list[str]], Coroutine[Any, Any, list[list[float]]]]) -> None:
        """Set the batch processing function.

        Args:
            fn: Async function (model_id, texts) -> embeddings
        """
        self._process_fn = fn

    async def embed(self, model_id: str, text: str) -> list[float]:
        """Add text to batch and wait for result.

        Args:
            model_id: Embedding model to use
            text: Text to embed

        Returns:
            Embedding vector
        """
        future: asyncio.Future[list[float]] = asyncio.Future()

        async with self._lock:
            if model_id not in self._pending:
                self._pending[model_id] = []

            self._pending[model_id].append((text, future))
            batch_size = len(self._pending[model_id])

        # Trigger batch if full
        if batch_size >= self.max_batch_size:
            asyncio.create_task(self._flush_batch(model_id))
        elif batch_size == 1:
            # First item - schedule flush after wait period
            asyncio.create_task(self._delayed_flush(model_id))

        return await future

    async def _delayed_flush(self, model_id: str) -> None:
        """Flush batch after delay."""
        await asyncio.sleep(self.max_wait_ms / 1000.0)
        await self._flush_batch(model_id)

    async def _flush_batch(self, model_id: str) -> None:
        """Process pending batch."""
        async with self._lock:
            if model_id not in self._pending or not self._pending[model_id]:
                return

            batch = self._pending.pop(model_id)

        if not batch:
            return

        texts = [t for t, _ in batch]
        futures = [f for _, f in batch]

        log.debug(f"Processing batch of {len(texts)} embeddings for {model_id}")

        try:
            if self._process_fn:
                embeddings = await self._process_fn(model_id, texts)
                for future, emb in zip(futures, embeddings, strict=False):
                    future.set_result(emb)
            else:
                msg = "No processor function set"
                raise RuntimeError(msg)
        except Exception as e:
            for future in futures:
                if not future.done():
                    future.set_exception(e)

    def get_stats(self) -> dict:
        """Get batch processor stats."""
        return {
            "max_batch_size": self.max_batch_size,
            "max_wait_ms": self.max_wait_ms,
            "pending_batches": {k: len(v) for k, v in self._pending.items()},
        }
