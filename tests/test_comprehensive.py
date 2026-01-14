"""Comprehensive Pantropic Test Suite.

All tests passing - covers config, types, sessions, queue, tokenizer, allocator.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

pytestmark = pytest.mark.asyncio


# ============================================================================
# CONFIG TESTS
# ============================================================================

class TestConfig:
    """Tests for pantropic.core.config."""

    def test_config_defaults(self):
        from pantropic.core.config import Config
        config = Config()
        assert config.host == "0.0.0.0"
        assert config.default_context == 8192
        assert config.flash_attention is True

    def test_config_validation_invalid_port(self):
        from pantropic.core.config import Config
        config = Config()
        config.port = 99999
        warnings = config.validate()
        assert any("port" in w.lower() for w in warnings)

    def test_config_from_dict(self):
        from pantropic.core.config import Config
        data = {"host": "127.0.0.1", "port": 9000}
        config = Config._from_dict(data)
        assert config.host == "127.0.0.1"
        assert config.port == 9000


# ============================================================================
# TYPES TESTS
# ============================================================================

class TestTypes:
    """Tests for pantropic.core.types."""

    def test_model_capabilities(self):
        from pantropic.core.types import ModelCapabilities
        caps = ModelCapabilities(chat=True, tools=True)
        assert caps.chat is True
        assert caps.tools is True
        assert "chat" in caps.as_list()


# ============================================================================
# SESSION TESTS
# ============================================================================

class TestSessions:
    """Tests for pantropic.inference.sessions."""

    def test_session_creation(self):
        from pantropic.inference.sessions import Session
        session = Session(id="test-123", model_id="llama-3b")
        assert session.id == "test-123"
        assert session.model_id == "llama-3b"

    def test_session_add_message(self):
        from pantropic.inference.sessions import Session
        session = Session(id="test-123", model_id="llama-3b")
        session.add_message("user", "Hello!")
        session.add_message("assistant", "Hi!")
        assert session.message_count == 2

    async def test_session_manager_create(self):
        from pantropic.inference.sessions import SessionManager
        manager = SessionManager(max_sessions=10)
        session = await manager.create("test-model")
        assert session.model_id == "test-model"

    async def test_session_manager_get(self):
        from pantropic.inference.sessions import SessionManager
        manager = SessionManager(max_sessions=10)
        created = await manager.create("test-model")
        retrieved = await manager.get(created.id)
        assert retrieved is not None

    async def test_session_manager_delete(self):
        from pantropic.inference.sessions import SessionManager
        manager = SessionManager(max_sessions=10)
        created = await manager.create("test-model")
        await manager.delete(created.id)
        retrieved = await manager.get(created.id)
        assert retrieved is None


# ============================================================================
# QUEUE TESTS
# ============================================================================

class TestRequestQueue:
    """Tests for pantropic.inference.queue."""

    def test_priority_ordering(self):
        from pantropic.inference.queue import Priority
        assert Priority.HIGH > Priority.NORMAL
        assert Priority.NORMAL > Priority.LOW

    def test_queue_stats(self):
        from pantropic.inference.queue import RequestQueue
        queue = RequestQueue(max_concurrent=2, max_queue_size=50)
        stats = queue.get_stats()
        assert stats["max_concurrent"] == 2


# ============================================================================
# TOKENIZER TESTS
# ============================================================================

class TestTokenizer:
    """Tests for pantropic.utils.tokenizer."""

    def test_count_tokens_empty(self):
        from pantropic.utils.tokenizer import count_tokens
        assert count_tokens("") == 0

    def test_count_tokens_text(self):
        from pantropic.utils.tokenizer import count_tokens
        tokens = count_tokens("Hello, world!")
        assert tokens > 0

    def test_count_message_tokens(self):
        from pantropic.utils.tokenizer import count_message_tokens
        messages = [{"role": "user", "content": "Hello!"}]
        tokens = count_message_tokens(messages)
        assert tokens > 0


# ============================================================================
# VRAM ALLOCATOR TESTS
# ============================================================================

class TestVRAMAllocator:
    """Tests for pantropic.hardware.allocator."""

    def _make_mock_gpu(self, free_vram=8.0, total_vram=16.0):
        mock_gpu = MagicMock()
        mock_status = MagicMock()
        mock_status.free_vram_gb = free_vram
        mock_status.total_vram_gb = total_vram
        mock_gpu.get_status.return_value = mock_status
        return mock_gpu

    def test_reserve_and_commit(self):
        from pantropic.hardware.allocator import VRAMAllocator
        allocator = VRAMAllocator(self._make_mock_gpu())
        reservation = allocator.reserve("model-1", 2.0)
        assert reservation is not None
        success = allocator.commit("model-1")
        assert success is True

    def test_reserve_and_cancel(self):
        from pantropic.hardware.allocator import VRAMAllocator
        allocator = VRAMAllocator(self._make_mock_gpu())
        allocator.reserve("model-1", 2.0)
        allocator.cancel("model-1")
        assert "model-1" not in allocator._reservations

    def test_free(self):
        from pantropic.hardware.allocator import VRAMAllocator
        allocator = VRAMAllocator(self._make_mock_gpu())
        allocator.reserve("model-1", 2.0)
        allocator.commit("model-1")
        allocator.free("model-1")
        assert "model-1" not in allocator._allocations

    def test_get_stats(self):
        from pantropic.hardware.allocator import VRAMAllocator
        allocator = VRAMAllocator(self._make_mock_gpu())
        stats = allocator.get_stats()
        assert "available_gb" in stats
