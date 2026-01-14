"""Test fixtures for Pantropic scanner tests."""

import struct
from io import BytesIO
from typing import Any

import pytest


@pytest.fixture
def sample_gguf_header() -> bytes:
    """Create a minimal valid GGUF v3 header."""
    header = b"GGUF"  # Magic
    header += struct.pack("<I", 3)  # Version 3
    header += struct.pack("<Q", 0)  # Tensor count
    header += struct.pack("<Q", 0)  # Metadata count
    return header


@pytest.fixture
def sample_metadata() -> dict[str, Any]:
    """Provide sample GGUF metadata for testing."""
    return {
        "general.architecture": "llama",
        "general.name": "test-model",
        "general.file_type": 2,  # Q4_0
        "llama.embedding_length": 4096,
        "llama.block_count": 32,
        "llama.head_count": 32,
        "llama.head_count_kv": 32,
        "llama.feed_forward_length": 11008,
        "llama.context_length": 2048,
        "tokenizer.ggml.model": "llama",
        "tokenizer.ggml.tokens": ["<unk>", "the", "a"],
        "tokenizer.ggml.bos_token_id": 1,
        "tokenizer.ggml.eos_token_id": 2,
    }


@pytest.fixture
def uint32_bytes() -> BytesIO:
    """Create BytesIO with a uint32 value."""
    return BytesIO(struct.pack("<I", 42))


@pytest.fixture
def uint64_bytes() -> BytesIO:
    """Create BytesIO with a uint64 value."""
    return BytesIO(struct.pack("<Q", 1234567890))


@pytest.fixture
def string_bytes() -> BytesIO:
    """Create BytesIO with a GGUF string (length + data)."""
    test_str = "test_string"
    data = struct.pack("<Q", len(test_str)) + test_str.encode("utf-8")
    return BytesIO(data)
