"""Pantropic - Tokenizer Utilities.

Fast token counting with fallback estimation.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pantropic.observability.logging import get_logger

log = get_logger("tokenizer")

# Try to import tiktoken for accurate counting
try:
    import tiktoken
    TIKTOKEN_OK = True
except ImportError:
    TIKTOKEN_OK = False
    log.debug("tiktoken not available - using estimation")


@lru_cache(maxsize=10)
def get_encoding(encoding_name: str = "cl100k_base"):
    """Get cached tokenizer encoding."""
    if not TIKTOKEN_OK:
        return None
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        return None


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text.

    Uses tiktoken if available, else estimates.
    """
    if not text:
        return 0

    enc = get_encoding(encoding_name)
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass

    # Fallback: ~4 chars per token
    return len(text) // 4


def count_message_tokens(
    messages: list[dict[str, Any]],
    encoding_name: str = "cl100k_base",
) -> int:
    """Count tokens in messages.

    Accounts for message formatting overhead.
    """
    total = 0

    for msg in messages:
        # Message overhead (~4 tokens per message)
        total += 4

        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content, encoding_name)
        elif isinstance(content, list):
            # Multimodal: text parts only
            for part in content:
                if part.get("type") == "text":
                    total += count_tokens(part.get("text", ""), encoding_name)

    # Final assistant prompt
    total += 3

    return total


def estimate_max_tokens(
    messages: list[dict[str, Any]],
    context_length: int,
    encoding_name: str = "cl100k_base",
) -> int:
    """Estimate maximum tokens available for generation.

    Returns context_length - prompt_tokens with safety margin.
    """
    prompt_tokens = count_message_tokens(messages, encoding_name)
    available = context_length - prompt_tokens - 100  # Safety margin
    return max(100, available)


def preload_tokenizers() -> bool:
    """Preload tokenizers for faster first use.

    Returns True if successful.
    """
    if not TIKTOKEN_OK:
        return False

    try:
        # Preload common encodings
        for enc_name in ["cl100k_base", "o200k_base"]:
            get_encoding(enc_name)
        log.info("Tokenizers preloaded")
        return True
    except Exception as e:
        log.warning(f"Failed to preload tokenizers: {e}")
        return False
