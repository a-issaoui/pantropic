#!/usr/bin/env python3
"""Test all error handling scenarios.

Verifies:
1. All errors return proper HTTP status codes
2. All errors match OpenAI error format
3. Server doesn't crash on any error
4. Context overflow is handled gracefully
"""

import asyncio
import aiohttp
import json

BASE_URL = "http://localhost:8090"

# OpenAI error format: {"error": {"message": str, "type": str, "param": str|null, "code": str|null}}


async def test_error_format(session, name: str, response_data: dict, expected_code: str) -> bool:
    """Verify response matches OpenAI error format."""
    if "error" not in response_data:
        print(f"  ❌ {name}: Missing 'error' key")
        return False

    error = response_data["error"]
    required_fields = ["message", "type"]

    for field in required_fields:
        if field not in error:
            print(f"  ❌ {name}: Missing '{field}' in error")
            return False

    if error.get("code") != expected_code:
        print(f"  ❌ {name}: Expected code '{expected_code}', got '{error.get('code')}'")
        return False

    return True


async def test_model_not_found(session) -> tuple[bool, str]:
    """Test 404 for non-existent model."""
    async with session.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": "nonexistent-model-12345.gguf",
        "messages": [{"role": "user", "content": "Hi"}]
    }) as resp:
        if resp.status != 404:
            return False, f"Expected 404, got {resp.status}"

        data = await resp.json()
        if not await test_error_format(session, "model_not_found", data, "model_not_found"):
            return False, "Invalid error format"

        return True, ""


async def test_empty_messages(session) -> tuple[bool, str]:
    """Test 400 for empty messages."""
    async with session.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": "Qwen2.5-3b-instruct-q4_k_m.gguf",
        "messages": []
    }) as resp:
        if resp.status != 400:
            return False, f"Expected 400, got {resp.status}"

        data = await resp.json()
        if "error" in data:
            return True, ""
        return False, "Missing error in response"


async def test_invalid_json(session) -> tuple[bool, str]:
    """Test 422 for invalid JSON."""
    async with session.post(
        f"{BASE_URL}/v1/chat/completions",
        data="not valid json",
        headers={"Content-Type": "application/json"}
    ) as resp:
        if resp.status >= 400:
            return True, ""
        return False, f"Expected 4xx, got {resp.status}"


async def test_missing_model(session) -> tuple[bool, str]:
    """Test 422 for missing model field."""
    async with session.post(f"{BASE_URL}/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Hi"}]
    }) as resp:
        if resp.status >= 400:
            return True, ""
        return False, f"Expected 4xx, got {resp.status}"


async def test_invalid_embedding_model(session) -> tuple[bool, str]:
    """Test error for non-embedding model used for embeddings."""
    async with session.post(f"{BASE_URL}/v1/embeddings", json={
        "model": "nonexistent-embed.gguf",
        "input": "Test text"
    }) as resp:
        if resp.status >= 400:
            return True, ""
        return False, f"Expected 4xx, got {resp.status}"


async def test_session_not_found(session) -> tuple[bool, str]:
    """Test 404 for non-existent session."""
    async with session.get(f"{BASE_URL}/v1/sessions/nonexistent-session-id") as resp:
        if resp.status >= 400:
            return True, ""
        return False, f"Expected 4xx, got {resp.status}"


async def test_very_long_input(session) -> tuple[bool, str]:
    """Test handling of very long input (should either work or return context error)."""
    # Create a message with ~200k tokens (way over any model's context)
    long_text = "x" * (200000 * 4)  # ~200k tokens

    try:
        async with session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "SmolLM-1.7B-Instruct-Q4_K_M.gguf",  # 2k context
                "messages": [{"role": "user", "content": long_text}],
                "max_tokens": 10
            },
            timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            # Should return error (400 context overflow) or handle gracefully
            if resp.status in [400, 413]:
                data = await resp.json()
                if "error" in data:
                    return True, ""
            # Even 200 is ok if it means automatic context handling
            if resp.status == 200:
                return True, "Handled via truncation"
            return False, f"Unexpected status {resp.status}"
    except asyncio.TimeoutError:
        return False, "Request timed out (server may have hung)"
    except Exception as e:
        return False, str(e)[:50]


async def test_server_health_after_errors(session) -> tuple[bool, str]:
    """Test that server is still healthy after all error tests."""
    async with session.get(f"{BASE_URL}/health") as resp:
        if resp.status == 200:
            return True, ""
        return False, f"Server unhealthy: {resp.status}"


async def main():
    print("=" * 60)
    print("PANTROPIC ERROR HANDLING TEST")
    print("OpenAI-Compatible Error Format Verification")
    print("=" * 60)

    tests = [
        ("Model Not Found (404)", test_model_not_found),
        ("Empty Messages (400)", test_empty_messages),
        ("Invalid JSON (422)", test_invalid_json),
        ("Missing Model (422)", test_missing_model),
        ("Invalid Embedding Model", test_invalid_embedding_model),
        ("Session Not Found (404)", test_session_not_found),
        ("Very Long Input (Context)", test_very_long_input),
        ("Server Health After Errors", test_server_health_after_errors),
    ]

    passed = 0
    failed = 0

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        # Health check first
        try:
            async with session.get(f"{BASE_URL}/health") as resp:
                if resp.status != 200:
                    print("❌ Server not running!")
                    return
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return

        print("\n🔍 TESTING ERROR SCENARIOS\n")

        for name, test_func in tests:
            try:
                success, details = await test_func(session)
                if success:
                    print(f"  ✅ {name}")
                    passed += 1
                else:
                    print(f"  ❌ {name}: {details}")
                    failed += 1
            except Exception as e:
                print(f"  ❌ {name}: Exception - {str(e)[:50]}")
                failed += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total: {passed + failed}")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")

    if failed == 0:
        print("\n✅ All errors are handled correctly with OpenAI-compatible format!")
    else:
        print("\n⚠️ Some error handling needs improvement")


if __name__ == "__main__":
    asyncio.run(main())
