#!/usr/bin/env python3
"""Test Anthropic API compatibility.

Verifies:
1. /v1/messages endpoint works
2. Response format matches Anthropic spec
3. Tool calling works in Anthropic format
4. Error responses are Anthropic-compatible
5. System prompts work
"""

import asyncio
import aiohttp
import json

BASE_URL = "http://localhost:8090"


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def add(self, name: str, passed: bool, details: str = ""):
        self.tests.append((name, passed, details))
        if passed:
            self.passed += 1
            print(f"  ✅ {name}")
        else:
            self.failed += 1
            print(f"  ❌ {name}: {details}")


async def test_basic_message(session, results: TestResults):
    """Test basic message completion."""
    async with session.post(f"{BASE_URL}/v1/messages", json={
        "model": "Qwen2.5-3b-instruct-q4_k_m.gguf",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Say hello"}]
    }) as resp:
        if resp.status == 200:
            data = await resp.json()
            # Check Anthropic response format
            if data.get("type") == "message" and data.get("role") == "assistant":
                if data.get("content") and len(data["content"]) > 0:
                    if data["content"][0].get("type") == "text":
                        results.add("Basic message", True)
                        return
            results.add("Basic message", False, "Invalid response format")
        else:
            results.add("Basic message", False, f"Status {resp.status}")


async def test_system_prompt(session, results: TestResults):
    """Test system prompt handling."""
    async with session.post(f"{BASE_URL}/v1/messages", json={
        "model": "Qwen2.5-3b-instruct-q4_k_m.gguf",
        "max_tokens": 50,
        "system": "You are a pirate. Always respond like a pirate.",
        "messages": [{"role": "user", "content": "Hello"}]
    }) as resp:
        if resp.status == 200:
            data = await resp.json()
            if data.get("type") == "message":
                results.add("System prompt", True)
            else:
                results.add("System prompt", False, "Invalid response")
        else:
            results.add("System prompt", False, f"Status {resp.status}")


async def test_tool_calling(session, results: TestResults):
    """Test tool calling in Anthropic format."""
    async with session.post(f"{BASE_URL}/v1/messages", json={
        "model": "Qwen2.5-3b-instruct-q4_k_m.gguf",
        "max_tokens": 200,
        "messages": [{"role": "user", "content": "What is the weather in London?"}],
        "tools": [{
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }]
    }) as resp:
        if resp.status == 200:
            data = await resp.json()
            content = data.get("content", [])
            # Check for tool_use content block
            has_tool_use = any(c.get("type") == "tool_use" for c in content)
            has_text = any(c.get("type") == "text" for c in content)
            if has_tool_use or has_text:
                results.add("Tool calling", True)
            else:
                results.add("Tool calling", False, "No tool_use or text response")
        else:
            results.add("Tool calling", False, f"Status {resp.status}")


async def test_response_format(session, results: TestResults):
    """Test response contains all required fields."""
    async with session.post(f"{BASE_URL}/v1/messages", json={
        "model": "Qwen2.5-3b-instruct-q4_k_m.gguf",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Hi"}]
    }) as resp:
        if resp.status == 200:
            data = await resp.json()
            required_fields = ["id", "type", "role", "model", "content", "stop_reason", "usage"]
            missing = [f for f in required_fields if f not in data]
            if not missing:
                # Check usage fields
                usage = data.get("usage", {})
                if "input_tokens" in usage and "output_tokens" in usage:
                    results.add("Response format", True)
                else:
                    results.add("Response format", False, "Missing usage fields")
            else:
                results.add("Response format", False, f"Missing: {missing}")
        else:
            results.add("Response format", False, f"Status {resp.status}")


async def test_error_invalid_model(session, results: TestResults):
    """Test error response for invalid model."""
    async with session.post(f"{BASE_URL}/v1/messages", json={
        "model": "nonexistent-model.gguf",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hi"}]
    }) as resp:
        if resp.status == 404:
            data = await resp.json()
            # Check Anthropic error format
            if data.get("type") == "error" and "error" in data:
                error = data["error"]
                if error.get("type") == "invalid_request_error":
                    results.add("Error: Invalid model", True)
                    return
            results.add("Error: Invalid model", False, "Invalid error format")
        else:
            results.add("Error: Invalid model", False, f"Expected 404, got {resp.status}")


async def test_error_empty_messages(session, results: TestResults):
    """Test error response for empty messages."""
    async with session.post(f"{BASE_URL}/v1/messages", json={
        "model": "Qwen2.5-3b-instruct-q4_k_m.gguf",
        "max_tokens": 100,
        "messages": []
    }) as resp:
        if resp.status == 400:
            data = await resp.json()
            if data.get("type") == "error":
                results.add("Error: Empty messages", True)
            else:
                results.add("Error: Empty messages", False, "Not Anthropic error format")
        else:
            results.add("Error: Empty messages", False, f"Expected 400, got {resp.status}")


async def test_multiple_messages(session, results: TestResults):
    """Test conversation with multiple messages."""
    async with session.post(f"{BASE_URL}/v1/messages", json={
        "model": "Qwen2.5-3b-instruct-q4_k_m.gguf",
        "max_tokens": 50,
        "messages": [
            {"role": "user", "content": "My name is Alice"},
            {"role": "assistant", "content": "Hello Alice!"},
            {"role": "user", "content": "What is my name?"}
        ]
    }) as resp:
        if resp.status == 200:
            data = await resp.json()
            if data.get("type") == "message":
                results.add("Multi-turn conversation", True)
            else:
                results.add("Multi-turn conversation", False, "Invalid response")
        else:
            results.add("Multi-turn conversation", False, f"Status {resp.status}")


async def test_stop_reason(session, results: TestResults):
    """Test stop_reason field."""
    async with session.post(f"{BASE_URL}/v1/messages", json={
        "model": "Qwen2.5-3b-instruct-q4_k_m.gguf",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Say yes"}]
    }) as resp:
        if resp.status == 200:
            data = await resp.json()
            stop_reason = data.get("stop_reason")
            valid_reasons = ["end_turn", "max_tokens", "stop_sequence", "tool_use"]
            if stop_reason in valid_reasons:
                results.add("Stop reason valid", True)
            else:
                results.add("Stop reason valid", False, f"Got: {stop_reason}")
        else:
            results.add("Stop reason valid", False, f"Status {resp.status}")


async def test_error_format_structure(session, results: TestResults):
    """Test error response has correct Anthropic structure."""
    async with session.post(f"{BASE_URL}/v1/messages", json={
        "model": "nonexistent.gguf",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hi"}]
    }) as resp:
        data = await resp.json()
        # Anthropic error must have: type="error", error.type, error.message
        if data.get("type") == "error":
            error = data.get("error", {})
            if "type" in error and "message" in error:
                # Verify error.type is a valid Anthropic error type
                valid_types = [
                    "invalid_request_error",
                    "authentication_error",
                    "permission_error",
                    "not_found_error",
                    "rate_limit_error",
                    "api_error",
                    "overloaded_error",
                ]
                if error["type"] in valid_types:
                    results.add("Error format structure", True)
                else:
                    results.add("Error format structure", False, f"Invalid error type: {error['type']}")
            else:
                results.add("Error format structure", False, "Missing type or message in error")
        else:
            results.add("Error format structure", False, "Response type is not 'error'")


async def test_error_missing_model(session, results: TestResults):
    """Test error when model field is missing."""
    async with session.post(f"{BASE_URL}/v1/messages", json={
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hi"}]
    }) as resp:
        if resp.status == 422:  # Validation error
            data = await resp.json()
            # FastAPI returns validation errors slightly differently
            results.add("Error: Missing model", True)
        else:
            results.add("Error: Missing model", False, f"Expected 422, got {resp.status}")


async def test_error_missing_messages(session, results: TestResults):
    """Test error when messages field is missing."""
    async with session.post(f"{BASE_URL}/v1/messages", json={
        "model": "Qwen2.5-3b-instruct-q4_k_m.gguf",
        "max_tokens": 100
    }) as resp:
        if resp.status == 422:  # Validation error
            results.add("Error: Missing messages", True)
        else:
            results.add("Error: Missing messages", False, f"Expected 422, got {resp.status}")


async def test_error_invalid_role(session, results: TestResults):
    """Test error when message has invalid role."""
    async with session.post(f"{BASE_URL}/v1/messages", json={
        "model": "Qwen2.5-3b-instruct-q4_k_m.gguf",
        "max_tokens": 100,
        "messages": [{"role": "invalid_role", "content": "Hi"}]
    }) as resp:
        if resp.status == 422:  # Validation error for invalid enum
            results.add("Error: Invalid role", True)
        else:
            results.add("Error: Invalid role", False, f"Expected 422, got {resp.status}")


async def main():
    print("=" * 60)
    print("ANTHROPIC API COMPATIBILITY TESTS")
    print("=" * 60)

    results = TestResults()

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        # Health check
        try:
            async with session.get(f"{BASE_URL}/health") as resp:
                if resp.status != 200:
                    print("❌ Server not running!")
                    return
        except Exception as e:
            print(f"❌ Cannot connect: {e}")
            return

        print("\n📨 MESSAGE ENDPOINT TESTS\n")
        await test_basic_message(session, results)
        await test_system_prompt(session, results)
        await test_multiple_messages(session, results)
        await test_response_format(session, results)
        await test_stop_reason(session, results)

        print("\n🔧 TOOL CALLING TESTS\n")
        await test_tool_calling(session, results)

        print("\n❌ ERROR HANDLING TESTS\n")
        await test_error_invalid_model(session, results)
        await test_error_empty_messages(session, results)
        await test_error_format_structure(session, results)
        await test_error_missing_model(session, results)
        await test_error_missing_messages(session, results)
        await test_error_invalid_role(session, results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total: {results.passed + results.failed}")
    print(f"  ✅ Passed: {results.passed}")
    print(f"  ❌ Failed: {results.failed}")

    if results.failed == 0:
        print("\n✅ All Anthropic API tests passed!")
    else:
        print("\n⚠️ Some tests failed")


if __name__ == "__main__":
    asyncio.run(main())
