#!/usr/bin/env python3
"""Test agent scenarios with model switching and context variations.

This test simulates real agent usage patterns:
1. Switching between different LLMs mid-conversation
2. Using different context sizes
3. Testing both OpenAI and Anthropic APIs
4. Tool calling as an agent would use it
"""

import asyncio
import aiohttp
import json
import time

BASE_URL = "http://localhost:8090"

# Models with different capabilities
MODELS = {
    "qwen": "Qwen2.5-3b-instruct-q4_k_m.gguf",
    "llama": "Llama-3.2-3B-Instruct-Q6_K.gguf",
    "phi": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
    "deepseek": "DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf",
}


class AgentTestResults:
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


async def test_openai_model_switch_different_contexts(session, results: AgentTestResults):
    """Agent scenario: Switch between models with different max_tokens."""
    # First request with Qwen, small context
    async with session.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODELS["qwen"],
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 10
    }) as resp:
        if resp.status != 200:
            results.add("OpenAI: Model 1 small context", False, f"Status {resp.status}")
            return
        data1 = await resp.json()

    # Second request with Llama, larger context
    async with session.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODELS["llama"],
        "messages": [{"role": "user", "content": "Explain quantum computing briefly."}],
        "max_tokens": 150
    }) as resp:
        if resp.status != 200:
            results.add("OpenAI: Model 2 larger context", False, f"Status {resp.status}")
            return
        data2 = await resp.json()

    # Third - back to first model with medium context
    async with session.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODELS["qwen"],
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 50
    }) as resp:
        if resp.status != 200:
            results.add("OpenAI: Back to Model 1", False, f"Status {resp.status}")
            return
        data3 = await resp.json()

    # All three should have content
    if all([
        data1.get("choices", [{}])[0].get("message", {}).get("content"),
        data2.get("choices", [{}])[0].get("message", {}).get("content"),
        data3.get("choices", [{}])[0].get("message", {}).get("content"),
    ]):
        results.add("OpenAI: Model switching with different contexts", True)
    else:
        results.add("OpenAI: Model switching with different contexts", False, "Missing content")


async def test_anthropic_model_switch_different_contexts(session, results: AgentTestResults):
    """Agent scenario: Switch between models using Anthropic API."""
    # First request with Qwen
    async with session.post(f"{BASE_URL}/v1/messages", json={
        "model": MODELS["qwen"],
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 20
    }) as resp:
        if resp.status != 200:
            results.add("Anthropic: Model 1", False, f"Status {resp.status}")
            return
        data1 = await resp.json()

    # Second with different model and larger context
    async with session.post(f"{BASE_URL}/v1/messages", json={
        "model": MODELS["llama"],
        "messages": [{"role": "user", "content": "Count from 1 to 10"}],
        "max_tokens": 100
    }) as resp:
        if resp.status != 200:
            results.add("Anthropic: Model 2", False, f"Status {resp.status}")
            return
        data2 = await resp.json()

    if data1.get("type") == "message" and data2.get("type") == "message":
        results.add("Anthropic: Model switching with different contexts", True)
    else:
        results.add("Anthropic: Model switching with different contexts", False, "Invalid response type")


async def test_mixed_api_model_switching(session, results: AgentTestResults):
    """Agent scenario: Mix OpenAI and Anthropic calls (like multi-provider agent)."""
    # OpenAI call
    async with session.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODELS["qwen"],
        "messages": [{"role": "user", "content": "Say 'OpenAI'"}],
        "max_tokens": 20
    }) as resp:
        if resp.status != 200:
            results.add("Mixed: OpenAI call", False, f"Status {resp.status}")
            return
        data1 = await resp.json()

    # Anthropic call
    async with session.post(f"{BASE_URL}/v1/messages", json={
        "model": MODELS["llama"],
        "messages": [{"role": "user", "content": "Say 'Anthropic'"}],
        "max_tokens": 20
    }) as resp:
        if resp.status != 200:
            results.add("Mixed: Anthropic call", False, f"Status {resp.status}")
            return
        data2 = await resp.json()

    # Back to OpenAI
    async with session.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODELS["qwen"],
        "messages": [{"role": "user", "content": "Say 'Mixed'"}],
        "max_tokens": 20
    }) as resp:
        if resp.status != 200:
            results.add("Mixed: Back to OpenAI", False, f"Status {resp.status}")
            return
        data3 = await resp.json()

    # Verify formats are correct
    has_openai_format = "choices" in data1 and "choices" in data3
    has_anthropic_format = data2.get("type") == "message"

    if has_openai_format and has_anthropic_format:
        results.add("Mixed API: OpenAI ↔ Anthropic switching", True)
    else:
        results.add("Mixed API: OpenAI ↔ Anthropic switching", False, "Format mismatch")


async def test_agent_tool_calling_openai(session, results: AgentTestResults):
    """Agent scenario: Tool calling workflow (OpenAI format)."""
    # Agent asks LLM to use a tool
    async with session.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODELS["qwen"],
        "messages": [{"role": "user", "content": "What's the weather in London?"}],
        "max_tokens": 200,
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        }]
    }) as resp:
        if resp.status != 200:
            results.add("OpenAI: Agent tool call", False, f"Status {resp.status}")
            return
        data = await resp.json()

    # Check if tool was called
    message = data.get("choices", [{}])[0].get("message", {})
    tool_calls = message.get("tool_calls")

    if tool_calls or message.get("content"):
        results.add("OpenAI: Agent tool calling", True)
    else:
        results.add("OpenAI: Agent tool calling", False, "No tool_calls or content")


async def test_agent_tool_calling_anthropic(session, results: AgentTestResults):
    """Agent scenario: Tool calling workflow (Anthropic format)."""
    async with session.post(f"{BASE_URL}/v1/messages", json={
        "model": MODELS["qwen"],
        "messages": [{"role": "user", "content": "Search for 'Python tutorials'"}],
        "max_tokens": 200,
        "tools": [{
            "name": "web_search",
            "description": "Search the web for information",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
        }]
    }) as resp:
        if resp.status != 200:
            results.add("Anthropic: Agent tool call", False, f"Status {resp.status}")
            return
        data = await resp.json()

    content = data.get("content", [])
    has_tool_use = any(c.get("type") == "tool_use" for c in content)
    has_text = any(c.get("type") == "text" for c in content)

    if has_tool_use or has_text:
        results.add("Anthropic: Agent tool calling", True)
    else:
        results.add("Anthropic: Agent tool calling", False, "No tool_use or text")


async def test_rapid_model_switching(session, results: AgentTestResults):
    """Agent scenario: Rapid switching between 4 different models."""
    models_to_test = list(MODELS.values())[:4]
    responses = []

    for i, model in enumerate(models_to_test):
        async with session.post(f"{BASE_URL}/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": f"Say '{i}'"}],
            "max_tokens": 10
        }) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get("choices", [{}])[0].get("message", {}).get("content"):
                    responses.append(True)
                else:
                    responses.append(False)
            else:
                responses.append(False)

    success_count = sum(responses)
    if success_count == len(models_to_test):
        results.add(f"Rapid model switching ({len(models_to_test)} models)", True)
    else:
        results.add(f"Rapid model switching ({len(models_to_test)} models)", False, f"{success_count}/{len(models_to_test)} worked")


async def test_context_size_variation(session, results: AgentTestResults):
    """Agent scenario: Same model with varying context sizes."""
    context_sizes = [10, 50, 100, 200, 500]
    success = True

    for max_tokens in context_sizes:
        async with session.post(f"{BASE_URL}/v1/chat/completions", json={
            "model": MODELS["qwen"],
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": max_tokens
        }) as resp:
            if resp.status != 200:
                success = False
                break

    if success:
        results.add(f"Context size variation ({len(context_sizes)} sizes)", True)
    else:
        results.add(f"Context size variation", False, f"Failed at max_tokens={max_tokens}")


async def test_streaming_model_switch(session, results: AgentTestResults):
    """Agent scenario: Streaming responses with model switching."""
    # OpenAI streaming
    async with session.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODELS["qwen"],
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 20,
        "stream": True
    }) as resp:
        if resp.status != 200:
            results.add("Streaming: OpenAI", False, f"Status {resp.status}")
            return
        chunks1 = []
        async for line in resp.content:
            if line:
                chunks1.append(line.decode())

    # Anthropic streaming
    async with session.post(f"{BASE_URL}/v1/messages", json={
        "model": MODELS["llama"],
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 20,
        "stream": True
    }) as resp:
        if resp.status != 200:
            results.add("Streaming: Anthropic", False, f"Status {resp.status}")
            return
        chunks2 = []
        async for line in resp.content:
            if line:
                chunks2.append(line.decode())

    if len(chunks1) > 1 and len(chunks2) > 1:
        results.add("Streaming: Both APIs with model switch", True)
    else:
        results.add("Streaming: Both APIs with model switch", False, "Not enough chunks")


async def main():
    print("=" * 60)
    print("AGENT SCENARIO TESTS")
    print("Model Switching & Context Variations")
    print("=" * 60)

    results = AgentTestResults()

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        # Health check
        try:
            async with session.get(f"{BASE_URL}/health") as resp:
                if resp.status != 200:
                    print("❌ Server not running!")
                    return
        except Exception as e:
            print(f"❌ Cannot connect: {e}")
            return

        print("\n🔄 MODEL SWITCHING TESTS\n")
        await test_openai_model_switch_different_contexts(session, results)
        await test_anthropic_model_switch_different_contexts(session, results)
        await test_mixed_api_model_switching(session, results)
        await test_rapid_model_switching(session, results)

        print("\n📏 CONTEXT SIZE TESTS\n")
        await test_context_size_variation(session, results)

        print("\n🔧 AGENT TOOL CALLING\n")
        await test_agent_tool_calling_openai(session, results)
        await test_agent_tool_calling_anthropic(session, results)

        print("\n📡 STREAMING TESTS\n")
        await test_streaming_model_switch(session, results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total: {results.passed + results.failed}")
    print(f"  ✅ Passed: {results.passed}")
    print(f"  ❌ Failed: {results.failed}")

    if results.failed == 0:
        print("\n✅ All agent scenario tests passed!")
    else:
        print("\n⚠️ Some tests failed")


if __name__ == "__main__":
    asyncio.run(main())
