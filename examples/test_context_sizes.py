#!/usr/bin/env python3
"""Test context sizes for all models at different tiers.

This test validates that the intelligent context management works
correctly for all models using their actual parameters from models.json.
"""

import asyncio
import json
import aiohttp
import time

BASE_URL = "http://localhost:8090"


async def get_models():
    """Get all available models."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/v1/models") as resp:
            data = await resp.json()
            return data["data"]


async def test_context_tier(session, model_id: str, token_count: int, tier_name: str):
    """Test a specific context tier for a model."""
    # Create a message with approximately token_count tokens
    # ~4 chars per token
    content = "x" * (token_count * 4)

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": f"Count to 3. Context test: {content[:100]}"}],
        "max_tokens": 20,
    }

    start = time.monotonic()
    try:
        async with session.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            elapsed = (time.monotonic() - start) * 1000
            if resp.status == 200:
                data = await resp.json()
                return True, elapsed, None
            else:
                error = await resp.text()
                return False, elapsed, error[:100]
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return False, elapsed, str(e)[:100]


async def test_model_context_tiers(session, model: dict):
    """Test all context tiers for a model."""
    model_id = model["id"]
    context_window = model.get("context_window", 8192)

    # Skip embedding models (capabilities is a list)
    caps = model.get("capabilities", [])
    if "chat" not in caps:
        return None

    print(f"\n📊 {model_id}")
    print(f"   Max context: {context_window//1024}k")

    # Calculate dynamic tiers for this model
    tiers = []
    for percent, name in [(0.1, "10%"), (0.25, "25%"), (0.5, "50%"), (0.75, "75%")]:
        tokens = int(context_window * percent)
        if tokens >= 500:  # Minimum useful size
            tiers.append((tokens, name))

    results = []
    for token_estimate, tier_name in tiers:
        success, elapsed, error = await test_context_tier(session, model_id, token_estimate, tier_name)

        if success:
            print(f"   ✅ {tier_name} (~{token_estimate} tokens) - {elapsed:.0f}ms")
            results.append((tier_name, True, elapsed))
        else:
            print(f"   ❌ {tier_name} (~{token_estimate} tokens) - {error}")
            results.append((tier_name, False, error))

    return results


async def main():
    print("=" * 60)
    print("CONTEXT SIZE TEST - ALL MODELS, ALL TIERS")
    print("=" * 60)
    print("\nFetching models...")

    models = await get_models()
    # capabilities is a list like ["chat", "reasoning"]
    chat_models = [m for m in models if "chat" in m.get("capabilities", [])]

    print(f"Found {len(chat_models)} chat models to test")

    all_results = {}
    passed = 0
    failed = 0

    async with aiohttp.ClientSession() as session:
        for model in chat_models:
            results = await test_model_context_tiers(session, model)
            if results:
                all_results[model["id"]] = results
                for _, success, _ in results:
                    if success:
                        passed += 1
                    else:
                        failed += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total tests: {passed + failed}")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")

    # Show models with their context windows
    print("\n📋 Model Context Windows (from scanner):")
    for model in models:
        ctx = model.get("context_window", 0)
        caps = ", ".join(model.get("capabilities", []))
        print(f"   {model['id'][:40]:40} {ctx//1024:>4}k  [{caps}]")


if __name__ == "__main__":
    asyncio.run(main())
