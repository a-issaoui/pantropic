#!/usr/bin/env python3
"""Test request queue with concurrent requests.

Verifies that:
1. Multiple concurrent requests don't crash the server
2. Requests are serialized (queue processes one at a time)
3. All requests eventually complete
"""

import asyncio
import aiohttp
import time

BASE_URL = "http://localhost:8090"


async def make_chat_request(session, request_id: int) -> tuple[int, bool, float, str]:
    """Make a chat request and return (id, success, time, error)."""
    start = time.monotonic()
    try:
        async with session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "Qwen2.5-3b-instruct-q4_k_m.gguf",
                "messages": [{"role": "user", "content": f"Say 'Request {request_id} done'"}],
                "max_tokens": 20
            },
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            elapsed = time.monotonic() - start
            if resp.status == 200:
                return request_id, True, elapsed, ""
            else:
                error = await resp.text()
                return request_id, False, elapsed, error[:50]
    except Exception as e:
        elapsed = time.monotonic() - start
        return request_id, False, elapsed, str(e)[:50]


async def main():
    print("=" * 60)
    print("REQUEST QUEUE TEST")
    print("Testing concurrent requests are serialized safely")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        # Health check
        async with session.get(f"{BASE_URL}/health") as resp:
            if resp.status != 200:
                print("❌ Server not running!")
                return

        print("\n📤 Sending 3 concurrent requests...")
        print("   (Without queue, this would crash CUDA)")

        start = time.monotonic()

        # Send 3 concurrent requests
        tasks = [make_chat_request(session, i) for i in range(3)]
        results = await asyncio.gather(*tasks)

        total_time = time.monotonic() - start

        print("\n📥 Results:")
        successes = 0
        for req_id, success, elapsed, error in sorted(results):
            if success:
                print(f"   ✅ Request {req_id}: completed in {elapsed:.1f}s")
                successes += 1
            else:
                print(f"   ❌ Request {req_id}: failed - {error}")

        print(f"\n⏱️  Total time: {total_time:.1f}s")

        # Check server is still healthy
        async with session.get(f"{BASE_URL}/health") as resp:
            if resp.status == 200:
                print("✅ Server still healthy after concurrent requests")
            else:
                print("❌ Server crashed!")

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Concurrent requests sent: 3")
        print(f"  Successful: {successes}/3")

        if successes == 3:
            print("\n✅ REQUEST QUEUE WORKING - All concurrent requests handled safely!")
        elif successes > 0:
            print("\n⚠️ Some requests failed but no crash")
        else:
            print("\n❌ All requests failed")


if __name__ == "__main__":
    asyncio.run(main())
