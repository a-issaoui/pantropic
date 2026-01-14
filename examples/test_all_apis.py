#!/usr/bin/env python3
"""Pantropic API Test Suite.

Comprehensive tests for all APIs and model capabilities.
Run with: python examples/test_all_apis.py

Requires server running at http://localhost:8090
"""

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from typing import Any

import aiohttp


@dataclass
class TestResult:
    """Test result container."""
    name: str
    passed: bool
    duration_ms: float
    error: str | None = None
    details: dict | None = None


class PantropicAPITester:
    """Comprehensive API tester for Pantropic."""

    def __init__(self, base_url: str = "http://localhost:8090"):
        self.base_url = base_url
        self.results: list[TestResult] = []
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    async def run_test(self, name: str, coro) -> TestResult:
        """Run a single test and record result."""
        start = time.monotonic()
        try:
            details = await coro
            duration = (time.monotonic() - start) * 1000
            result = TestResult(name=name, passed=True, duration_ms=duration, details=details)
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            result = TestResult(name=name, passed=False, duration_ms=duration, error=str(e))
        
        self.results.append(result)
        status = "✅" if result.passed else "❌"
        print(f"  {status} {name} ({result.duration_ms:.0f}ms)")
        if result.error:
            print(f"     Error: {result.error}")
        return result

    # =========================================================================
    # HEALTH & SYSTEM ENDPOINTS
    # =========================================================================

    async def test_health(self) -> dict:
        """Test /health endpoint."""
        async with self.session.get(f"{self.base_url}/health") as resp:
            assert resp.status == 200, f"Status {resp.status}"
            data = await resp.json()
            assert data["status"] == "healthy"
            return data

    async def test_metrics(self) -> dict:
        """Test /metrics endpoint."""
        async with self.session.get(f"{self.base_url}/metrics") as resp:
            assert resp.status == 200, f"Status {resp.status}"
            data = await resp.json()
            assert "system" in data
            assert "models" in data
            return data

    async def test_root(self) -> dict:
        """Test / endpoint."""
        async with self.session.get(f"{self.base_url}/") as resp:
            assert resp.status == 200, f"Status {resp.status}"
            data = await resp.json()
            assert data["service"] == "Pantropic"
            return data

    # =========================================================================
    # MODELS ENDPOINTS
    # =========================================================================

    async def test_list_models(self) -> dict:
        """Test GET /v1/models."""
        async with self.session.get(f"{self.base_url}/v1/models") as resp:
            assert resp.status == 200, f"Status {resp.status}"
            data = await resp.json()
            assert data["object"] == "list"
            assert len(data["data"]) > 0, "No models found"
            return {"count": len(data["data"])}

    async def test_get_model(self, model_id: str) -> dict:
        """Test GET /v1/models/{model_id}."""
        async with self.session.get(f"{self.base_url}/v1/models/{model_id}") as resp:
            assert resp.status == 200, f"Status {resp.status}"
            data = await resp.json()
            assert data["id"] == model_id
            return data

    async def test_list_loaded_models(self) -> dict:
        """Test GET /v1/models/loaded."""
        async with self.session.get(f"{self.base_url}/v1/models/loaded") as resp:
            assert resp.status == 200, f"Status {resp.status}"
            data = await resp.json()
            return {"loaded": data.get("loaded_models", [])}

    async def test_load_model(self, model_id: str) -> dict:
        """Test POST /v1/models/{model_id}/load."""
        async with self.session.post(f"{self.base_url}/v1/models/{model_id}/load") as resp:
            assert resp.status == 200, f"Status {resp.status}"
            data = await resp.json()
            return data

    async def test_unload_model(self, model_id: str) -> dict:
        """Test DELETE /v1/models/{model_id}/unload."""
        async with self.session.delete(f"{self.base_url}/v1/models/{model_id}/unload") as resp:
            assert resp.status == 200, f"Status {resp.status}"
            data = await resp.json()
            return data

    # =========================================================================
    # CHAT COMPLETIONS
    # =========================================================================

    async def test_chat_completion(self, model_id: str) -> dict:
        """Test POST /v1/chat/completions (non-streaming)."""
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 20,
            "temperature": 0.7,
        }
        async with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        ) as resp:
            assert resp.status == 200, f"Status {resp.status}"
            data = await resp.json()
            assert "choices" in data
            assert len(data["choices"]) > 0
            content = data["choices"][0]["message"]["content"]
            return {"response": content[:50]}

    async def test_chat_completion_stream(self, model_id: str) -> dict:
        """Test POST /v1/chat/completions (streaming)."""
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Count from 1 to 3."}],
            "max_tokens": 30,
            "stream": True,
        }
        chunks = []
        async with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        ) as resp:
            assert resp.status == 200, f"Status {resp.status}"
            async for line in resp.content:
                line = line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunks.append(line[6:])
        
        assert len(chunks) > 0, "No streaming chunks received"
        return {"chunks": len(chunks)}

    # =========================================================================
    # EMBEDDINGS
    # =========================================================================

    async def test_embedding_single(self, model_id: str) -> dict:
        """Test POST /v1/embeddings (single text)."""
        payload = {
            "model": model_id,
            "input": "Hello world",
        }
        async with self.session.post(
            f"{self.base_url}/v1/embeddings",
            json=payload,
        ) as resp:
            assert resp.status == 200, f"Status {resp.status}"
            data = await resp.json()
            assert "data" in data
            assert len(data["data"]) == 1
            dim = len(data["data"][0]["embedding"])
            return {"dimension": dim}

    async def test_embedding_batch(self, model_id: str) -> dict:
        """Test POST /v1/embeddings (batch)."""
        payload = {
            "model": model_id,
            "input": ["Hello", "World", "Test"],
        }
        async with self.session.post(
            f"{self.base_url}/v1/embeddings",
            json=payload,
        ) as resp:
            assert resp.status == 200, f"Status {resp.status}"
            data = await resp.json()
            assert len(data["data"]) == 3
            return {"count": 3}

    # =========================================================================
    # SESSIONS
    # =========================================================================

    async def test_session_create(self) -> dict:
        """Test POST /v1/sessions."""
        payload = {"model_id": "test-model"}
        async with self.session.post(
            f"{self.base_url}/v1/sessions",
            json=payload,
        ) as resp:
            assert resp.status == 200, f"Status {resp.status}"
            data = await resp.json()
            return {"session_id": data["id"][:8]}

    async def test_session_list(self) -> dict:
        """Test GET /v1/sessions."""
        async with self.session.get(f"{self.base_url}/v1/sessions") as resp:
            assert resp.status == 200, f"Status {resp.status}"
            data = await resp.json()
            return {"count": data["stats"]["active_sessions"]}

    async def test_session_get(self, session_id: str) -> dict:
        """Test GET /v1/sessions/{session_id}."""
        async with self.session.get(f"{self.base_url}/v1/sessions/{session_id}") as resp:
            assert resp.status == 200, f"Status {resp.status}"
            data = await resp.json()
            return {"messages": data["message_count"]}

    async def test_session_add_message(self, session_id: str) -> dict:
        """Test POST /v1/sessions/{session_id}/messages."""
        payload = {"role": "user", "content": "Test message"}
        async with self.session.post(
            f"{self.base_url}/v1/sessions/{session_id}/messages",
            json=payload,
        ) as resp:
            assert resp.status == 200, f"Status {resp.status}"
            return {"status": "added"}

    async def test_session_delete(self, session_id: str) -> dict:
        """Test DELETE /v1/sessions/{session_id}."""
        async with self.session.delete(f"{self.base_url}/v1/sessions/{session_id}") as resp:
            assert resp.status == 200, f"Status {resp.status}"
            return {"status": "deleted"}

    # =========================================================================
    # CAPABILITY TESTS
    # =========================================================================

    async def test_model_capabilities(self, model_id: str, capabilities: list[str]) -> dict:
        """Test model based on its capabilities."""
        results = {}
        
        if "chat" in capabilities:
            try:
                await self.test_chat_completion(model_id)
                results["chat"] = "✅"
            except Exception as e:
                results["chat"] = f"❌ {e}"
        
        if "embed" in capabilities:
            try:
                await self.test_embedding_single(model_id)
                results["embed"] = "✅"
            except Exception as e:
                results["embed"] = f"❌ {e}"
        
        return results

    # =========================================================================
    # MAIN TEST RUNNER
    # =========================================================================

    async def run_all_tests(self):
        """Run complete test suite."""
        print("\n" + "=" * 60)
        print("PANTROPIC API TEST SUITE")
        print("=" * 60)

        # 1. Health & System Tests
        print("\n📡 HEALTH & SYSTEM")
        await self.run_test("Health Check", self.test_health())
        await self.run_test("Metrics", self.test_metrics())
        await self.run_test("Root Endpoint", self.test_root())

        # 2. Models Tests
        print("\n📦 MODELS")
        models_result = await self.run_test("List Models", self.test_list_models())
        await self.run_test("List Loaded Models", self.test_list_loaded_models())

        # Get first chat model and embed model
        async with self.session.get(f"{self.base_url}/v1/models") as resp:
            models_data = await resp.json()
            models = models_data["data"]

        chat_model = None
        embed_model = None
        for m in models:
            caps = m.get("capabilities", {})
            # Handle both list and dict formats
            if isinstance(caps, list):
                cap_list = caps
            else:
                cap_list = [k for k, v in caps.items() if v]
            
            if "chat" in cap_list and not chat_model:
                chat_model = m["id"]
            if "embed" in cap_list and not embed_model:
                embed_model = m["id"]

        if chat_model:
            await self.run_test(f"Get Model ({chat_model[:20]})", self.test_get_model(chat_model))

        # 3. Chat Completion Tests
        print("\n💬 CHAT COMPLETIONS")
        if chat_model:
            await self.run_test("Chat (non-stream)", self.test_chat_completion(chat_model))
            await self.run_test("Chat (streaming)", self.test_chat_completion_stream(chat_model))
        else:
            print("  ⚠️  No chat model found")

        # 4. Embeddings Tests
        print("\n📊 EMBEDDINGS")
        if embed_model:
            await self.run_test("Embedding (single)", self.test_embedding_single(embed_model))
            await self.run_test("Embedding (batch)", self.test_embedding_batch(embed_model))
        else:
            print("  ⚠️  No embedding model found")

        # 5. Sessions Tests
        print("\n🗂️  SESSIONS")
        session_result = await self.run_test("Create Session", self.test_session_create())
        await self.run_test("List Sessions", self.test_session_list())
        
        if session_result.passed and session_result.details:
            # Get full session ID
            async with self.session.get(f"{self.base_url}/v1/sessions") as resp:
                sessions_data = await resp.json()
                if sessions_data["sessions"]:
                    session_id = sessions_data["sessions"][0]["id"]
                    await self.run_test("Get Session", self.test_session_get(session_id))
                    await self.run_test("Add Message", self.test_session_add_message(session_id))
                    await self.run_test("Delete Session", self.test_session_delete(session_id))

        # 6. Test ALL Models by Capability
        print("\n🧪 MODEL CAPABILITY TESTS")
        tested = 0
        for m in models:  # Test ALL models
            model_id = m["id"]
            caps = m.get("capabilities", {})
            # Handle both list and dict formats
            if isinstance(caps, list):
                cap_list = caps
            else:
                cap_list = [k for k, v in caps.items() if v]
            
            if cap_list:
                tested += 1
                await self.run_test(
                    f"{model_id[:30]} [{', '.join(cap_list)}]",
                    self.test_model_capabilities(model_id, cap_list)
                )
        
        if tested == 0:
            print("  ⚠️  No models with capabilities to test")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total_time = sum(r.duration_ms for r in self.results)
        
        print(f"  Total: {len(self.results)} tests")
        print(f"  ✅ Passed: {passed}")
        print(f"  ❌ Failed: {failed}")
        print(f"  ⏱️  Time: {total_time/1000:.2f}s")
        print()

        if failed > 0:
            print("Failed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  ❌ {r.name}: {r.error}")
            print()

        return failed == 0


async def main():
    """Main entry point."""
    print("Starting Pantropic API Test Suite...")
    print("Connecting to http://localhost:8090")
    
    async with PantropicAPITester("http://localhost:8090") as tester:
        try:
            success = await tester.run_all_tests()
            sys.exit(0 if success else 1)
        except aiohttp.ClientConnectorError:
            print("\n❌ ERROR: Cannot connect to server at http://localhost:8090")
            print("   Make sure Pantropic is running: python -m pantropic.main")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
