"""Pantropic - Session Manager with aiosqlite.

Persistent session storage with:
- aiosqlite for async database operations
- In-memory LRU cache for fast access
- Background cleanup of expired sessions
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pantropic.observability.logging import get_logger

log = get_logger("sessions")

# Try to import aiosqlite
try:
    import aiosqlite
    AIOSQLITE_OK = True
except ImportError:
    AIOSQLITE_OK = False
    log.warning("aiosqlite not available - using memory-only sessions")


@dataclass
class Session:
    """A conversation session."""
    id: str
    model_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def token_estimate(self) -> int:
        total = 0
        for msg in self.messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content) // 4
        return total

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self.last_accessed = time.time()

    def touch(self) -> None:
        self.last_accessed = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_id": self.model_id,
            "messages": self.messages,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "message_count": self.message_count,
            "token_estimate": self.token_estimate,
            "metadata": self.metadata,
        }


class SessionManager:
    """Async session manager with SQLite persistence.

    Features:
    - aiosqlite for persistent storage
    - In-memory LRU cache (fast reads)
    - Write-through caching (immediate persistence)
    - Background expired session cleanup
    """

    def __init__(
        self,
        max_sessions: int = 100,
        session_timeout: float = 3600.0,
        storage_dir: Path | str | None = None,
    ) -> None:
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        self.storage_dir = Path(storage_dir) if storage_dir else None

        # In-memory cache (LRU)
        self._cache: OrderedDict[str, Session] = OrderedDict()
        self._lock = asyncio.Lock()

        # Database
        self._db: aiosqlite.Connection | None = None
        self._db_path: Path | None = None

        if self.storage_dir and AIOSQLITE_OK:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = self.storage_dir / "sessions.db"

        log.info(f"Session manager: max={max_sessions}, persist={self._db_path is not None}")

    async def initialize(self) -> None:
        """Initialize database connection."""
        if self._db_path and AIOSQLITE_OK:
            self._db = await aiosqlite.connect(str(self._db_path))
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    messages TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    metadata TEXT
                )
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed ON sessions(last_accessed)
            """)
            await self._db.commit()

            # Load recent sessions into cache
            async with self._db.execute(
                "SELECT * FROM sessions ORDER BY last_accessed DESC LIMIT ?",
                (self.max_sessions // 2,)
            ) as cursor:
                async for row in cursor:
                    session = self._row_to_session(row)
                    self._cache[session.id] = session

            log.info(f"Loaded {len(self._cache)} sessions from database")

    def _row_to_session(self, row: tuple) -> Session:
        """Convert database row to Session."""
        return Session(
            id=row[0],
            model_id=row[1],
            messages=json.loads(row[2]),
            created_at=row[3],
            last_accessed=row[4],
            metadata=json.loads(row[5]) if row[5] else {},
        )

    async def create(self, model_id: str, session_id: str | None = None) -> Session:
        """Create a new session."""
        async with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self.max_sessions:
                oldest_id, _ = self._cache.popitem(last=False)
                log.debug(f"Evicted session {oldest_id} from cache")

            sid = session_id or str(uuid.uuid4())
            session = Session(id=sid, model_id=model_id)
            self._cache[sid] = session

            # Persist
            if self._db:
                await self._db.execute(
                    """INSERT OR REPLACE INTO sessions
                       (id, model_id, messages, created_at, last_accessed, metadata)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (sid, model_id, "[]", session.created_at, session.last_accessed, "{}")
                )
                await self._db.commit()

            log.debug(f"Created session {sid}")
            return session

    async def get(self, session_id: str) -> Session | None:
        """Get session by ID (cache first, then database)."""
        async with self._lock:
            # Check cache first
            if session_id in self._cache:
                session = self._cache[session_id]
                session.touch()
                self._cache.move_to_end(session_id)
                return session

            # Try database
            if self._db:
                async with self._db.execute(
                    "SELECT * FROM sessions WHERE id = ?", (session_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        session = self._row_to_session(row)
                        session.touch()
                        self._cache[session_id] = session
                        return session

            return None

    async def update(self, session_id: str, messages: list[dict[str, Any]]) -> bool:
        """Update session messages."""
        async with self._lock:
            session = self._cache.get(session_id)
            if not session:
                return False

            session.messages = messages
            session.touch()
            self._cache.move_to_end(session_id)

            # Persist
            if self._db:
                await self._db.execute(
                    "UPDATE sessions SET messages = ?, last_accessed = ? WHERE id = ?",
                    (json.dumps(messages), session.last_accessed, session_id)
                )
                await self._db.commit()

            return True

    async def add_message(self, session_id: str, role: str, content: str) -> bool:
        """Add a message to session."""
        async with self._lock:
            session = self._cache.get(session_id)
            if not session:
                return False

            session.add_message(role, content)
            self._cache.move_to_end(session_id)

            # Persist
            if self._db:
                await self._db.execute(
                    "UPDATE sessions SET messages = ?, last_accessed = ? WHERE id = ?",
                    (json.dumps(session.messages), session.last_accessed, session_id)
                )
                await self._db.commit()

            return True

    async def delete(self, session_id: str) -> bool:
        """Delete a session."""
        async with self._lock:
            if session_id in self._cache:
                del self._cache[session_id]

            if self._db:
                await self._db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                await self._db.commit()

            log.debug(f"Deleted session {session_id}")
            return True

    async def list(self) -> list[dict[str, Any]]:
        """List all sessions (summary only)."""
        async with self._lock:
            return [
                {
                    "id": s.id,
                    "model_id": s.model_id,
                    "message_count": s.message_count,
                    "created_at": s.created_at,
                    "last_accessed": s.last_accessed,
                }
                for s in self._cache.values()
            ]

    async def cleanup_expired(self) -> int:
        """Remove expired sessions."""
        now = time.time()
        expired = []

        async with self._lock:
            for sid, session in list(self._cache.items()):
                if now - session.last_accessed > self.session_timeout:
                    expired.append(sid)

            for sid in expired:
                del self._cache[sid]

            if self._db and expired:
                await self._db.execute(
                    f"DELETE FROM sessions WHERE id IN ({','.join('?' * len(expired))})",
                    expired
                )
                await self._db.commit()

        if expired:
            log.info(f"Cleaned up {len(expired)} expired sessions")

        return len(expired)

    def get_stats(self) -> dict[str, Any]:
        """Get manager statistics."""
        total_tokens = sum(s.token_estimate for s in self._cache.values())
        return {
            "active_sessions": len(self._cache),
            "max_sessions": self.max_sessions,
            "total_messages": sum(s.message_count for s in self._cache.values()),
            "total_tokens_estimate": total_tokens,
            "storage_enabled": self._db is not None,
            "storage_type": "aiosqlite" if self._db else "memory",
        }

    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            log.info("Session database closed")
