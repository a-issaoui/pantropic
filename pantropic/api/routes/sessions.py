"""Pantropic - Session API Routes (Async).

Endpoints for managing conversation sessions with aiosqlite.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from pantropic.inference.sessions import SessionManager

router = APIRouter()


# Pydantic models
class CreateSessionRequest(BaseModel):
    model_id: str
    session_id: str | None = None


class AddMessageRequest(BaseModel):
    role: str
    content: str


class CreateSessionResponse(BaseModel):
    id: str
    model_id: str
    created_at: float


# Session manager singleton
_session_manager: SessionManager | None = None
_initialized: bool = False


async def get_session_manager() -> SessionManager:
    """Get or create session manager (async)."""
    global _session_manager, _initialized
    if _session_manager is None:
        from pathlib import Path
        _session_manager = SessionManager(
            max_sessions=100,
            session_timeout=3600.0,
            storage_dir=Path("sessions"),  # Persist to sessions/sessions.db
        )
    if not _initialized:
        await _session_manager.initialize()
        _initialized = True
    return _session_manager


@router.post("", response_model=CreateSessionResponse)
async def create_session(
    request: CreateSessionRequest,
    session_mgr: SessionManager = Depends(get_session_manager),
):
    """Create a new conversation session."""
    session = await session_mgr.create(
        model_id=request.model_id,
        session_id=request.session_id,
    )
    return CreateSessionResponse(
        id=session.id,
        model_id=session.model_id,
        created_at=session.created_at,
    )


@router.get("")
async def list_sessions(
    session_mgr: SessionManager = Depends(get_session_manager),
):
    """List all active sessions."""
    sessions = await session_mgr.list()
    return {
        "sessions": sessions,
        "stats": session_mgr.get_stats(),
    }


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    session_mgr: SessionManager = Depends(get_session_manager),
):
    """Get session details with full message history."""
    session = await session_mgr.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


@router.post("/{session_id}/messages")
async def add_message(
    session_id: str,
    request: AddMessageRequest,
    session_mgr: SessionManager = Depends(get_session_manager),
):
    """Add a message to session."""
    success = await session_mgr.add_message(session_id, request.role, request.content)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "added"}


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    session_mgr: SessionManager = Depends(get_session_manager),
):
    """Delete a session."""
    success = await session_mgr.delete(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


@router.post("/cleanup")
async def cleanup_sessions(
    session_mgr: SessionManager = Depends(get_session_manager),
):
    """Remove expired sessions."""
    count = await session_mgr.cleanup_expired()
    return {"removed": count}
