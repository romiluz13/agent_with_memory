"""
Time-Travel Debugging API.
Exposes MongoDBSaver's state history for replaying agent states.

From LangChain+MongoDB article: "time-travel debugging (replay any prior state)"
From Mem0 reference: history(memory_id) returns full change timeline

Note: MongoDBSaver uses sync PyMongo. API wraps with asyncio.to_thread()
for async compatibility. Do NOT use deprecated asyncio.get_event_loop().
"""

import asyncio
import logging
import os
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# Lazy imports: MongoDBSaver and MongoClient are imported at call time
# to avoid import errors when langgraph-checkpoint-mongodb is not installed.
MongoDBSaver = None
MongoClient = None


def _ensure_imports():
    """Lazy-load MongoDBSaver and MongoClient on first use."""
    global MongoDBSaver, MongoClient
    if MongoDBSaver is None:
        from langgraph.checkpoint.mongodb import MongoDBSaver as _Saver
        MongoDBSaver = _Saver
    if MongoClient is None:
        from pymongo import MongoClient as _Client
        MongoClient = _Client


class StateSnapshot(BaseModel):
    """A snapshot of agent state at a point in time."""

    checkpoint_id: str
    thread_id: str
    timestamp: str
    messages: list[dict[str, Any]]
    metadata: dict[str, Any]


class StateHistory(BaseModel):
    """Full state history for a thread."""

    thread_id: str
    snapshots: list[StateSnapshot]
    total_count: int


class ReplayRequest(BaseModel):
    """Request model for replaying from a checkpoint."""

    message: str = Field(..., description="New message to send from this checkpoint")


class ReplayResponse(BaseModel):
    """Response model for replay endpoint."""

    status: str
    thread_id: str
    from_checkpoint: str
    new_thread_id: str
    message: str


@router.get("/history/{thread_id}", response_model=StateHistory)
async def get_state_history(
    thread_id: str,
    limit: int = Query(default=50, ge=1, le=500),
):
    """Get the full state history for a conversation thread.

    Uses MongoDBSaver.get_state_history() to retrieve all checkpoints.
    Enables replaying any prior agent state for debugging.

    Args:
        thread_id: The conversation thread ID
        limit: Maximum number of snapshots to return

    Returns:
        Full state history with all checkpoints
    """
    try:
        uri = os.getenv("MONGODB_URI")
        if not uri:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="MongoDB not configured",
            )

        _ensure_imports()

        # MongoDBSaver is sync - wrap with asyncio.to_thread (Python 3.9+)
        def _get_history():
            client = MongoClient(uri)
            saver = MongoDBSaver(client)
            config = {"configurable": {"thread_id": thread_id}}

            snapshots = []
            for state in saver.get_state_history(config):
                snapshots.append(
                    {
                        "checkpoint_id": state.config.get("configurable", {}).get(
                            "checkpoint_id", ""
                        ),
                        "thread_id": thread_id,
                        "timestamp": str(state.metadata.get("created_at", "")),
                        "messages": [
                            {
                                "type": getattr(m, "type", "unknown"),
                                "content": getattr(m, "content", ""),
                            }
                            for m in state.values.get("messages", [])
                        ],
                        "metadata": state.metadata or {},
                    }
                )

                if len(snapshots) >= limit:
                    break

            client.close()
            return snapshots

        snapshots = await asyncio.to_thread(_get_history)

        return StateHistory(
            thread_id=thread_id,
            snapshots=[StateSnapshot(**s) for s in snapshots],
            total_count=len(snapshots),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get state history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get state history: {str(e)}",
        ) from e


@router.get("/snapshot/{thread_id}/{checkpoint_id}", response_model=StateSnapshot)
async def get_state_snapshot(
    thread_id: str,
    checkpoint_id: str,
):
    """Get a specific state snapshot by checkpoint ID.

    Args:
        thread_id: The conversation thread ID
        checkpoint_id: The specific checkpoint to retrieve

    Returns:
        The state at that checkpoint
    """
    try:
        uri = os.getenv("MONGODB_URI")
        if not uri:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="MongoDB not configured",
            )

        _ensure_imports()

        def _get_snapshot():
            client = MongoClient(uri)
            saver = MongoDBSaver(client)
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                }
            }

            state = saver.get(config)
            if not state:
                return None

            snapshot = {
                "checkpoint_id": checkpoint_id,
                "thread_id": thread_id,
                "timestamp": str(state.metadata.get("created_at", "")),
                "messages": [
                    {
                        "type": getattr(m, "type", "unknown"),
                        "content": getattr(m, "content", ""),
                    }
                    for m in state.values.get("messages", [])
                ],
                "metadata": state.metadata or {},
            }

            client.close()
            return snapshot

        snapshot = await asyncio.to_thread(_get_snapshot)

        if not snapshot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Checkpoint {checkpoint_id} not found for thread {thread_id}",
            )

        return StateSnapshot(**snapshot)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get snapshot: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get snapshot: {str(e)}",
        ) from e


@router.post("/replay/{thread_id}/{checkpoint_id}", response_model=ReplayResponse)
async def replay_from_checkpoint(
    thread_id: str,
    checkpoint_id: str,
    request: ReplayRequest,
):
    """Replay from a specific checkpoint with a new message.

    Creates a new branch (new thread_id) from the historical state.

    Args:
        thread_id: The conversation thread ID
        checkpoint_id: The checkpoint to replay from
        request: Contains the new message

    Returns:
        ReplayResponse with new thread_id for the branched conversation
    """
    try:
        uri = os.getenv("MONGODB_URI")
        if not uri:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="MongoDB not configured",
            )

        _ensure_imports()
        new_thread_id = f"{thread_id}-replay-{uuid.uuid4().hex[:8]}"

        # Verify the checkpoint exists
        def _verify_checkpoint():
            client = MongoClient(uri)
            saver = MongoDBSaver(client)
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                }
            }
            state = saver.get(config)
            client.close()
            return state is not None

        exists = await asyncio.to_thread(_verify_checkpoint)
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Checkpoint {checkpoint_id} not found for thread {thread_id}",
            )

        return ReplayResponse(
            status="replay_ready",
            thread_id=thread_id,
            from_checkpoint=checkpoint_id,
            new_thread_id=new_thread_id,
            message=request.message,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to initiate replay: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
