"""
Chat routes backed by the canonical LangGraph runtime.
"""

import json
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter()


class ChatRequest(BaseModel):
    """Request model for chat interactions."""

    message: str = Field(..., description="User message")
    agent_id: str = Field(..., description="Agent to chat with")
    user_id: str | None = Field(default=None, description="User identifier")
    session_id: str | None = Field(default=None, description="Session identifier")
    conversation_id: str | None = Field(default=None, description="Conversation identifier")
    stream: bool = Field(default=False, description="Enable streaming response")


class ChatResponse(BaseModel):
    """Response model for chat interactions."""

    response: str
    agent_id: str
    conversation_id: str
    thread_id: str
    timestamp: str
    metadata: dict[str, Any]


class ConversationHistory(BaseModel):
    """Model for conversation history."""

    agent_id: str
    thread_id: str
    conversations: list[dict[str, Any]]
    total_count: int


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, app_request: Request) -> ChatResponse:
    """Send a message to an agent and get a response."""
    registry = app_request.app.state.agent_registry
    try:
        agent = await registry.ensure_agent(request.agent_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {request.agent_id} not found",
        ) from exc

    thread_id = agent.build_thread_id(
        user_id=request.user_id,
        session_id=request.session_id,
        conversation_id=request.conversation_id,
    )

    try:
        response = await agent.invoke(
            message=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
            conversation_id=request.conversation_id,
        )

        return ChatResponse(
            response=response,
            agent_id=request.agent_id,
            conversation_id=agent.extract_conversation_id(thread_id),
            thread_id=thread_id,
            timestamp=datetime.now(UTC).isoformat(),
            metadata={
                "user_id": request.user_id,
                "session_id": request.session_id,
                "model": agent.model_name,
                "provider": agent.model_provider,
                "temperature": agent.temperature,
            },
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(exc)}",
        ) from exc


@router.post("/stream")
async def chat_stream(request: ChatRequest, app_request: Request):
    """Stream actual graph execution events over SSE."""
    registry = app_request.app.state.agent_registry
    try:
        agent = await registry.ensure_agent(request.agent_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {request.agent_id} not found",
        ) from exc

    async def generate():
        try:
            async for event in agent.stream_events(
                message=request.message,
                user_id=request.user_id,
                session_id=request.session_id,
                conversation_id=request.conversation_id,
            ):
                yield f"data: {json.dumps(event)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            error_event = {
                "type": "error",
                "error": str(exc),
                "timestamp": datetime.now(UTC).isoformat(),
            }
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history/{agent_id}/{thread_id}", response_model=ConversationHistory)
async def get_conversation_history(
    agent_id: str,
    thread_id: str,
    app_request: Request,
    limit: int = 50,
) -> ConversationHistory:
    """Get conversation history for a specific persisted thread."""
    memory_manager = app_request.app.state.memory_manager
    if memory_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory runtime is not configured",
        )

    try:
        memories = await memory_manager.episodic.get_conversation_history(
            agent_id=agent_id,
            thread_id=thread_id,
            limit=limit,
        )

        conversations = [
            {
                "content": memory.content,
                "timestamp": memory.created_at.isoformat(),
                "importance": memory.importance,
                "metadata": memory.metadata,
            }
            for memory in memories
        ]

        return ConversationHistory(
            agent_id=agent_id,
            thread_id=thread_id,
            conversations=conversations,
            total_count=len(conversations),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get history: {str(exc)}",
        ) from exc


@router.delete("/history/{agent_id}/{thread_id}")
async def clear_conversation_history(
    agent_id: str,
    thread_id: str,
    app_request: Request,
) -> dict[str, Any]:
    """Clear conversation history for a specific persisted thread."""
    memory_manager = app_request.app.state.memory_manager
    if memory_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory runtime is not configured",
        )

    try:
        memories = await memory_manager.episodic.list_memories(
            filters={"agent_id": agent_id, "metadata.thread_id": thread_id}
        )

        deleted_count = 0
        for memory in memories:
            if memory.id and await memory_manager.episodic.delete(memory.id):
                deleted_count += 1

        return {
            "status": "cleared",
            "agent_id": agent_id,
            "thread_id": thread_id,
            "deleted_count": deleted_count,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear history: {str(exc)}",
        ) from exc
