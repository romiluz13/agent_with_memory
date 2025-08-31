"""
Chat Routes
Handle conversations with agents
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio

from fastapi import APIRouter, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .agents import active_agents

router = APIRouter()


class ChatRequest(BaseModel):
    """Request model for chat interactions."""
    message: str = Field(..., description="User message")
    agent_id: str = Field(..., description="Agent to chat with")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    conversation_id: Optional[str] = Field(default=None, description="Conversation identifier")
    stream: bool = Field(default=False, description="Enable streaming response")


class ChatResponse(BaseModel):
    """Response model for chat interactions."""
    response: str
    agent_id: str
    conversation_id: str
    timestamp: str
    metadata: Dict[str, Any]


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a message to an agent and get a response.
    
    Args:
        request: Chat request parameters
        
    Returns:
        Agent response
    """
    # Check if agent exists
    if request.agent_id not in active_agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {request.agent_id} not found"
        )
    
    agent = active_agents[request.agent_id]
    
    try:
        # Get response from agent
        response = await agent.invoke(
            message=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
            conversation_id=request.conversation_id
        )
        
        # Create conversation ID if not provided
        conversation_id = request.conversation_id or f"conv_{agent.conversation_count}"
        
        return ChatResponse(
            response=response,
            agent_id=request.agent_id,
            conversation_id=conversation_id,
            timestamp=datetime.utcnow().isoformat(),
            metadata={
                "user_id": request.user_id,
                "session_id": request.session_id,
                "model": agent.config.model_name,
                "temperature": agent.config.temperature
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}"
        )


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream a chat response.
    
    Args:
        request: Chat request parameters
        
    Returns:
        Streaming response
    """
    # Check if agent exists
    if request.agent_id not in active_agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {request.agent_id} not found"
        )
    
    agent = active_agents[request.agent_id]
    
    async def generate():
        """Generate streaming response."""
        try:
            # This is a simplified streaming implementation
            # In production, you would use the actual streaming from the LLM
            response = await agent.invoke(
                message=request.message,
                user_id=request.user_id,
                session_id=request.session_id,
                conversation_id=request.conversation_id
            )
            
            # Simulate streaming by sending chunks
            chunk_size = 20
            for i in range(0, len(response), chunk_size):
                chunk = response[i:i+chunk_size]
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0.05)  # Simulate processing time
            
            yield f"data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


class ConversationHistory(BaseModel):
    """Model for conversation history."""
    agent_id: str
    user_id: str
    conversations: List[Dict[str, Any]]
    total_count: int


@router.get("/history/{agent_id}/{user_id}", response_model=ConversationHistory)
async def get_conversation_history(
    agent_id: str,
    user_id: str,
    limit: int = 50,
    offset: int = 0,
    app_request: Request = None
) -> ConversationHistory:
    """
    Get conversation history between an agent and user.
    
    Args:
        agent_id: Agent identifier
        user_id: User identifier
        limit: Maximum number of conversations
        offset: Pagination offset
        app_request: FastAPI request
        
    Returns:
        Conversation history
    """
    try:
        # Get memory manager
        memory_manager = app_request.app.state.memory_manager
        
        # Get episodic memories (conversation history)
        memories = await memory_manager.episodic.get_conversation_history(
            agent_id=agent_id,
            user_id=user_id,
            limit=limit
        )
        
        # Format conversations
        conversations = []
        for memory in memories:
            conversations.append({
                "content": memory.content,
                "timestamp": memory.created_at.isoformat(),
                "importance": memory.importance,
                "metadata": memory.metadata
            })
        
        return ConversationHistory(
            agent_id=agent_id,
            user_id=user_id,
            conversations=conversations,
            total_count=len(conversations)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get history: {str(e)}"
        )


@router.delete("/history/{agent_id}/{user_id}")
async def clear_conversation_history(
    agent_id: str,
    user_id: str,
    app_request: Request
) -> Dict[str, Any]:
    """
    Clear conversation history between an agent and user.
    
    Args:
        agent_id: Agent identifier
        user_id: User identifier
        app_request: FastAPI request
        
    Returns:
        Deletion confirmation
    """
    try:
        # Get memory manager
        memory_manager = app_request.app.state.memory_manager
        
        # Clear episodic memories for this pair
        memories = await memory_manager.episodic.list_memories(
            filters={"agent_id": agent_id, "user_id": user_id}
        )
        
        deleted_count = 0
        for memory in memories:
            if await memory_manager.episodic.delete(memory.id):
                deleted_count += 1
        
        return {
            "status": "cleared",
            "agent_id": agent_id,
            "user_id": user_id,
            "deleted_count": deleted_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear history: {str(e)}"
        )
