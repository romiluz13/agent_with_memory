"""
Memory Management Routes
CRUD operations for agent memories
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Request, Query
from pydantic import BaseModel, Field

from ...memory.base import MemoryType, Memory

router = APIRouter()


class StoreMemoryRequest(BaseModel):
    """Request model for storing a memory."""
    content: str = Field(..., description="Memory content")
    memory_type: MemoryType = Field(..., description="Type of memory")
    agent_id: str = Field(..., description="Agent that owns this memory")
    user_id: Optional[str] = Field(default=None, description="Associated user")
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryResponse(BaseModel):
    """Response model for memory operations."""
    id: str
    memory_type: str
    content: str
    agent_id: str
    user_id: Optional[str]
    importance: float
    created_at: str
    metadata: Dict[str, Any]


class SearchMemoryRequest(BaseModel):
    """Request model for searching memories."""
    query: str = Field(..., description="Search query")
    memory_types: Optional[List[MemoryType]] = Field(default=None)
    limit: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    use_cache: bool = Field(default=True)


@router.post("/", response_model=MemoryResponse)
async def store_memory(
    request: StoreMemoryRequest,
    app_request: Request
) -> MemoryResponse:
    """
    Store a new memory.
    
    Args:
        request: Memory storage parameters
        app_request: FastAPI request
        
    Returns:
        Stored memory details
    """
    try:
        # Get memory manager
        memory_manager = app_request.app.state.memory_manager
        
        # Store memory
        memory_id = await memory_manager.store_memory(
            content=request.content,
            memory_type=request.memory_type,
            agent_id=request.agent_id,
            user_id=request.user_id,
            metadata=request.metadata,
            importance=request.importance,
            tags=request.tags
        )
        
        # Get the stored memory
        store = memory_manager.stores[request.memory_type]
        memory = await store.get_by_id(memory_id)
        
        return MemoryResponse(
            id=memory_id,
            memory_type=memory.memory_type.value,
            content=memory.content,
            agent_id=memory.agent_id,
            user_id=memory.user_id,
            importance=memory.importance,
            created_at=memory.created_at.isoformat(),
            metadata=memory.metadata
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store memory: {str(e)}"
        )


@router.post("/search", response_model=List[MemoryResponse])
async def search_memories(
    request: SearchMemoryRequest,
    app_request: Request
) -> List[MemoryResponse]:
    """
    Search for relevant memories.
    
    Args:
        request: Search parameters
        app_request: FastAPI request
        
    Returns:
        List of relevant memories
    """
    try:
        # Get memory manager
        memory_manager = app_request.app.state.memory_manager
        
        # Search memories
        memories = await memory_manager.retrieve_memories(
            query=request.query,
            memory_types=request.memory_types,
            limit=request.limit,
            threshold=request.threshold,
            use_cache=request.use_cache
        )
        
        # Format response
        results = []
        for memory in memories:
            results.append(MemoryResponse(
                id=memory.id or "unknown",
                memory_type=memory.memory_type.value,
                content=memory.content,
                agent_id=memory.agent_id,
                user_id=memory.user_id,
                importance=memory.importance,
                created_at=memory.created_at.isoformat(),
                metadata=memory.metadata
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search memories: {str(e)}"
        )


@router.get("/{memory_type}", response_model=List[MemoryResponse])
async def list_memories(
    memory_type: MemoryType,
    app_request: Request,
    agent_id: Optional[str] = Query(default=None),
    user_id: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0)
) -> List[MemoryResponse]:
    """
    List memories of a specific type.
    
    Args:
        memory_type: Type of memories to list
        app_request: FastAPI request
        agent_id: Optional agent filter
        user_id: Optional user filter
        limit: Maximum memories to return
        offset: Pagination offset
        
    Returns:
        List of memories
    """
    try:
        # Get memory manager
        memory_manager = app_request.app.state.memory_manager
        
        # Build filters
        filters = {}
        if agent_id:
            filters["agent_id"] = agent_id
        if user_id:
            filters["user_id"] = user_id
        
        # Get memories
        store = memory_manager.stores[memory_type]
        memories = await store.list_memories(
            filters=filters,
            limit=limit,
            offset=offset
        )
        
        # Format response
        results = []
        for memory in memories:
            results.append(MemoryResponse(
                id=memory.id or "unknown",
                memory_type=memory.memory_type.value,
                content=memory.content,
                agent_id=memory.agent_id,
                user_id=memory.user_id,
                importance=memory.importance,
                created_at=memory.created_at.isoformat(),
                metadata=memory.metadata
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list memories: {str(e)}"
        )


@router.get("/stats/summary")
async def get_memory_stats(app_request: Request) -> Dict[str, Any]:
    """
    Get memory system statistics.
    
    Args:
        app_request: FastAPI request
        
    Returns:
        Memory statistics
    """
    try:
        # Get memory manager
        memory_manager = app_request.app.state.memory_manager
        
        # Get stats
        stats = await memory_manager.get_stats()
        
        # Add cache stats if available
        cache_stats = await memory_manager.cache.get_stats()
        stats["cache"] = cache_stats
        
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


@router.post("/consolidate")
async def consolidate_memories(
    app_request: Request,
    agent_id: str = Query(...),
    memory_type: Optional[MemoryType] = Query(default=None)
) -> Dict[str, Any]:
    """
    Consolidate similar memories to reduce redundancy.
    
    Args:
        app_request: FastAPI request
        agent_id: Agent identifier
        memory_type: Optional specific type to consolidate
        
    Returns:
        Consolidation results
    """
    try:
        # Get memory manager
        memory_manager = app_request.app.state.memory_manager
        
        # Consolidate memories
        consolidated = await memory_manager.consolidate_memories(
            agent_id=agent_id,
            memory_type=memory_type
        )
        
        return {
            "status": "completed",
            "agent_id": agent_id,
            "memory_type": memory_type.value if memory_type else "all",
            "consolidated_count": consolidated,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to consolidate: {str(e)}"
        )


@router.delete("/{memory_type}/{memory_id}")
async def delete_memory(
    memory_type: MemoryType,
    memory_id: str,
    app_request: Request
) -> Dict[str, str]:
    """
    Delete a specific memory.
    
    Args:
        memory_type: Type of memory
        memory_id: Memory identifier
        app_request: FastAPI request
        
    Returns:
        Deletion confirmation
    """
    try:
        # Get memory manager
        memory_manager = app_request.app.state.memory_manager
        
        # Delete memory
        store = memory_manager.stores[memory_type]
        success = await store.delete(memory_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory {memory_id} not found"
            )
        
        return {
            "status": "deleted",
            "memory_id": memory_id,
            "memory_type": memory_type.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete memory: {str(e)}"
        )
