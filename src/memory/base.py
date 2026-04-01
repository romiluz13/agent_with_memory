"""
Base Memory Interface
Defines the contract for all memory types
"""

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Types of memory in the system."""

    EPISODIC = "episodic"  # Past conversations and events
    PROCEDURAL = "procedural"  # Workflows and skills
    SEMANTIC = "semantic"  # Knowledge and facts
    WORKING = "working"  # Active session context
    CACHE = "cache"  # Semantic cache for performance
    ENTITY = "entity"  # Extracted entities (people, places, systems)
    SUMMARY = "summary"  # Compressed context summaries


class MemoryImportance(str, Enum):
    """Importance levels for memories."""

    CRITICAL = "critical"  # Must never forget
    HIGH = "high"  # Very important
    MEDIUM = "medium"  # Normal importance
    LOW = "low"  # Can be forgotten if needed
    TRIVIAL = "trivial"  # First to be forgotten


class Memory(BaseModel):
    """Base memory model."""

    id: str | None = Field(default=None, description="Memory ID")
    agent_id: str = Field(..., description="Agent that owns this memory")
    user_id: str | None = Field(default=None, description="User associated with memory")
    memory_type: MemoryType = Field(..., description="Type of memory")
    content: str = Field(..., description="Memory content")
    embedding: list[float] | None = Field(default=None, description="Vector embedding")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance score")
    importance_level: MemoryImportance = Field(default=MemoryImportance.MEDIUM)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    accessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    access_count: int = Field(default=0, description="Number of times accessed")
    ttl: datetime | None = Field(default=None, description="Time to live")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    source: str | None = Field(default=None, description="Source of the memory")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in memory")
    summary_id: str | None = Field(default=None, description="ID of summary if compressed")


class MemoryStore(ABC):
    """Abstract base class for memory stores."""

    @abstractmethod
    async def store(self, memory: Memory) -> str:
        """
        Store a memory.

        Args:
            memory: Memory to store

        Returns:
            Memory ID
        """
        pass

    @abstractmethod
    async def retrieve(
        self, query: str, limit: int = 10, threshold: float = 0.7, agent_id: str | None = None
    ) -> list[Memory]:
        """
        Retrieve memories based on similarity.

        Args:
            query: Query text
            limit: Maximum memories to return
            threshold: Minimum similarity threshold
            agent_id: Filter by agent (for multi-tenant isolation)

        Returns:
            List of relevant memories
        """
        pass

    @abstractmethod
    async def get_by_id(self, memory_id: str) -> Memory | None:
        """
        Get a specific memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            Memory if found
        """
        pass

    @abstractmethod
    async def update(self, memory_id: str, updates: dict[str, Any]) -> bool:
        """
        Update a memory.

        Args:
            memory_id: Memory ID
            updates: Fields to update

        Returns:
            True if updated
        """
        pass

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: Memory ID

        Returns:
            True if deleted
        """
        pass

    @abstractmethod
    async def list_memories(
        self, filters: dict[str, Any] | None = None, limit: int = 100, offset: int = 0
    ) -> list[Memory]:
        """
        List memories with optional filters.

        Args:
            filters: MongoDB-style filters
            limit: Maximum memories to return
            offset: Pagination offset

        Returns:
            List of memories
        """
        pass

    @abstractmethod
    async def clear_all(self, confirm: bool = False) -> int:
        """
        Clear all memories (dangerous operation).

        Args:
            confirm: Must be True to proceed

        Returns:
            Number of memories deleted
        """
        pass

    async def consolidate(self, agent_id: str | None = None) -> int:
        """
        Consolidate similar memories (optional operation).

        Args:
            agent_id: Filter by agent (CRITICAL for multi-tenant isolation)

        Returns:
            Number of memories consolidated
        """
        return 0

    async def update_importance(self, memory_id: str, delta: float) -> bool:
        """
        Update memory importance based on usage.

        Args:
            memory_id: Memory ID
            delta: Change in importance

        Returns:
            True if updated
        """
        updates = {"importance": delta, "accessed_at": datetime.now(UTC)}
        return await self.update(memory_id, updates)

    async def cleanup_expired(self) -> int:
        """
        Remove expired memories based on TTL.

        Returns:
            Number of memories removed
        """
        # Default implementation
        expired = await self.list_memories(filters={"ttl": {"$lt": datetime.now(UTC)}})

        count = 0
        for memory in expired:
            if await self.delete(memory.id):
                count += 1

        return count
