"""
Summary Memory Store
Stores compressed context summaries with UUID references for JIT expansion.
Based on Oracle Memory Engineering pattern.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection

from ..embeddings.voyage_client import get_embedding_service
from ..retrieval.vector_search import VectorSearchEngine
from .base import Memory, MemoryStore, MemoryType

logger = logging.getLogger(__name__)


class SummaryMemory(MemoryStore):
    """
    Summary memory for storing compressed context.
    Enables JIT (Just-In-Time) retrieval via summary IDs.
    """

    def __init__(self, collection: AsyncIOMotorCollection):
        """Initialize with MongoDB collection."""
        self.collection = collection
        self.search_engine = VectorSearchEngine(collection)
        self.embedding_service = get_embedding_service()

    async def store(self, memory: Memory) -> str:
        """Store a summary memory."""
        try:
            memory.memory_type = MemoryType.SUMMARY

            # Ensure summary metadata
            if "summary_id" not in memory.metadata:
                memory.metadata["summary_id"] = str(uuid.uuid4())[:8]

            # Generate embedding if not present
            if not memory.embedding:
                # Embed the summary + description for retrieval
                embed_text = f"{memory.metadata.get('description', '')} {memory.content[:500]}"
                embedding_result = await self.embedding_service.generate_embedding(
                    embed_text, input_type="document"
                )
                memory.embedding = embedding_result.embedding

            # Convert to dict for MongoDB
            memory_dict = memory.model_dump(exclude={"id"})
            memory_dict["_id"] = ObjectId()

            result = await self.collection.insert_one(memory_dict)
            logger.debug(f"Stored summary memory: {result.inserted_id}")
            return str(result.inserted_id)

        except Exception as e:
            logger.error(f"Failed to store summary memory: {e}")
            raise

    async def store_summary(
        self,
        summary_id: str,
        full_content: str,
        summary: str,
        description: str,
        agent_id: str,
        user_id: str | None = None,
        thread_id: str | None = None,
    ) -> str:
        """
        Store a summary with full content for JIT expansion.

        Args:
            summary_id: Unique ID for referencing this summary
            full_content: Original full content (for expansion)
            summary: Compressed summary text
            description: Brief label (10 words max)
            agent_id: Agent ID for memory ownership
            user_id: Optional user ID
            thread_id: Optional thread/conversation ID

        Returns:
            Database ID of stored summary
        """
        memory = Memory(
            agent_id=agent_id,
            user_id=user_id,
            memory_type=MemoryType.SUMMARY,
            content=summary,
            metadata={
                "summary_id": summary_id,
                "full_content": full_content,
                "description": description,
                "thread_id": thread_id,
                "word_count_original": len(full_content.split()),
                "word_count_summary": len(summary.split()),
                "compression_ratio": round(len(summary) / max(len(full_content), 1), 2),
            },
            tags=["summary", "compressed"],
        )

        return await self.store(memory)

    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.7,
        agent_id: str | None = None,
        search_mode: str = "hybrid",
    ) -> list[Memory]:
        """
        Retrieve summary memories by similarity.

        Uses hybrid search (vector + full-text) by default for best results.
        Based on MongoDB's official GenAI-Showcase pattern with $rankFusion.

        Args:
            query: Search query text
            limit: Maximum memories to return
            threshold: Minimum similarity threshold
            agent_id: Filter by agent (CRITICAL for multi-tenant isolation)
            search_mode: Search strategy - "hybrid" (default), "semantic", or "text"

        Returns:
            List of relevant memories
        """
        try:
            embedding_result = await self.embedding_service.generate_embedding(
                query, input_type="query"
            )

            # Execute search based on mode
            filter_query = {"memory_type": "summary"}
            if agent_id:
                filter_query["agent_id"] = agent_id

            if search_mode == "hybrid":
                # Hybrid search: vector + full-text with $rankFusion (DEFAULT)
                results = await self.search_engine.hybrid_search(
                    query_text=query,
                    query_embedding=embedding_result.embedding,
                    limit=limit,
                    filter_query=filter_query,
                )
            else:
                # Vector-only search (semantic mode)
                results = await self.search_engine.search(
                    query_embedding=embedding_result.embedding,
                    limit=limit,
                    filter_query=filter_query,
                )

            memories = []
            for result in results:
                if result.score >= threshold:
                    doc = await self.collection.find_one({"_id": ObjectId(result.id)})
                    if doc:
                        doc["id"] = str(doc.pop("_id"))
                        memory = Memory(**doc)
                        memory.metadata["search_score"] = result.score
                        memories.append(memory)

            return memories

        except Exception as e:
            logger.error(f"Failed to retrieve summary memories: {e}")
            return []

    async def retrieve_by_summary_id(
        self, summary_id: str, agent_id: str | None = None
    ) -> Memory | None:
        """
        Retrieve a specific summary by its summary_id (JIT expansion).

        Args:
            summary_id: The summary ID (e.g., "abc12345")
            agent_id: Filter by agent (CRITICAL for multi-tenant isolation)

        Returns:
            Memory with full_content in metadata for expansion
        """
        try:
            query = {"memory_type": "summary", "metadata.summary_id": summary_id}
            if agent_id:
                query["agent_id"] = agent_id

            doc = await self.collection.find_one(query)

            if doc:
                doc["id"] = str(doc.pop("_id"))
                return Memory(**doc)
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve summary {summary_id}: {e}")
            return None

    async def expand_summary(self, summary_id: str, agent_id: str | None = None) -> str | None:
        """
        Expand a summary to get the full original content.

        Args:
            summary_id: The summary ID to expand
            agent_id: Filter by agent (CRITICAL for multi-tenant isolation)

        Returns:
            Full original content or None if not found
        """
        memory = await self.retrieve_by_summary_id(summary_id, agent_id=agent_id)
        if memory:
            return memory.metadata.get("full_content")
        return None

    async def get_by_id(self, memory_id: str) -> Memory | None:
        """Get summary memory by database ID."""
        try:
            doc = await self.collection.find_one({"_id": ObjectId(memory_id)})
            if doc:
                doc["id"] = str(doc.pop("_id"))
                return Memory(**doc)
            return None
        except Exception as e:
            logger.error(f"Failed to get summary memory {memory_id}: {e}")
            return None

    async def update(self, memory_id: str, updates: dict[str, Any]) -> bool:
        """Update a summary memory."""
        try:
            updates["updated_at"] = datetime.now(UTC)
            result = await self.collection.update_one(
                {"_id": ObjectId(memory_id)}, {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update summary memory {memory_id}: {e}")
            return False

    async def delete(self, memory_id: str) -> bool:
        """Delete a summary memory."""
        try:
            result = await self.collection.delete_one({"_id": ObjectId(memory_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to delete summary memory {memory_id}: {e}")
            return False

    async def list_memories(
        self, filters: dict[str, Any] | None = None, limit: int = 100, offset: int = 0
    ) -> list[Memory]:
        """List summary memories with filters."""
        try:
            query = filters or {}
            query["memory_type"] = "summary"

            cursor = self.collection.find(query).skip(offset).limit(limit)
            cursor = cursor.sort([("created_at", -1)])

            memories = []
            async for doc in cursor:
                doc["id"] = str(doc.pop("_id"))
                memories.append(Memory(**doc))

            return memories
        except Exception as e:
            logger.error(f"Failed to list summary memories: {e}")
            return []

    async def clear_all(self, confirm: bool = False) -> int:
        """Clear all summary memories."""
        if not confirm:
            raise ValueError("Must confirm deletion")

        try:
            result = await self.collection.delete_many({"memory_type": "summary"})
            return result.deleted_count
        except Exception as e:
            logger.error(f"Failed to clear summary memories: {e}")
            return 0

    async def list_summary_references(
        self, agent_id: str, thread_id: str | None = None, limit: int = 20
    ) -> list[dict[str, str]]:
        """
        List available summaries as references for context.

        Args:
            agent_id: Agent ID
            thread_id: Optional thread filter
            limit: Max references to return

        Returns:
            List of {summary_id, description} for context inclusion
        """
        try:
            query = {"memory_type": "summary", "agent_id": agent_id}
            if thread_id:
                query["metadata.thread_id"] = thread_id

            cursor = (
                self.collection.find(query, {"metadata.summary_id": 1, "metadata.description": 1})
                .limit(limit)
                .sort("created_at", -1)
            )

            references = []
            async for doc in cursor:
                references.append(
                    {
                        "summary_id": doc["metadata"].get("summary_id"),
                        "description": doc["metadata"].get("description", "No description"),
                    }
                )

            return references

        except Exception as e:
            logger.error(f"Failed to list summary references: {e}")
            return []

    async def get_summaries_for_thread(
        self, thread_id: str, agent_id: str | None = None, limit: int = 10
    ) -> list[Memory]:
        """Get all summaries for a specific thread.

        Args:
            thread_id: Thread/conversation ID
            agent_id: Filter by agent (CRITICAL for multi-tenant isolation)
            limit: Maximum summaries to return

        Returns:
            List of summary memories for the thread
        """
        filters = {"metadata.thread_id": thread_id}
        if agent_id:
            filters["agent_id"] = agent_id
        return await self.list_memories(filters=filters, limit=limit)
