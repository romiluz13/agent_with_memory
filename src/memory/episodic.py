"""
Episodic Memory Store
Stores past conversations, events, and experiences
"""

import logging
from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection

from ..embeddings.voyage_client import get_embedding_service
from ..retrieval.vector_search import SearchResult, VectorSearchEngine
from .base import Memory, MemoryStore, MemoryType

logger = logging.getLogger(__name__)


class EpisodicMemory(MemoryStore):
    """
    Episodic memory for storing conversations and events.
    Remembers what happened, when, and with whom.
    """

    def __init__(self, collection: AsyncIOMotorCollection):
        """Initialize with MongoDB collection."""
        self.collection = collection
        self.search_engine = VectorSearchEngine(collection)
        self.embedding_service = get_embedding_service()

    async def _memory_from_search_result(
        self, result: SearchResult, search_mode: str
    ) -> Memory | None:
        """Convert an aggregated search result into a Memory instance."""
        doc = dict(result.document) if result.document else None
        if doc is None and result.id:
            doc = await self.collection.find_one({"_id": ObjectId(result.id)})
        if not doc:
            return None

        doc = dict(doc)
        if "_id" in doc:
            doc["id"] = str(doc.pop("_id"))
        elif result.id:
            doc["id"] = result.id

        memory = Memory(**doc)
        memory.metadata["search_score"] = result.score
        memory.metadata["search_mode"] = search_mode
        return memory

    async def store(self, memory: Memory) -> str:
        """Store an episodic memory."""
        try:
            # Ensure it's episodic type
            memory.memory_type = MemoryType.EPISODIC

            # Add temporal metadata
            memory.metadata["timestamp"] = datetime.now(UTC).isoformat()
            memory.metadata["day_of_week"] = datetime.now(UTC).strftime("%A")

            # Generate embedding only if not already present
            # mongodb-search-and-ai: "Generate embeddings once at orchestration layer"
            if not memory.embedding:
                embedding_result = await self.embedding_service.generate_embedding(
                    memory.content, input_type="document"
                )
                memory.embedding = embedding_result.embedding

            # Convert to dict for MongoDB
            memory_dict = memory.model_dump(exclude={"id"})
            memory_dict["_id"] = ObjectId()
            memory_dict["embedding"] = memory.embedding

            # Insert into collection
            result = await self.collection.insert_one(memory_dict)

            logger.debug(f"Stored episodic memory: {result.inserted_id}")
            return str(result.inserted_id)

        except Exception as e:
            logger.error(f"Failed to store episodic memory: {e}")
            raise

    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.7,
        agent_id: str | None = None,
        user_id: str | None = None,
        search_mode: str = "hybrid",
    ) -> list[Memory]:
        """
        Retrieve episodic memories by similarity.

        Uses hybrid search (vector + full-text) by default for best results.
        Based on MongoDB's official GenAI-Showcase pattern with $rankFusion.

        Args:
            query: Search query text
            limit: Maximum memories to return
            threshold: Minimum similarity threshold
            agent_id: Filter by agent (CRITICAL for multi-tenant isolation)
            user_id: Filter by user (for user isolation)
            search_mode: Search strategy - "hybrid" (default), "semantic", or "text"

        Returns:
            List of relevant memories
        """
        try:
            # Generate query embedding (needed for hybrid and semantic modes)
            embedding_result = await self.embedding_service.generate_embedding(
                query, input_type="query"
            )

            # Build filter for multi-tenant isolation
            filter_query = {}
            if agent_id:
                filter_query["agent_id"] = agent_id
            if user_id:
                filter_query["user_id"] = user_id

            # Execute search based on mode
            if search_mode == "hybrid":
                # Hybrid search: vector + full-text with $rankFusion (DEFAULT)
                # Falls back to vector-only if text index unavailable
                results = await self.search_engine.hybrid_search(
                    query_text=query,
                    query_embedding=embedding_result.embedding,
                    limit=limit,
                    filter_query=filter_query if filter_query else None,
                )
            else:
                # Vector-only search (semantic mode or fallback)
                results = await self.search_engine.search(
                    query_embedding=embedding_result.embedding,
                    limit=limit,
                    filter_query=filter_query if filter_query else None,
                )

            # Convert results to Memory objects.
            # For hybrid search, skip threshold: $rankFusion returns RRF scores (0.008-0.03)
            # which are not comparable to cosine similarity scores (0-1).
            # Results are already relevance-ranked by the search engine.
            memories = []
            for result in results:
                if search_mode == "hybrid" or result.score >= threshold:
                    memory = await self._memory_from_search_result(result, search_mode)
                    if memory:
                        memories.append(memory)

            logger.debug(f"Retrieved {len(memories)} memories using {search_mode} search")
            return memories

        except Exception as e:
            logger.error(f"Failed to retrieve episodic memories: {e}")
            return []

    async def get_by_id(self, memory_id: str) -> Memory | None:
        """Get episodic memory by ID."""
        try:
            doc = await self.collection.find_one({"_id": ObjectId(memory_id)})
            if doc:
                doc["id"] = str(doc.pop("_id"))
                return Memory(**doc)
            return None
        except Exception as e:
            logger.error(f"Failed to get episodic memory {memory_id}: {e}")
            return None

    async def update(self, memory_id: str, updates: dict[str, Any]) -> bool:
        """Update an episodic memory."""
        try:
            updates["updated_at"] = datetime.now(UTC)
            result = await self.collection.update_one(
                {"_id": ObjectId(memory_id)}, {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update episodic memory {memory_id}: {e}")
            return False

    async def delete(self, memory_id: str) -> bool:
        """Delete an episodic memory."""
        try:
            result = await self.collection.delete_one({"_id": ObjectId(memory_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to delete episodic memory {memory_id}: {e}")
            return False

    async def clear_all(self, confirm: bool = False) -> int:
        """Clear all episodic memories."""
        if not confirm:
            raise ValueError("Must confirm deletion")

        try:
            result = await self.collection.delete_many({"memory_type": "episodic"})
            return result.deleted_count
        except Exception as e:
            logger.error(f"Failed to clear episodic memories: {e}")
            return 0

    async def mark_as_summarized(self, agent_id: str, thread_id: str, summary_id: str) -> int:
        """
        Mark messages as summarized without deleting them.
        This preserves the full audit trail while excluding them from normal retrieval.

        Based on Oracle's mark-instead-of-delete pattern.

        Args:
            agent_id: Agent ID
            thread_id: Thread/conversation ID
            summary_id: ID of the summary that replaced these messages

        Returns:
            Number of messages marked
        """
        try:
            result = await self.collection.update_many(
                {
                    "memory_type": "episodic",
                    "agent_id": agent_id,
                    "metadata.thread_id": thread_id,
                    "summary_id": None,  # Only mark unsummarized messages
                },
                {"$set": {"summary_id": summary_id, "updated_at": datetime.now(UTC)}},
            )
            logger.info(
                f"Marked {result.modified_count} messages as summarized (summary_id: {summary_id})"
            )
            return result.modified_count
        except Exception as e:
            logger.error(f"Failed to mark messages as summarized: {e}")
            return 0

    async def list_memories(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0,
        include_summarized: bool = False,
        agent_id: str | None = None,
        user_id: str | None = None,
    ) -> list[Memory]:
        """
        List episodic memories with filters.

        Args:
            filters: MongoDB-style filters
            limit: Maximum memories to return
            offset: Pagination offset
            include_summarized: If False, exclude summarized messages
            agent_id: Filter by agent (for multi-tenant isolation)
            user_id: Filter by user (for user isolation)

        Returns:
            List of memories
        """
        try:
            query = filters or {}
            query["memory_type"] = "episodic"

            # Add isolation filters
            if agent_id:
                query["agent_id"] = agent_id
            if user_id:
                query["user_id"] = user_id

            # Exclude summarized messages by default
            if not include_summarized:
                query["summary_id"] = None

            cursor = self.collection.find(query).skip(offset).limit(limit)
            cursor = cursor.sort("created_at", -1)  # Most recent first

            memories = []
            async for doc in cursor:
                doc["id"] = str(doc.pop("_id"))
                memories.append(Memory(**doc))

            return memories
        except Exception as e:
            logger.error(f"Failed to list episodic memories: {e}")
            return []

    async def get_conversation_history(
        self, agent_id: str, thread_id: str, limit: int = 50, include_summarized: bool = False
    ) -> list[Memory]:
        """
        Get conversation history for a thread.

        Args:
            agent_id: Agent ID
            thread_id: Thread/conversation ID
            limit: Maximum messages to return
            include_summarized: Include summarized messages

        Returns:
            List of conversation memories (chronological order)
        """
        filters = {"agent_id": agent_id, "metadata.thread_id": thread_id}
        memories = await self.list_memories(
            filters=filters, limit=limit, include_summarized=include_summarized
        )
        # Return in chronological order
        return sorted(memories, key=lambda m: m.created_at)
