"""
Semantic Memory Store
Stores knowledge, facts, and domain information
"""

import logging
import math
from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection

from ..embeddings.voyage_client import get_embedding_service
from ..retrieval.vector_search import VectorSearchEngine
from .base import Memory, MemoryStore, MemoryType

logger = logging.getLogger(__name__)


class SemanticMemory(MemoryStore):
    """
    Semantic memory for storing knowledge and facts.
    Accumulates domain knowledge over time.
    """

    def __init__(self, collection: AsyncIOMotorCollection):
        """Initialize with MongoDB collection."""
        self.collection = collection
        self.search_engine = VectorSearchEngine(collection)
        self.embedding_service = get_embedding_service()

    async def store(self, memory: Memory) -> str:
        """Store a semantic memory."""
        try:
            # Ensure it's semantic type
            memory.memory_type = MemoryType.SEMANTIC

            # Add knowledge metadata
            memory.metadata["domain"] = memory.metadata.get("domain", "general")
            memory.metadata["verified"] = memory.metadata.get("verified", False)
            memory.metadata["source_count"] = memory.metadata.get("source_count", 1)

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

            # Check for existing similar knowledge (scoped to agent)
            existing = await self._find_similar_knowledge(memory.content, agent_id=memory.agent_id)
            if existing:
                # Merge with existing knowledge
                return await self._merge_knowledge(existing, memory)

            # Insert new knowledge
            result = await self.collection.insert_one(memory_dict)

            logger.debug(f"Stored semantic memory: {result.inserted_id}")
            return str(result.inserted_id)

        except Exception as e:
            logger.error(f"Failed to store semantic memory: {e}")
            raise

    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.7,
        agent_id: str | None = None,
        search_mode: str = "hybrid",
    ) -> list[Memory]:
        """
        Retrieve semantic memories by similarity.

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
            # Generate query embedding
            embedding_result = await self.embedding_service.generate_embedding(
                query, input_type="query"
            )

            # Execute search based on mode
            filter_query = {"memory_type": "semantic"}
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

            # Convert results to Memory objects
            memories = []
            for result in results:
                if result.score >= threshold:
                    # Fetch full document
                    doc = await self.collection.find_one({"_id": ObjectId(result.id)})
                    if doc:
                        doc["id"] = str(doc.pop("_id"))
                        memory = Memory(**doc)
                        memory.metadata["search_score"] = result.score
                        memories.append(memory)

            # Prioritize verified knowledge
            memories.sort(
                key=lambda m: (
                    m.metadata.get("verified", False),
                    m.metadata.get("source_count", 1),
                    m.metadata.get("search_score", 0),
                ),
                reverse=True,
            )

            return memories

        except Exception as e:
            logger.error(f"Failed to retrieve semantic memories: {e}")
            return []

    async def get_by_id(self, memory_id: str) -> Memory | None:
        """Get semantic memory by ID."""
        try:
            doc = await self.collection.find_one({"_id": ObjectId(memory_id)})
            if doc:
                doc["id"] = str(doc.pop("_id"))
                return Memory(**doc)
            return None
        except Exception as e:
            logger.error(f"Failed to get semantic memory {memory_id}: {e}")
            return None

    async def update(self, memory_id: str, updates: dict[str, Any]) -> bool:
        """Update a semantic memory."""
        try:
            updates["updated_at"] = datetime.now(UTC)
            result = await self.collection.update_one(
                {"_id": ObjectId(memory_id)}, {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update semantic memory {memory_id}: {e}")
            return False

    async def delete(self, memory_id: str) -> bool:
        """Delete a semantic memory."""
        try:
            result = await self.collection.delete_one({"_id": ObjectId(memory_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to delete semantic memory {memory_id}: {e}")
            return False

    async def list_memories(
        self, filters: dict[str, Any] | None = None, limit: int = 100, offset: int = 0
    ) -> list[Memory]:
        """List semantic memories with filters."""
        try:
            query = filters or {}
            query["memory_type"] = "semantic"

            # Sort by importance and verification status
            # mongodb-query-optimizer: exclude embedding array from fetch results
            cursor = self.collection.find(query, {"embedding": 0}).skip(offset).limit(limit)
            cursor = cursor.sort(
                [("metadata.verified", -1), ("importance", -1), ("metadata.source_count", -1)]
            )

            memories = []
            async for doc in cursor:
                doc["id"] = str(doc.pop("_id"))
                memories.append(Memory(**doc))

            return memories
        except Exception as e:
            logger.error(f"Failed to list semantic memories: {e}")
            return []

    async def clear_all(self, confirm: bool = False) -> int:
        """Clear all semantic memories."""
        if not confirm:
            raise ValueError("Must confirm deletion")

        try:
            result = await self.collection.delete_many({"memory_type": "semantic"})
            return result.deleted_count
        except Exception as e:
            logger.error(f"Failed to clear semantic memories: {e}")
            return 0

    async def consolidate(self, agent_id: str | None = None, batch_size: int = 100) -> int:
        """Consolidate similar semantic memories using vector search.

        mongodb-query-optimizer: Use vector search to find similar documents
        server-side instead of O(n^2) pairwise comparison.

        Args:
            agent_id: Filter by agent (CRITICAL for multi-tenant isolation)
            batch_size: Number of documents to process per batch
        """
        consolidated = 0
        processed_ids: set = set()

        try:
            # Fetch a batch of semantic memories
            query: dict[str, Any] = {"memory_type": "semantic"}
            if agent_id:
                query["agent_id"] = agent_id

            cursor = self.collection.find(
                query, {"embedding": 1, "content": 1, "_id": 1, "metadata": 1, "importance": 1}
            ).limit(batch_size)

            async for doc in cursor:
                doc_id = str(doc["_id"])
                if doc_id in processed_ids:
                    continue

                # Use vector search to find similar documents
                embedding = doc.get("embedding")
                if not embedding:
                    continue

                filter_query: dict[str, Any] = {"memory_type": "semantic"}
                if agent_id:
                    filter_query["agent_id"] = agent_id

                similar = await self.search_engine.search(
                    query_embedding=embedding,
                    limit=5,
                    filter_query=filter_query,
                )
                for match in similar:
                    match_id = match.id
                    if match_id != doc_id and match_id not in processed_ids:
                        if match.score >= 0.9:
                            # Merge and delete duplicate
                            doc["id"] = doc_id
                            existing_mem = Memory(
                                **{
                                    "id": doc_id,
                                    "content": doc.get("content", ""),
                                    "memory_type": "semantic",
                                    "agent_id": agent_id or "",
                                    "metadata": doc.get("metadata", {}),
                                    "importance": doc.get("importance", 0.5),
                                }
                            )
                            match_doc = await self.collection.find_one({"_id": ObjectId(match_id)})
                            if match_doc:
                                new_mem = Memory(
                                    **{
                                        "id": match_id,
                                        "content": match_doc.get("content", ""),
                                        "memory_type": "semantic",
                                        "agent_id": agent_id or "",
                                        "metadata": match_doc.get("metadata", {}),
                                        "importance": match_doc.get("importance", 0.5),
                                    }
                                )
                                await self._merge_knowledge(existing_mem, new_mem)
                                await self.delete(match_id)
                                processed_ids.add(match_id)
                                consolidated += 1

            logger.info(f"Consolidated {consolidated} semantic memories")
            return consolidated

        except Exception as e:
            logger.error(f"Failed to consolidate semantic memories: {e}")
            return 0

    async def _find_similar_knowledge(
        self, content: str, agent_id: str | None = None
    ) -> Memory | None:
        """Find existing similar knowledge.

        Args:
            content: Content to find similar knowledge for
            agent_id: Filter by agent (CRITICAL for multi-tenant isolation)

        Returns:
            Existing similar Memory if found, None otherwise
        """
        results = await self.retrieve(content, limit=1, threshold=0.9, agent_id=agent_id)
        return results[0] if results else None

    async def _merge_knowledge(self, existing: Memory, new: Memory) -> str:
        """Merge new knowledge with existing."""
        # Update source count
        existing.metadata["source_count"] = existing.metadata.get("source_count", 1) + 1

        # Update importance (average)
        existing.importance = (existing.importance + new.importance) / 2

        # Mark as verified if multiple sources
        if existing.metadata["source_count"] >= 3:
            existing.metadata["verified"] = True

        # Update with merged information
        await self.update(
            existing.id,
            {
                "metadata": existing.metadata,
                "importance": existing.importance,
                "confidence": min(1.0, existing.confidence + 0.1),
            },
        )

        return existing.id

    async def _calculate_similarity(self, mem1: Memory, mem2: Memory) -> float:
        """Calculate cosine similarity between two memories.

        Uses ALL dimensions of the embedding vector for accurate results.
        mongodb-search-and-ai: "Cosine similarity MUST use ALL dimensions."

        Args:
            mem1: First memory
            mem2: Second memory

        Returns:
            Cosine similarity score between 0.0 and 1.0
        """
        if mem1.content == mem2.content:
            return 1.0

        if mem1.embedding and mem2.embedding:
            # Full cosine similarity using all dimensions
            dot_product = sum(a * b for a, b in zip(mem1.embedding, mem2.embedding))
            norm1 = math.sqrt(sum(a * a for a in mem1.embedding))
            norm2 = math.sqrt(sum(b * b for b in mem2.embedding))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)

        return 0.0

    async def verify_knowledge(self, memory_id: str, verified: bool = True) -> bool:
        """Mark knowledge as verified."""
        return await self.update(memory_id, {"metadata.verified": verified})

    async def get_knowledge_by_domain(
        self, domain: str, verified_only: bool = False, limit: int = 50
    ) -> list[Memory]:
        """Get knowledge for a specific domain."""
        filters = {"metadata.domain": domain}
        if verified_only:
            filters["metadata.verified"] = True

        return await self.list_memories(filters=filters, limit=limit)
