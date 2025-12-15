"""
Semantic Memory Store
Stores knowledge, facts, and domain information
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection

from .base import Memory, MemoryStore, MemoryType
from ..retrieval.vector_search import VectorSearchEngine
from ..embeddings.voyage_client import get_embedding_service

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
            
            # Convert to dict for MongoDB
            memory_dict = memory.model_dump(exclude={"id"})
            memory_dict["_id"] = ObjectId()
            
            # Check for existing similar knowledge
            existing = await self._find_similar_knowledge(memory.content)
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
        search_mode: str = "hybrid"
    ) -> List[Memory]:
        """
        Retrieve semantic memories by similarity.

        Uses hybrid search (vector + full-text) by default for best results.
        Based on MongoDB's official GenAI-Showcase pattern with $rankFusion.

        Args:
            query: Search query text
            limit: Maximum memories to return
            threshold: Minimum similarity threshold
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

            if search_mode == "hybrid":
                # Hybrid search: vector + full-text with $rankFusion (DEFAULT)
                results = await self.search_engine.hybrid_search(
                    query_text=query,
                    query_embedding=embedding_result.embedding,
                    limit=limit,
                    filter_query=filter_query
                )
            else:
                # Vector-only search (semantic mode)
                results = await self.search_engine.search(
                    query_embedding=embedding_result.embedding,
                    limit=limit,
                    filter_query=filter_query
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
                    m.metadata.get("search_score", 0)
                ),
                reverse=True
            )
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve semantic memories: {e}")
            return []
    
    async def get_by_id(self, memory_id: str) -> Optional[Memory]:
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
    
    async def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a semantic memory."""
        try:
            updates["updated_at"] = datetime.utcnow()
            result = await self.collection.update_one(
                {"_id": ObjectId(memory_id)},
                {"$set": updates}
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
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Memory]:
        """List semantic memories with filters."""
        try:
            query = filters or {}
            query["memory_type"] = "semantic"
            
            # Sort by importance and verification status
            cursor = self.collection.find(query).skip(offset).limit(limit)
            cursor = cursor.sort([
                ("metadata.verified", -1),
                ("importance", -1),
                ("metadata.source_count", -1)
            ])
            
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
    
    async def consolidate(self) -> int:
        """Consolidate similar semantic memories."""
        consolidated = 0
        
        try:
            # Get all semantic memories
            all_memories = await self.list_memories(limit=1000)
            
            # Group by domain
            domains = {}
            for memory in all_memories:
                domain = memory.metadata.get("domain", "general")
                if domain not in domains:
                    domains[domain] = []
                domains[domain].append(memory)
            
            # Consolidate within each domain
            for domain, memories in domains.items():
                if len(memories) < 2:
                    continue
                
                # Find similar memories (simplified approach)
                for i, mem1 in enumerate(memories):
                    for mem2 in memories[i+1:]:
                        similarity = await self._calculate_similarity(mem1, mem2)
                        if similarity > 0.9:  # Very similar
                            # Merge mem2 into mem1
                            await self._merge_knowledge(mem1, mem2)
                            await self.delete(mem2.id)
                            consolidated += 1
            
            logger.info(f"Consolidated {consolidated} semantic memories")
            return consolidated
            
        except Exception as e:
            logger.error(f"Failed to consolidate semantic memories: {e}")
            return 0
    
    async def _find_similar_knowledge(self, content: str) -> Optional[Memory]:
        """Find existing similar knowledge."""
        results = await self.retrieve(content, limit=1, threshold=0.9)
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
                "confidence": min(1.0, existing.confidence + 0.1)
            }
        )
        
        return existing.id
    
    async def _calculate_similarity(self, mem1: Memory, mem2: Memory) -> float:
        """Calculate similarity between two memories."""
        # Simplified - in production, use embeddings
        if mem1.content == mem2.content:
            return 1.0
        
        # Check embedding similarity if available
        if mem1.embedding and mem2.embedding:
            # Cosine similarity (simplified)
            dot_product = sum(a * b for a, b in zip(mem1.embedding[:10], mem2.embedding[:10]))
            return min(1.0, max(0.0, dot_product / 10))
        
        return 0.0
    
    async def verify_knowledge(self, memory_id: str, verified: bool = True) -> bool:
        """Mark knowledge as verified."""
        return await self.update(
            memory_id,
            {"metadata.verified": verified}
        )
    
    async def get_knowledge_by_domain(
        self,
        domain: str,
        verified_only: bool = False,
        limit: int = 50
    ) -> List[Memory]:
        """Get knowledge for a specific domain."""
        filters = {"metadata.domain": domain}
        if verified_only:
            filters["metadata.verified"] = True
        
        return await self.list_memories(filters=filters, limit=limit)
