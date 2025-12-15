"""
Working Memory Store
Stores active session context and current task state
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection

from .base import Memory, MemoryStore, MemoryType
from ..retrieval.vector_search import VectorSearchEngine
from ..embeddings.voyage_client import get_embedding_service

logger = logging.getLogger(__name__)


class WorkingMemory(MemoryStore):
    """
    Working memory for active session context.
    Short-term, high-priority memory for current tasks.
    """
    
    def __init__(self, collection: AsyncIOMotorCollection):
        """Initialize with MongoDB collection."""
        self.collection = collection
        self.search_engine = VectorSearchEngine(collection)
        self.embedding_service = get_embedding_service()
        
        # Working memory has limited capacity
        self.max_capacity = 20  # Miller's law: 7±2, but we allow more for agents
    
    async def store(self, memory: Memory) -> str:
        """Store a working memory."""
        try:
            # Ensure it's working type
            memory.memory_type = MemoryType.WORKING
            
            # Working memory has short TTL
            if not memory.ttl:
                memory.ttl = datetime.utcnow() + timedelta(minutes=60)
            
            # Add session metadata
            memory.metadata["session_id"] = memory.metadata.get("session_id", "default")
            memory.metadata["task_id"] = memory.metadata.get("task_id", None)
            memory.metadata["priority"] = memory.metadata.get("priority", "normal")
            
            # Check capacity
            await self._manage_capacity(memory.agent_id, memory.metadata["session_id"])
            
            # Convert to dict for MongoDB
            memory_dict = memory.model_dump(exclude={"id"})
            memory_dict["_id"] = ObjectId()
            
            # Insert into collection
            result = await self.collection.insert_one(memory_dict)
            
            logger.debug(f"Stored working memory: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to store working memory: {e}")
            raise
    
    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.7,
        search_mode: str = "hybrid"
    ) -> List[Memory]:
        """
        Retrieve working memories by similarity.

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

            # Search only active working memories
            filter_query = {
                "memory_type": "working",
                "ttl": {"$gt": datetime.utcnow()}  # Not expired
            }

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
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve working memories: {e}")
            return []
    
    async def get_by_id(self, memory_id: str) -> Optional[Memory]:
        """Get working memory by ID."""
        try:
            doc = await self.collection.find_one({"_id": ObjectId(memory_id)})
            if doc:
                doc["id"] = str(doc.pop("_id"))
                return Memory(**doc)
            return None
        except Exception as e:
            logger.error(f"Failed to get working memory {memory_id}: {e}")
            return None
    
    async def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a working memory."""
        try:
            updates["updated_at"] = datetime.utcnow()
            updates["accessed_at"] = datetime.utcnow()
            
            # Extend TTL on update
            if "ttl" not in updates:
                updates["ttl"] = datetime.utcnow() + timedelta(minutes=30)
            
            result = await self.collection.update_one(
                {"_id": ObjectId(memory_id)},
                {"$set": updates, "$inc": {"access_count": 1}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update working memory {memory_id}: {e}")
            return False
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a working memory."""
        try:
            result = await self.collection.delete_one({"_id": ObjectId(memory_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to delete working memory {memory_id}: {e}")
            return False
    
    async def list_memories(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Memory]:
        """List working memories with filters."""
        try:
            query = filters or {}
            query["memory_type"] = "working"
            
            # Only show non-expired memories by default
            if "ttl" not in query:
                query["ttl"] = {"$gt": datetime.utcnow()}
            
            # Sort by priority and recency
            cursor = self.collection.find(query).skip(offset).limit(limit)
            cursor = cursor.sort([
                ("metadata.priority", -1),
                ("accessed_at", -1)
            ])
            
            memories = []
            async for doc in cursor:
                doc["id"] = str(doc.pop("_id"))
                memories.append(Memory(**doc))
            
            return memories
        except Exception as e:
            logger.error(f"Failed to list working memories: {e}")
            return []
    
    async def clear_all(self, confirm: bool = False) -> int:
        """Clear all working memories."""
        if not confirm:
            raise ValueError("Must confirm deletion")
        
        try:
            result = await self.collection.delete_many({"memory_type": "working"})
            return result.deleted_count
        except Exception as e:
            logger.error(f"Failed to clear working memories: {e}")
            return 0
    
    async def _manage_capacity(self, agent_id: str, session_id: str):
        """Manage working memory capacity by removing least important items."""
        # Count current working memories for this session
        count = await self.collection.count_documents({
            "agent_id": agent_id,
            "metadata.session_id": session_id,
            "memory_type": "working",
            "ttl": {"$gt": datetime.utcnow()}
        })
        
        if count >= self.max_capacity:
            # Remove least important/oldest memories
            to_remove = count - self.max_capacity + 1
            
            cursor = self.collection.find({
                "agent_id": agent_id,
                "metadata.session_id": session_id,
                "memory_type": "working"
            }).sort([
                ("importance", 1),  # Lowest importance first
                ("accessed_at", 1)  # Oldest access first
            ]).limit(to_remove)
            
            async for doc in cursor:
                await self.delete(str(doc["_id"]))
                logger.debug(f"Removed working memory {doc['_id']} due to capacity")
    
    async def get_session_context(
        self,
        agent_id: str,
        session_id: str
    ) -> List[Memory]:
        """Get all working memories for a session."""
        return await self.list_memories(
            filters={
                "agent_id": agent_id,
                "metadata.session_id": session_id
            }
        )
    
    async def clear_session(
        self,
        agent_id: str,
        session_id: str
    ) -> int:
        """Clear all memories for a specific session."""
        result = await self.collection.delete_many({
            "agent_id": agent_id,
            "metadata.session_id": session_id,
            "memory_type": "working"
        })
        return result.deleted_count
    
    async def extend_ttl(
        self,
        memory_id: str,
        minutes: int = 30
    ) -> bool:
        """Extend the TTL of a working memory."""
        new_ttl = datetime.utcnow() + timedelta(minutes=minutes)
        return await self.update(memory_id, {"ttl": new_ttl})
    
    async def cleanup_expired(self) -> int:
        """Remove expired working memories."""
        result = await self.collection.delete_many({
            "memory_type": "working",
            "ttl": {"$lt": datetime.utcnow()}
        })
        
        if result.deleted_count > 0:
            logger.info(f"Cleaned up {result.deleted_count} expired working memories")
        
        return result.deleted_count
