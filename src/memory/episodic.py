"""
Episodic Memory Store
Stores past conversations, events, and experiences
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection

from .base import Memory, MemoryStore, MemoryType
from ..retrieval.vector_search import VectorSearchEngine, SearchResult
from ..embeddings.voyage_client import get_embedding_service

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
    
    async def store(self, memory: Memory) -> str:
        """Store an episodic memory."""
        try:
            # Ensure it's episodic type
            memory.memory_type = MemoryType.EPISODIC
            
            # Add temporal metadata
            memory.metadata["timestamp"] = datetime.utcnow().isoformat()
            memory.metadata["day_of_week"] = datetime.utcnow().strftime("%A")
            
            # Convert to dict for MongoDB
            memory_dict = memory.model_dump(exclude={"id"})
            memory_dict["_id"] = ObjectId()
            
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
        threshold: float = 0.7
    ) -> List[Memory]:
        """Retrieve episodic memories by similarity."""
        try:
            # Generate query embedding
            embedding_result = await self.embedding_service.generate_embedding(
                query, input_type="query"
            )
            
            # Search for similar memories
            results = await self.search_engine.search(
                query_embedding=embedding_result["embedding"],
                limit=limit,
                filter_query={"memory_type": "episodic"}
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
            logger.error(f"Failed to retrieve episodic memories: {e}")
            return []
    
    async def get_by_id(self, memory_id: str) -> Optional[Memory]:
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
    
    async def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update an episodic memory."""
        try:
            updates["updated_at"] = datetime.utcnow()
            result = await self.collection.update_one(
                {"_id": ObjectId(memory_id)},
                {"$set": updates}
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
    
    async def list_memories(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Memory]:
        """List episodic memories with filters."""
        try:
            query = filters or {}
            query["memory_type"] = "episodic"
            
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