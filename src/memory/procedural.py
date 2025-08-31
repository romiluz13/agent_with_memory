"""
Procedural Memory Store
Stores workflows, skills, and learned procedures
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


class ProceduralMemory(MemoryStore):
    """
    Procedural memory for storing learned workflows and skills.
    Improves through repetition and feedback.
    """
    
    def __init__(self, collection: AsyncIOMotorCollection):
        """Initialize with MongoDB collection."""
        self.collection = collection
        self.search_engine = VectorSearchEngine(collection)
        self.embedding_service = get_embedding_service()
    
    async def store(self, memory: Memory) -> str:
        """Store a procedural memory."""
        try:
            # Ensure it's procedural type
            memory.memory_type = MemoryType.PROCEDURAL
            
            # Add skill metadata
            memory.metadata["skill_level"] = memory.metadata.get("skill_level", 1)
            memory.metadata["success_rate"] = memory.metadata.get("success_rate", 0.0)
            memory.metadata["execution_count"] = memory.metadata.get("execution_count", 0)
            
            # Convert to dict for MongoDB
            memory_dict = memory.model_dump(exclude={"id"})
            memory_dict["_id"] = ObjectId()
            
            # Insert into collection
            result = await self.collection.insert_one(memory_dict)
            
            logger.debug(f"Stored procedural memory: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to store procedural memory: {e}")
            raise
    
    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Memory]:
        """Retrieve procedural memories by similarity."""
        try:
            # Generate query embedding
            embedding_result = await self.embedding_service.generate_embedding(
                query, input_type="query"
            )
            
            # Search for similar procedures
            results = await self.search_engine.search(
                query_embedding=embedding_result.embedding,
                limit=limit,
                filter_query={"memory_type": "procedural"}
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
            
            # Sort by skill level and success rate
            memories.sort(
                key=lambda m: (
                    m.metadata.get("skill_level", 0) * 
                    m.metadata.get("success_rate", 0)
                ),
                reverse=True
            )
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve procedural memories: {e}")
            return []
    
    async def get_by_id(self, memory_id: str) -> Optional[Memory]:
        """Get procedural memory by ID."""
        try:
            doc = await self.collection.find_one({"_id": ObjectId(memory_id)})
            if doc:
                doc["id"] = str(doc.pop("_id"))
                return Memory(**doc)
            return None
        except Exception as e:
            logger.error(f"Failed to get procedural memory {memory_id}: {e}")
            return None
    
    async def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a procedural memory."""
        try:
            updates["updated_at"] = datetime.utcnow()
            result = await self.collection.update_one(
                {"_id": ObjectId(memory_id)},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update procedural memory {memory_id}: {e}")
            return False
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a procedural memory."""
        try:
            result = await self.collection.delete_one({"_id": ObjectId(memory_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to delete procedural memory {memory_id}: {e}")
            return False
    
    async def list_memories(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Memory]:
        """List procedural memories with filters."""
        try:
            query = filters or {}
            query["memory_type"] = "procedural"
            
            # Sort by skill level and success rate
            cursor = self.collection.find(query).skip(offset).limit(limit)
            cursor = cursor.sort([
                ("metadata.skill_level", -1),
                ("metadata.success_rate", -1)
            ])
            
            memories = []
            async for doc in cursor:
                doc["id"] = str(doc.pop("_id"))
                memories.append(Memory(**doc))
            
            return memories
        except Exception as e:
            logger.error(f"Failed to list procedural memories: {e}")
            return []
    
    async def clear_all(self, confirm: bool = False) -> int:
        """Clear all procedural memories."""
        if not confirm:
            raise ValueError("Must confirm deletion")
        
        try:
            result = await self.collection.delete_many({"memory_type": "procedural"})
            return result.deleted_count
        except Exception as e:
            logger.error(f"Failed to clear procedural memories: {e}")
            return 0
    
    async def improve_skill(
        self,
        memory_id: str,
        success: bool,
        feedback: Optional[str] = None
    ) -> bool:
        """
        Improve a procedural memory based on execution feedback.
        
        Args:
            memory_id: Memory ID
            success: Whether execution was successful
            feedback: Optional feedback text
            
        Returns:
            True if updated
        """
        try:
            memory = await self.get_by_id(memory_id)
            if not memory:
                return False
            
            # Update execution count
            exec_count = memory.metadata.get("execution_count", 0) + 1
            
            # Update success rate
            current_rate = memory.metadata.get("success_rate", 0.0)
            new_rate = ((current_rate * (exec_count - 1)) + (1.0 if success else 0.0)) / exec_count
            
            # Update skill level (increases with successful executions)
            skill_level = memory.metadata.get("skill_level", 1)
            if success and exec_count % 5 == 0:  # Level up every 5 successful executions
                skill_level += 1
            
            updates = {
                "metadata.execution_count": exec_count,
                "metadata.success_rate": new_rate,
                "metadata.skill_level": skill_level,
                "metadata.last_execution": datetime.utcnow().isoformat(),
                "metadata.last_feedback": feedback
            }
            
            # Increase importance if consistently successful
            if new_rate > 0.8:
                updates["importance"] = min(1.0, memory.importance + 0.1)
            
            return await self.update(memory_id, updates)
            
        except Exception as e:
            logger.error(f"Failed to improve skill {memory_id}: {e}")
            return False
    
    async def get_best_procedures(
        self,
        agent_id: str,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[Memory]:
        """Get the best performing procedures."""
        filters = {"agent_id": agent_id}
        if category:
            filters["tags"] = category
        
        return await self.list_memories(filters=filters, limit=limit)
