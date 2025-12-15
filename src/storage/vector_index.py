"""
MongoDB Vector Index Management
Creates and manages Atlas Vector Search indexes
"""

import logging
from motor.motor_asyncio import AsyncIOMotorCollection
from typing import Dict, Any

logger = logging.getLogger(__name__)


class VectorIndexManager:
    """Manages MongoDB Atlas Vector Search indexes."""
    
    def __init__(self, collection: AsyncIOMotorCollection):
        """
        Initialize the index manager.
        
        Args:
            collection: MongoDB collection
        """
        self.collection = collection
    
    async def create_index(
        self,
        index_name: str = "vector_index",
        embedding_field: str = "embedding",  # Field used by AWM memory stores
        dimensions: int = 1024,  # Voyage AI dimensions
        similarity: str = "cosine"
    ) -> Dict[str, Any]:
        """
        Create a vector search index.
        
        Args:
            index_name: Name of the index
            embedding_field: Field containing embeddings
            dimensions: Embedding dimensions
            similarity: Similarity metric
            
        Returns:
            Index creation result
        """
        index_definition = {
            "name": index_name,
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": embedding_field,
                        "similarity": similarity,
                        "numDimensions": dimensions
                    },
                    {
                        "type": "filter",
                        "path": "agent_id"
                    },
                    {
                        "type": "filter",
                        "path": "user_id"
                    },
                    {
                        "type": "filter",
                        "path": "memory_type"
                    }
                ]
            }
        }
        
        try:
            # Check if index exists
            existing_indexes = await self.collection.list_search_indexes().to_list(length=None)
            index_exists = any(idx.get("name") == index_name for idx in existing_indexes)
            
            if index_exists:
                logger.info(f"Index '{index_name}' already exists on collection '{self.collection.name}'")
                return {"status": "exists", "index_name": index_name}
            
            # Create the index
            result = await self.collection.create_search_index(index_definition)
            logger.info(f"Successfully created index '{index_name}' on collection '{self.collection.name}'")
            return {"status": "created", "index_name": index_name, "result": result}
            
        except Exception as e:
            # MongoDB Atlas free tier may have limitations
            if "already exists" in str(e) or "duplicate" in str(e).lower():
                logger.info(f"Index '{index_name}' already exists")
                return {"status": "exists", "index_name": index_name}
            elif "not allowed" in str(e) or "permission" in str(e).lower():
                logger.warning(f"Cannot create index (may need Atlas permissions): {e}")
                return {"status": "permission_error", "error": str(e)}
            else:
                logger.error(f"Error creating index: {e}")
                return {"status": "error", "error": str(e)}


async def create_vector_index(collection: AsyncIOMotorCollection):
    """
    Create vector search index (backward compatible function).

    Args:
        collection: MongoDB collection
    """
    manager = VectorIndexManager(collection)
    return await manager.create_index()


# All memory collections that need vector indexes
MEMORY_COLLECTIONS = [
    "episodic_memories",
    "semantic_memories",
    "procedural_memories",
    "working_memories",
    "cache_memories",
    "entity_memories",
    "summary_memories"
]


async def ensure_all_vector_indexes(db) -> Dict[str, Any]:
    """
    Create vector search indexes for all memory collections.

    Args:
        db: MongoDB database instance (AsyncIOMotorDatabase)

    Returns:
        Dict mapping collection names to creation results
    """
    results = {}

    for collection_name in MEMORY_COLLECTIONS:
        collection = db[collection_name]
        manager = VectorIndexManager(collection)
        result = await manager.create_index()
        results[collection_name] = result
        logger.info(f"{collection_name}: {result['status']}")

    return results


async def wait_for_indexes_ready(db, timeout_seconds: int = 120) -> bool:
    """
    Wait for all vector indexes to become queryable.

    Note: Atlas vector indexes can take 1-5 minutes to build initially.

    Args:
        db: MongoDB database instance
        timeout_seconds: Maximum time to wait

    Returns:
        True if all indexes are ready
    """
    import asyncio

    waited = 0
    interval = 10

    while waited < timeout_seconds:
        all_ready = True

        for collection_name in MEMORY_COLLECTIONS:
            collection = db[collection_name]
            ready = False

            async for index in collection.list_search_indexes():
                if index.get("name") == "vector_index":
                    status = index.get("status", "UNKNOWN")
                    if status == "READY":
                        ready = True
                    else:
                        logger.info(f"{collection_name} index status: {status}")
                        all_ready = False
                    break

            if not ready:
                all_ready = False

        if all_ready:
            logger.info("All vector indexes are READY")
            return True

        await asyncio.sleep(interval)
        waited += interval
        logger.info(f"Waiting for indexes... ({waited}s / {timeout_seconds}s)")

    logger.warning("Timeout waiting for indexes to be ready")
    return False