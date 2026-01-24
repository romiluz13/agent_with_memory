"""
MongoDB Vector Index Management
Creates and manages Atlas Vector Search indexes and Text Search indexes for hybrid search.

Based on MongoDB's official GenAI-Showcase pattern.
"""

import logging
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase

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
        similarity: str = "cosine",
    ) -> dict[str, Any]:
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
                        "numDimensions": dimensions,
                    },
                    # Core filters
                    {"type": "filter", "path": "agent_id"},
                    {"type": "filter", "path": "user_id"},
                    {"type": "filter", "path": "memory_type"},
                    # Enhanced filters for advanced queries
                    {"type": "filter", "path": "thread_id"},
                    {"type": "filter", "path": "timestamp"},
                    {"type": "filter", "path": "importance"},
                    {"type": "filter", "path": "metadata.tags"},
                    {"type": "filter", "path": "metadata.entity_type"},
                ]
            },
        }

        try:
            # Check if index exists
            existing_indexes = await self.collection.list_search_indexes().to_list(length=None)
            index_exists = any(idx.get("name") == index_name for idx in existing_indexes)

            if index_exists:
                logger.info(
                    f"Index '{index_name}' already exists on collection '{self.collection.name}'"
                )
                return {"status": "exists", "index_name": index_name}

            # Create the index
            result = await self.collection.create_search_index(index_definition)
            logger.info(
                f"Successfully created index '{index_name}' on collection '{self.collection.name}'"
            )
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

    async def create_text_index(self, index_name: str = "text_search_index") -> dict[str, Any]:
        """
        Create a text search index for hybrid search.

        Based on MongoDB GenAI-Showcase pattern.

        Args:
            index_name: Name of the text search index

        Returns:
            Index creation result
        """
        index_definition = {
            "name": index_name,
            "type": "search",
            "definition": {
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        # Primary content field with full-text analysis
                        "content": {"type": "string", "analyzer": "lucene.standard"},
                        # Core fields
                        "agent_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        # Enhanced text search fields
                        "metadata.tags": {"type": "string", "analyzer": "lucene.standard"},
                        "metadata.entity_name": {
                            "type": "string",
                            "analyzer": "lucene.keyword",  # Exact match
                        },
                        "metadata.entity_type": {"type": "string", "analyzer": "lucene.keyword"},
                        # Date fields for range queries
                        "timestamp": {"type": "date"},
                        "created_at": {"type": "date"},
                    },
                }
            },
        }

        try:
            # Check if index exists
            existing_indexes = await self.collection.list_search_indexes().to_list(length=None)
            index_exists = any(idx.get("name") == index_name for idx in existing_indexes)

            if index_exists:
                logger.info(
                    f"Text index '{index_name}' already exists on collection '{self.collection.name}'"
                )
                return {"status": "exists", "index_name": index_name}

            # Create the index
            result = await self.collection.create_search_index(index_definition)
            logger.info(
                f"Successfully created text index '{index_name}' on collection '{self.collection.name}'"
            )
            return {"status": "created", "index_name": index_name, "result": result}

        except Exception as e:
            if "already exists" in str(e) or "duplicate" in str(e).lower():
                logger.info(f"Text index '{index_name}' already exists")
                return {"status": "exists", "index_name": index_name}
            elif "not allowed" in str(e) or "permission" in str(e).lower():
                logger.warning(f"Cannot create text index (may need Atlas permissions): {e}")
                return {"status": "permission_error", "error": str(e)}
            else:
                logger.error(f"Error creating text index: {e}")
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
    "summary_memories",
]


async def ensure_all_vector_indexes(
    db: AsyncIOMotorDatabase, include_text_indexes: bool = True
) -> dict[str, Any]:
    """
    Create vector search indexes (and optionally text indexes) for all memory collections.

    Args:
        db: MongoDB database instance (AsyncIOMotorDatabase)
        include_text_indexes: Also create text search indexes for hybrid search

    Returns:
        Dict mapping collection names to creation results
    """
    results = {}

    for collection_name in MEMORY_COLLECTIONS:
        collection = db[collection_name]
        manager = VectorIndexManager(collection)

        # Create vector index
        vector_result = await manager.create_index()
        results[collection_name] = {"vector": vector_result}
        logger.info(f"{collection_name} vector: {vector_result['status']}")

        # Create text index for hybrid search
        if include_text_indexes:
            text_result = await manager.create_text_index()
            results[collection_name]["text"] = text_result
            logger.info(f"{collection_name} text: {text_result['status']}")

    return results


async def wait_for_indexes_ready(db: AsyncIOMotorDatabase, timeout_seconds: int = 120) -> bool:
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


async def migrate_vector_indexes(
    db: AsyncIOMotorDatabase,
    force_recreate: bool = False,
) -> dict[str, Any]:
    """
    Migrate existing vector indexes to new schema.

    WARNING: This will drop and recreate indexes if force_recreate=True.
    Index rebuilding can take several minutes for large collections.

    Args:
        db: MongoDB database instance
        force_recreate: Drop existing indexes before creating

    Returns:
        Dict with migration results per collection
    """
    results = {}

    for collection_name in MEMORY_COLLECTIONS:
        collection = db[collection_name]
        manager = VectorIndexManager(collection)
        collection_results = {"vector": None, "text": None}

        if force_recreate:
            # Drop existing indexes first
            try:
                existing_indexes = await collection.list_search_indexes().to_list(length=None)
                for idx in existing_indexes:
                    idx_name = idx.get("name")
                    if idx_name:
                        logger.info(f"Dropping index '{idx_name}' from {collection_name}")
                        await collection.drop_search_index(idx_name)
            except Exception as e:
                logger.warning(f"Error dropping indexes for {collection_name}: {e}")
                collection_results["drop_error"] = str(e)

        # Create updated indexes
        vector_result = await manager.create_index()
        collection_results["vector"] = vector_result
        logger.info(f"{collection_name} vector: {vector_result['status']}")

        text_result = await manager.create_text_index()
        collection_results["text"] = text_result
        logger.info(f"{collection_name} text: {text_result['status']}")

        results[collection_name] = collection_results

    return results


async def get_index_status(db: AsyncIOMotorDatabase) -> dict[str, Any]:
    """
    Get the current status of all search indexes.

    Args:
        db: MongoDB database instance

    Returns:
        Dict mapping collection names to index status
    """
    status = {}

    for collection_name in MEMORY_COLLECTIONS:
        collection = db[collection_name]
        collection_status = {"indexes": [], "ready": False}

        try:
            indexes = await collection.list_search_indexes().to_list(length=None)
            for idx in indexes:
                idx_info = {
                    "name": idx.get("name"),
                    "type": idx.get("type"),
                    "status": idx.get("status", "UNKNOWN"),
                }
                collection_status["indexes"].append(idx_info)

            # Check if vector index is ready
            vector_ready = any(
                idx.get("name") == "vector_index" and idx.get("status") == "READY"
                for idx in indexes
            )
            collection_status["ready"] = vector_ready

        except Exception as e:
            collection_status["error"] = str(e)
            logger.warning(f"Error getting index status for {collection_name}: {e}")

        status[collection_name] = collection_status

    return status
