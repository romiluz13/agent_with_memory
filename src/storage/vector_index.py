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
        embedding_field: str = "vector_embeddings",
        dimensions: int = 1024,  # voyage-2 dimensions
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