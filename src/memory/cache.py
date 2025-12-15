"""
Semantic Cache Store
High-performance cache for frequently accessed queries
Saves 80% compute by reusing knowledge
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection

from .base import Memory, MemoryStore, MemoryType
from ..retrieval.vector_search import VectorSearchEngine
from ..embeddings.voyage_client import get_embedding_service

logger = logging.getLogger(__name__)


class SemanticCache(MemoryStore):
    """
    Semantic cache for performance optimization.
    Caches query results to avoid redundant processing.
    """
    
    def __init__(self, collection: AsyncIOMotorCollection):
        """Initialize with MongoDB collection."""
        self.collection = collection
        self.search_engine = VectorSearchEngine(collection)
        self.embedding_service = get_embedding_service()
        
        # Cache configuration
        self.default_ttl_seconds = 3600  # 1 hour default
        self.similarity_threshold = 0.95  # High threshold for cache hits
        self.max_cache_size = 1000  # Maximum cached queries
    
    async def store(self, memory: Memory) -> str:
        """Store a cache entry."""
        try:
            # Ensure it's cache type
            memory.memory_type = MemoryType.CACHE
            
            # Set TTL if not provided
            if not memory.ttl:
                memory.ttl = datetime.utcnow() + timedelta(seconds=self.default_ttl_seconds)
            
            # Generate cache key
            cache_key = self._generate_cache_key(memory.content)
            memory.metadata["cache_key"] = cache_key
            memory.metadata["hit_count"] = 0
            memory.metadata["last_hit"] = None
            
            # Check if already cached
            existing = await self.collection.find_one({"metadata.cache_key": cache_key})
            if existing:
                # Update existing cache entry
                await self.collection.update_one(
                    {"_id": existing["_id"]},
                    {
                        "$set": {
                            "content": memory.content,
                            "metadata.response": memory.metadata.get("response"),
                            "ttl": memory.ttl,
                            "updated_at": datetime.utcnow()
                        },
                        "$inc": {"metadata.hit_count": 1}
                    }
                )
                return str(existing["_id"])
            
            # Manage cache size
            await self._evict_if_needed()
            
            # Convert to dict for MongoDB
            memory_dict = memory.model_dump(exclude={"id"})
            memory_dict["_id"] = ObjectId()
            
            # Insert into collection
            result = await self.collection.insert_one(memory_dict)
            
            logger.debug(f"Cached query: {cache_key[:8]}...")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to store cache entry: {e}")
            raise
    
    async def retrieve(
        self,
        query: str,
        limit: int = 1,
        threshold: float = 0.95,
        search_mode: str = "hybrid"
    ) -> List[Memory]:
        """
        Retrieve cached results for a query.

        Uses hybrid search (vector + full-text) by default for best results.
        Based on MongoDB's official GenAI-Showcase pattern with $rankFusion.

        Args:
            query: Search query text
            limit: Maximum memories to return
            threshold: Minimum similarity threshold (high for cache)
            search_mode: Search strategy - "hybrid" (default), "semantic", or "text"

        Returns:
            List of cached results
        """
        try:
            # First try exact match with cache key
            cache_key = self._generate_cache_key(query)
            exact_match = await self.collection.find_one({
                "metadata.cache_key": cache_key,
                "ttl": {"$gt": datetime.utcnow()}
            })

            if exact_match:
                # Update hit statistics
                await self.collection.update_one(
                    {"_id": exact_match["_id"]},
                    {
                        "$inc": {"metadata.hit_count": 1},
                        "$set": {
                            "metadata.last_hit": datetime.utcnow(),
                            "accessed_at": datetime.utcnow()
                        }
                    }
                )

                exact_match["id"] = str(exact_match.pop("_id"))
                memory = Memory(**exact_match)
                logger.debug(f"Cache hit (exact): {cache_key[:8]}...")
                return [memory]

            # Try similarity search (hybrid or semantic)
            embedding_result = await self.embedding_service.generate_embedding(
                query, input_type="query"
            )

            filter_query = {
                "memory_type": "cache",
                "ttl": {"$gt": datetime.utcnow()}
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
            
            # Only return if similarity is very high
            memories = []
            for result in results:
                if result.score >= threshold:
                    doc = await self.collection.find_one({"_id": ObjectId(result.id)})
                    if doc:
                        # Update hit statistics
                        await self.collection.update_one(
                            {"_id": doc["_id"]},
                            {
                                "$inc": {"metadata.hit_count": 1},
                                "$set": {
                                    "metadata.last_hit": datetime.utcnow(),
                                    "accessed_at": datetime.utcnow()
                                }
                            }
                        )
                        
                        doc["id"] = str(doc.pop("_id"))
                        memory = Memory(**doc)
                        memory.metadata["search_score"] = result.score
                        memories.append(memory)
                        logger.debug(f"Cache hit (semantic): score={result.score:.3f}")
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve from cache: {e}")
            return []
    
    async def get_by_id(self, memory_id: str) -> Optional[Memory]:
        """Get cache entry by ID."""
        try:
            doc = await self.collection.find_one({"_id": ObjectId(memory_id)})
            if doc:
                doc["id"] = str(doc.pop("_id"))
                return Memory(**doc)
            return None
        except Exception as e:
            logger.error(f"Failed to get cache entry {memory_id}: {e}")
            return None
    
    async def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a cache entry."""
        try:
            updates["updated_at"] = datetime.utcnow()
            result = await self.collection.update_one(
                {"_id": ObjectId(memory_id)},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update cache entry {memory_id}: {e}")
            return False
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a cache entry."""
        try:
            result = await self.collection.delete_one({"_id": ObjectId(memory_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to delete cache entry {memory_id}: {e}")
            return False
    
    async def list_memories(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Memory]:
        """List cache entries with filters."""
        try:
            query = filters or {}
            query["memory_type"] = "cache"
            
            # Only show non-expired by default
            if "ttl" not in query:
                query["ttl"] = {"$gt": datetime.utcnow()}
            
            # Sort by hit count and recency
            cursor = self.collection.find(query).skip(offset).limit(limit)
            cursor = cursor.sort([
                ("metadata.hit_count", -1),
                ("metadata.last_hit", -1)
            ])
            
            memories = []
            async for doc in cursor:
                doc["id"] = str(doc.pop("_id"))
                memories.append(Memory(**doc))
            
            return memories
        except Exception as e:
            logger.error(f"Failed to list cache entries: {e}")
            return []
    
    async def clear_all(self, confirm: bool = False) -> int:
        """Clear all cache entries."""
        if not confirm:
            raise ValueError("Must confirm deletion")
        
        try:
            result = await self.collection.delete_many({"memory_type": "cache"})
            logger.info(f"Cleared {result.deleted_count} cache entries")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return 0
    
    def _generate_cache_key(self, content: str) -> str:
        """Generate deterministic cache key for content."""
        # Normalize content (lowercase, strip whitespace)
        normalized = content.lower().strip()
        # Generate hash
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    async def _evict_if_needed(self):
        """Evict old cache entries if cache is full."""
        count = await self.collection.count_documents({"memory_type": "cache"})
        
        if count >= self.max_cache_size:
            # Remove expired entries first
            expired_result = await self.collection.delete_many({
                "memory_type": "cache",
                "ttl": {"$lt": datetime.utcnow()}
            })
            
            if expired_result.deleted_count > 0:
                logger.debug(f"Evicted {expired_result.deleted_count} expired cache entries")
            
            # If still over limit, remove least recently used
            count = await self.collection.count_documents({"memory_type": "cache"})
            if count >= self.max_cache_size:
                to_remove = count - self.max_cache_size + 1
                
                # Find LRU entries
                cursor = self.collection.find({"memory_type": "cache"}).sort([
                    ("metadata.last_hit", 1),  # Oldest hit first
                    ("accessed_at", 1)  # Oldest access first
                ]).limit(to_remove)
                
                ids_to_remove = []
                async for doc in cursor:
                    ids_to_remove.append(doc["_id"])
                
                if ids_to_remove:
                    result = await self.collection.delete_many({"_id": {"$in": ids_to_remove}})
                    logger.debug(f"Evicted {result.deleted_count} LRU cache entries")
    
    async def cleanup_expired(self) -> int:
        """Remove expired cache entries."""
        result = await self.collection.delete_many({
            "memory_type": "cache",
            "ttl": {"$lt": datetime.utcnow()}
        })
        
        if result.deleted_count > 0:
            logger.info(f"Cleaned up {result.deleted_count} expired cache entries")
        
        return result.deleted_count
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        # Total entries
        total = await self.collection.count_documents({"memory_type": "cache"})
        
        # Active entries
        active = await self.collection.count_documents({
            "memory_type": "cache",
            "ttl": {"$gt": datetime.utcnow()}
        })
        
        # Get hit statistics
        pipeline = [
            {"$match": {"memory_type": "cache"}},
            {"$group": {
                "_id": None,
                "total_hits": {"$sum": "$metadata.hit_count"},
                "avg_hits": {"$avg": "$metadata.hit_count"},
                "max_hits": {"$max": "$metadata.hit_count"}
            }}
        ]
        
        stats_cursor = self.collection.aggregate(pipeline)
        hit_stats = await stats_cursor.to_list(1)
        
        stats = {
            "total_entries": total,
            "active_entries": active,
            "expired_entries": total - active,
            "cache_utilization": (total / self.max_cache_size) * 100 if self.max_cache_size > 0 else 0
        }
        
        if hit_stats:
            stats.update({
                "total_hits": hit_stats[0].get("total_hits", 0),
                "avg_hits_per_entry": hit_stats[0].get("avg_hits", 0),
                "max_hits": hit_stats[0].get("max_hits", 0)
            })
        
        return stats
    
    async def warm_cache(self, common_queries: List[str]):
        """Pre-populate cache with common queries."""
        warmed = 0
        
        for query in common_queries:
            # Check if already cached
            cache_key = self._generate_cache_key(query)
            existing = await self.collection.find_one({"metadata.cache_key": cache_key})
            
            if not existing:
                # Create cache entry (response will be filled later)
                memory = Memory(
                    agent_id="system",
                    memory_type=MemoryType.CACHE,
                    content=query,
                    metadata={"warmed": True},
                    importance=0.8
                )
                await self.store(memory)
                warmed += 1
        
        logger.info(f"Warmed cache with {warmed} queries")
        return warmed
