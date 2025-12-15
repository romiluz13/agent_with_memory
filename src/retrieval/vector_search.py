"""
MongoDB Vector Search Implementation
Supports both pure vector search and hybrid search ($rankFusion).

Based on MongoDB's official GenAI-Showcase patterns.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from motor.motor_asyncio import AsyncIOMotorCollection

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from vector search."""
    content: str
    metadata: Dict[str, Any]
    score: float
    id: Optional[str] = None


class VectorSearchEngine:
    """
    Performs vector similarity search using MongoDB Atlas.
    Uses exact patterns from MongoDB examples.
    """
    
    def __init__(self, collection: AsyncIOMotorCollection):
        """Initialize with MongoDB collection."""
        self.collection = collection
    
    async def search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        num_candidates: int = 100,
        filter_query: Optional[Dict[str, Any]] = None,
        index_name: str = "vector_index"
    ) -> List[SearchResult]:
        """
        Perform vector similarity search.
        
        Args:
            query_embedding: Query vector (1024 dimensions for Voyage AI)
            limit: Number of results to return
            num_candidates: Number of candidates to consider (affects accuracy)
            filter_query: Optional MongoDB filter query
            index_name: Name of the vector index
            
        Returns:
            List of search results sorted by similarity
        """
        try:
            # Build the aggregation pipeline - EXACT pattern from MongoDB examples
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": index_name,
                        "path": "embedding",  # Field containing embeddings
                        "queryVector": query_embedding,
                        "numCandidates": num_candidates,  # MongoDB recommends 20:1 ratio
                        "limit": limit
                    }
                }
            ]
            
            # Add filter if provided
            if filter_query:
                pipeline[0]["$vectorSearch"]["filter"] = filter_query
            
            # Add projection to include score
            pipeline.append({
                "$project": {
                    "_id": 1,
                    "content": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}  # Similarity score
                }
            })
            
            # Execute search
            results = []
            async for doc in self.collection.aggregate(pipeline):
                result = SearchResult(
                    id=str(doc.get("_id")),
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    score=doc.get("score", 0.0)
                )
                results.append(result)
            
            logger.debug(f"Vector search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise
    
    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        limit: int = 5,
        vector_weight: float = 0.5,
        text_weight: float = 0.5,
        filter_query: Optional[Dict[str, Any]] = None,
        num_candidates: int = 100,
        index_name: str = "vector_index",
        text_index_name: str = "text_search_index"
    ) -> List[SearchResult]:
        """
        Perform hybrid search using MongoDB's $rankFusion operator.

        Based on MongoDB's official GenAI-Showcase pattern.
        Combines vector similarity search with full-text search for best results.

        Args:
            query_text: Text search query
            query_embedding: Query vector (1024 dimensions for Voyage AI)
            limit: Number of results to return
            vector_weight: Weight for vector search (0-1)
            text_weight: Weight for text search (0-1)
            filter_query: Optional MongoDB filter query for multi-tenant isolation
            num_candidates: Number of candidates to consider for vector search
            index_name: Name of the vector search index
            text_index_name: Name of the text search index

        Returns:
            Combined search results sorted by fused score
        """
        try:
            # Build vector search pipeline (matches GenAI-Showcase pattern)
            vector_pipeline = [
                {
                    "$vectorSearch": {
                        "index": index_name,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": num_candidates,
                        "limit": limit,
                    }
                }
            ]

            # Add filter to vector pipeline if provided
            if filter_query:
                vector_pipeline[0]["$vectorSearch"]["filter"] = filter_query

            # Build text search pipeline
            text_pipeline = [
                {
                    "$search": {
                        "index": text_index_name,
                        "text": {
                            "query": query_text,
                            "path": ["content"]
                        },
                    }
                }
            ]

            # Add filter for text pipeline (as $match stage after $search)
            if filter_query:
                text_pipeline.append({"$match": filter_query})

            # Add limit for text pipeline
            text_pipeline.append({"$limit": limit})

            # Build $rankFusion pipeline (MongoDB official pattern)
            pipeline = [
                {
                    "$rankFusion": {
                        "input": {
                            "pipelines": {
                                "vectorPipeline": vector_pipeline,
                                "textPipeline": text_pipeline,
                            }
                        },
                        "combination": {
                            "weights": {
                                "vectorPipeline": vector_weight,
                                "textPipeline": text_weight,
                            }
                        },
                        "scoreDetails": True,
                    }
                },
                {"$addFields": {"scoreDetails": {"$meta": "scoreDetails"}}},
                {"$addFields": {"score": "$scoreDetails.value"}},
                {"$limit": limit},
                {
                    "$project": {
                        "_id": 1,
                        "content": 1,
                        "metadata": 1,
                        "score": 1,
                    }
                },
            ]

            # Execute hybrid search
            results = []
            async for doc in self.collection.aggregate(pipeline):
                result = SearchResult(
                    id=str(doc.get("_id")),
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    score=doc.get("score", 0.0)
                )
                results.append(result)

            logger.debug(f"Hybrid search returned {len(results)} results")
            return results

        except Exception as e:
            # CRITICAL: Graceful fallback to vector-only search
            # This ensures the system works even if text index is not ready
            logger.warning(f"Hybrid search failed, falling back to vector search: {e}")
            return await self.search(
                query_embedding=query_embedding,
                limit=limit,
                num_candidates=num_candidates,
                filter_query=filter_query,
                index_name=index_name
            )
    
    async def search_with_reranking(
        self,
        query_embedding: List[float],
        limit: int = 5,
        rerank_factor: int = 3
    ) -> List[SearchResult]:
        """
        Search with reranking for better results.
        
        Args:
            query_embedding: Query vector
            limit: Final number of results
            rerank_factor: How many times more results to fetch for reranking
            
        Returns:
            Reranked search results
        """
        # Get more results for reranking
        initial_results = await self.search(
            query_embedding=query_embedding,
            limit=limit * rerank_factor,
            num_candidates=200  # More candidates for better initial set
        )
        
        # TODO: Implement sophisticated reranking logic
        # For now, just return top results (already sorted by score)
        return initial_results[:limit]
    
    async def search_memories_by_type(
        self,
        query_embedding: List[float],
        memory_type: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 5
    ) -> List[SearchResult]:
        """
        Search memories filtered by type and ownership.
        
        Args:
            query_embedding: Query vector
            memory_type: Type of memory to search
            user_id: Optional user filter
            agent_id: Optional agent filter
            limit: Number of results
            
        Returns:
            Filtered search results
        """
        # Build filter query
        filter_query = {"memory_type": memory_type}
        
        if user_id:
            filter_query["user_id"] = user_id
        
        if agent_id:
            filter_query["agent_id"] = agent_id
        
        # Perform filtered search
        return await self.search(
            query_embedding=query_embedding,
            limit=limit,
            filter_query=filter_query
        )


class MultiCollectionSearch:
    """Search across multiple collections."""
    
    def __init__(self, collections: Dict[str, AsyncIOMotorCollection]):
        """
        Initialize with multiple collections.
        
        Args:
            collections: Dict mapping collection names to collections
        """
        self.search_engines = {
            name: VectorSearchEngine(collection)
            for name, collection in collections.items()
        }
    
    async def search_all(
        self,
        query_embedding: List[float],
        limit_per_collection: int = 3
    ) -> Dict[str, List[SearchResult]]:
        """
        Search across all collections.
        
        Args:
            query_embedding: Query vector
            limit_per_collection: Results per collection
            
        Returns:
            Results grouped by collection
        """
        results = {}
        
        for name, engine in self.search_engines.items():
            try:
                collection_results = await engine.search(
                    query_embedding=query_embedding,
                    limit=limit_per_collection
                )
                results[name] = collection_results
                logger.debug(f"Found {len(collection_results)} results in {name}")
            except Exception as e:
                logger.error(f"Search failed for {name}: {e}")
                results[name] = []
        
        return results


# Example usage
if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv
    from ..storage.mongodb_client import initialize_mongodb
    from ..embeddings.voyage_client import get_embedding_service
    
    load_dotenv()
    
    async def main():
        # Initialize MongoDB
        client = await initialize_mongodb(
            uri=os.getenv("MONGODB_URI"),
            database=os.getenv("MONGODB_DB_NAME")
        )
        
        # Get collection
        collection = client.get_collection("memories")
        
        # Initialize search engine
        search_engine = VectorSearchEngine(collection)
        
        # Generate query embedding
        embedding_service = get_embedding_service()
        result = await embedding_service.generate_embedding(
            "How do AI agents work?",
            input_type="query"
        )
        
        # Perform search
        results = await search_engine.search(
            query_embedding=result.embedding,
            limit=5
        )
        
        for r in results:
            print(f"Score: {r.score:.4f} - Content: {r.content[:100]}...")
        
        await client.close()
    
    asyncio.run(main())
