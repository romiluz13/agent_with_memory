"""
MongoDB Vector Search Implementation
Ported from MongoDB's retrieve-documents.js
Maintains exact aggregation pipeline patterns
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
        query_embedding: List[float],
        text_query: str,
        limit: int = 5,
        vector_weight: float = 0.7,
        text_weight: float = 0.3
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and text search.
        
        Args:
            query_embedding: Query vector
            text_query: Text search query
            limit: Number of results
            vector_weight: Weight for vector search (0-1)
            text_weight: Weight for text search (0-1)
            
        Returns:
            Combined search results
        """
        # Normalize weights
        total_weight = vector_weight + text_weight
        vector_weight = vector_weight / total_weight
        text_weight = text_weight / total_weight
        
        try:
            # Vector search pipeline
            vector_pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 100,
                        "limit": limit * 2  # Get more for merging
                    }
                },
                {
                    "$addFields": {
                        "vector_score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            # Text search pipeline
            text_pipeline = [
                {
                    "$search": {
                        "index": "text_index",  # Requires text index
                        "text": {
                            "query": text_query,
                            "path": "content"
                        }
                    }
                },
                {
                    "$limit": limit * 2
                },
                {
                    "$addFields": {
                        "text_score": {"$meta": "searchScore"}
                    }
                }
            ]
            
            # Run both searches
            vector_results = []
            async for doc in self.collection.aggregate(vector_pipeline):
                vector_results.append(doc)
            
            text_results = []
            async for doc in self.collection.aggregate(text_pipeline):
                text_results.append(doc)
            
            # Combine and score
            combined = {}
            
            # Add vector results
            for doc in vector_results:
                doc_id = str(doc["_id"])
                combined[doc_id] = {
                    "doc": doc,
                    "combined_score": doc["vector_score"] * vector_weight
                }
            
            # Add/update with text results
            for doc in text_results:
                doc_id = str(doc["_id"])
                if doc_id in combined:
                    combined[doc_id]["combined_score"] += doc["text_score"] * text_weight
                else:
                    combined[doc_id] = {
                        "doc": doc,
                        "combined_score": doc["text_score"] * text_weight
                    }
            
            # Sort by combined score and return top results
            sorted_results = sorted(
                combined.values(),
                key=lambda x: x["combined_score"],
                reverse=True
            )[:limit]
            
            results = []
            for item in sorted_results:
                doc = item["doc"]
                result = SearchResult(
                    id=str(doc.get("_id")),
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    score=item["combined_score"]
                )
                results.append(result)
            
            logger.debug(f"Hybrid search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise
    
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
