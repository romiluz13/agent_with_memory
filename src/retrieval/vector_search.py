"""
MongoDB Vector Search Implementation
Supports both pure vector search and hybrid search ($rankFusion).

Based on MongoDB's official GenAI-Showcase patterns.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

from motor.motor_asyncio import AsyncIOMotorCollection

from .config import NUM_CANDIDATES_MULTIPLIER, HybridSearchConfig, LexicalPrefilterConfig
from .rrf import reciprocal_rank_fusion
from .tier_support import SearchTier, get_strategy

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from vector search."""

    content: str
    metadata: dict[str, Any]
    score: float
    id: str | None = None
    document: dict[str, Any] | None = None


SEARCH_RESULT_SOURCE_PROJECTION = {
    "_id": 1,
    "agent_id": 1,
    "user_id": 1,
    "memory_type": 1,
    "content": 1,
    "metadata": 1,
    "importance": 1,
    "importance_level": 1,
    "created_at": 1,
    "updated_at": 1,
    "accessed_at": 1,
    "access_count": 1,
    "ttl": 1,
    "tags": 1,
    "source": 1,
    "confidence": 1,
    "summary_id": 1,
}


class VectorSearchEngine:
    """
    Performs vector similarity search using MongoDB Atlas.
    Uses exact patterns from MongoDB examples.
    """

    def __init__(self, collection: AsyncIOMotorCollection):
        """Initialize with MongoDB collection."""
        self.collection = collection

    @staticmethod
    def _search_result_from_doc(doc: dict[str, Any]) -> SearchResult:
        """Build a SearchResult from an aggregated MongoDB document."""
        projected_document = {key: value for key, value in doc.items() if key != "score"}
        return SearchResult(
            id=str(doc.get("_id")),
            content=doc.get("content", ""),
            metadata=doc.get("metadata", {}),
            score=doc.get("score", 0.0),
            document=projected_document,
        )

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        num_candidates: int | None = None,
        filter_query: dict[str, Any] | None = None,
        index_name: str = "vector_index",
    ) -> list[SearchResult]:
        """
        Perform vector similarity search.

        Args:
            query_embedding: Query vector (1024 dimensions for Voyage AI)
            limit: Number of results to return
            num_candidates: Number of candidates to consider (affects accuracy).
                           If None, uses limit * NUM_CANDIDATES_MULTIPLIER (default 20:1 ratio)
            filter_query: Optional MongoDB filter query
            index_name: Name of the vector index

        Returns:
            List of search results sorted by similarity
        """
        try:
            # Calculate numCandidates dynamically if not provided
            actual_num_candidates = (
                num_candidates
                if num_candidates is not None
                else (limit * NUM_CANDIDATES_MULTIPLIER)
            )

            # Build the aggregation pipeline - EXACT pattern from MongoDB examples
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": index_name,
                        "path": "embedding",  # Field containing embeddings
                        "queryVector": query_embedding,
                        "numCandidates": actual_num_candidates,  # MongoDB recommends 20:1 ratio
                        "limit": limit,
                    }
                }
            ]

            # Add filter if provided
            if filter_query:
                pipeline[0]["$vectorSearch"]["filter"] = filter_query

            # Add projection to include score
            pipeline.append(
                {
                    "$project": {
                        **SEARCH_RESULT_SOURCE_PROJECTION,
                        "score": {"$meta": "vectorSearchScore"},  # Similarity score
                    }
                }
            )

            # Execute search
            results = []
            async for doc in self.collection.aggregate(pipeline):
                results.append(self._search_result_from_doc(doc))

            logger.debug(
                f"Vector search returned {len(results)} results (numCandidates={actual_num_candidates})"
            )
            return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        limit: int = 5,
        vector_weight: float = 0.5,
        text_weight: float = 0.5,
        filter_query: dict[str, Any] | None = None,
        num_candidates: int | None = None,
        index_name: str = "vector_index",
        text_index_name: str = "text_search_index",
    ) -> list[SearchResult]:
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
            actual_num_candidates = (
                num_candidates
                if num_candidates is not None
                else (limit * NUM_CANDIDATES_MULTIPLIER)
            )

            # Build vector search pipeline (matches GenAI-Showcase pattern)
            vector_pipeline = [
                {
                    "$vectorSearch": {
                        "index": index_name,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": actual_num_candidates,
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
                        "text": {"query": query_text, "path": ["content"]},
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
                        **SEARCH_RESULT_SOURCE_PROJECTION,
                        "score": 1,
                    }
                },
            ]

            # Execute hybrid search
            results = []
            async for doc in self.collection.aggregate(pipeline):
                results.append(self._search_result_from_doc(doc))

            logger.debug(f"Hybrid search returned {len(results)} results")
            return results

        except Exception as e:
            # CRITICAL: Graceful fallback to vector-only search
            # This ensures the system works even if text index is not ready
            logger.warning(f"Hybrid search failed, falling back to vector search: {e}")
            return await self.search(
                query_embedding=query_embedding,
                limit=limit,
                num_candidates=actual_num_candidates,
                filter_query=filter_query,
                index_name=index_name,
            )

    async def search_with_reranking(
        self, query_embedding: list[float], limit: int = 5, rerank_factor: int = 3
    ) -> list[SearchResult]:
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
            num_candidates=200,  # More candidates for better initial set
        )

        # TODO: Implement sophisticated reranking logic
        # For now, just return top results (already sorted by score)
        return initial_results[:limit]

    async def search_memories_by_type(
        self,
        query_embedding: list[float],
        memory_type: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
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
            query_embedding=query_embedding, limit=limit, filter_query=filter_query
        )

    async def search_with_lexical_prefilters(
        self,
        query_embedding: list[float],
        lexical_config: Optional["LexicalPrefilterConfig"] = None,
        limit: int = 5,
        index_name: str = "vector_index",
    ) -> list[SearchResult]:
        """
        Vector search with lexical prefiltering (MongoDB 8.2+).

        Uses $search.vectorSearch to apply Atlas Search filters
        BEFORE vector similarity calculation.

        Args:
            query_embedding: Query vector
            lexical_config: Lexical prefilter configuration
            limit: Number of results
            index_name: Name of the search index

        Returns:
            Search results with lexical prefiltering applied
        """
        from .filters.lexical_prefilters import (
            build_lexical_prefilters,
            build_search_vector_search_stage,
        )

        # Build lexical filters
        lexical_filters = build_lexical_prefilters(lexical_config)

        # Build the $search.vectorSearch stage
        search_stage = build_search_vector_search_stage(
            index_name=index_name,
            query_vector=query_embedding,
            limit=limit,
            num_candidates_multiplier=NUM_CANDIDATES_MULTIPLIER,
            lexical_filters=lexical_filters if lexical_filters else None,
        )

        # Build pipeline
        pipeline = [
            search_stage,
            {"$addFields": {"score": {"$meta": "searchScore"}}},
            {
                "$project": {
                    "_id": 1,
                    "content": 1,
                    "metadata": 1,
                    "score": 1,
                }
            },
        ]

        try:
            results = []
            async for doc in self.collection.aggregate(pipeline):
                results.append(
                    SearchResult(
                        id=str(doc.get("_id")),
                        content=doc.get("content", ""),
                        metadata=doc.get("metadata", {}),
                        score=doc.get("score", 0.0),
                    )
                )

            logger.debug(f"Lexical prefilter search returned {len(results)} results")
            return results

        except Exception as e:
            # Fall back to standard vector search if lexical prefilters not supported
            logger.warning(f"Lexical prefilter search failed, falling back to vector: {e}")
            return await self.search(
                query_embedding=query_embedding,
                limit=limit,
                index_name=index_name,
            )

    async def hybrid_search_with_fallback(
        self,
        query_text: str,
        query_embedding: list[float],
        limit: int = 5,
        config: HybridSearchConfig | None = None,
        vector_filters: dict[str, Any] | None = None,
        atlas_filters: list[dict[str, Any]] | None = None,
        tier: SearchTier = SearchTier.M10_PLUS,
    ) -> list[SearchResult]:
        """
        Hybrid search with automatic tier fallback.

        Fallback cascade:
        1. Try native $rankFusion (M10+)
        2. Fall back to manual RRF (M0/M2)
        3. Fall back to vector-only (last resort)

        Args:
            query_text: Text query for full-text search
            query_embedding: Query vector
            limit: Number of results
            config: Hybrid search configuration
            vector_filters: MQL filters for $vectorSearch
            atlas_filters: Atlas filters for $search
            tier: Detected cluster tier

        Returns:
            Combined search results
        """
        config = config or HybridSearchConfig()
        strategy = get_strategy(tier)

        try:
            if strategy.should_use_rank_fusion():
                # Tier 1: Native $rankFusion
                return await self._rank_fusion_search(
                    query_text=query_text,
                    query_embedding=query_embedding,
                    limit=limit,
                    config=config,
                    vector_filters=vector_filters,
                    atlas_filters=atlas_filters,
                )
        except Exception as e:
            logger.warning(f"$rankFusion failed: {e}, trying manual RRF")

        try:
            if strategy.should_use_manual_rrf() or strategy.should_use_rank_fusion():
                # Tier 2: Manual RRF
                return await self._manual_rrf_search(
                    query_text=query_text,
                    query_embedding=query_embedding,
                    limit=limit,
                    config=config,
                    vector_filters=vector_filters,
                )
        except Exception as e:
            logger.warning(f"Manual RRF failed: {e}, falling back to vector-only")

        # Tier 3: Vector-only fallback
        return await self.search(
            query_embedding=query_embedding,
            limit=limit,
            filter_query=vector_filters,
            index_name=config.vector_index_name,
        )

    async def _rank_fusion_search(
        self,
        query_text: str,
        query_embedding: list[float],
        limit: int,
        config: HybridSearchConfig,
        vector_filters: dict[str, Any] | None = None,
        atlas_filters: list[dict[str, Any]] | None = None,
    ) -> list[SearchResult]:
        """
        Execute native $rankFusion search (M10+ only).
        """
        fetch_count = limit * config.over_fetch_multiplier
        num_candidates = fetch_count * config.num_candidates_multiplier

        # Vector pipeline
        vector_pipeline = [
            {
                "$vectorSearch": {
                    "index": config.vector_index_name,
                    "path": config.vector_path,
                    "queryVector": query_embedding,
                    "numCandidates": num_candidates,
                    "limit": fetch_count,
                }
            }
        ]

        if vector_filters:
            vector_pipeline[0]["$vectorSearch"]["filter"] = vector_filters

        # Text pipeline with optional Atlas filters
        text_search = {
            "index": config.text_index_name,
            "text": {
                "query": query_text,
                "path": config.text_search_path,
                "fuzzy": {
                    "maxEdits": config.fuzzy_max_edits,
                    "prefixLength": config.fuzzy_prefix_length,
                },
            },
        }

        text_pipeline = [{"$search": text_search}]

        # Apply Atlas filters as $match after $search
        if atlas_filters:
            # Convert Atlas filters to $match (simplified)
            match_filter = {}
            for f in atlas_filters:
                if "equals" in f:
                    match_filter[f["equals"]["path"]] = f["equals"]["value"]
            if match_filter:
                text_pipeline.append({"$match": match_filter})

        text_pipeline.append({"$limit": fetch_count})

        # $rankFusion pipeline
        pipeline = [
            {
                "$rankFusion": {
                    "input": {
                        "pipelines": {
                            "vector": vector_pipeline,
                            "text": text_pipeline,
                        }
                    },
                    "combination": {
                        "weights": {
                            "vector": config.vector_weight,
                            "text": config.text_weight,
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
                    "scoreDetails": 1,
                }
            },
        ]

        results = []
        async for doc in self.collection.aggregate(pipeline, allowDiskUse=True):
            results.append(
                SearchResult(
                    id=str(doc.get("_id")),
                    content=doc.get("content", ""),
                    metadata={
                        **doc.get("metadata", {}),
                        "score_details": doc.get("scoreDetails"),
                    },
                    score=doc.get("score", 0.0),
                )
            )

        logger.debug(f"$rankFusion search returned {len(results)} results")
        return results

    async def _manual_rrf_search(
        self,
        query_text: str,
        query_embedding: list[float],
        limit: int,
        config: HybridSearchConfig,
        vector_filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Execute manual RRF search (M0/M2 fallback).

        Runs vector and text search concurrently, then fuses with RRF.
        """
        fetch_count = limit * config.over_fetch_multiplier

        # Run both searches concurrently
        vector_task = self.search(
            query_embedding=query_embedding,
            limit=fetch_count,
            filter_query=vector_filters,
            index_name=config.vector_index_name,
        )

        text_task = self._text_only_search(
            query_text=query_text,
            limit=fetch_count,
            index_name=config.text_index_name,
            filter_query=vector_filters,  # Simplified filter for text
        )

        # Wait for both with exception handling
        vector_results, text_results = await asyncio.gather(
            vector_task, text_task, return_exceptions=True
        )

        # Handle exceptions
        if isinstance(vector_results, Exception):
            logger.warning(f"Vector search failed in RRF: {vector_results}")
            vector_results = []

        if isinstance(text_results, Exception):
            logger.warning(f"Text search failed in RRF: {text_results}")
            text_results = []

        # Fuse results using RRF
        merged = reciprocal_rank_fusion(
            result_lists={"vector": vector_results, "text": text_results},
            k=config.rrf_constant,
            top_k=limit,
            weights={"vector": config.vector_weight, "text": config.text_weight},
        )

        logger.debug(f"Manual RRF search returned {len(merged)} results")
        return merged

    async def _text_only_search(
        self,
        query_text: str,
        limit: int,
        index_name: str = "text_search_index",
        filter_query: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Execute text-only search.
        """
        pipeline = [
            {
                "$search": {
                    "index": index_name,
                    "text": {
                        "query": query_text,
                        "path": "content",
                    },
                }
            },
            {"$addFields": {"score": {"$meta": "searchScore"}}},
        ]

        if filter_query:
            pipeline.append({"$match": filter_query})

        pipeline.append({"$limit": limit})
        pipeline.append(
            {
                "$project": {
                    "_id": 1,
                    "content": 1,
                    "metadata": 1,
                    "score": 1,
                }
            }
        )

        results = []
        async for doc in self.collection.aggregate(pipeline):
            results.append(
                SearchResult(
                    id=str(doc.get("_id")),
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    score=doc.get("score", 0.0),
                )
            )

        return results


class MultiCollectionSearch:
    """Search across multiple collections with concurrent execution."""

    def __init__(self, collections: dict[str, AsyncIOMotorCollection]):
        """
        Initialize with multiple collections.

        Args:
            collections: Dict mapping collection names to collections
        """
        self.search_engines = {
            name: VectorSearchEngine(collection) for name, collection in collections.items()
        }

    async def search_all(
        self,
        query_embedding: list[float],
        limit_per_collection: int = 3,
        filter_query: dict[str, Any] | None = None,
    ) -> dict[str, list[SearchResult]]:
        """
        Search across all collections concurrently.

        Args:
            query_embedding: Query vector
            limit_per_collection: Results per collection
            filter_query: Optional filter (applied to all collections)

        Returns:
            Results grouped by collection
        """
        # Create tasks for concurrent execution
        tasks = {}
        for name, engine in self.search_engines.items():
            tasks[name] = engine.search(
                query_embedding=query_embedding,
                limit=limit_per_collection,
                filter_query=filter_query,
            )

        # Execute all searches concurrently
        task_list = list(tasks.items())
        results_list = await asyncio.gather(
            *[task for _, task in task_list], return_exceptions=True
        )

        # Map results back to collection names
        results = {}
        for (name, _), result in zip(task_list, results_list):
            if isinstance(result, Exception):
                logger.error(f"Search failed for {name}: {result}")
                results[name] = []
            else:
                results[name] = result
                logger.debug(f"Found {len(result)} results in {name}")

        return results

    async def hybrid_search_all(
        self,
        query_text: str,
        query_embedding: list[float],
        limit_per_collection: int = 3,
        config: HybridSearchConfig | None = None,
        vector_filters: dict[str, Any] | None = None,
        tier: SearchTier = SearchTier.M10_PLUS,
    ) -> dict[str, list[SearchResult]]:
        """
        Hybrid search across all collections concurrently.

        Args:
            query_text: Text query
            query_embedding: Query vector
            limit_per_collection: Results per collection
            config: Hybrid search config
            vector_filters: MQL filters
            tier: Cluster tier

        Returns:
            Results grouped by collection
        """
        config = config or HybridSearchConfig()

        # Create tasks for concurrent execution
        tasks = {}
        for name, engine in self.search_engines.items():
            tasks[name] = engine.hybrid_search_with_fallback(
                query_text=query_text,
                query_embedding=query_embedding,
                limit=limit_per_collection,
                config=config,
                vector_filters=vector_filters,
                tier=tier,
            )

        # Execute all searches concurrently
        task_list = list(tasks.items())
        results_list = await asyncio.gather(
            *[task for _, task in task_list], return_exceptions=True
        )

        # Map results
        results = {}
        for (name, _), result in zip(task_list, results_list):
            if isinstance(result, Exception):
                logger.error(f"Hybrid search failed for {name}: {result}")
                results[name] = []
            else:
                results[name] = result
                logger.debug(f"Found {len(result)} hybrid results in {name}")

        return results

    async def search_and_merge(
        self,
        query_embedding: list[float],
        limit: int = 10,
        limit_per_collection: int = 5,
        filter_query: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search all collections and merge results by score.

        Args:
            query_embedding: Query vector
            limit: Total results to return
            limit_per_collection: Max results per collection before merging
            filter_query: Optional filter

        Returns:
            Merged and sorted results from all collections
        """
        # Get results from all collections
        all_results = await self.search_all(
            query_embedding=query_embedding,
            limit_per_collection=limit_per_collection,
            filter_query=filter_query,
        )

        # Flatten and tag with source collection
        merged = []
        for collection_name, results in all_results.items():
            for result in results:
                # Add source collection to metadata
                enriched_metadata = {
                    **result.metadata,
                    "source_collection": collection_name,
                }
                merged.append(
                    SearchResult(
                        id=result.id,
                        content=result.content,
                        metadata=enriched_metadata,
                        score=result.score,
                    )
                )

        # Sort by score descending and limit
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:limit]


# Example usage
if __name__ == "__main__":
    import asyncio
    import os

    from dotenv import load_dotenv

    from ..embeddings.voyage_client import get_embedding_service
    from ..storage.mongodb_client import initialize_mongodb

    load_dotenv()

    async def main():
        # Initialize MongoDB
        client = await initialize_mongodb(
            uri=os.getenv("MONGODB_URI"), database=os.getenv("MONGODB_DB_NAME")
        )

        # Get collection
        collection = client.get_collection("memories")

        # Initialize search engine
        search_engine = VectorSearchEngine(collection)

        # Generate query embedding
        embedding_service = get_embedding_service()
        result = await embedding_service.generate_embedding(
            "How do AI agents work?", input_type="query"
        )

        # Perform search
        results = await search_engine.search(query_embedding=result.embedding, limit=5)

        for r in results:
            print(f"Score: {r.score:.4f} - Content: {r.content[:100]}...")

        await client.close()

    asyncio.run(main())
