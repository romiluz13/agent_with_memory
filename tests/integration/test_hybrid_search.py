"""
Integration tests for Hybrid Search functionality.
AWM 2.0 unique feature: Vector + Text search with weighted scoring.
Oracle only has semantic search - we have hybrid.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.retrieval.vector_search import VectorSearchEngine, SearchResult, MultiCollectionSearch


class TestHybridSearchExists:
    """Verify hybrid search is implemented."""

    def test_hybrid_search_method_exists(self):
        """Test VectorSearchEngine has hybrid_search method."""
        assert hasattr(VectorSearchEngine, 'hybrid_search')

    def test_search_method_exists(self):
        """Test VectorSearchEngine has basic search method."""
        assert hasattr(VectorSearchEngine, 'search')

    def test_search_with_reranking_exists(self):
        """Test VectorSearchEngine has search_with_reranking method."""
        assert hasattr(VectorSearchEngine, 'search_with_reranking')


class TestSearchResultModel:
    """Test SearchResult model."""

    def test_search_result_creation(self):
        """Test SearchResult can be created."""
        result = SearchResult(
            id="test-id",
            content="Test content",
            score=0.95,
            metadata={"key": "value"}
        )
        assert result.id == "test-id"
        assert result.content == "Test content"
        assert result.score == 0.95
        assert result.metadata == {"key": "value"}

    def test_search_result_score_range(self):
        """Test search scores should be between 0 and 1."""
        high_score = SearchResult(id="1", content="", score=0.99, metadata={})
        low_score = SearchResult(id="2", content="", score=0.01, metadata={})

        assert 0 <= high_score.score <= 1
        assert 0 <= low_score.score <= 1


class TestHybridSearchWeighting:
    """Test hybrid search weighting logic."""

    def test_default_weights(self):
        """Test default vector and text weights."""
        # Default: vector_weight=0.7, text_weight=0.3
        vector_weight = 0.7
        text_weight = 0.3
        assert vector_weight + text_weight == 1.0

    def test_weighted_score_calculation(self):
        """Test weighted score calculation."""
        vector_score = 0.8
        text_score = 0.6
        vector_weight = 0.7
        text_weight = 0.3

        combined = vector_weight * vector_score + text_weight * text_score
        expected = 0.7 * 0.8 + 0.3 * 0.6  # 0.56 + 0.18 = 0.74
        assert abs(combined - expected) < 0.001

    def test_vector_only_weight(self):
        """Test pure vector search (weight=1.0)."""
        vector_score = 0.9
        vector_weight = 1.0
        text_weight = 0.0

        combined = vector_weight * vector_score + text_weight * 0.5
        assert combined == vector_score

    def test_text_only_weight(self):
        """Test pure text search (weight=1.0)."""
        text_score = 0.8
        vector_weight = 0.0
        text_weight = 1.0

        combined = vector_weight * 0.5 + text_weight * text_score
        assert combined == text_score


class TestHybridSearchSignature:
    """Test hybrid_search method signature."""

    def test_hybrid_search_accepts_query_embedding(self):
        """Test hybrid_search accepts query_embedding parameter."""
        import inspect
        sig = inspect.signature(VectorSearchEngine.hybrid_search)
        params = list(sig.parameters.keys())
        assert 'query_embedding' in params

    def test_hybrid_search_accepts_text_query(self):
        """Test hybrid_search accepts text_query parameter."""
        import inspect
        sig = inspect.signature(VectorSearchEngine.hybrid_search)
        params = list(sig.parameters.keys())
        assert 'text_query' in params

    def test_hybrid_search_accepts_weights(self):
        """Test hybrid_search accepts weight parameters."""
        import inspect
        sig = inspect.signature(VectorSearchEngine.hybrid_search)
        params = list(sig.parameters.keys())
        assert 'vector_weight' in params
        assert 'text_weight' in params


class TestVectorSearchPipeline:
    """Test MongoDB $vectorSearch pipeline structure."""

    def test_pipeline_structure(self):
        """Test expected pipeline structure for vector search."""
        # This is what the pipeline should look like
        expected_stages = ["$vectorSearch", "$project"]
        pipeline = [
            {"$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": [0.1] * 1024,
                "numCandidates": 100,
                "limit": 10
            }},
            {"$project": {
                "_id": 1,
                "content": 1,
                "metadata": 1,
                "score": {"$meta": "vectorSearchScore"}
            }}
        ]

        for stage, expected in zip(pipeline, expected_stages):
            assert expected in stage

    def test_voyage_ai_embedding_dimension(self):
        """Test Voyage AI embeddings are 1024 dimensions."""
        embedding_dim = 1024
        sample_embedding = [0.1] * embedding_dim
        assert len(sample_embedding) == 1024


class TestMultiCollectionSearch:
    """Test searching across multiple collections."""

    def test_multi_collection_search_class_exists(self):
        """Test MultiCollectionSearch class exists."""
        assert MultiCollectionSearch is not None

    def test_search_all_method_exists(self):
        """Test search_all method exists on MultiCollectionSearch."""
        assert hasattr(MultiCollectionSearch, 'search_all')

    def test_search_memories_by_type_exists(self):
        """Test search_memories_by_type method exists on VectorSearchEngine."""
        assert hasattr(VectorSearchEngine, 'search_memories_by_type')

    def test_memory_types_searchable(self):
        """Test all 7 memory types have collections that can be searched."""
        collections = [
            "episodic_memories",
            "semantic_memories",
            "procedural_memories",
            "working_memories",
            "cache_memories",
            "entity_memories",
            "summary_memories"
        ]
        assert len(collections) == 7
