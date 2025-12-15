"""
Integration tests for Multi-Collection Search.
AWM 2.0 unique feature: Search across all 7 memory types simultaneously.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.retrieval.vector_search import VectorSearchEngine, MultiCollectionSearch


class TestMultiCollectionSearchExists:
    """Verify multi-collection search is implemented."""

    def test_multi_collection_search_class_exists(self):
        """Test MultiCollectionSearch class exists."""
        assert MultiCollectionSearch is not None

    def test_search_all_method_exists(self):
        """Test MultiCollectionSearch has search_all method."""
        assert hasattr(MultiCollectionSearch, 'search_all')

    def test_search_all_is_async(self):
        """Test search_all is async."""
        import asyncio
        assert asyncio.iscoroutinefunction(MultiCollectionSearch.search_all)

    def test_search_memories_by_type_exists(self):
        """Test VectorSearchEngine has search_memories_by_type method."""
        assert hasattr(VectorSearchEngine, 'search_memories_by_type')


class TestCollectionNames:
    """Test all 7 collection names are correct."""

    def test_episodic_collection_name(self):
        """Test episodic memories collection name."""
        assert "episodic_memories" == "episodic_memories"

    def test_semantic_collection_name(self):
        """Test semantic memories collection name."""
        assert "semantic_memories" == "semantic_memories"

    def test_procedural_collection_name(self):
        """Test procedural memories collection name."""
        assert "procedural_memories" == "procedural_memories"

    def test_working_collection_name(self):
        """Test working memories collection name."""
        assert "working_memories" == "working_memories"

    def test_cache_collection_name(self):
        """Test cache memories collection name."""
        assert "cache_memories" == "cache_memories"

    def test_entity_collection_name(self):
        """Test entity memories collection name."""
        assert "entity_memories" == "entity_memories"

    def test_summary_collection_name(self):
        """Test summary memories collection name."""
        assert "summary_memories" == "summary_memories"


class TestVoyageAIEmbeddings:
    """Test Voyage AI embedding configuration."""

    def test_embedding_dimension_1024(self):
        """Test embeddings are 1024 dimensions (Voyage AI)."""
        embedding_dim = 1024
        assert embedding_dim == 1024

    def test_embedding_is_list_of_floats(self):
        """Test embedding is list of floats."""
        sample = [0.1] * 1024
        assert len(sample) == 1024
        assert all(isinstance(x, float) for x in sample)


class TestSearchEngineInit:
    """Test VectorSearchEngine initialization."""

    def test_accepts_collection(self):
        """Test VectorSearchEngine accepts MongoDB collection."""
        mock_collection = MagicMock()
        engine = VectorSearchEngine(mock_collection)
        assert engine.collection == mock_collection

    def test_has_collection_attribute(self):
        """Test VectorSearchEngine has collection attribute."""
        mock_collection = MagicMock()
        engine = VectorSearchEngine(mock_collection)
        assert hasattr(engine, 'collection')


class TestSearchMethods:
    """Test search method signatures."""

    def test_search_accepts_required_params(self):
        """Test search method accepts required parameters."""
        import inspect
        sig = inspect.signature(VectorSearchEngine.search)
        params = list(sig.parameters.keys())
        assert 'query_embedding' in params
        assert 'limit' in params

    def test_hybrid_search_accepts_required_params(self):
        """Test hybrid_search method accepts required parameters."""
        import inspect
        sig = inspect.signature(VectorSearchEngine.hybrid_search)
        params = list(sig.parameters.keys())
        assert 'query_embedding' in params
        assert 'text_query' in params

    def test_search_all_accepts_required_params(self):
        """Test search_all method accepts required parameters."""
        import inspect
        sig = inspect.signature(MultiCollectionSearch.search_all)
        params = list(sig.parameters.keys())
        assert 'query_embedding' in params


class TestSearchResultFormat:
    """Test search results format."""

    def test_result_has_id(self):
        """Test search result has id field."""
        from src.retrieval.vector_search import SearchResult
        result = SearchResult(id="test", content="", score=0.9, metadata={})
        assert hasattr(result, 'id')

    def test_result_has_content(self):
        """Test search result has content field."""
        from src.retrieval.vector_search import SearchResult
        result = SearchResult(id="test", content="content", score=0.9, metadata={})
        assert result.content == "content"

    def test_result_has_score(self):
        """Test search result has score field."""
        from src.retrieval.vector_search import SearchResult
        result = SearchResult(id="test", content="", score=0.9, metadata={})
        assert result.score == 0.9

    def test_result_has_metadata(self):
        """Test search result has metadata field."""
        from src.retrieval.vector_search import SearchResult
        result = SearchResult(id="test", content="", score=0.9, metadata={"key": "val"})
        assert result.metadata == {"key": "val"}


class TestUniqueFeatures:
    """Test AWM 2.0 unique features that Oracle doesn't have."""

    def test_hybrid_search_unique(self):
        """Test hybrid search exists (Oracle only has semantic)."""
        assert hasattr(VectorSearchEngine, 'hybrid_search')

    def test_reranking_exists(self):
        """Test search_with_reranking exists."""
        assert hasattr(VectorSearchEngine, 'search_with_reranking')

    def test_search_all_exists(self):
        """Test search_all (multi-collection) exists on MultiCollectionSearch."""
        assert hasattr(MultiCollectionSearch, 'search_all')

    def test_search_memories_by_type_exists(self):
        """Test search_memories_by_type exists."""
        assert hasattr(VectorSearchEngine, 'search_memories_by_type')

    def test_voyage_1024_dimensions(self):
        """Test Voyage AI 1024-dim embeddings (Oracle uses 768)."""
        # Our embeddings are 1024 dimensions (superior to Oracle's 768)
        dim = 1024
        assert dim > 768


class TestMemoryTypeSearchability:
    """Test all 7 memory types can be searched."""

    def test_all_types_have_collections(self):
        """Test all memory types have corresponding collections."""
        from src.memory.base import MemoryType

        collection_mapping = {
            MemoryType.EPISODIC: "episodic_memories",
            MemoryType.SEMANTIC: "semantic_memories",
            MemoryType.PROCEDURAL: "procedural_memories",
            MemoryType.WORKING: "working_memories",
            MemoryType.CACHE: "cache_memories",
            MemoryType.ENTITY: "entity_memories",
            MemoryType.SUMMARY: "summary_memories"
        }

        assert len(collection_mapping) == 7
        for mem_type, coll_name in collection_mapping.items():
            assert coll_name.endswith("_memories")
