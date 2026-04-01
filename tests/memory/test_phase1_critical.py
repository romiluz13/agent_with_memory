"""
Tests for Phase 1 CRITICAL fixes: MongoDB audit remediation.
- Issue #1: No hardcoded credentials
- Issue #2: Collection name consistency (cache_memories, not semantic_cache)
- Issue #3: Multi-tenant isolation (agent_id in all retrieve() methods)
"""

import inspect
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Issue #1: No hardcoded credentials in scripts/
# ---------------------------------------------------------------------------


class TestNoHardcodedCredentials:
    """Verify no hardcoded MongoDB URIs exist in source or scripts."""

    def test_setup_test_indexes_uses_env_var(self):
        """setup_test_indexes.py must load MONGODB_URI from environment, not hardcode it."""
        script_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "scripts", "setup_test_indexes.py"
        )
        with open(script_path) as f:
            source = f.read()

        # Must NOT contain hardcoded connection strings
        assert (
            "mongodb+srv://" not in source
        ), "Hardcoded MongoDB URI found in setup_test_indexes.py"
        # Must use os.getenv for MONGODB_URI
        assert "os.getenv" in source, "setup_test_indexes.py must use os.getenv to load MONGODB_URI"

    def test_no_hardcoded_uris_in_src(self):
        """No hardcoded MongoDB URIs in any src/ Python file."""
        src_dir = os.path.join(os.path.dirname(__file__), "..", "..", "src")
        for root, _dirs, files in os.walk(src_dir):
            for fname in files:
                if fname.endswith(".py"):
                    fpath = os.path.join(root, fname)
                    with open(fpath) as f:
                        content = f.read()
                    assert (
                        "mongodb+srv://" not in content
                    ), f"Hardcoded MongoDB URI found in {fpath}"


# ---------------------------------------------------------------------------
# Issue #2: Collection name consistency
# ---------------------------------------------------------------------------


class TestCollectionNameConsistency:
    """Verify cache uses 'cache_memories' collection, not 'semantic_cache'."""

    def test_manager_uses_cache_memories_collection(self):
        """MemoryManager must use db['cache_memories'] for the cache store."""
        manager_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "memory", "manager.py"
        )
        with open(manager_path) as f:
            source = f.read()

        # Must NOT reference "semantic_cache"
        assert (
            "semantic_cache" not in source
        ), "manager.py still references 'semantic_cache' collection"
        # Must reference "cache_memories"
        assert (
            "cache_memories" in source
        ), "manager.py must use 'cache_memories' collection for cache"

    def test_no_semantic_cache_in_src(self):
        """No file in src/ should reference 'semantic_cache' collection name."""
        src_dir = os.path.join(os.path.dirname(__file__), "..", "..", "src")
        for root, _dirs, files in os.walk(src_dir):
            for fname in files:
                if fname.endswith(".py"):
                    fpath = os.path.join(root, fname)
                    with open(fpath) as f:
                        content = f.read()
                    assert (
                        "semantic_cache" not in content
                    ), f"'semantic_cache' collection name found in {fpath}"


# ---------------------------------------------------------------------------
# Issue #3: Multi-tenant isolation - agent_id in all retrieve() methods
# ---------------------------------------------------------------------------


class TestBaseMemoryStoreABC:
    """The MemoryStore ABC retrieve() must accept agent_id parameter."""

    def test_base_retrieve_accepts_agent_id(self):
        """MemoryStore.retrieve() ABC must include agent_id: Optional[str] = None."""
        from src.memory.base import MemoryStore

        sig = inspect.signature(MemoryStore.retrieve)
        assert "agent_id" in sig.parameters, "MemoryStore ABC retrieve() missing agent_id parameter"
        param = sig.parameters["agent_id"]
        assert param.default is None, "agent_id parameter must default to None for backward compat"


class TestSemanticMemoryIsolation:
    """SemanticMemory.retrieve() must accept and filter by agent_id."""

    @pytest.fixture
    def mock_collection(self):
        return AsyncMock()

    @pytest.fixture
    def semantic_memory(self, mock_collection):
        with (
            patch("src.memory.semantic.VectorSearchEngine"),
            patch("src.memory.semantic.get_embedding_service") as mock_embed,
        ):
            mock_embed_instance = MagicMock()
            mock_embed_instance.generate_embedding = AsyncMock(
                return_value=MagicMock(embedding=[0.1] * 1024)
            )
            mock_embed.return_value = mock_embed_instance
            mem = __import__("src.memory.semantic", fromlist=["SemanticMemory"]).SemanticMemory(
                mock_collection
            )
            # Make hybrid_search return empty by default
            mem.search_engine.hybrid_search = AsyncMock(return_value=[])
            mem.search_engine.search = AsyncMock(return_value=[])
            return mem

    def test_retrieve_signature_has_agent_id(self):
        """SemanticMemory.retrieve() must accept agent_id parameter."""
        from src.memory.semantic import SemanticMemory

        sig = inspect.signature(SemanticMemory.retrieve)
        assert "agent_id" in sig.parameters, "SemanticMemory.retrieve() missing agent_id parameter"

    @pytest.mark.asyncio
    async def test_retrieve_passes_agent_id_to_filter(self, semantic_memory):
        """When agent_id is provided, it must appear in the filter_query."""
        await semantic_memory.retrieve("test query", agent_id="agent-123")

        # Check that agent_id was included in the filter passed to search
        call_args = semantic_memory.search_engine.hybrid_search.call_args
        filter_query = call_args.kwargs.get("filter_query", {})
        assert (
            filter_query.get("agent_id") == "agent-123"
        ), "agent_id not forwarded to filter_query in SemanticMemory.retrieve()"


class TestProceduralMemoryIsolation:
    """ProceduralMemory.retrieve() must accept and filter by agent_id."""

    @pytest.fixture
    def mock_collection(self):
        return AsyncMock()

    @pytest.fixture
    def procedural_memory(self, mock_collection):
        with (
            patch("src.memory.procedural.VectorSearchEngine"),
            patch("src.memory.procedural.get_embedding_service") as mock_embed,
        ):
            mock_embed_instance = MagicMock()
            mock_embed_instance.generate_embedding = AsyncMock(
                return_value=MagicMock(embedding=[0.1] * 1024)
            )
            mock_embed.return_value = mock_embed_instance
            mem = __import__(
                "src.memory.procedural", fromlist=["ProceduralMemory"]
            ).ProceduralMemory(mock_collection)
            mem.search_engine.hybrid_search = AsyncMock(return_value=[])
            mem.search_engine.search = AsyncMock(return_value=[])
            return mem

    def test_retrieve_signature_has_agent_id(self):
        """ProceduralMemory.retrieve() must accept agent_id parameter."""
        from src.memory.procedural import ProceduralMemory

        sig = inspect.signature(ProceduralMemory.retrieve)
        assert "agent_id" in sig.parameters

    @pytest.mark.asyncio
    async def test_retrieve_passes_agent_id_to_filter(self, procedural_memory):
        """When agent_id is provided, it must appear in the filter_query."""
        await procedural_memory.retrieve("test query", agent_id="agent-456")

        call_args = procedural_memory.search_engine.hybrid_search.call_args
        filter_query = call_args.kwargs.get("filter_query", {})
        assert filter_query.get("agent_id") == "agent-456"


class TestWorkingMemoryIsolation:
    """WorkingMemory.retrieve() must accept and filter by agent_id."""

    @pytest.fixture
    def mock_collection(self):
        return AsyncMock()

    @pytest.fixture
    def working_memory(self, mock_collection):
        with (
            patch("src.memory.working.VectorSearchEngine"),
            patch("src.memory.working.get_embedding_service") as mock_embed,
        ):
            mock_embed_instance = MagicMock()
            mock_embed_instance.generate_embedding = AsyncMock(
                return_value=MagicMock(embedding=[0.1] * 1024)
            )
            mock_embed.return_value = mock_embed_instance
            mem = __import__("src.memory.working", fromlist=["WorkingMemory"]).WorkingMemory(
                mock_collection
            )
            mem.search_engine.hybrid_search = AsyncMock(return_value=[])
            mem.search_engine.search = AsyncMock(return_value=[])
            return mem

    def test_retrieve_signature_has_agent_id(self):
        """WorkingMemory.retrieve() must accept agent_id parameter."""
        from src.memory.working import WorkingMemory

        sig = inspect.signature(WorkingMemory.retrieve)
        assert "agent_id" in sig.parameters

    @pytest.mark.asyncio
    async def test_retrieve_passes_agent_id_to_filter(self, working_memory):
        """When agent_id is provided, it must appear in the filter_query."""
        await working_memory.retrieve("test query", agent_id="agent-789")

        call_args = working_memory.search_engine.hybrid_search.call_args
        filter_query = call_args.kwargs.get("filter_query", {})
        assert filter_query.get("agent_id") == "agent-789"


class TestCacheMemoryIsolation:
    """SemanticCache.retrieve() must accept and filter by agent_id."""

    @pytest.fixture
    def mock_collection(self):
        coll = AsyncMock()
        # find_one for exact match returns None (no cache hit)
        coll.find_one = AsyncMock(return_value=None)
        return coll

    @pytest.fixture
    def cache_memory(self, mock_collection):
        with (
            patch("src.memory.cache.VectorSearchEngine"),
            patch("src.memory.cache.get_embedding_service") as mock_embed,
        ):
            mock_embed_instance = MagicMock()
            mock_embed_instance.generate_embedding = AsyncMock(
                return_value=MagicMock(embedding=[0.1] * 1024)
            )
            mock_embed.return_value = mock_embed_instance
            mem = __import__("src.memory.cache", fromlist=["SemanticCache"]).SemanticCache(
                mock_collection
            )
            mem.search_engine.hybrid_search = AsyncMock(return_value=[])
            mem.search_engine.search = AsyncMock(return_value=[])
            return mem

    def test_retrieve_signature_has_agent_id(self):
        """SemanticCache.retrieve() must accept agent_id parameter."""
        from src.memory.cache import SemanticCache

        sig = inspect.signature(SemanticCache.retrieve)
        assert "agent_id" in sig.parameters

    @pytest.mark.asyncio
    async def test_retrieve_passes_agent_id_to_exact_match(self, cache_memory, mock_collection):
        """When agent_id is provided, exact-match query must include agent_id filter."""
        await cache_memory.retrieve("test query", agent_id="agent-cache-1")

        # Check the exact-match find_one call included agent_id
        find_one_call = mock_collection.find_one.call_args
        query = find_one_call[0][0] if find_one_call[0] else find_one_call.kwargs.get("filter", {})
        assert (
            query.get("agent_id") == "agent-cache-1"
        ), "agent_id not included in cache exact-match query"

    @pytest.mark.asyncio
    async def test_retrieve_passes_agent_id_to_hybrid_search(self, cache_memory):
        """When agent_id is provided, hybrid search filter must include agent_id."""
        await cache_memory.retrieve("test query", agent_id="agent-cache-2")

        call_args = cache_memory.search_engine.hybrid_search.call_args
        filter_query = call_args.kwargs.get("filter_query", {})
        assert filter_query.get("agent_id") == "agent-cache-2"


class TestSummaryMemoryIsolation:
    """SummaryMemory.retrieve() must accept and filter by agent_id."""

    @pytest.fixture
    def mock_collection(self):
        return AsyncMock()

    @pytest.fixture
    def summary_memory(self, mock_collection):
        with (
            patch("src.memory.summary.VectorSearchEngine"),
            patch("src.memory.summary.get_embedding_service") as mock_embed,
        ):
            mock_embed_instance = MagicMock()
            mock_embed_instance.generate_embedding = AsyncMock(
                return_value=MagicMock(embedding=[0.1] * 1024)
            )
            mock_embed.return_value = mock_embed_instance
            mem = __import__("src.memory.summary", fromlist=["SummaryMemory"]).SummaryMemory(
                mock_collection
            )
            mem.search_engine.hybrid_search = AsyncMock(return_value=[])
            mem.search_engine.search = AsyncMock(return_value=[])
            return mem

    def test_retrieve_signature_has_agent_id(self):
        """SummaryMemory.retrieve() must accept agent_id parameter."""
        from src.memory.summary import SummaryMemory

        sig = inspect.signature(SummaryMemory.retrieve)
        assert "agent_id" in sig.parameters

    @pytest.mark.asyncio
    async def test_retrieve_passes_agent_id_to_filter(self, summary_memory):
        """When agent_id is provided, it must appear in the filter_query."""
        await summary_memory.retrieve("test query", agent_id="agent-sum-1")

        call_args = summary_memory.search_engine.hybrid_search.call_args
        filter_query = call_args.kwargs.get("filter_query", {})
        assert filter_query.get("agent_id") == "agent-sum-1"


class TestManagerForwardsAgentId:
    """MemoryManager.retrieve_memories() must forward agent_id to all stores."""

    def test_retrieve_memories_signature_has_agent_id(self):
        """MemoryManager.retrieve_memories() must accept agent_id parameter."""
        from src.memory.manager import MemoryManager

        sig = inspect.signature(MemoryManager.retrieve_memories)
        assert (
            "agent_id" in sig.parameters
        ), "MemoryManager.retrieve_memories() missing agent_id parameter"
