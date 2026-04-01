"""
Tests for REM-FIX #2: Remaining multi-tenant isolation gaps.
6 issues: 2 CRITICAL + 4 HIGH.

CRITICAL-1: semantic.py consolidate() ignores agent_id in list_memories call
CRITICAL-2: cache.py store() dedup lookup missing agent_id
HIGH-1: summary.py expand_summary() missing agent_id parameter
HIGH-2: entity.py get_entities_by_type() missing agent_id parameter
HIGH-3: summary.py get_summaries_for_thread() missing agent_id parameter
HIGH-4: cache.py _evict_if_needed() global eviction (not scoped by agent_id)

Each test verifies the TERMINAL behavior: the actual MongoDB query/filter dict
contains agent_id when provided.
"""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# CRITICAL-1: semantic.py consolidate() must pass agent_id to list_memories
# ---------------------------------------------------------------------------


class TestSemanticConsolidateAgentIdFilter:
    """consolidate() must scope list_memories by agent_id."""

    @pytest.fixture
    def mock_collection(self):
        coll = AsyncMock()
        # find() returns async iterator with no documents
        mock_cursor = AsyncMock()
        mock_cursor.__aiter__ = AsyncMock(return_value=iter([]))
        mock_cursor.skip = MagicMock(return_value=mock_cursor)
        mock_cursor.limit = MagicMock(return_value=mock_cursor)
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        coll.find = MagicMock(return_value=mock_cursor)
        return coll

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
            from src.memory.semantic import SemanticMemory

            return SemanticMemory(mock_collection)

    @pytest.mark.asyncio
    async def test_consolidate_filters_by_agent_id(self, semantic_memory, mock_collection):
        """When agent_id is provided, consolidate() must pass it as a filter
        to list_memories, resulting in the collection.find() query containing
        agent_id."""
        await semantic_memory.consolidate(agent_id="agent-cons-A")

        # list_memories calls collection.find(query)
        mock_collection.find.assert_called()
        query = mock_collection.find.call_args[0][0]
        assert query.get("agent_id") == "agent-cons-A", (
            f"consolidate() did not pass agent_id to list_memories filter. "
            f"Actual query: {query}"
        )

    @pytest.mark.asyncio
    async def test_consolidate_without_agent_id_has_no_agent_filter(
        self, semantic_memory, mock_collection
    ):
        """When agent_id is None, consolidate() must NOT add agent_id to the filter."""
        await semantic_memory.consolidate(agent_id=None)

        mock_collection.find.assert_called()
        query = mock_collection.find.call_args[0][0]
        assert "agent_id" not in query, (
            f"consolidate(agent_id=None) should not add agent_id to query. "
            f"Actual query: {query}"
        )


# ---------------------------------------------------------------------------
# CRITICAL-2: cache.py store() dedup lookup must include agent_id
# ---------------------------------------------------------------------------


class TestCacheStoreDedupAgentId:
    """cache.store() dedup lookup must include memory.agent_id."""

    @pytest.fixture
    def mock_collection(self):
        coll = AsyncMock()
        coll.find_one = AsyncMock(return_value=None)
        coll.count_documents = AsyncMock(return_value=0)
        coll.insert_one = AsyncMock(return_value=MagicMock(inserted_id="new-id"))
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
            from src.memory.cache import SemanticCache

            return SemanticCache(mock_collection)

    @pytest.mark.asyncio
    async def test_store_dedup_includes_agent_id(self, cache_memory, mock_collection):
        """When storing a cache entry with agent_id, the dedup find_one must include
        agent_id in its query to prevent cross-tenant dedup collisions."""
        from src.memory.base import Memory, MemoryType

        memory = Memory(
            agent_id="agent-cache-1",
            memory_type=MemoryType.CACHE,
            content="What is Python?",
            metadata={"response": "A programming language"},
            embedding=[0.1] * 1024,
        )

        await cache_memory.store(memory)

        # The first find_one call is the dedup check
        first_call = mock_collection.find_one.call_args_list[0]
        dedup_query = first_call[0][0] if first_call[0] else first_call.kwargs.get("filter", {})
        assert (
            dedup_query.get("agent_id") == "agent-cache-1"
        ), f"store() dedup lookup missing agent_id. Actual query: {dedup_query}"

    @pytest.mark.asyncio
    async def test_store_dedup_without_agent_id_omits_filter(self, cache_memory, mock_collection):
        """When agent_id is empty string (falsy), the dedup query should not
        include agent_id filter. Note: Memory.agent_id is required str, so we
        use empty string to test the falsy branch."""
        from src.memory.base import Memory, MemoryType

        memory = Memory(
            agent_id="",
            memory_type=MemoryType.CACHE,
            content="What is Python?",
            metadata={"response": "A programming language"},
            embedding=[0.1] * 1024,
        )

        await cache_memory.store(memory)

        first_call = mock_collection.find_one.call_args_list[0]
        dedup_query = first_call[0][0] if first_call[0] else first_call.kwargs.get("filter", {})
        assert "agent_id" not in dedup_query, (
            f"store() dedup with empty agent_id should not include agent_id filter. "
            f"Actual query: {dedup_query}"
        )


# ---------------------------------------------------------------------------
# HIGH-1: summary.py expand_summary() must accept and forward agent_id
# ---------------------------------------------------------------------------


class TestSummaryExpandAgentId:
    """expand_summary() must accept agent_id and forward to retrieve_by_summary_id."""

    @pytest.fixture
    def mock_collection(self):
        coll = AsyncMock()
        coll.find_one = AsyncMock(return_value=None)
        return coll

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
            from src.memory.summary import SummaryMemory

            return SummaryMemory(mock_collection)

    def test_expand_summary_accepts_agent_id_param(self):
        """expand_summary() must accept agent_id parameter."""
        from src.memory.summary import SummaryMemory

        sig = inspect.signature(SummaryMemory.expand_summary)
        assert "agent_id" in sig.parameters, "expand_summary() missing agent_id parameter"

    @pytest.mark.asyncio
    async def test_expand_summary_forwards_agent_id_to_query(self, summary_memory, mock_collection):
        """expand_summary(agent_id=X) must result in find_one query containing agent_id=X."""
        await summary_memory.expand_summary("sum-123", agent_id="agent-exp-1")

        call_args = mock_collection.find_one.call_args
        query = call_args[0][0] if call_args[0] else call_args.kwargs.get("filter", {})
        assert query.get("agent_id") == "agent-exp-1", (
            f"expand_summary() did not forward agent_id to MongoDB query. " f"Actual query: {query}"
        )


# ---------------------------------------------------------------------------
# HIGH-2: entity.py get_entities_by_type() must accept and filter by agent_id
# ---------------------------------------------------------------------------


class TestEntityGetByTypeAgentId:
    """get_entities_by_type() must accept agent_id and add it to filters."""

    @pytest.fixture
    def mock_collection(self):
        coll = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.__aiter__ = AsyncMock(return_value=iter([]))
        mock_cursor.skip = MagicMock(return_value=mock_cursor)
        mock_cursor.limit = MagicMock(return_value=mock_cursor)
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        coll.find = MagicMock(return_value=mock_cursor)
        return coll

    @pytest.fixture
    def entity_memory(self, mock_collection):
        with (
            patch("src.memory.entity.VectorSearchEngine"),
            patch("src.memory.entity.get_embedding_service") as mock_embed,
        ):
            mock_embed_instance = MagicMock()
            mock_embed_instance.generate_embedding = AsyncMock(
                return_value=MagicMock(embedding=[0.1] * 1024)
            )
            mock_embed.return_value = mock_embed_instance
            from src.memory.entity import EntityMemory

            return EntityMemory(mock_collection)

    def test_get_entities_by_type_accepts_agent_id(self):
        """get_entities_by_type() must accept agent_id parameter."""
        from src.memory.entity import EntityMemory

        sig = inspect.signature(EntityMemory.get_entities_by_type)
        assert "agent_id" in sig.parameters, "get_entities_by_type() missing agent_id parameter"

    @pytest.mark.asyncio
    async def test_get_entities_by_type_filters_by_agent_id(self, entity_memory, mock_collection):
        """When agent_id is provided, the collection.find() query must include it."""
        await entity_memory.get_entities_by_type("PERSON", agent_id="agent-ent-type-1")

        mock_collection.find.assert_called()
        query = mock_collection.find.call_args[0][0]
        assert query.get("agent_id") == "agent-ent-type-1", (
            f"get_entities_by_type() did not add agent_id to query. " f"Actual query: {query}"
        )

    @pytest.mark.asyncio
    async def test_get_entities_by_type_without_agent_id(self, entity_memory, mock_collection):
        """When agent_id is None, the query must not include agent_id."""
        await entity_memory.get_entities_by_type("PERSON")

        mock_collection.find.assert_called()
        query = mock_collection.find.call_args[0][0]
        assert "agent_id" not in query, (
            f"get_entities_by_type(agent_id=None) should not include agent_id. "
            f"Actual query: {query}"
        )


# ---------------------------------------------------------------------------
# HIGH-3: summary.py get_summaries_for_thread() must accept and filter by agent_id
# ---------------------------------------------------------------------------


class TestSummaryGetForThreadAgentId:
    """get_summaries_for_thread() must accept agent_id and pass it in filters."""

    @pytest.fixture
    def mock_collection(self):
        coll = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.__aiter__ = AsyncMock(return_value=iter([]))
        mock_cursor.skip = MagicMock(return_value=mock_cursor)
        mock_cursor.limit = MagicMock(return_value=mock_cursor)
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        coll.find = MagicMock(return_value=mock_cursor)
        return coll

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
            from src.memory.summary import SummaryMemory

            return SummaryMemory(mock_collection)

    def test_get_summaries_for_thread_accepts_agent_id(self):
        """get_summaries_for_thread() must accept agent_id parameter."""
        from src.memory.summary import SummaryMemory

        sig = inspect.signature(SummaryMemory.get_summaries_for_thread)
        assert "agent_id" in sig.parameters, "get_summaries_for_thread() missing agent_id parameter"

    @pytest.mark.asyncio
    async def test_get_summaries_for_thread_filters_by_agent_id(
        self, summary_memory, mock_collection
    ):
        """When agent_id is provided, collection.find() query must include it."""
        await summary_memory.get_summaries_for_thread("thread-1", agent_id="agent-thread-1")

        mock_collection.find.assert_called()
        query = mock_collection.find.call_args[0][0]
        assert query.get("agent_id") == "agent-thread-1", (
            f"get_summaries_for_thread() did not add agent_id to query. " f"Actual query: {query}"
        )

    @pytest.mark.asyncio
    async def test_get_summaries_for_thread_without_agent_id(self, summary_memory, mock_collection):
        """When agent_id is None, the query must not include agent_id."""
        await summary_memory.get_summaries_for_thread("thread-1")

        mock_collection.find.assert_called()
        query = mock_collection.find.call_args[0][0]
        assert "agent_id" not in query, (
            f"get_summaries_for_thread(agent_id=None) should not include agent_id. "
            f"Actual query: {query}"
        )


# ---------------------------------------------------------------------------
# HIGH-4: cache.py _evict_if_needed() must scope eviction by agent_id
# ---------------------------------------------------------------------------


class TestCacheEvictionAgentIdScope:
    """_evict_if_needed() must scope count and LRU by agent_id when provided."""

    @pytest.fixture
    def mock_collection(self):
        coll = AsyncMock()
        coll.count_documents = AsyncMock(return_value=0)
        coll.delete_many = AsyncMock(return_value=MagicMock(deleted_count=0))
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
            from src.memory.cache import SemanticCache

            return SemanticCache(mock_collection)

    def test_evict_if_needed_accepts_agent_id(self):
        """_evict_if_needed must accept agent_id parameter."""
        from src.memory.cache import SemanticCache

        sig = inspect.signature(SemanticCache._evict_if_needed)
        assert "agent_id" in sig.parameters, "_evict_if_needed() missing agent_id parameter"

    @pytest.mark.asyncio
    async def test_evict_count_scoped_by_agent_id(self, cache_memory, mock_collection):
        """When agent_id is provided, count_documents must include agent_id in query."""
        # Set count high enough to trigger eviction check
        mock_collection.count_documents = AsyncMock(return_value=1001)
        mock_collection.delete_many = AsyncMock(return_value=MagicMock(deleted_count=500))

        # After expired delete, still over limit
        mock_collection.count_documents = AsyncMock(side_effect=[1001, 600])

        # Mock find for LRU
        mock_cursor = AsyncMock()
        mock_cursor.__aiter__ = AsyncMock(return_value=iter([]))
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.limit = MagicMock(return_value=mock_cursor)
        mock_collection.find = MagicMock(return_value=mock_cursor)

        await cache_memory._evict_if_needed(agent_id="agent-evict-1")

        # The first count_documents call should include agent_id
        first_count_call = mock_collection.count_documents.call_args_list[0]
        count_query = first_count_call[0][0]
        assert count_query.get("agent_id") == "agent-evict-1", (
            f"_evict_if_needed() count_documents not scoped by agent_id. "
            f"Actual query: {count_query}"
        )

    @pytest.mark.asyncio
    async def test_evict_without_agent_id_is_global(self, cache_memory, mock_collection):
        """When agent_id is None, eviction should operate globally (no agent_id filter)."""
        mock_collection.count_documents = AsyncMock(return_value=0)

        await cache_memory._evict_if_needed(agent_id=None)

        first_count_call = mock_collection.count_documents.call_args_list[0]
        count_query = first_count_call[0][0]
        assert "agent_id" not in count_query, (
            f"_evict_if_needed(agent_id=None) should not filter by agent_id. "
            f"Actual query: {count_query}"
        )


# ---------------------------------------------------------------------------
# HIGH-1 (extended): summary_tools.py must forward agent_id to expand_summary
# ---------------------------------------------------------------------------


class TestSummaryToolsAgentIdForwarding:
    """create_expand_summary_tool must support agent_id forwarding."""

    @pytest.mark.asyncio
    async def test_expand_tool_calls_expand_summary_with_agent_id(self):
        """The expand_summary tool should be able to forward agent_id
        to memory_manager.summary.expand_summary()."""
        from src.tools.summary_tools import create_expand_summary_tool

        mock_summary = AsyncMock()
        mock_summary.expand_summary = AsyncMock(return_value="Full content here")

        mock_manager = MagicMock()
        mock_manager.summary = mock_summary

        tool_fn = create_expand_summary_tool(mock_manager, agent_id="agent-tool-1")

        # Invoke the tool
        await tool_fn.ainvoke({"summary_id": "sum-abc"})

        # Verify agent_id was forwarded
        mock_summary.expand_summary.assert_called_once()
        call_kwargs = mock_summary.expand_summary.call_args
        # Check that agent_id was passed
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        all_args = call_kwargs.args if call_kwargs.args else ()
        assert all_kwargs.get("agent_id") == "agent-tool-1" or (
            len(all_args) > 1 and all_args[1] == "agent-tool-1"
        ), (
            f"expand_summary tool did not forward agent_id to "
            f"memory_manager.summary.expand_summary(). "
            f"Args: {all_args}, Kwargs: {all_kwargs}"
        )
