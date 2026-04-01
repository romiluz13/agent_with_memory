"""
Tests for multi-tenant isolation bypass fixes (REM-FIX).
7 issues: 3 CRITICAL + 4 HIGH.

CRITICAL-1: API search endpoint missing agent_id
CRITICAL-2: Agent graph retrieve_memory_node missing agent_id
CRITICAL-3: Entity dedup query missing agent_id
HIGH-1: Semantic dedup query missing agent_id
HIGH-2: Consolidation missing agent_id
HIGH-3: Summary JIT expansion missing agent_id
HIGH-4: Cache writes use wrong agent_id
"""

import ast
import inspect
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# CRITICAL-1: SearchMemoryRequest must have agent_id field
# ---------------------------------------------------------------------------


class TestSearchEndpointIsolation:
    """The /search endpoint must require agent_id and forward it."""

    def _get_memories_source(self):
        """Read the memories.py source file."""
        fpath = os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "api", "routes", "memories.py"
        )
        with open(fpath) as f:
            return f.read()

    def test_search_memory_request_has_agent_id_field(self):
        """SearchMemoryRequest model must include agent_id field."""
        source = self._get_memories_source()
        tree = ast.parse(source)

        # Find SearchMemoryRequest class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "SearchMemoryRequest":
                # Check for agent_id assignment in class body
                field_names = []
                for item in node.body:
                    if isinstance(item, (ast.AnnAssign, ast.Assign)):
                        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                            field_names.append(item.target.id)
                assert "agent_id" in field_names, "SearchMemoryRequest missing agent_id field"
                return

        pytest.fail("SearchMemoryRequest class not found in memories.py")

    def test_search_endpoint_forwards_agent_id(self):
        """The /search endpoint must pass agent_id to retrieve_memories()."""
        source = self._get_memories_source()

        # The call to retrieve_memories must include agent_id=request.agent_id
        assert (
            "agent_id=request.agent_id" in source
        ), "search_memories() does not forward request.agent_id to retrieve_memories()"


# ---------------------------------------------------------------------------
# CRITICAL-2: retrieve_memory_node must pass agent_id
# ---------------------------------------------------------------------------


class TestRetrieveMemoryNodeIsolation:
    """retrieve_memory_node must pass agent_id to retrieve_memories()."""

    def _get_agent_source(self):
        """Read the agent.py source file."""
        fpath = os.path.join(os.path.dirname(__file__), "..", "..", "src", "core", "agent.py")
        with open(fpath) as f:
            return f.read()

    def test_retrieve_memory_node_passes_agent_id(self):
        """retrieve_memory_node must pass self.agent_id to retrieve_memories()."""
        source = self._get_agent_source()

        # Find the retrieve_memory_node method and verify it passes agent_id
        # The call to retrieve_memories must include agent_id=self.agent_id
        assert (
            "agent_id=self.agent_id" in source
        ), "retrieve_memory_node does not pass self.agent_id to retrieve_memories()"


# ---------------------------------------------------------------------------
# CRITICAL-3: Entity dedup _find_existing_entity must include agent_id
# ---------------------------------------------------------------------------


class TestEntityDedupIsolation:
    """_find_existing_entity must filter by agent_id."""

    @pytest.fixture
    def mock_collection(self):
        coll = AsyncMock()
        coll.find_one = AsyncMock(return_value=None)
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

    def test_find_existing_entity_accepts_agent_id(self):
        """_find_existing_entity must accept agent_id parameter."""
        from src.memory.entity import EntityMemory

        sig = inspect.signature(EntityMemory._find_existing_entity)
        assert "agent_id" in sig.parameters, "_find_existing_entity missing agent_id parameter"

    @pytest.mark.asyncio
    async def test_find_existing_entity_filters_by_agent_id(self, entity_memory, mock_collection):
        """When agent_id is provided, the dedup query must include it."""
        await entity_memory._find_existing_entity("John", "PERSON", agent_id="agent-ent-1")

        call_args = mock_collection.find_one.call_args
        query = call_args[0][0] if call_args[0] else call_args.kwargs.get("filter", {})
        assert (
            query.get("agent_id") == "agent-ent-1"
        ), "_find_existing_entity does not filter by agent_id"

    @pytest.mark.asyncio
    async def test_store_passes_agent_id_to_find_existing(self, entity_memory, mock_collection):
        """store() must pass the memory's agent_id to _find_existing_entity."""
        from src.memory.base import Memory, MemoryType

        memory = Memory(
            agent_id="agent-ent-2",
            memory_type=MemoryType.ENTITY,
            content="John (PERSON): A person",
            metadata={"entity_name": "John", "entity_type": "PERSON"},
            embedding=[0.1] * 1024,
        )

        mock_collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id="abc123"))

        await entity_memory.store(memory)

        # The find_one call from _find_existing_entity must include agent_id
        call_args = mock_collection.find_one.call_args
        query = call_args[0][0] if call_args[0] else call_args.kwargs.get("filter", {})
        assert (
            query.get("agent_id") == "agent-ent-2"
        ), "store() does not pass memory.agent_id to _find_existing_entity"


# ---------------------------------------------------------------------------
# HIGH-1: Semantic _find_similar_knowledge must pass agent_id
# ---------------------------------------------------------------------------


class TestSemanticDedupIsolation:
    """_find_similar_knowledge must pass agent_id to retrieve()."""

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
            from src.memory.semantic import SemanticMemory

            mem = SemanticMemory(mock_collection)
            mem.search_engine.hybrid_search = AsyncMock(return_value=[])
            mem.search_engine.search = AsyncMock(return_value=[])
            return mem

    def test_find_similar_knowledge_accepts_agent_id(self):
        """_find_similar_knowledge must accept agent_id parameter."""
        from src.memory.semantic import SemanticMemory

        sig = inspect.signature(SemanticMemory._find_similar_knowledge)
        assert "agent_id" in sig.parameters, "_find_similar_knowledge missing agent_id parameter"

    @pytest.mark.asyncio
    async def test_find_similar_knowledge_forwards_agent_id(self, semantic_memory):
        """_find_similar_knowledge must pass agent_id to retrieve()."""
        # Patch retrieve to capture arguments
        semantic_memory.retrieve = AsyncMock(return_value=[])
        await semantic_memory._find_similar_knowledge("some knowledge", agent_id="agent-sem-1")

        call_kwargs = semantic_memory.retrieve.call_args.kwargs
        assert (
            call_kwargs.get("agent_id") == "agent-sem-1"
        ), "_find_similar_knowledge does not forward agent_id to retrieve()"

    @pytest.mark.asyncio
    async def test_store_passes_agent_id_to_find_similar(self, semantic_memory, mock_collection):
        """store() must pass agent_id to _find_similar_knowledge."""
        from src.memory.base import Memory, MemoryType

        memory = Memory(
            agent_id="agent-sem-2",
            memory_type=MemoryType.SEMANTIC,
            content="Python is a language",
            embedding=[0.1] * 1024,
        )

        # Patch _find_similar_knowledge to capture args
        semantic_memory._find_similar_knowledge = AsyncMock(return_value=None)
        mock_collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id="abc123"))

        await semantic_memory.store(memory)

        call_kwargs = semantic_memory._find_similar_knowledge.call_args
        # Check positional or keyword agent_id
        all_args = call_kwargs.kwargs if call_kwargs.kwargs else {}
        positional = call_kwargs.args if call_kwargs.args else ()
        assert all_args.get("agent_id") == "agent-sem-2" or (
            len(positional) > 1 and positional[1] == "agent-sem-2"
        ), "store() does not pass memory.agent_id to _find_similar_knowledge"


# ---------------------------------------------------------------------------
# HIGH-2: Consolidation must pass agent_id
# ---------------------------------------------------------------------------


class TestConsolidationIsolation:
    """consolidate_memories must pass agent_id to store.consolidate()."""

    def test_semantic_consolidate_accepts_agent_id(self):
        """SemanticMemory.consolidate() must accept agent_id parameter."""
        from src.memory.semantic import SemanticMemory

        sig = inspect.signature(SemanticMemory.consolidate)
        assert (
            "agent_id" in sig.parameters
        ), "SemanticMemory.consolidate() missing agent_id parameter"

    @pytest.mark.asyncio
    async def test_manager_passes_agent_id_to_consolidate(self):
        """MemoryManager.consolidate_memories() must pass agent_id to store.consolidate()."""
        from src.memory.manager import MemoryManager

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=AsyncMock())

        with (
            patch("src.memory.manager.EpisodicMemory"),
            patch("src.memory.manager.ProceduralMemory"),
            patch("src.memory.manager.SemanticMemory") as MockSemantic,
            patch("src.memory.manager.WorkingMemory"),
            patch("src.memory.manager.SemanticCache"),
            patch("src.memory.manager.EntityMemory"),
            patch("src.memory.manager.SummaryMemory"),
            patch("src.memory.manager.get_embedding_service"),
            patch("src.memory.manager.MultiCollectionSearch"),
        ):

            mock_semantic_store = AsyncMock()
            mock_semantic_store.consolidate = AsyncMock(return_value=0)
            MockSemantic.return_value = mock_semantic_store

            manager = MemoryManager(mock_db)
            # Override the stores dict to use our mock
            manager.stores[
                __import__("src.memory.base", fromlist=["MemoryType"]).MemoryType.SEMANTIC
            ] = mock_semantic_store

            # Also mock episodic consolidate
            mock_episodic_store = AsyncMock()
            mock_episodic_store.consolidate = AsyncMock(return_value=0)
            manager.stores[
                __import__("src.memory.base", fromlist=["MemoryType"]).MemoryType.EPISODIC
            ] = mock_episodic_store

            await manager.consolidate_memories(agent_id="agent-cons-1")

            # Check semantic store consolidate was called with agent_id
            mock_semantic_store.consolidate.assert_called()
            call_kwargs = mock_semantic_store.consolidate.call_args.kwargs
            assert (
                call_kwargs.get("agent_id") == "agent-cons-1"
            ), "consolidate_memories does not forward agent_id to store.consolidate()"


# ---------------------------------------------------------------------------
# HIGH-3: Summary retrieve_by_summary_id must include agent_id filter
# ---------------------------------------------------------------------------


class TestSummaryJITIsolation:
    """retrieve_by_summary_id must support agent_id filtering."""

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

    def test_retrieve_by_summary_id_accepts_agent_id(self):
        """retrieve_by_summary_id must accept optional agent_id parameter."""
        from src.memory.summary import SummaryMemory

        sig = inspect.signature(SummaryMemory.retrieve_by_summary_id)
        assert "agent_id" in sig.parameters, "retrieve_by_summary_id missing agent_id parameter"

    @pytest.mark.asyncio
    async def test_retrieve_by_summary_id_filters_by_agent_id(
        self, summary_memory, mock_collection
    ):
        """When agent_id is provided, query must include it."""
        await summary_memory.retrieve_by_summary_id("sum-abc", agent_id="agent-sum-1")

        call_args = mock_collection.find_one.call_args
        query = call_args[0][0] if call_args[0] else call_args.kwargs.get("filter", {})
        assert (
            query.get("agent_id") == "agent-sum-1"
        ), "retrieve_by_summary_id does not filter by agent_id"


# ---------------------------------------------------------------------------
# HIGH-4: Cache writes must use actual agent_id, not "system"
# ---------------------------------------------------------------------------


class TestCacheWriteAgentId:
    """Cache entries in retrieve_memories must use actual agent_id."""

    @pytest.mark.asyncio
    async def test_cache_write_uses_actual_agent_id(self):
        """retrieve_memories must cache results with the requesting agent_id, not 'system'."""
        from src.memory.base import Memory, MemoryType
        from src.memory.manager import MemoryManager

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=AsyncMock())

        with (
            patch("src.memory.manager.EpisodicMemory"),
            patch("src.memory.manager.ProceduralMemory"),
            patch("src.memory.manager.SemanticMemory"),
            patch("src.memory.manager.WorkingMemory"),
            patch("src.memory.manager.SemanticCache"),
            patch("src.memory.manager.EntityMemory"),
            patch("src.memory.manager.SummaryMemory"),
            patch("src.memory.manager.get_embedding_service") as mock_embed,
            patch("src.memory.manager.MultiCollectionSearch"),
        ):

            mock_embed_instance = MagicMock()
            mock_embed_instance.generate_embedding = AsyncMock(
                return_value=MagicMock(embedding=[0.1] * 1024)
            )
            mock_embed.return_value = mock_embed_instance

            manager = MemoryManager(mock_db)

            # Mock cache retrieve to return empty (cache miss)
            manager.cache.retrieve = AsyncMock(return_value=[])

            # Mock store retrieve to return a result
            test_memory = Memory(
                agent_id="agent-x", memory_type=MemoryType.SEMANTIC, content="test content"
            )

            # Mock all stores' retrieve to return empty except semantic
            for _mem_type, store in manager.stores.items():
                store.retrieve = AsyncMock(return_value=[])
            manager.stores[MemoryType.SEMANTIC].retrieve = AsyncMock(return_value=[test_memory])

            # Spy on store_memory to capture arguments
            manager.store_memory = AsyncMock(return_value="cached-id")

            await manager.retrieve_memories(query="test", agent_id="agent-actual-1", use_cache=True)

            # store_memory should have been called with agent_id="agent-actual-1" not "system"
            if manager.store_memory.called:
                call_kwargs = manager.store_memory.call_args.kwargs
                actual_agent_id = call_kwargs.get("agent_id")
                assert actual_agent_id == "agent-actual-1", (
                    f"Cache write uses agent_id='{actual_agent_id}' instead of "
                    f"'agent-actual-1'. Must use requesting agent's ID."
                )
