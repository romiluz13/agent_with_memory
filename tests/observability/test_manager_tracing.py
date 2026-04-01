"""Tests for tracer integration with MemoryManager."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.base import MemoryType


class TestMemoryManagerTracing:
    """Verify that MemoryManager calls tracer on store and retrieve."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock MongoDB database."""
        db = MagicMock()
        db.__getitem__ = MagicMock(return_value=AsyncMock())
        return db

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        service = MagicMock()
        result = MagicMock()
        result.embedding = [0.1] * 1024
        service.generate_embedding = AsyncMock(return_value=result)
        return service

    @patch("src.memory.manager.get_tracer")
    @patch("src.memory.manager.get_embedding_service")
    @patch("src.memory.episodic.get_embedding_service")
    @patch("src.memory.semantic.get_embedding_service")
    @patch("src.memory.procedural.get_embedding_service")
    @patch("src.memory.working.get_embedding_service")
    @patch("src.memory.cache.get_embedding_service")
    @patch("src.memory.entity.get_embedding_service")
    @patch("src.memory.summary.get_embedding_service")
    async def test_store_memory_traces_operation(
        self,
        mock_summary_embed,
        mock_entity_embed,
        mock_cache_embed,
        mock_working_embed,
        mock_procedural_embed,
        mock_semantic_embed,
        mock_episodic_embed,
        mock_manager_embed,
        mock_get_tracer,
        mock_db,
        mock_embedding_service,
    ):
        """store_memory() should call trace_memory_operation after successful store."""
        from src.memory.manager import MemoryManager

        # All embedding service mocks return the same service
        for m in [
            mock_summary_embed,
            mock_entity_embed,
            mock_cache_embed,
            mock_working_embed,
            mock_procedural_embed,
            mock_semantic_embed,
            mock_episodic_embed,
            mock_manager_embed,
        ]:
            m.return_value = mock_embedding_service

        # Setup tracer mock
        mock_tracer = MagicMock()
        mock_tracer.trace_memory_operation = MagicMock()
        mock_get_tracer.return_value = mock_tracer

        manager = MemoryManager(mock_db)

        # Mock the store method on the episodic store
        store_mock = AsyncMock(return_value="memory_123")
        manager.stores[MemoryType.EPISODIC].store = store_mock

        # Act
        memory_id = await manager.store_memory(
            content="test memory",
            memory_type=MemoryType.EPISODIC,
            agent_id="agent_1",
        )

        # Assert
        assert memory_id == "memory_123"
        mock_tracer.trace_memory_operation.assert_called_once_with(
            "store",
            "episodic",
            "agent_1",
            metadata={"importance": 0.5, "memory_id": "memory_123"},
        )

    @patch("src.memory.manager.get_tracer")
    @patch("src.memory.manager.get_embedding_service")
    @patch("src.memory.episodic.get_embedding_service")
    @patch("src.memory.semantic.get_embedding_service")
    @patch("src.memory.procedural.get_embedding_service")
    @patch("src.memory.working.get_embedding_service")
    @patch("src.memory.cache.get_embedding_service")
    @patch("src.memory.entity.get_embedding_service")
    @patch("src.memory.summary.get_embedding_service")
    async def test_retrieve_memories_traces_operation(
        self,
        mock_summary_embed,
        mock_entity_embed,
        mock_cache_embed,
        mock_working_embed,
        mock_procedural_embed,
        mock_semantic_embed,
        mock_episodic_embed,
        mock_manager_embed,
        mock_get_tracer,
        mock_db,
        mock_embedding_service,
    ):
        """retrieve_memories() should call trace_memory_operation with result count."""
        from src.memory.manager import MemoryManager

        # All embedding service mocks return the same service
        for m in [
            mock_summary_embed,
            mock_entity_embed,
            mock_cache_embed,
            mock_working_embed,
            mock_procedural_embed,
            mock_semantic_embed,
            mock_episodic_embed,
            mock_manager_embed,
        ]:
            m.return_value = mock_embedding_service

        # Setup tracer mock
        mock_tracer = MagicMock()
        mock_tracer.trace_memory_operation = MagicMock()
        mock_get_tracer.return_value = mock_tracer

        manager = MemoryManager(mock_db)

        # Mock retrieve to return empty
        manager.cache.retrieve = AsyncMock(return_value=[])
        for store in manager.stores.values():
            store.retrieve = AsyncMock(return_value=[])
        manager.store_memory = AsyncMock()

        # Act
        results = await manager.retrieve_memories(
            query="test query",
            agent_id="agent_1",
        )

        # Assert
        assert results == []
        mock_tracer.trace_memory_operation.assert_called_once_with(
            "retrieve",
            "multi",
            "agent_1",
            metadata={"query_prefix": "test query", "results": 0},
        )
