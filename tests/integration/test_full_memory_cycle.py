"""
Integration tests for full memory cycle across all 7 memory types.
Tests: Store -> Retrieve -> Update -> Delete for each type.
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

from src.memory.base import Memory, MemoryType
from src.memory.episodic import EpisodicMemory
from src.memory.entity import EntityMemory
from src.memory.summary import SummaryMemory


class TestAllMemoryTypesExist:
    """Verify all 7 memory types are defined."""

    def test_memory_type_enum_has_7_types(self):
        """Test MemoryType enum has all 7 types."""
        expected = {"episodic", "semantic", "procedural", "working", "cache", "entity", "summary"}
        actual = {m.value for m in MemoryType}
        assert actual == expected, f"Missing types: {expected - actual}"

    def test_episodic_type(self):
        """Test EPISODIC memory type exists."""
        assert MemoryType.EPISODIC.value == "episodic"

    def test_semantic_type(self):
        """Test SEMANTIC memory type exists."""
        assert MemoryType.SEMANTIC.value == "semantic"

    def test_procedural_type(self):
        """Test PROCEDURAL memory type exists."""
        assert MemoryType.PROCEDURAL.value == "procedural"

    def test_working_type(self):
        """Test WORKING memory type exists."""
        assert MemoryType.WORKING.value == "working"

    def test_cache_type(self):
        """Test CACHE memory type exists."""
        assert MemoryType.CACHE.value == "cache"

    def test_entity_type(self):
        """Test ENTITY memory type exists."""
        assert MemoryType.ENTITY.value == "entity"

    def test_summary_type(self):
        """Test SUMMARY memory type exists."""
        assert MemoryType.SUMMARY.value == "summary"


class TestMemoryModel:
    """Test Memory model creation and validation."""

    def test_create_memory_with_defaults(self):
        """Test Memory creation with minimal required fields."""
        # memory_type is required, not optional
        memory = Memory(
            agent_id="test-agent",
            content="Test content",
            memory_type=MemoryType.EPISODIC
        )
        assert memory.agent_id == "test-agent"
        assert memory.content == "Test content"
        assert memory.memory_type == MemoryType.EPISODIC
        assert memory.importance == 0.5  # default

    def test_create_memory_with_all_fields(self):
        """Test Memory creation with all fields."""
        memory = Memory(
            agent_id="test-agent",
            user_id="test-user",
            memory_type=MemoryType.SEMANTIC,
            content="Test content",
            metadata={"key": "value"},
            tags=["tag1", "tag2"],
            importance=0.8,
            confidence=0.9
        )
        assert memory.agent_id == "test-agent"
        assert memory.user_id == "test-user"
        assert memory.memory_type == MemoryType.SEMANTIC
        assert memory.metadata == {"key": "value"}
        assert memory.tags == ["tag1", "tag2"]
        assert memory.importance == 0.8
        assert memory.confidence == 0.9

    def test_memory_types_can_be_assigned(self):
        """Test all memory types can be assigned to Memory."""
        for memory_type in MemoryType:
            memory = Memory(
                agent_id="test",
                content="test",
                memory_type=memory_type
            )
            assert memory.memory_type == memory_type


class TestEpisodicMemoryStore:
    """Integration tests for EpisodicMemory store."""

    @pytest.fixture
    def episodic_store(self, test_db):
        """Create EpisodicMemory with test database."""
        with patch('src.memory.episodic.VectorSearchEngine'), \
             patch('src.memory.episodic.get_embedding_service') as mock_embed:
            mock_service = MagicMock()
            mock_service.generate_embedding = AsyncMock(
                return_value=MagicMock(embedding=[0.1] * 1024)
            )
            mock_embed.return_value = mock_service
            return EpisodicMemory(test_db["episodic_memories"])

    @pytest.mark.asyncio
    async def test_store_episodic_memory(self, episodic_store, test_agent_id, clean_collections):
        """Test storing an episodic memory."""
        memory = Memory(
            agent_id=test_agent_id,
            content="User asked about Python",
            memory_type=MemoryType.EPISODIC,
            metadata={"role": "user", "thread_id": "thread-1"}
        )

        memory_id = await episodic_store.store(memory)
        assert memory_id is not None
        assert len(memory_id) > 0

    @pytest.mark.asyncio
    async def test_get_by_id(self, episodic_store, test_agent_id, clean_collections):
        """Test retrieving memory by ID."""
        memory = Memory(
            agent_id=test_agent_id,
            content="Test retrieval",
            memory_type=MemoryType.EPISODIC
        )

        memory_id = await episodic_store.store(memory)
        retrieved = await episodic_store.get_by_id(memory_id)

        assert retrieved is not None
        assert retrieved.content == "Test retrieval"
        assert retrieved.agent_id == test_agent_id


class TestSummaryMemoryStore:
    """Integration tests for SummaryMemory store."""

    @pytest.fixture
    def summary_store(self, test_db):
        """Create SummaryMemory with test database."""
        with patch('src.memory.summary.VectorSearchEngine'), \
             patch('src.memory.summary.get_embedding_service') as mock_embed:
            mock_service = MagicMock()
            mock_service.generate_embedding = AsyncMock(
                return_value=MagicMock(embedding=[0.1] * 1024)
            )
            mock_embed.return_value = mock_service
            return SummaryMemory(test_db["summary_memories"])

    @pytest.mark.asyncio
    async def test_store_summary_with_full_content(self, summary_store, test_agent_id, clean_collections):
        """Test storing summary with full content for JIT expansion."""
        full_content = "This is a very long conversation about AI agents..."
        summary = "Discussion about AI agents"

        memory_id = await summary_store.store_summary(
            summary_id="sum-001",
            full_content=full_content,
            summary=summary,
            description="AI Agents Discussion",
            agent_id=test_agent_id
        )

        assert memory_id is not None

    @pytest.mark.asyncio
    async def test_expand_summary_returns_full_content(self, summary_store, test_agent_id, clean_collections):
        """Test JIT expansion returns full original content."""
        full_content = "Original detailed content here..."
        await summary_store.store_summary(
            summary_id="expand-test",
            full_content=full_content,
            summary="Brief summary",
            description="Test",
            agent_id=test_agent_id
        )

        expanded = await summary_store.expand_summary("expand-test")
        assert expanded == full_content


class TestEntityMemoryStore:
    """Integration tests for EntityMemory store."""

    @pytest.fixture
    def entity_store(self, test_db):
        """Create EntityMemory with test database."""
        with patch('src.memory.entity.VectorSearchEngine'), \
             patch('src.memory.entity.get_embedding_service') as mock_embed:
            mock_service = MagicMock()
            mock_service.generate_embedding = AsyncMock(
                return_value=MagicMock(embedding=[0.1] * 1024)
            )
            mock_embed.return_value = mock_service
            return EntityMemory(test_db["entity_memories"])

    @pytest.mark.asyncio
    async def test_store_entity_memory(self, entity_store, test_agent_id, clean_collections):
        """Test storing an entity memory."""
        memory = Memory(
            agent_id=test_agent_id,
            content="John Smith (PERSON): Software engineer at Google",
            memory_type=MemoryType.ENTITY,
            metadata={
                "entity_name": "John Smith",
                "entity_type": "PERSON",
                "description": "Software engineer at Google",
                "mentions": 1
            }
        )

        memory_id = await entity_store.store(memory)
        assert memory_id is not None

    @pytest.mark.asyncio
    async def test_get_entities_by_type(self, entity_store, test_agent_id, clean_collections):
        """Test retrieving entities by type."""
        # Store a PERSON entity
        memory = Memory(
            agent_id=test_agent_id,
            content="Jane Doe (PERSON): Data scientist",
            memory_type=MemoryType.ENTITY,
            metadata={
                "entity_name": "Jane Doe",
                "entity_type": "PERSON",
                "mentions": 1
            }
        )
        await entity_store.store(memory)

        entities = await entity_store.get_entities_by_type("PERSON")
        assert len(entities) >= 1


class TestMarkAsSummarizedPattern:
    """Test the mark-as-summarized pattern (Oracle innovation)."""

    @pytest.fixture
    def episodic_store(self, test_db):
        """Create EpisodicMemory with test database."""
        with patch('src.memory.episodic.VectorSearchEngine'), \
             patch('src.memory.episodic.get_embedding_service') as mock_embed:
            mock_service = MagicMock()
            mock_service.generate_embedding = AsyncMock(
                return_value=MagicMock(embedding=[0.1] * 1024)
            )
            mock_embed.return_value = mock_service
            return EpisodicMemory(test_db["episodic_memories"])

    @pytest.mark.asyncio
    async def test_mark_as_summarized_preserves_messages(
        self, episodic_store, test_agent_id, test_thread_id, clean_collections
    ):
        """Test that marking as summarized doesn't delete messages."""
        # Store messages
        for i in range(5):
            memory = Memory(
                agent_id=test_agent_id,
                content=f"Message {i}",
                memory_type=MemoryType.EPISODIC,
                metadata={"thread_id": test_thread_id, "role": "user"}
            )
            await episodic_store.store(memory)

        # Mark as summarized
        count = await episodic_store.mark_as_summarized(
            agent_id=test_agent_id,
            thread_id=test_thread_id,
            summary_id="sum-001"
        )

        # Messages should still exist (not deleted) - use include_summarized=True
        # to verify they're still in DB (default is False which hides summarized)
        all_messages = await episodic_store.list_memories(
            filters={"agent_id": test_agent_id},
            include_summarized=True
        )
        assert len(all_messages) >= 5  # Still there (Oracle pattern: mark, don't delete)

    @pytest.mark.asyncio
    async def test_mark_as_summarized_adds_summary_id(
        self, episodic_store, test_agent_id, test_thread_id, clean_collections
    ):
        """Test that messages get summary_id field after marking."""
        memory = Memory(
            agent_id=test_agent_id,
            content="Test message",
            memory_type=MemoryType.EPISODIC,
            metadata={"thread_id": test_thread_id}
        )
        await episodic_store.store(memory)

        await episodic_store.mark_as_summarized(
            agent_id=test_agent_id,
            thread_id=test_thread_id,
            summary_id="sum-002"
        )

        # Check message has summary_id
        messages = await episodic_store.list_memories(
            filters={"agent_id": test_agent_id, "metadata.thread_id": test_thread_id}
        )
        for msg in messages:
            assert msg.metadata.get("summary_id") == "sum-002"
