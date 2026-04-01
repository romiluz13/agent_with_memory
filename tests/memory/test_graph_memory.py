"""
Tests for GraphMemory - Knowledge graph with $graphLookup traversal.
Phase 4: GraphRAG entity relationships.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.graph import GraphMemory, Relationship


@pytest.fixture
def mock_db():
    """Mock Motor database."""
    db = MagicMock()
    collection = AsyncMock()
    db.__getitem__ = MagicMock(return_value=collection)
    return db


@pytest.fixture
def mock_entity_memory():
    """Mock EntityMemory."""
    return AsyncMock()


@pytest.fixture
def graph_memory(mock_db, mock_entity_memory):
    """Create GraphMemory with mocked dependencies."""
    with patch("src.memory.graph.VectorSearchEngine"):
        return GraphMemory(db=mock_db, entity_memory=mock_entity_memory)


class TestAddRelationship:
    """Tests for adding relationships between entities."""

    @pytest.mark.asyncio
    async def test_add_relationship_success(self, graph_memory):
        """Add relationship between two existing entities."""
        # Source entity exists
        graph_memory.collection.find_one = AsyncMock(
            return_value={
                "_id": "source_id",
                "metadata": {"entity_name": "John", "entity_type": "PERSON"},
                "agent_id": "agent_1",
            }
        )
        graph_memory.collection.update_one = AsyncMock(
            return_value=MagicMock(modified_count=1)
        )

        rel = Relationship(
            source_entity="John",
            target_entity="Acme Corp",
            relationship_type="WORKS_AT",
            agent_id="agent_1",
        )
        result = await graph_memory.add_relationship(rel)

        assert result is True
        # Verify agent_id scoping in query
        find_call = graph_memory.collection.find_one.call_args[0][0]
        assert find_call["agent_id"] == "agent_1"
        assert find_call["metadata.entity_name"] == "John"

    @pytest.mark.asyncio
    async def test_add_relationship_missing_source_entity(self, graph_memory):
        """Return False when source entity not found."""
        graph_memory.collection.find_one = AsyncMock(return_value=None)

        rel = Relationship(
            source_entity="NonExistent",
            target_entity="Acme Corp",
            relationship_type="WORKS_AT",
            agent_id="agent_1",
        )
        result = await graph_memory.add_relationship(rel)

        assert result is False

    @pytest.mark.asyncio
    async def test_add_relationship_uses_addToSet(self, graph_memory):
        """Verify $addToSet is used to prevent duplicate relationships."""
        graph_memory.collection.find_one = AsyncMock(
            return_value={"_id": "source_id", "metadata": {"entity_name": "John"}, "agent_id": "a1"}
        )
        graph_memory.collection.update_one = AsyncMock(
            return_value=MagicMock(modified_count=1)
        )

        rel = Relationship(
            source_entity="John",
            target_entity="Acme",
            relationship_type="WORKS_AT",
            agent_id="a1",
        )
        await graph_memory.add_relationship(rel)

        update_call = graph_memory.collection.update_one.call_args
        update_doc = update_call[0][1]
        assert "$addToSet" in update_doc
        assert "metadata.relationships" in update_doc["$addToSet"]


class TestGraphLookup:
    """Tests for $graphLookup traversal."""

    @pytest.mark.asyncio
    async def test_graph_lookup_pipeline_structure(self, graph_memory):
        """Verify $graphLookup aggregation pipeline is correctly structured."""
        # Mock aggregate to return empty results but capture pipeline
        captured_pipeline = None

        def mock_aggregate(pipeline):
            nonlocal captured_pipeline
            captured_pipeline = pipeline
            return AsyncIteratorMock([])

        graph_memory.collection.aggregate = mock_aggregate

        await graph_memory.graph_lookup(
            start_entity="John", agent_id="agent_1", max_depth=2
        )

        assert captured_pipeline is not None

        # Check $match stage
        match_stage = captured_pipeline[0]
        assert "$match" in match_stage
        assert match_stage["$match"]["metadata.entity_name"] == "John"
        assert match_stage["$match"]["agent_id"] == "agent_1"

        # Check $graphLookup stage
        graph_lookup_stage = captured_pipeline[1]
        assert "$graphLookup" in graph_lookup_stage
        gl = graph_lookup_stage["$graphLookup"]
        assert gl["from"] == "entity_memories"
        assert gl["startWith"] == "$metadata.relationships.target"
        assert gl["connectFromField"] == "metadata.relationships.target"
        assert gl["connectToField"] == "metadata.entity_name"
        assert gl["as"] == "connected_entities"
        assert gl["maxDepth"] == 2
        assert gl["depthField"] == "depth"

    @pytest.mark.asyncio
    async def test_graph_lookup_agent_isolation(self, graph_memory):
        """Verify agent_id is in restrictSearchWithMatch."""
        captured_pipeline = None

        def mock_aggregate(pipeline):
            nonlocal captured_pipeline
            captured_pipeline = pipeline
            return AsyncIteratorMock([])

        graph_memory.collection.aggregate = mock_aggregate

        await graph_memory.graph_lookup(
            start_entity="John", agent_id="agent_42", max_depth=1
        )

        gl = captured_pipeline[1]["$graphLookup"]
        assert "restrictSearchWithMatch" in gl
        assert gl["restrictSearchWithMatch"]["agent_id"] == "agent_42"

    @pytest.mark.asyncio
    async def test_graph_lookup_returns_connected_entities(self, graph_memory):
        """Verify results include connected entity data."""
        mock_result = {
            "entity_name": "John",
            "entity_type": "PERSON",
            "relationships": [{"target": "Acme", "type": "WORKS_AT"}],
            "connected_entities": [
                {"name": "Acme", "type": "ORGANIZATION", "depth": 0}
            ],
        }

        def mock_aggregate(pipeline):
            return AsyncIteratorMock([mock_result])

        graph_memory.collection.aggregate = mock_aggregate

        results = await graph_memory.graph_lookup(
            start_entity="John", agent_id="agent_1"
        )

        assert len(results) == 1
        assert results[0]["entity_name"] == "John"
        assert len(results[0]["connected_entities"]) == 1
        assert results[0]["connected_entities"][0]["name"] == "Acme"


class TestExtractAndStoreRelationships:
    """Tests for LLM-based relationship extraction."""

    @pytest.mark.asyncio
    async def test_extract_relationships_from_text(self, graph_memory):
        """Extract relationships via LLM and store them."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content='[{"source": "John", "target": "Acme Corp", "type": "WORKS_AT"}]'
        )

        # Source entity exists for add_relationship
        graph_memory.collection.find_one = AsyncMock(
            return_value={
                "_id": "id1",
                "metadata": {"entity_name": "John"},
                "agent_id": "agent_1",
            }
        )
        graph_memory.collection.update_one = AsyncMock(
            return_value=MagicMock(modified_count=1)
        )

        result = await graph_memory.extract_and_store_relationships(
            text="John works at Acme Corp in New York.",
            agent_id="agent_1",
            llm=mock_llm,
        )

        assert len(result) == 1
        assert result[0]["source"] == "John"
        assert result[0]["target"] == "Acme Corp"

    @pytest.mark.asyncio
    async def test_extract_relationships_empty_text(self, graph_memory):
        """Return empty list for short text."""
        mock_llm = AsyncMock()
        result = await graph_memory.extract_and_store_relationships(
            text="Hi", agent_id="agent_1", llm=mock_llm
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_relationships_invalid_json(self, graph_memory):
        """Handle invalid JSON from LLM gracefully."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(content="Not valid JSON at all")

        result = await graph_memory.extract_and_store_relationships(
            text="John works at Acme Corp in New York.",
            agent_id="agent_1",
            llm=mock_llm,
        )
        assert result == []


class TestEntityBoostedSearch:
    """Tests for entity-boosted hybrid search."""

    @pytest.mark.asyncio
    async def test_entity_boost_increases_score(self, graph_memory):
        """Verify entity boost increases score for entity-matching results."""
        from src.retrieval.vector_search import SearchResult

        # Mock embedding service
        mock_embedding = MagicMock()
        mock_embedding.generate_embedding = AsyncMock(
            return_value=MagicMock(embedding=[0.1] * 1024)
        )

        # Mock hybrid search results
        graph_memory.search_engine.hybrid_search = AsyncMock(
            return_value=[
                SearchResult(content="John is a developer at Acme", metadata={}, score=0.8, id="1"),
                SearchResult(content="Weather is nice today", metadata={}, score=0.85, id="2"),
            ]
        )

        # Mock entity lookup - "John" is an entity
        def mock_find(query, projection):
            return AsyncIteratorMock([
                {"metadata": {"entity_name": "John"}},
            ])

        graph_memory.collection.find = mock_find

        with patch("src.memory.graph.get_embedding_service", return_value=mock_embedding):
            results = await graph_memory.entity_boosted_search(
                query="Tell me about John",
                agent_id="agent_1",
                entity_boost=0.3,
            )

        # The result mentioning "John" should be boosted
        assert len(results) > 0
        # First result should mention John (boosted)
        john_result = [r for r in results if "John" in r.content][0]
        other_result = [r for r in results if "John" not in r.content][0]
        assert john_result.score > other_result.score


# Helper for async iteration in tests
class AsyncIteratorMock:
    """Mock async iterator for Motor cursors."""

    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item
