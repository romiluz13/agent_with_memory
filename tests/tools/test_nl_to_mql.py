"""Tests for NL-to-MQL Generator (src/tools/nl_to_mql.py).

Tests cover:
- NLToMQLGenerator initialization with allowed_collections whitelist
- Schema context extraction from sample documents
- Query generation with agent_id injection
- Collection whitelist enforcement
- Read-only validation (no writes)
- Dangerous operator rejection ($where, JavaScript)
- Graceful error handling
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.tools.nl_to_mql import NLToMQLGenerator


class TestNLToMQLInit:
    """Tests for NLToMQLGenerator initialization."""

    def test_default_allowed_collections(self):
        """Default collections are the 7 memory collections."""
        mock_db = MagicMock()
        gen = NLToMQLGenerator(db=mock_db)
        expected = [
            "episodic_memories",
            "semantic_memories",
            "procedural_memories",
            "working_memories",
            "cache_memories",
            "entity_memories",
            "summary_memories",
        ]
        assert gen._allowed_collections == expected

    def test_custom_allowed_collections(self):
        """Custom collection whitelist overrides default."""
        mock_db = MagicMock()
        gen = NLToMQLGenerator(
            db=mock_db, allowed_collections=["custom_collection"]
        )
        assert gen._allowed_collections == ["custom_collection"]


class TestSchemaContext:
    """Tests for schema context extraction."""

    @pytest.mark.asyncio
    async def test_get_schema_context_returns_doc_structure(self):
        """_get_schema_context fetches a sample doc and returns its keys."""
        mock_collection = _make_mock_collection(
            sample_doc={
                "_id": "abc123",
                "content": "Hello world",
                "agent_id": "agent_1",
                "metadata": {"type": "test"},
                "embedding": [0.1, 0.2],
            },
        )

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        gen = NLToMQLGenerator(db=mock_db)
        schema = await gen._get_schema_context("episodic_memories")
        assert isinstance(schema, str)
        assert "content" in schema
        assert "agent_id" in schema
        assert "metadata" in schema

    @pytest.mark.asyncio
    async def test_get_schema_context_empty_collection(self):
        """_get_schema_context returns fallback when collection is empty."""
        mock_collection = _make_mock_collection(sample_doc=None)

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        gen = NLToMQLGenerator(db=mock_db)
        schema = await gen._get_schema_context("episodic_memories")
        assert isinstance(schema, str)
        assert len(schema) > 0  # Should return some fallback text


def _make_mock_collection(
    sample_doc: dict | None = None,
    find_results: list | None = None,
):
    """Create a mock Motor collection with sync find() and async find_one/to_list.

    Motor's find() is synchronous (returns cursor), but find_one() and
    cursor.to_list() are async. This helper sets up mocks correctly.
    """
    mock_collection = MagicMock()
    # find_one is async
    mock_collection.find_one = AsyncMock(return_value=sample_doc)

    # find() is sync -- returns a cursor object
    mock_cursor = MagicMock()
    mock_cursor.to_list = AsyncMock(return_value=find_results or [])
    mock_collection.find = MagicMock(return_value=mock_cursor)

    return mock_collection


class TestQueryGeneration:
    """Tests for query generation with agent_id injection."""

    @pytest.mark.asyncio
    async def test_generate_query_injects_agent_id(self):
        """Generated query MUST contain agent_id in the filter."""
        mock_collection = _make_mock_collection(
            sample_doc={"content": "test", "agent_id": "agent_1"},
            find_results=[{"_id": "1", "content": "Result 1", "agent_id": "agent_1"}],
        )

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "collection": "episodic_memories",
            "filter": {"content": {"$regex": "hello"}},
            "projection": {"content": 1, "agent_id": 1},
            "limit": 10,
        })
        mock_llm.ainvoke.return_value = mock_response

        gen = NLToMQLGenerator(db=mock_db)
        result = await gen.generate_query(
            question="Find messages containing hello",
            agent_id="agent_1",
            llm=mock_llm,
        )

        assert "error" not in result or result.get("error") is None
        assert result.get("generated_mql") is not None
        # Verify agent_id was injected into the filter
        generated_filter = result["generated_mql"]["filter"]
        assert generated_filter.get("agent_id") == "agent_1"

    @pytest.mark.asyncio
    async def test_generate_query_returns_results(self):
        """generate_query returns query results."""
        mock_collection = _make_mock_collection(
            sample_doc={"content": "test", "agent_id": "agent_1"},
            find_results=[{"_id": "1", "content": "Result", "agent_id": "agent_1"}],
        )

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "collection": "episodic_memories",
            "filter": {},
            "projection": None,
            "limit": 10,
        })
        mock_llm.ainvoke.return_value = mock_response

        gen = NLToMQLGenerator(db=mock_db)
        result = await gen.generate_query(
            question="Show all memories",
            agent_id="agent_1",
            llm=mock_llm,
        )

        assert "results" in result
        assert isinstance(result["results"], list)

    @pytest.mark.asyncio
    async def test_generate_query_includes_execution_time(self):
        """generate_query reports execution_time in response."""
        mock_collection = _make_mock_collection(
            sample_doc={"content": "x", "agent_id": "a"},
            find_results=[],
        )

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "collection": "episodic_memories",
            "filter": {},
            "projection": None,
            "limit": 10,
        })
        mock_llm.ainvoke.return_value = mock_response

        gen = NLToMQLGenerator(db=mock_db)
        result = await gen.generate_query(
            question="List all",
            agent_id="agent_1",
            llm=mock_llm,
        )
        assert "execution_time" in result
        assert isinstance(result["execution_time"], float)


class TestCollectionWhitelist:
    """Tests for collection whitelist enforcement."""

    @pytest.mark.asyncio
    async def test_rejects_disallowed_collection(self):
        """Queries targeting non-whitelisted collections are rejected."""
        mock_collection = _make_mock_collection(sample_doc={"content": "x"})

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "collection": "system_users",
            "filter": {},
            "projection": None,
            "limit": 10,
        })
        mock_llm.ainvoke.return_value = mock_response

        gen = NLToMQLGenerator(db=mock_db)
        result = await gen.generate_query(
            question="Show system users",
            agent_id="agent_1",
            llm=mock_llm,
        )

        assert "error" in result
        assert "not in allowlist" in result["error"].lower() or "not allowed" in result["error"].lower()


class TestReadOnlyValidation:
    """Tests for read-only enforcement."""

    @pytest.mark.asyncio
    async def test_rejects_where_operator(self):
        """Queries containing $where are rejected."""
        mock_collection = _make_mock_collection(sample_doc={"content": "x"})

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "collection": "episodic_memories",
            "filter": {"$where": "this.content.length > 10"},
            "projection": None,
            "limit": 10,
        })
        mock_llm.ainvoke.return_value = mock_response

        gen = NLToMQLGenerator(db=mock_db)
        result = await gen.generate_query(
            question="Find long messages",
            agent_id="agent_1",
            llm=mock_llm,
        )

        assert "error" in result
        assert "forbidden" in result["error"].lower() or "not allowed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_rejects_javascript_in_filter(self):
        """Queries containing JavaScript function() calls are rejected."""
        mock_collection = _make_mock_collection(sample_doc={"content": "x"})

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "collection": "episodic_memories",
            "filter": {"$expr": {"$function": {"body": "function(){return true}", "args": [], "lang": "js"}}},
            "projection": None,
            "limit": 10,
        })
        mock_llm.ainvoke.return_value = mock_response

        gen = NLToMQLGenerator(db=mock_db)
        result = await gen.generate_query(
            question="Run custom function",
            agent_id="agent_1",
            llm=mock_llm,
        )

        assert "error" in result


class TestErrorHandling:
    """Tests for error handling in query generation."""

    @pytest.mark.asyncio
    async def test_malformed_llm_response(self):
        """Generator handles non-JSON LLM responses gracefully."""
        mock_collection = _make_mock_collection(sample_doc={"content": "x"})

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "I don't understand the question."
        mock_llm.ainvoke.return_value = mock_response

        gen = NLToMQLGenerator(db=mock_db)
        result = await gen.generate_query(
            question="garbled input",
            agent_id="agent_1",
            llm=mock_llm,
        )

        assert "error" in result
        assert result.get("results", []) == []

    @pytest.mark.asyncio
    async def test_llm_exception(self):
        """Generator handles LLM invocation failures gracefully."""
        mock_collection = _make_mock_collection(sample_doc={"content": "x"})

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = RuntimeError("LLM down")

        gen = NLToMQLGenerator(db=mock_db)
        result = await gen.generate_query(
            question="anything",
            agent_id="agent_1",
            llm=mock_llm,
        )

        assert "error" in result
        assert result.get("results", []) == []

    @pytest.mark.asyncio
    async def test_no_llm_returns_error(self):
        """generate_query without LLM returns error."""
        mock_db = MagicMock()
        gen = NLToMQLGenerator(db=mock_db)
        result = await gen.generate_query(
            question="anything",
            agent_id="agent_1",
        )

        assert "error" in result
        assert "llm required" in result["error"].lower()
