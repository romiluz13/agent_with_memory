"""
Tests for Phase 3 (MEDIUM) and Phase 4 (LOW) audit fixes.
Covers Issues #9-15 (Phase 3) and #16-22 (Phase 4).
"""

import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Base directory for source files
SRC_DIR = Path(__file__).parent.parent / "src"
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


# ============================================================
# Phase 3: MEDIUM Issues
# ============================================================


class TestTTLIndexes:
    """Issue #9: TTL indexes must be created for working_memories and cache_memories.
    Skill: mongodb-query-optimizer - TTL indexes auto-remove expired documents.
    """

    def test_setup_indexes_has_ttl_creation(self):
        """setup_indexes.py must create TTL indexes."""
        setup_file = SCRIPTS_DIR / "setup_indexes.py"
        content = setup_file.read_text()
        # Must contain TTL index creation with expireAfterSeconds
        assert (
            "expireAfterSeconds" in content
        ), "setup_indexes.py must create TTL indexes with expireAfterSeconds"

    def test_ttl_index_for_working_memories(self):
        """TTL index must target working_memories collection."""
        setup_file = SCRIPTS_DIR / "setup_indexes.py"
        content = setup_file.read_text()
        # The TTL index creation should reference working_memories
        assert "working_memories" in content and "expireAfterSeconds" in content


class TestApprovalRequestIndexes:
    """Issue #10 follow-up: approval requests need the compound access-path index."""

    def test_setup_indexes_has_approval_request_compound_index(self):
        """setup_indexes.py must create the approval request compound index."""
        setup_file = SCRIPTS_DIR / "setup_indexes.py"
        content = setup_file.read_text()
        assert "approval_requests_agent_status_created_at" in content
        assert '("agent_id", 1)' in content
        assert '("status", 1)' in content
        assert '("created_at", -1)' in content


class TestNPlusOneRetrievalFixes:
    """Issue #6: retrieval should use projected search docs instead of extra fetches."""

    @staticmethod
    def _search_result(document: dict, score: float = 0.92):
        from src.retrieval.vector_search import SearchResult

        return SearchResult(
            id="memory-1",
            content=document["content"],
            metadata=document["metadata"],
            score=score,
            document=document,
        )

    @staticmethod
    def _memory_doc(memory_type: str) -> dict:
        return {
            "_id": "memory-1",
            "agent_id": "agent-1",
            "user_id": "user-1",
            "memory_type": memory_type,
            "content": f"{memory_type} content",
            "metadata": {"kind": memory_type},
            "importance": 0.5,
            "importance_level": "medium",
        }

    @pytest.mark.asyncio
    async def test_episodic_retrieve_uses_projected_document(self):
        from src.memory.episodic import EpisodicMemory

        collection = AsyncMock()
        collection.find_one = AsyncMock()

        with (
            patch("src.memory.episodic.VectorSearchEngine"),
            patch("src.memory.episodic.get_embedding_service") as mock_embed,
        ):
            embed_instance = MagicMock()
            embed_instance.generate_embedding = AsyncMock(
                return_value=MagicMock(embedding=[0.1] * 1024)
            )
            mock_embed.return_value = embed_instance

            store = EpisodicMemory(collection)
            store.search_engine.hybrid_search = AsyncMock(
                return_value=[self._search_result(self._memory_doc("episodic"))]
            )

            results = await store.retrieve("query", agent_id="agent-1")

        assert len(results) == 1
        collection.find_one.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_semantic_retrieve_uses_projected_document(self):
        from src.memory.semantic import SemanticMemory

        collection = AsyncMock()
        collection.find_one = AsyncMock()

        with (
            patch("src.memory.semantic.VectorSearchEngine"),
            patch("src.memory.semantic.get_embedding_service") as mock_embed,
        ):
            embed_instance = MagicMock()
            embed_instance.generate_embedding = AsyncMock(
                return_value=MagicMock(embedding=[0.1] * 1024)
            )
            mock_embed.return_value = embed_instance

            store = SemanticMemory(collection)
            store.search_engine.hybrid_search = AsyncMock(
                return_value=[self._search_result(self._memory_doc("semantic"))]
            )

            results = await store.retrieve("query", agent_id="agent-1")

        assert len(results) == 1
        collection.find_one.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_procedural_retrieve_uses_projected_document(self):
        from src.memory.procedural import ProceduralMemory

        collection = AsyncMock()
        collection.find_one = AsyncMock()

        with (
            patch("src.memory.procedural.VectorSearchEngine"),
            patch("src.memory.procedural.get_embedding_service") as mock_embed,
        ):
            embed_instance = MagicMock()
            embed_instance.generate_embedding = AsyncMock(
                return_value=MagicMock(embedding=[0.1] * 1024)
            )
            mock_embed.return_value = embed_instance

            store = ProceduralMemory(collection)
            store.search_engine.hybrid_search = AsyncMock(
                return_value=[self._search_result(self._memory_doc("procedural"))]
            )

            results = await store.retrieve("query", agent_id="agent-1")

        assert len(results) == 1
        collection.find_one.assert_not_awaited()


class TestEmbeddingFieldUnification:
    """Issue #10: All embedding fields must be 'embedding', not 'vector_embeddings'.
    Skill: mongodb-schema-design - Embedding field names must be consistent.
    """

    def test_no_vector_embeddings_in_ingestion(self):
        """mongodb_ingestion.py must not use 'vector_embeddings'."""
        ingestion_file = SRC_DIR / "ingestion" / "mongodb_ingestion.py"
        content = ingestion_file.read_text()
        assert (
            "vector_embeddings" not in content
        ), "mongodb_ingestion.py still uses 'vector_embeddings' — must be 'embedding'"

    def test_no_vector_embeddings_in_langgraph(self):
        """agent_langgraph.py must not use 'vector_embeddings'."""
        langgraph_file = SRC_DIR / "core" / "agent_langgraph.py"
        content = langgraph_file.read_text()
        assert (
            "vector_embeddings" not in content
        ), "agent_langgraph.py still uses 'vector_embeddings' — must be 'embedding'"

    def test_no_vector_embeddings_anywhere_in_src(self):
        """No file in src/ should reference 'vector_embeddings'."""
        for py_file in SRC_DIR.rglob("*.py"):
            content = py_file.read_text()
            assert (
                "vector_embeddings" not in content
            ), f"{py_file.relative_to(SRC_DIR.parent)} still uses 'vector_embeddings'"


class TestSchemaValidation:
    """Issue #11: Schema validation script must exist.
    Skill: mongodb-schema-design - $jsonSchema prevents malformed documents.
    """

    def test_schema_validation_script_exists(self):
        """scripts/setup_schema_validation.py must exist."""
        script = SCRIPTS_DIR / "setup_schema_validation.py"
        assert script.exists(), "scripts/setup_schema_validation.py not found"

    def test_schema_validation_has_required_fields(self):
        """Schema validation must enforce content, memory_type, agent_id, created_at."""
        script = SCRIPTS_DIR / "setup_schema_validation.py"
        content = script.read_text()
        for field in ["content", "memory_type", "agent_id", "created_at"]:
            assert field in content, f"Schema validation missing required field: {field}"

    def test_schema_validation_uses_warn_action(self):
        """Schema validation must use validationAction: 'warn' initially."""
        script = SCRIPTS_DIR / "setup_schema_validation.py"
        content = script.read_text()
        assert "warn" in content, "Schema validation must use validationAction: 'warn'"


class TestDatetimeDeprecation:
    """Issue #12: datetime.utcnow() must be replaced with datetime.now(timezone.utc).
    Skill: mongodb-connection - datetime.utcnow() deprecated in Python 3.12+.
    """

    def test_no_utcnow_in_base(self):
        """base.py must not use datetime.utcnow()."""
        base_file = SRC_DIR / "memory" / "base.py"
        content = base_file.read_text()
        assert "datetime.utcnow" not in content, "base.py still uses datetime.utcnow()"

    def test_no_utcnow_anywhere_in_src(self):
        """No file in src/ should use datetime.utcnow()."""
        for py_file in SRC_DIR.rglob("*.py"):
            content = py_file.read_text()
            assert (
                "datetime.utcnow" not in content
            ), f"{py_file.relative_to(SRC_DIR.parent)} still uses datetime.utcnow()"


class TestEmbeddingProjection:
    """Issue #13: Retrieval must exclude embedding arrays from fetch results.
    Skill: mongodb-query-optimizer - Use projection to exclude embedding arrays.
    """

    def test_list_memories_uses_projection_in_semantic(self):
        """semantic.py list_memories() should use projection to exclude embedding."""
        semantic_file = SRC_DIR / "memory" / "semantic.py"
        content = semantic_file.read_text()
        # The find() call in list_memories should have a projection arg
        assert (
            '"embedding": 0' in content or "'embedding': 0" in content
        ), "semantic.py list_memories() must exclude embedding with projection"


class TestConsolidationFix:
    """Issue #14: O(n^2) consolidation must be replaced.
    Skill: mongodb-query-optimizer - Use vector search instead of nested loop.
    """

    def test_consolidate_has_batch_size_param(self):
        """consolidate() should accept batch_size parameter."""
        from src.memory.semantic import SemanticMemory

        sig = inspect.signature(SemanticMemory.consolidate)
        assert "batch_size" in sig.parameters, "consolidate() must accept batch_size parameter"

    def test_consolidate_tracks_processed_ids(self):
        """consolidate() implementation should track processed IDs."""
        semantic_file = SRC_DIR / "memory" / "semantic.py"
        content = semantic_file.read_text()
        assert (
            "processed_ids" in content
        ), "consolidate() must track processed_ids to avoid duplicate merges"


class TestVectorIndexFilterFields:
    """Issue #15: Vector index filter fields must be unified to 8 fields.
    Skill: mongodb-search-and-ai - Filter fields must match between index definitions.
    """

    def test_setup_indexes_has_8_filter_fields(self):
        """setup_indexes.py must define all 8 filter fields."""
        setup_file = SCRIPTS_DIR / "setup_indexes.py"
        content = setup_file.read_text()
        required_fields = [
            "agent_id",
            "user_id",
            "memory_type",
            "thread_id",
            "timestamp",
            "importance",
            "metadata.tags",
            "metadata.entity_type",
        ]
        for field in required_fields:
            assert field in content, f"setup_indexes.py missing filter field: {field}"


# ============================================================
# Phase 4: LOW Issues
# ============================================================


class TestSingletonReset:
    """Issue #16: Singleton must have reset() classmethod for testing.
    Skill: mongodb-connection - Managed singletons with reset for testing.
    """

    def test_mongodb_client_has_reset(self):
        """MongoDBClient must have a reset() classmethod."""
        from src.storage.mongodb_client import MongoDBClient

        assert hasattr(MongoDBClient, "reset"), "MongoDBClient must have reset() classmethod"

    def test_reset_is_classmethod(self):
        """reset() must be a classmethod."""
        from src.storage.mongodb_client import MongoDBClient

        assert isinstance(
            inspect.getattr_static(MongoDBClient, "reset"), classmethod
        ), "MongoDBClient.reset must be a classmethod"


class TestTestParameterName:
    """Issue #18: Test assertions must use 'query_text', not 'text_query'.
    Skill: mongodb-natural-language-querying - Test assertions must match signatures.
    """

    def test_no_text_query_in_hybrid_search_test(self):
        """test_hybrid_search.py must not assert 'text_query'."""
        test_file = Path(__file__).parent.parent / "tests" / "integration" / "test_hybrid_search.py"
        content = test_file.read_text()
        # Should not have assert 'text_query' in params
        assert (
            "assert 'text_query' in params" not in content
        ), "test_hybrid_search.py still asserts 'text_query' — should be 'query_text'"

    def test_no_text_query_in_multi_collection_test(self):
        """test_multi_collection.py must not assert 'text_query'."""
        test_file = (
            Path(__file__).parent.parent / "tests" / "integration" / "test_multi_collection.py"
        )
        content = test_file.read_text()
        assert (
            "assert 'text_query' in params" not in content
        ), "test_multi_collection.py still asserts 'text_query' — should be 'query_text'"


class TestCORSWildcard:
    """Issue #19: CORS must not use wildcard origins."""

    def test_no_wildcard_cors(self):
        """main.py must not have allow_origins=["*"]."""
        main_file = SRC_DIR / "api" / "main.py"
        content = main_file.read_text()
        assert (
            'allow_origins=["*"]' not in content
        ), "main.py still uses CORS wildcard allow_origins=['*']"


class TestHealthCheckPublicAPI:
    """Issue #20: Health check must use public API, not private members.
    Skill: mongodb-connection - Use public APIs, not private members.
    """

    def test_no_private_db_access_in_health(self):
        """health.py must not access mongodb_client._db directly."""
        health_file = SRC_DIR / "api" / "routes" / "health.py"
        content = health_file.read_text()
        assert (
            "mongodb_client._db" not in content
        ), "health.py still accesses mongodb_client._db — use health_check()"

    def test_no_private_client_access_in_health(self):
        """health.py must not access mongodb_client._client directly."""
        health_file = SRC_DIR / "api" / "routes" / "health.py"
        content = health_file.read_text()
        assert (
            "mongodb_client._client" not in content
        ), "health.py still accesses mongodb_client._client — use health_check()"


class TestEfficientStats:
    """Issue #21: get_stats() must use count_documents(), not list_memories().
    Skill: mongodb-query-optimizer - Use count_documents() for counting.
    """

    def test_get_stats_uses_count_documents(self):
        """manager.py get_stats() must use count_documents."""
        manager_file = SRC_DIR / "memory" / "manager.py"
        content = manager_file.read_text()
        assert "count_documents" in content, "manager.py get_stats() must use count_documents()"

    def test_get_stats_does_not_use_list_memories_for_count(self):
        """manager.py get_stats() must not use len(list_memories(limit=1))."""
        manager_file = SRC_DIR / "memory" / "manager.py"
        content = manager_file.read_text()
        assert (
            "len(await store.list_memories(limit=1))" not in content
        ), "manager.py still uses len(list_memories(limit=1)) for counting"


class TestIngestionSafeDelete:
    """Issue #22: ingest_pdf() must not auto-delete all documents.
    Skill: mongodb-schema-design - Use additive patterns, not blanket deletion.
    """

    def test_ingest_pdf_has_clear_existing_param(self):
        """ingest_pdf() must accept clear_existing parameter."""
        ingestion_file = SRC_DIR / "ingestion" / "mongodb_ingestion.py"
        content = ingestion_file.read_text()
        assert "clear_existing" in content, "ingest_pdf() must accept clear_existing parameter"

    def test_ingest_pdf_clear_existing_defaults_false(self):
        """clear_existing must default to False."""
        ingestion_file = SRC_DIR / "ingestion" / "mongodb_ingestion.py"
        content = ingestion_file.read_text()
        assert (
            "clear_existing: bool = False" in content
        ), "clear_existing must default to False (additive pattern)"

    def test_delete_many_gated_behind_flag(self):
        """delete_many must be gated behind clear_existing flag."""
        ingestion_file = SRC_DIR / "ingestion" / "mongodb_ingestion.py"
        content = ingestion_file.read_text()
        assert (
            "if clear_existing:" in content
        ), "delete_many must be gated behind 'if clear_existing:'"


class TestMainHealthCheckNoPymongo:
    """Issue #20 (related): main.py health check must not access private _client."""

    def test_no_private_client_in_main_health(self):
        """main.py health check must not use mongodb_client._client."""
        main_file = SRC_DIR / "api" / "main.py"
        content = main_file.read_text()
        assert (
            "mongodb_client._client" not in content
        ), "main.py health check still accesses _client directly"
