"""
Tests for Phase 2 HIGH fixes: MongoDB audit remediation.
- Issue #4: Double embedding generation (check before re-generating)
- Issue #5: Sync PyMongo in API health check (must use async Motor)
- Issue #6: Regex injection in entity lookup (re.escape or collation)
- Issue #7: Broken similarity calculation (use all dimensions)
- Issue #8: No write concern / retry config
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.base import Memory, MemoryType

# ---------------------------------------------------------------------------
# Issue #4: Double Embedding Generation
# MongoDB Skill: mongodb-search-and-ai -- "Generate embeddings once at the
# orchestration layer, not per-store."
# ---------------------------------------------------------------------------


class TestEpisodicNoDoubleEmbedding:
    """episodic.py store() must NOT regenerate embedding when already present."""

    @pytest.fixture
    def mock_collection(self):
        coll = AsyncMock()
        coll.insert_one = AsyncMock(return_value=MagicMock(inserted_id="test-id-123"))
        return coll

    @pytest.fixture
    def episodic_memory(self, mock_collection):
        with (
            patch("src.memory.episodic.VectorSearchEngine"),
            patch("src.memory.episodic.get_embedding_service") as mock_embed,
        ):
            mock_embed_instance = MagicMock()
            mock_embed_instance.generate_embedding = AsyncMock(
                return_value=MagicMock(embedding=[0.2] * 1024)
            )
            mock_embed.return_value = mock_embed_instance
            from src.memory.episodic import EpisodicMemory

            mem = EpisodicMemory(mock_collection)
            return mem

    @pytest.mark.asyncio
    async def test_store_skips_embedding_when_already_present(self, episodic_memory):
        """When memory.embedding is already populated, store() must NOT call generate_embedding."""
        pre_existing_embedding = [0.5] * 1024
        memory = Memory(
            agent_id="agent-1",
            memory_type=MemoryType.EPISODIC,
            content="Test content",
            embedding=pre_existing_embedding,
        )

        await episodic_memory.store(memory)

        # generate_embedding must NOT have been called
        episodic_memory.embedding_service.generate_embedding.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_generates_embedding_when_absent(self, episodic_memory):
        """When memory.embedding is None/empty, store() MUST generate embedding."""
        memory = Memory(
            agent_id="agent-1",
            memory_type=MemoryType.EPISODIC,
            content="Test content",
            embedding=None,
        )

        await episodic_memory.store(memory)

        # generate_embedding MUST have been called
        episodic_memory.embedding_service.generate_embedding.assert_called_once()


class TestProceduralNoDoubleEmbedding:
    """procedural.py store() must check for existing embedding."""

    @pytest.fixture
    def mock_collection(self):
        coll = AsyncMock()
        coll.insert_one = AsyncMock(return_value=MagicMock(inserted_id="test-id-456"))
        return coll

    @pytest.fixture
    def procedural_memory(self, mock_collection):
        with (
            patch("src.memory.procedural.VectorSearchEngine"),
            patch("src.memory.procedural.get_embedding_service") as mock_embed,
        ):
            mock_embed_instance = MagicMock()
            mock_embed_instance.generate_embedding = AsyncMock(
                return_value=MagicMock(embedding=[0.2] * 1024)
            )
            mock_embed.return_value = mock_embed_instance
            from src.memory.procedural import ProceduralMemory

            mem = ProceduralMemory(mock_collection)
            return mem

    @pytest.mark.asyncio
    async def test_store_uses_existing_embedding(self, procedural_memory, mock_collection):
        """When memory.embedding is populated, procedural store() must use it, not regenerate."""
        pre_existing_embedding = [0.7] * 1024
        memory = Memory(
            agent_id="agent-1",
            memory_type=MemoryType.PROCEDURAL,
            content="How to deploy",
            embedding=pre_existing_embedding,
        )

        await procedural_memory.store(memory)

        # The inserted document must contain the pre-existing embedding
        insert_call = mock_collection.insert_one.call_args
        inserted_doc = insert_call[0][0]
        assert (
            inserted_doc["embedding"] == pre_existing_embedding
        ), "procedural store() did not preserve pre-existing embedding"


class TestWorkingNoDoubleEmbedding:
    """working.py store() must check for existing embedding."""

    @pytest.fixture
    def mock_collection(self):
        coll = AsyncMock()
        coll.insert_one = AsyncMock(return_value=MagicMock(inserted_id="test-id-789"))
        coll.count_documents = AsyncMock(return_value=0)
        return coll

    @pytest.fixture
    def working_memory(self, mock_collection):
        with (
            patch("src.memory.working.VectorSearchEngine"),
            patch("src.memory.working.get_embedding_service") as mock_embed,
        ):
            mock_embed_instance = MagicMock()
            mock_embed_instance.generate_embedding = AsyncMock(
                return_value=MagicMock(embedding=[0.2] * 1024)
            )
            mock_embed.return_value = mock_embed_instance
            from src.memory.working import WorkingMemory

            mem = WorkingMemory(mock_collection)
            return mem

    @pytest.mark.asyncio
    async def test_store_uses_existing_embedding(self, working_memory, mock_collection):
        """When memory.embedding is populated, working store() must use it."""
        pre_existing_embedding = [0.3] * 1024
        memory = Memory(
            agent_id="agent-1",
            memory_type=MemoryType.WORKING,
            content="Current task context",
            embedding=pre_existing_embedding,
        )

        await working_memory.store(memory)

        insert_call = mock_collection.insert_one.call_args
        inserted_doc = insert_call[0][0]
        assert (
            inserted_doc["embedding"] == pre_existing_embedding
        ), "working store() did not preserve pre-existing embedding"


class TestCacheNoDoubleEmbedding:
    """cache.py store() must check for existing embedding."""

    @pytest.fixture
    def mock_collection(self):
        coll = AsyncMock()
        coll.insert_one = AsyncMock(return_value=MagicMock(inserted_id="test-id-cache"))
        coll.find_one = AsyncMock(return_value=None)
        coll.count_documents = AsyncMock(return_value=0)
        return coll

    @pytest.fixture
    def cache_memory(self, mock_collection):
        with (
            patch("src.memory.cache.VectorSearchEngine"),
            patch("src.memory.cache.get_embedding_service") as mock_embed,
        ):
            mock_embed_instance = MagicMock()
            mock_embed_instance.generate_embedding = AsyncMock(
                return_value=MagicMock(embedding=[0.2] * 1024)
            )
            mock_embed.return_value = mock_embed_instance
            from src.memory.cache import SemanticCache

            mem = SemanticCache(mock_collection)
            return mem

    @pytest.mark.asyncio
    async def test_store_uses_existing_embedding(self, cache_memory, mock_collection):
        """When memory.embedding is populated, cache store() must use it."""
        pre_existing_embedding = [0.4] * 1024
        memory = Memory(
            agent_id="agent-1",
            memory_type=MemoryType.CACHE,
            content="Cached query",
            embedding=pre_existing_embedding,
        )

        await cache_memory.store(memory)

        insert_call = mock_collection.insert_one.call_args
        inserted_doc = insert_call[0][0]
        assert (
            inserted_doc["embedding"] == pre_existing_embedding
        ), "cache store() did not preserve pre-existing embedding"


class TestSemanticNoDoubleEmbedding:
    """semantic.py store() must check for existing embedding."""

    @pytest.fixture
    def mock_collection(self):
        coll = AsyncMock()
        coll.insert_one = AsyncMock(return_value=MagicMock(inserted_id="test-id-sem"))
        return coll

    @pytest.fixture
    def semantic_memory(self, mock_collection):
        with (
            patch("src.memory.semantic.VectorSearchEngine"),
            patch("src.memory.semantic.get_embedding_service") as mock_embed,
        ):
            mock_embed_instance = MagicMock()
            mock_embed_instance.generate_embedding = AsyncMock(
                return_value=MagicMock(embedding=[0.2] * 1024)
            )
            mock_embed.return_value = mock_embed_instance
            from src.memory.semantic import SemanticMemory

            mem = SemanticMemory(mock_collection)
            # Make _find_similar_knowledge return None (no merge)
            mem._find_similar_knowledge = AsyncMock(return_value=None)
            return mem

    @pytest.mark.asyncio
    async def test_store_uses_existing_embedding(self, semantic_memory, mock_collection):
        """When memory.embedding is populated, semantic store() must use it."""
        pre_existing_embedding = [0.6] * 1024
        memory = Memory(
            agent_id="agent-1",
            memory_type=MemoryType.SEMANTIC,
            content="Knowledge fact",
            embedding=pre_existing_embedding,
        )

        await semantic_memory.store(memory)

        insert_call = mock_collection.insert_one.call_args
        inserted_doc = insert_call[0][0]
        assert (
            inserted_doc["embedding"] == pre_existing_embedding
        ), "semantic store() did not preserve pre-existing embedding"


# ---------------------------------------------------------------------------
# Issue #5: Sync PyMongo in API health check
# MongoDB Skill: mongodb-connection -- "In async apps, use Motor exclusively.
# Health checks should reuse the existing async client."
# ---------------------------------------------------------------------------


class TestNoSyncPyMongoInHealthCheck:
    """src/api/main.py health check must NOT use sync PyMongo."""

    def test_no_pymongo_import_in_main(self):
        """src/api/main.py must not import from pymongo."""
        main_path = os.path.join(os.path.dirname(__file__), "..", "..", "src", "api", "main.py")
        with open(main_path) as f:
            source = f.read()

        assert "from pymongo" not in source, "src/api/main.py still imports sync pymongo"
        assert "import pymongo" not in source, "src/api/main.py still imports sync pymongo"

    def test_no_sync_mongoclient_in_main(self):
        """src/api/main.py must not use sync MongoClient."""
        main_path = os.path.join(os.path.dirname(__file__), "..", "..", "src", "api", "main.py")
        with open(main_path) as f:
            source = f.read()

        # Must not have MongoClient (sync) - only Motor's AsyncIOMotorClient is allowed
        # Check specifically for pymongo.MongoClient usage pattern
        assert "MongoClient(" not in source, "src/api/main.py still uses sync MongoClient"


# ---------------------------------------------------------------------------
# Issue #6: Regex Injection in entity lookup
# MongoDB Skill: mongodb-natural-language-querying -- "Escape special chars in
# regex patterns. Validate all field references."
# ---------------------------------------------------------------------------


class TestRegexInjectionPrevention:
    """entity.py _find_existing_entity must escape regex metacharacters."""

    def test_entity_file_uses_re_escape(self):
        """entity.py must use re.escape() when building regex for entity name lookup."""
        entity_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "memory", "entity.py"
        )
        with open(entity_path) as f:
            source = f.read()

        assert (
            "re.escape" in source
        ), "entity.py does not use re.escape() -- regex injection vulnerability"

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

    @pytest.mark.asyncio
    async def test_find_existing_entity_escapes_metacharacters(
        self, entity_memory, mock_collection
    ):
        """Entity names with regex metacharacters must be escaped before MongoDB query."""
        # A name with regex metacharacters that could match unintended patterns
        malicious_name = "test.*+?value"

        await entity_memory._find_existing_entity(malicious_name, "PERSON", agent_id="agent-1")

        # Get the query that was passed to find_one
        call_args = mock_collection.find_one.call_args
        query = call_args[0][0] if call_args[0] else call_args.kwargs.get("filter", {})

        # The regex pattern must contain escaped metacharacters
        regex_pattern = query.get("metadata.entity_name", {})
        if isinstance(regex_pattern, dict) and "$regex" in regex_pattern:
            pattern = regex_pattern["$regex"]
            # re.escape("test.*+?value") should produce "test\\.\\*\\+\\?value"
            # The key chars .* +? must be escaped
            assert ".*" not in pattern, f"Regex metacharacter .* not escaped in pattern: {pattern}"
            assert "+?" not in pattern, f"Regex metacharacter +? not escaped in pattern: {pattern}"


# ---------------------------------------------------------------------------
# Issue #7: Broken similarity calculation
# MongoDB Skill: mongodb-search-and-ai -- "Cosine similarity MUST use ALL
# dimensions of the embedding vector."
# ---------------------------------------------------------------------------


class TestFullCosineSimlarity:
    """semantic.py _calculate_similarity must use ALL dimensions."""

    def test_no_embedding_truncation_in_source(self):
        """semantic.py must not truncate embeddings with [:10] or any slice."""
        semantic_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "memory", "semantic.py"
        )
        with open(semantic_path) as f:
            source = f.read()

        assert (
            "embedding[:10]" not in source
        ), "semantic.py still truncates embeddings to 10 dimensions"
        # Check for any suspicious embedding slicing in _calculate_similarity
        assert "[:10]" not in source, "semantic.py still uses [:10] slicing on embeddings"

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
            mock_embed.return_value = mock_embed_instance
            from src.memory.semantic import SemanticMemory

            return SemanticMemory(mock_collection)

    @pytest.mark.asyncio
    async def test_identical_vectors_return_one(self, semantic_memory):
        """Two memories with identical embeddings must have similarity 1.0."""
        embedding = [0.1] * 1024
        mem1 = Memory(
            agent_id="a", memory_type=MemoryType.SEMANTIC, content="x", embedding=embedding
        )
        mem2 = Memory(
            agent_id="a", memory_type=MemoryType.SEMANTIC, content="y", embedding=embedding
        )

        similarity = await semantic_memory._calculate_similarity(mem1, mem2)
        assert (
            abs(similarity - 1.0) < 1e-6
        ), f"Identical embeddings should have similarity 1.0, got {similarity}"

    @pytest.mark.asyncio
    async def test_orthogonal_vectors_return_zero(self, semantic_memory):
        """Orthogonal vectors must have similarity ~0.0."""
        # Create two orthogonal vectors
        vec1 = [1.0] + [0.0] * 1023
        vec2 = [0.0, 1.0] + [0.0] * 1022
        mem1 = Memory(agent_id="a", memory_type=MemoryType.SEMANTIC, content="x", embedding=vec1)
        mem2 = Memory(agent_id="a", memory_type=MemoryType.SEMANTIC, content="y", embedding=vec2)

        similarity = await semantic_memory._calculate_similarity(mem1, mem2)
        assert (
            abs(similarity) < 1e-6
        ), f"Orthogonal vectors should have similarity ~0.0, got {similarity}"

    @pytest.mark.asyncio
    async def test_known_angle_similarity(self, semantic_memory):
        """Known vectors must produce correct cosine similarity."""
        # vec1 = [3, 4], vec2 = [4, 3], cosine = (12+12)/(5*5) = 24/25 = 0.96
        vec1 = [3.0, 4.0] + [0.0] * 1022
        vec2 = [4.0, 3.0] + [0.0] * 1022
        mem1 = Memory(agent_id="a", memory_type=MemoryType.SEMANTIC, content="x", embedding=vec1)
        mem2 = Memory(agent_id="a", memory_type=MemoryType.SEMANTIC, content="y", embedding=vec2)

        similarity = await semantic_memory._calculate_similarity(mem1, mem2)
        expected = 24.0 / 25.0
        assert (
            abs(similarity - expected) < 1e-6
        ), f"Expected similarity {expected}, got {similarity}"

    @pytest.mark.asyncio
    async def test_zero_vector_returns_zero(self, semantic_memory):
        """Zero vector must return 0.0 similarity (avoid division by zero)."""
        vec1 = [0.0] * 1024
        vec2 = [1.0] * 1024
        mem1 = Memory(agent_id="a", memory_type=MemoryType.SEMANTIC, content="x", embedding=vec1)
        mem2 = Memory(agent_id="a", memory_type=MemoryType.SEMANTIC, content="y", embedding=vec2)

        similarity = await semantic_memory._calculate_similarity(mem1, mem2)
        assert similarity == 0.0, f"Zero vector should return 0.0 similarity, got {similarity}"


# ---------------------------------------------------------------------------
# Issue #8: No write concern / retry config
# MongoDB Skill: mongodb-connection -- "Memory systems require w:'majority'
# write concern for durability. Add retryWrites and retryReads."
# ---------------------------------------------------------------------------


class TestWriteConcernAndRetryConfig:
    """mongodb_client.py must include w='majority', retryWrites, retryReads, journal."""

    def test_connection_config_in_source(self):
        """mongodb_client.py must include production-grade connection parameters."""
        client_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "storage", "mongodb_client.py"
        )
        with open(client_path) as f:
            source = f.read()

        assert "retryWrites" in source, "mongodb_client.py missing retryWrites parameter"
        assert "retryReads" in source, "mongodb_client.py missing retryReads parameter"
        assert (
            'w="majority"' in source or "w='majority'" in source
        ), "mongodb_client.py missing w='majority' write concern"
        assert "journal" in source, "mongodb_client.py missing journal parameter"
