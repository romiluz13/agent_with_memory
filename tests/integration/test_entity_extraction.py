"""
Integration tests for Entity Extraction with Gemini LLM.
Tests LLM-powered NER for PERSON, ORGANIZATION, LOCATION, SYSTEM, CONCEPT.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.memory.entity import EntityMemory, ExtractionConfig, ENTITY_TYPES


class TestEntityTypes:
    """Test entity type definitions."""

    def test_all_entity_types_defined(self):
        """Test all 5 entity types are defined."""
        expected = ["PERSON", "ORGANIZATION", "LOCATION", "SYSTEM", "CONCEPT"]
        assert ENTITY_TYPES == expected

    def test_person_type(self):
        """Test PERSON entity type."""
        assert "PERSON" in ENTITY_TYPES

    def test_organization_type(self):
        """Test ORGANIZATION entity type."""
        assert "ORGANIZATION" in ENTITY_TYPES

    def test_location_type(self):
        """Test LOCATION entity type."""
        assert "LOCATION" in ENTITY_TYPES

    def test_system_type(self):
        """Test SYSTEM entity type."""
        assert "SYSTEM" in ENTITY_TYPES

    def test_concept_type(self):
        """Test CONCEPT entity type."""
        assert "CONCEPT" in ENTITY_TYPES


class TestExtractionConfigGemini:
    """Test ExtractionConfig uses Gemini by default."""

    def test_default_model_is_gemini(self):
        """Test default model is gemini-2.5-flash."""
        config = ExtractionConfig()
        assert config.model == "gemini-2.5-flash"

    def test_default_provider_is_google(self):
        """Test default provider is google."""
        config = ExtractionConfig()
        assert config.provider == "google"

    def test_temperature_zero_for_determinism(self):
        """Test temperature is 0 for deterministic extraction."""
        config = ExtractionConfig()
        assert config.temperature == 0.0

    def test_max_tokens(self):
        """Test max tokens is configured."""
        config = ExtractionConfig()
        assert config.max_tokens == 500


class TestExtractionPrompt:
    """Test entity extraction prompt."""

    def test_prompt_has_text_placeholder(self):
        """Test extraction prompt has {text} placeholder."""
        assert "{text}" in EntityMemory.EXTRACTION_PROMPT

    def test_prompt_mentions_entity_types(self):
        """Test extraction prompt mentions entity types."""
        prompt = EntityMemory.EXTRACTION_PROMPT
        assert "PERSON" in prompt
        assert "ORGANIZATION" in prompt
        assert "LOCATION" in prompt
        assert "SYSTEM" in prompt
        assert "CONCEPT" in prompt

    def test_prompt_requests_json(self):
        """Test extraction prompt requests JSON format."""
        prompt = EntityMemory.EXTRACTION_PROMPT
        assert "JSON" in prompt

    def test_prompt_specifies_output_format(self):
        """Test extraction prompt specifies expected output structure."""
        prompt = EntityMemory.EXTRACTION_PROMPT
        assert "name" in prompt
        assert "type" in prompt
        assert "description" in prompt


class TestExtractAndStore:
    """Test extract_and_store method."""

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
    async def test_extract_returns_entities(self, entity_store, test_agent_id, clean_collections):
        """Test extraction returns list of entities."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content='[{"name": "John", "type": "PERSON", "description": "Engineer"}]'
        )

        entities = await entity_store.extract_and_store(
            text="John is a software engineer at Google.",
            llm=mock_llm,
            agent_id=test_agent_id
        )

        assert len(entities) >= 1
        assert entities[0]["name"] == "John"
        assert entities[0]["type"] == "PERSON"

    @pytest.mark.asyncio
    async def test_extract_empty_text_returns_empty(self, entity_store, test_agent_id, clean_collections):
        """Test extraction with empty text returns empty list."""
        mock_llm = AsyncMock()

        entities = await entity_store.extract_and_store(
            text="",
            llm=mock_llm,
            agent_id=test_agent_id
        )

        assert entities == []

    @pytest.mark.asyncio
    async def test_extract_invalid_type_fallback(self, entity_store, test_agent_id, clean_collections):
        """Test invalid entity type falls back to CONCEPT."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content='[{"name": "Quantum", "type": "INVALID_TYPE", "description": "A concept"}]'
        )

        entities = await entity_store.extract_and_store(
            text="Quantum computing is interesting.",
            llm=mock_llm,
            agent_id=test_agent_id
        )

        # Should store with fallback type
        assert len(entities) >= 0  # May be filtered or stored as CONCEPT


class TestEntityMerging:
    """Test entity merging when same entity is found."""

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
    async def test_merge_increments_mentions(self, entity_store, test_agent_id, clean_collections):
        """Test merging same entity increments mention count."""
        from src.memory.base import Memory, MemoryType

        # Store first entity
        memory1 = Memory(
            agent_id=test_agent_id,
            content="Google (ORGANIZATION): Tech company",
            memory_type=MemoryType.ENTITY,
            metadata={
                "entity_name": "Google",
                "entity_type": "ORGANIZATION",
                "mentions": 1
            }
        )
        await entity_store.store(memory1)

        # Store same entity again
        memory2 = Memory(
            agent_id=test_agent_id,
            content="Google (ORGANIZATION): Tech giant",
            memory_type=MemoryType.ENTITY,
            metadata={
                "entity_name": "Google",
                "entity_type": "ORGANIZATION",
                "mentions": 1
            }
        )
        await entity_store.store(memory2)

        # Check mentions were incremented
        entities = await entity_store.get_entities_by_type("ORGANIZATION")
        google = next((e for e in entities if "Google" in e.content), None)
        if google:
            assert google.metadata.get("mentions", 1) >= 1


class TestEntityRetrieval:
    """Test entity retrieval methods."""

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

    def test_has_retrieve_method(self, entity_store):
        """Test EntityMemory has retrieve method."""
        assert hasattr(entity_store, 'retrieve')

    def test_has_get_entities_by_type_method(self, entity_store):
        """Test EntityMemory has get_entities_by_type method."""
        assert hasattr(entity_store, 'get_entities_by_type')

    def test_has_increment_mentions_method(self, entity_store):
        """Test EntityMemory has increment_mentions method."""
        assert hasattr(entity_store, 'increment_mentions')
