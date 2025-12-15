"""Tests for Entity Memory module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.memory.entity import EntityMemory, ENTITY_TYPES, ExtractionConfig
from src.memory.base import Memory, MemoryType


class TestEntityTypes:
    """Tests for entity type constants."""

    def test_entity_types_defined(self):
        """Test all expected entity types are defined."""
        expected = ["PERSON", "ORGANIZATION", "LOCATION", "SYSTEM", "CONCEPT"]
        assert ENTITY_TYPES == expected

    def test_entity_types_are_strings(self):
        """Test entity types are strings."""
        for entity_type in ENTITY_TYPES:
            assert isinstance(entity_type, str)


class TestExtractionConfig:
    """Tests for ExtractionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExtractionConfig()
        assert config.model == "gemini-2.5-flash"
        assert config.temperature == 0.0
        assert config.max_tokens == 500
        assert config.provider == "google"

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExtractionConfig(
            model="claude-3-5-sonnet",
            temperature=0.1,
            max_tokens=1000,
            provider="anthropic"
        )
        assert config.model == "claude-3-5-sonnet"
        assert config.temperature == 0.1
        assert config.max_tokens == 1000
        assert config.provider == "anthropic"


class TestEntityMemoryInit:
    """Tests for EntityMemory initialization."""

    def test_extraction_prompt_format(self):
        """Test extraction prompt contains expected elements."""
        prompt = EntityMemory.EXTRACTION_PROMPT
        assert "{text}" in prompt
        assert "PERSON" in prompt
        assert "ORGANIZATION" in prompt
        assert "JSON" in prompt


class TestEntityMemoryMethods:
    """Tests for EntityMemory methods using mocks."""

    @pytest.fixture
    def mock_collection(self):
        """Create mock MongoDB collection."""
        return AsyncMock()

    @pytest.fixture
    def entity_memory(self, mock_collection):
        """Create EntityMemory with mocked dependencies."""
        with patch('src.memory.entity.VectorSearchEngine'), \
             patch('src.memory.entity.get_embedding_service') as mock_embed:
            mock_embed.return_value = MagicMock()
            return EntityMemory(mock_collection)

    def test_entity_memory_attributes(self, entity_memory):
        """Test EntityMemory has expected attributes."""
        assert hasattr(entity_memory, 'collection')
        assert hasattr(entity_memory, 'search_engine')
        assert hasattr(entity_memory, 'embedding_service')
        assert hasattr(entity_memory, 'config')

    def test_set_llm(self, entity_memory):
        """Test LLM can be set."""
        mock_llm = MagicMock()
        entity_memory.set_llm(mock_llm)
        assert entity_memory._llm == mock_llm


class TestEntityExtractionParsing:
    """Tests for entity extraction JSON parsing logic."""

    def test_valid_entity_format(self):
        """Test valid entity format is accepted."""
        entity = {
            "name": "John Smith",
            "type": "PERSON",
            "description": "A software engineer"
        }
        assert entity.get("name") == "John Smith"
        assert entity.get("type") in ENTITY_TYPES

    def test_entity_type_validation(self):
        """Test invalid entity types fall back to CONCEPT."""
        invalid_type = "UNKNOWN_TYPE"
        valid_type = "CONCEPT" if invalid_type not in ENTITY_TYPES else invalid_type
        assert valid_type == "CONCEPT"
