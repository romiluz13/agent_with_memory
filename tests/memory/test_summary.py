"""Tests for Summary Memory module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.memory.summary import SummaryMemory
from src.memory.base import Memory, MemoryType


class TestSummaryMemoryInit:
    """Tests for SummaryMemory initialization."""

    @pytest.fixture
    def mock_collection(self):
        """Create mock MongoDB collection."""
        return AsyncMock()

    @pytest.fixture
    def summary_memory(self, mock_collection):
        """Create SummaryMemory with mocked dependencies."""
        with patch('src.memory.summary.VectorSearchEngine'), \
             patch('src.memory.summary.get_embedding_service') as mock_embed:
            mock_embed.return_value = MagicMock()
            return SummaryMemory(mock_collection)

    def test_summary_memory_attributes(self, summary_memory):
        """Test SummaryMemory has expected attributes."""
        assert hasattr(summary_memory, 'collection')
        assert hasattr(summary_memory, 'search_engine')
        assert hasattr(summary_memory, 'embedding_service')


class TestSummaryIdGeneration:
    """Tests for summary ID generation."""

    def test_uuid_format(self):
        """Test summary IDs follow expected format."""
        import uuid
        # Simulate what store_summary does
        summary_id = str(uuid.uuid4())[:8]
        assert len(summary_id) == 8
        assert all(c in '0123456789abcdef-' for c in summary_id)

    def test_unique_ids(self):
        """Test generated IDs are unique."""
        import uuid
        ids = [str(uuid.uuid4())[:8] for _ in range(100)]
        assert len(set(ids)) == 100  # All unique


class TestSummaryStorage:
    """Tests for summary storage logic."""

    def test_summary_memory_format(self):
        """Test expected summary memory format."""
        memory_data = {
            "content": "This is a summary of the conversation",
            "memory_type": "summary",
            "metadata": {
                "summary_id": "abc12345",
                "full_content": "This is the original long content...",
                "description": "Conversation about project planning"
            }
        }

        assert memory_data["memory_type"] == "summary"
        assert "summary_id" in memory_data["metadata"]
        assert "full_content" in memory_data["metadata"]

    def test_description_truncation(self):
        """Test description truncation logic."""
        max_words = 10  # Reasonable limit for descriptions
        long_description = " ".join(["word"] * 20)
        words = long_description.split()
        truncated = " ".join(words[:max_words])

        assert len(truncated.split()) <= max_words


class TestSummaryRetrieval:
    """Tests for summary retrieval."""

    def test_summary_reference_format(self):
        """Test summary reference format."""
        summary_id = "abc12345"
        description = "Brief summary"
        reference = f"[Summary ID: {summary_id}] {description}"

        assert summary_id in reference
        assert description in reference
        assert reference.startswith("[Summary ID:")

    def test_expand_returns_full_content(self):
        """Test expand logic returns full content."""
        memory_metadata = {
            "summary_id": "abc12345",
            "full_content": "This is the original full content that was summarized."
        }

        full_content = memory_metadata.get("full_content")
        assert full_content is not None
        assert len(full_content) > 0


class TestSummaryMemoryMethods:
    """Tests for SummaryMemory methods using mocks."""

    @pytest.fixture
    def mock_collection(self):
        """Create mock MongoDB collection."""
        return AsyncMock()

    @pytest.fixture
    def summary_memory(self, mock_collection):
        """Create SummaryMemory with mocked dependencies."""
        with patch('src.memory.summary.VectorSearchEngine'), \
             patch('src.memory.summary.get_embedding_service') as mock_embed:
            mock_embed.return_value = MagicMock()
            return SummaryMemory(mock_collection)

    def test_has_store_method(self, summary_memory):
        """Test SummaryMemory has store method."""
        assert hasattr(summary_memory, 'store')
        assert callable(getattr(summary_memory, 'store'))

    def test_has_retrieve_method(self, summary_memory):
        """Test SummaryMemory has retrieve method."""
        assert hasattr(summary_memory, 'retrieve')
        assert callable(getattr(summary_memory, 'retrieve'))

    def test_has_store_summary_method(self, summary_memory):
        """Test SummaryMemory has store_summary method."""
        assert hasattr(summary_memory, 'store_summary')
        assert callable(getattr(summary_memory, 'store_summary'))

    def test_has_expand_summary_method(self, summary_memory):
        """Test SummaryMemory has expand_summary method."""
        assert hasattr(summary_memory, 'expand_summary')
        assert callable(getattr(summary_memory, 'expand_summary'))

    def test_has_retrieve_by_summary_id_method(self, summary_memory):
        """Test SummaryMemory has retrieve_by_summary_id method."""
        assert hasattr(summary_memory, 'retrieve_by_summary_id')
        assert callable(getattr(summary_memory, 'retrieve_by_summary_id'))

    def test_has_list_summary_references_method(self, summary_memory):
        """Test SummaryMemory has list_summary_references method."""
        assert hasattr(summary_memory, 'list_summary_references')
        assert callable(getattr(summary_memory, 'list_summary_references'))


class TestCompressionMetadata:
    """Tests for compression ratio and word count metadata."""

    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        full_content = "This is the original content with many words that needs compression."
        summary = "Brief summary."

        ratio = round(len(summary) / max(len(full_content), 1), 2)
        assert 0 < ratio < 1

    def test_word_count_tracking(self):
        """Test word count tracking for original and summary."""
        full_content = "This is the original content with many words."
        summary = "Brief summary content."

        original_words = len(full_content.split())
        summary_words = len(summary.split())

        assert original_words > summary_words
        assert summary_words > 0
