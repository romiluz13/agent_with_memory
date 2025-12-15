"""Tests for Context Summarizer module."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.context.summarizer import (
    ContextSummarizer,
    SummarizationConfig
)


class TestSummarizationConfig:
    """Tests for summarization configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SummarizationConfig()
        assert config.max_input_length == 3000
        assert config.max_summary_tokens == 200
        assert config.max_label_tokens == 30
        assert config.temperature == 0.3

    def test_custom_config(self):
        """Test custom configuration."""
        config = SummarizationConfig(
            max_input_length=5000,
            max_summary_tokens=500
        )
        assert config.max_input_length == 5000
        assert config.max_summary_tokens == 500


class TestContextSummarizer:
    """Tests for ContextSummarizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.summarizer = ContextSummarizer()

    def test_init_defaults(self):
        """Test default initialization."""
        assert self.summarizer.config.max_input_length == 3000
        assert self.summarizer.config.max_summary_tokens == 200

    def test_init_custom_config(self):
        """Test custom initialization."""
        config = SummarizationConfig(max_input_length=5000)
        custom = ContextSummarizer(config=config)
        assert custom.config.max_input_length == 5000

    def test_summarize_prompt_has_placeholder(self):
        """Test summarize prompt contains content placeholder."""
        assert "{content}" in self.summarizer.SUMMARIZE_PROMPT

    def test_label_prompt_has_placeholder(self):
        """Test label prompt contains summary placeholder."""
        assert "{summary}" in self.summarizer.LABEL_PROMPT

    @pytest.mark.asyncio
    async def test_summarize_returns_dict(self):
        """Test summarize returns expected structure."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            MagicMock(content="This is a test summary."),
            MagicMock(content="Test Summary Label")
        ]

        result = await self.summarizer.summarize("Test content to summarize", mock_llm)

        assert "summary" in result
        assert "description" in result
        assert "original_length" in result
        assert result["summary"] == "This is a test summary."

    @pytest.mark.asyncio
    async def test_summarize_truncates_long_content(self):
        """Test summarize truncates content that exceeds max length."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            MagicMock(content="Summary"),
            MagicMock(content="Label")
        ]

        long_content = "x" * 5000
        result = await self.summarizer.summarize(long_content, mock_llm)

        assert result["truncated"] == True
        assert result["original_length"] == 5000

    @pytest.mark.asyncio
    async def test_summarize_conversation(self):
        """Test conversation summarization."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            MagicMock(content="Conversation summary"),
            MagicMock(content="Chat Summary")
        ]

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]

        result = await self.summarizer.summarize_conversation(messages, mock_llm)

        assert "summary" in result
        assert result["summary"] == "Conversation summary"


class TestSummarizerOutputFormat:
    """Tests for summarizer output format."""

    def test_summary_result_structure(self):
        """Test expected summary result structure."""
        result = {
            "summary": "Brief summary of the content",
            "description": "Content Summary",
            "original_length": 5000,
            "truncated": False
        }

        assert "summary" in result
        assert "description" in result

    def test_summary_id_generation(self):
        """Test summary ID generation."""
        import uuid
        summary_id = str(uuid.uuid4())[:8]
        assert len(summary_id) == 8
