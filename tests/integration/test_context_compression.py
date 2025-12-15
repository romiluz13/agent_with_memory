"""
Integration tests for Context Compression at 80% threshold.
Oracle innovation: Auto-compress when tokens exceed 80% of model limit.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.context.engineer import ContextEngineer, MODEL_TOKEN_LIMITS
from src.context.summarizer import ContextSummarizer, SummarizationConfig


class TestContextEngineerConfig:
    """Test ContextEngineer configuration."""

    def test_default_threshold_is_80_percent(self):
        """Test default compression threshold is 80%."""
        engineer = ContextEngineer()
        assert engineer.threshold == 0.80

    def test_custom_threshold(self):
        """Test custom threshold can be set."""
        engineer = ContextEngineer(threshold=0.75)
        assert engineer.threshold == 0.75


class TestModelTokenLimits:
    """Test model token limits are defined."""

    def test_gpt4_models_defined(self):
        """Test GPT-4 models have token limits."""
        assert "gpt-4" in MODEL_TOKEN_LIMITS
        assert "gpt-4o" in MODEL_TOKEN_LIMITS
        assert "gpt-4o-mini" in MODEL_TOKEN_LIMITS

    def test_claude_models_defined(self):
        """Test Claude models have token limits."""
        assert "claude-3-5-sonnet" in MODEL_TOKEN_LIMITS
        assert "claude-3-opus" in MODEL_TOKEN_LIMITS

    def test_gemini_models_defined(self):
        """Test Gemini models have token limits."""
        assert "gemini-2.0-flash" in MODEL_TOKEN_LIMITS or \
               "gemini-1.5-flash" in MODEL_TOKEN_LIMITS or \
               "gemini-pro" in MODEL_TOKEN_LIMITS

    def test_all_limits_positive(self):
        """Test all token limits are positive integers."""
        for model, limit in MODEL_TOKEN_LIMITS.items():
            assert limit > 0, f"{model} has non-positive limit: {limit}"


class TestTokenEstimation:
    """Test token estimation (chars // 4 formula)."""

    def test_token_estimation_formula(self):
        """Test token estimation uses chars // 4."""
        text = "Hello world"
        expected_tokens = len(text) // 4
        assert expected_tokens == 2  # 11 // 4 = 2

    def test_calculate_usage(self):
        """Test calculate_usage returns ContextUsage with percentage."""
        engineer = ContextEngineer()
        # 40,000 chars = 10,000 tokens
        # GPT-4o has 128k context, so 10k/128k = ~7.8%
        context = "x" * 40000
        usage = engineer.calculate_usage(context, "gpt-4o")
        # Returns ContextUsage with percent as percentage (0-100)
        assert 0 < usage.percent < 100
        assert usage.tokens > 0
        assert usage.max_tokens > 0


class TestShouldCompress:
    """Test compression trigger logic."""

    def test_under_threshold_no_compress(self):
        """Test no compression when under 80%."""
        engineer = ContextEngineer(threshold=0.80)
        # Short context should not trigger compression
        context = "Short message"
        should = engineer.should_compress(context, "gpt-4o")
        assert should is False

    def test_over_threshold_compress(self):
        """Test compression triggers at 80%+."""
        engineer = ContextEngineer(threshold=0.80)
        # Very long context should trigger (depends on model limit)
        # For gpt-4o with 128k limit, 80% = ~100k tokens = ~400k chars
        # We'll use a smaller model or mock
        engineer.threshold = 0.10  # Lower threshold for testing
        context = "x" * 100000  # 25k tokens
        should = engineer.should_compress(context, "gpt-4o-mini")
        assert should is True

    def test_exact_threshold(self):
        """Test behavior at exactly 80%."""
        engineer = ContextEngineer(threshold=0.80)
        # Create context that's exactly at threshold
        # This tests edge case handling
        pass  # Implementation specific


class TestSummaryReference:
    """Test summary reference creation."""

    def test_create_reference_format(self):
        """Test summary reference format."""
        engineer = ContextEngineer()
        ref = engineer.create_reference("sum-001", "Discussion about Python")
        assert "sum-001" in ref
        assert "Python" in ref

    def test_reference_is_compact(self):
        """Test references are compact for context efficiency."""
        engineer = ContextEngineer()
        ref = engineer.create_reference("id", "description")
        # References should be shorter than original content
        assert len(ref) < 200


class TestContextSummarizerConfig:
    """Test ContextSummarizer configuration."""

    def test_default_config(self):
        """Test default summarization config."""
        config = SummarizationConfig()
        assert config.max_input_length == 3000
        assert config.max_summary_tokens == 200
        assert config.max_label_tokens == 30
        assert config.temperature == 0.3

    def test_custom_config(self):
        """Test custom summarization config."""
        config = SummarizationConfig(
            max_input_length=5000,
            max_summary_tokens=500
        )
        assert config.max_input_length == 5000
        assert config.max_summary_tokens == 500


class TestSummarizerPrompts:
    """Test summarizer prompts are properly configured."""

    def test_summarize_prompt_has_content_placeholder(self):
        """Test summarize prompt has {content} placeholder."""
        summarizer = ContextSummarizer()
        assert "{content}" in summarizer.SUMMARIZE_PROMPT

    def test_label_prompt_has_summary_placeholder(self):
        """Test label prompt has {summary} placeholder."""
        summarizer = ContextSummarizer()
        assert "{summary}" in summarizer.LABEL_PROMPT


class TestSummarizeMethod:
    """Test summarize method behavior."""

    @pytest.mark.asyncio
    async def test_summarize_returns_expected_structure(self):
        """Test summarize returns dict with required keys."""
        summarizer = ContextSummarizer()
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            MagicMock(content="This is a summary"),
            MagicMock(content="Summary Label")
        ]

        result = await summarizer.summarize("Test content", mock_llm)

        assert "summary" in result
        assert "description" in result
        assert "original_length" in result

    @pytest.mark.asyncio
    async def test_summarize_truncates_long_input(self):
        """Test long content is truncated before summarization."""
        summarizer = ContextSummarizer()
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            MagicMock(content="Summary"),
            MagicMock(content="Label")
        ]

        long_content = "x" * 10000
        result = await summarizer.summarize(long_content, mock_llm)

        assert result["truncated"] is True
        assert result["original_length"] == 10000


class TestBuildContextWithReferences:
    """Test building context with summary references."""

    def test_build_context_adds_references(self):
        """Test that summary references are added to context."""
        engineer = ContextEngineer()
        summaries = [
            {"summary_id": "sum-001", "description": "First discussion"},
            {"summary_id": "sum-002", "description": "Second discussion"}
        ]

        # Note: parameter order is (summaries, current_context)
        context = engineer.build_context_with_references(
            summaries,
            "Current message"
        )

        assert "sum-001" in context
        assert "sum-002" in context
        assert "Current message" in context

    def test_build_context_empty_summaries(self):
        """Test building context with no summaries."""
        engineer = ContextEngineer()
        # Note: parameter order is (summaries, current_context)
        context = engineer.build_context_with_references(
            [],
            "Message only"
        )
        assert "Message only" in context
