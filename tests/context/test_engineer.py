"""Tests for Context Engineer module."""

import pytest
from src.context.engineer import ContextEngineer, ContextUsage, MODEL_TOKEN_LIMITS


class TestContextEngineer:
    """Tests for ContextEngineer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engineer = ContextEngineer(threshold=0.80, default_model="gpt-4o-mini")

    def test_calculate_usage_basic(self):
        """Test basic token usage calculation."""
        context = "a" * 400  # 400 chars = ~100 tokens
        usage = self.engineer.calculate_usage(context, "gpt-4o-mini")

        assert isinstance(usage, ContextUsage)
        assert usage.tokens == 100
        assert usage.max_tokens == 128000
        assert usage.percent < 1.0
        assert usage.remaining > 0

    def test_calculate_usage_different_models(self):
        """Test token limits for different models."""
        context = "test content"

        # GPT-4o-mini
        usage = self.engineer.calculate_usage(context, "gpt-4o-mini")
        assert usage.max_tokens == 128000

        # Claude models
        usage = self.engineer.calculate_usage(context, "claude-3-5-sonnet")
        assert usage.max_tokens == 200000

    def test_should_compress_under_threshold(self):
        """Test should_compress returns False under threshold."""
        context = "x" * 1000  # Small context
        assert not self.engineer.should_compress(context, "gpt-4o-mini")

    def test_should_compress_over_threshold(self):
        """Test should_compress returns True over threshold."""
        # Create context that exceeds 80% of 128k tokens
        # 80% of 128000 tokens = 102400 tokens
        # 102400 tokens * 4 chars = 409600 chars
        context = "x" * 500000  # Well over 80%
        assert self.engineer.should_compress(context, "gpt-4o-mini")

    def test_create_reference(self):
        """Test summary reference creation."""
        ref = self.engineer.create_reference("abc123", "Brief description")
        assert ref == "[Summary ID: abc123] Brief description"

    def test_build_context_with_references(self):
        """Test building context with summary references."""
        summaries = [
            {"summary_id": "id1", "description": "First summary"},
            {"summary_id": "id2", "description": "Second summary"}
        ]
        current = "Current context"

        result = self.engineer.build_context_with_references(summaries, current)

        assert "## Summary Memory" in result
        assert "[ID: id1]" in result
        assert "[ID: id2]" in result
        assert "Current context" in result

    def test_build_context_no_summaries(self):
        """Test building context with no summaries returns current context."""
        current = "Current context"
        result = self.engineer.build_context_with_references([], current)
        assert result == current

    def test_unknown_model_fallback(self):
        """Test fallback for unknown models."""
        usage = self.engineer.calculate_usage("test", "unknown-model-xyz")
        assert usage.max_tokens == 128000  # Default fallback

    def test_format_usage_string(self):
        """Test usage string formatting."""
        usage = ContextUsage(tokens=50000, max_tokens=128000, percent=39.1, remaining=78000)
        formatted = self.engineer.format_usage_string(usage)
        assert "39.1%" in formatted
        assert "50,000" in formatted
        assert "128,000" in formatted


class TestModelTokenLimits:
    """Tests for model token limits."""

    def test_openai_models_present(self):
        """Test OpenAI models are in limits."""
        assert "gpt-4o" in MODEL_TOKEN_LIMITS
        assert "gpt-4o-mini" in MODEL_TOKEN_LIMITS

    def test_anthropic_models_present(self):
        """Test Anthropic models are in limits."""
        assert "claude-3-5-sonnet" in MODEL_TOKEN_LIMITS
        assert "claude-3-opus" in MODEL_TOKEN_LIMITS

    def test_all_limits_positive(self):
        """Test all token limits are positive integers."""
        for model, limit in MODEL_TOKEN_LIMITS.items():
            assert isinstance(limit, int)
            assert limit > 0
