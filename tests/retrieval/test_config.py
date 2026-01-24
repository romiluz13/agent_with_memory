"""Tests for retrieval configuration dataclasses."""

from datetime import datetime

from src.retrieval.config import (
    NUM_CANDIDATES_MULTIPLIER,
    AtlasSearchFilterConfig,
    HybridSearchConfig,
    LexicalPrefilterConfig,
    SearchTier,
    VectorSearchFilterConfig,
)


class TestSearchTier:
    """Tests for SearchTier enum."""

    def test_tier_values_defined(self):
        """Test all expected tier values are defined."""
        assert SearchTier.M10_PLUS.value == "m10_plus"
        assert SearchTier.M0_M2.value == "m0_m2"
        assert SearchTier.VECTOR_ONLY.value == "vector_only"


class TestConstants:
    """Tests for module constants."""

    def test_num_candidates_multiplier(self):
        """Test NUM_CANDIDATES_MULTIPLIER is set correctly."""
        assert NUM_CANDIDATES_MULTIPLIER == 20


class TestVectorSearchFilterConfig:
    """Tests for VectorSearchFilterConfig dataclass."""

    def test_default_config(self):
        """Test default configuration is empty."""
        config = VectorSearchFilterConfig()
        assert config.agent_id is None
        assert config.user_id is None
        assert config.memory_types is None
        assert config.start_date is None
        assert config.end_date is None
        assert config.importance_min is None
        assert config.importance_max is None
        assert config.tags is None
        assert config.thread_id is None
        assert config.equality_filters == {}
        assert config.in_filters == {}
        assert config.comparison_filters == {}

    def test_custom_config(self):
        """Test custom configuration."""
        start = datetime(2024, 1, 1)
        config = VectorSearchFilterConfig(
            agent_id="agent_123",
            user_id="user_456",
            memory_types=["episodic", "semantic"],
            start_date=start,
            importance_min=0.5,
        )
        assert config.agent_id == "agent_123"
        assert config.user_id == "user_456"
        assert config.memory_types == ["episodic", "semantic"]
        assert config.start_date == start
        assert config.importance_min == 0.5


class TestAtlasSearchFilterConfig:
    """Tests for AtlasSearchFilterConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = AtlasSearchFilterConfig()
        assert config.agent_id is None
        assert config.user_id is None
        assert config.start_date is None
        assert config.end_date is None
        assert config.timestamp_field == "timestamp"
        assert config.equality_filters == {}
        assert config.in_filters == {}
        assert config.range_filters == {}

    def test_custom_timestamp_field(self):
        """Test custom timestamp field."""
        config = AtlasSearchFilterConfig(timestamp_field="created_at")
        assert config.timestamp_field == "created_at"


class TestLexicalPrefilterConfig:
    """Tests for LexicalPrefilterConfig dataclass."""

    def test_default_config(self):
        """Test default configuration is empty."""
        config = LexicalPrefilterConfig()
        assert config.text_filters == []
        assert config.phrase_filters == []
        assert config.wildcard_filters == []
        assert config.range_filters == {}
        assert config.geo_filters == []


class TestHybridSearchConfig:
    """Tests for HybridSearchConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HybridSearchConfig()
        assert config.vector_index_name == "vector_index"
        assert config.vector_path == "embedding"
        assert config.num_candidates_multiplier == NUM_CANDIDATES_MULTIPLIER
        assert config.text_index_name == "text_search_index"
        assert config.text_search_path == "content"
        assert config.fuzzy_max_edits == 2
        assert config.fuzzy_prefix_length == 3
        assert config.vector_weight == 0.6
        assert config.text_weight == 0.4
        assert config.use_rank_fusion is True
        assert config.rrf_constant == 60
        assert config.cosine_threshold == 0.3
        assert config.over_fetch_multiplier == 2
        assert config.enable_tier_fallback is True
        assert config.detected_tier == SearchTier.M10_PLUS

    def test_custom_weights(self):
        """Test custom weights configuration."""
        config = HybridSearchConfig(
            vector_weight=0.7,
            text_weight=0.3,
        )
        assert config.vector_weight == 0.7
        assert config.text_weight == 0.3
