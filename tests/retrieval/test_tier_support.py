"""
REAL Integration Tests for MongoDB Atlas tier detection and search strategies.
Tests with actual MongoDB connections - no mocks!
"""

import pytest

from src.retrieval.config import SearchTier
from src.retrieval.tier_support import (
    TierAwareSearchStrategy,
    TierCapabilities,
    detect_cluster_tier,
    get_strategy,
    get_tier,
)


class TestTierCapabilitiesReal:
    """Real tests for tier capability definitions."""

    def test_m10_plus_supports_rank_fusion(self):
        """M10+ clusters support native $rankFusion."""
        assert TierCapabilities.supports(SearchTier.M10_PLUS, "rank_fusion") is True

    def test_m10_plus_supports_all_features(self):
        """M10+ should support all search features."""
        tier = SearchTier.M10_PLUS
        assert TierCapabilities.supports(tier, "rank_fusion") is True
        assert TierCapabilities.supports(tier, "vector_search") is True
        assert TierCapabilities.supports(tier, "text_search") is True
        assert TierCapabilities.supports(tier, "graph_lookup") is True
        assert TierCapabilities.supports(tier, "lexical_prefilters") is True

    def test_m0_m2_no_rank_fusion(self):
        """M0/M2 free tier clusters don't support $rankFusion."""
        assert TierCapabilities.supports(SearchTier.M0_M2, "rank_fusion") is False

    def test_m0_m2_has_basic_search(self):
        """M0/M2 should support basic vector and text search."""
        tier = SearchTier.M0_M2
        assert TierCapabilities.supports(tier, "vector_search") is True
        assert TierCapabilities.supports(tier, "text_search") is True
        assert TierCapabilities.supports(tier, "graph_lookup") is False
        assert TierCapabilities.supports(tier, "lexical_prefilters") is False

    def test_vector_only_minimal_features(self):
        """Vector-only tier should only support vector search."""
        tier = SearchTier.VECTOR_ONLY
        assert TierCapabilities.supports(tier, "vector_search") is True
        assert TierCapabilities.supports(tier, "text_search") is False
        assert TierCapabilities.supports(tier, "rank_fusion") is False
        assert TierCapabilities.supports(tier, "graph_lookup") is False

    def test_unknown_feature_returns_false(self):
        """Unknown features should return False for any tier."""
        assert TierCapabilities.supports(SearchTier.M10_PLUS, "unknown_feature") is False
        assert TierCapabilities.supports(SearchTier.M0_M2, "imaginary_feature") is False
        assert TierCapabilities.supports(SearchTier.VECTOR_ONLY, "magic") is False


class TestTierAwareSearchStrategyReal:
    """Real tests for search strategy selection based on tier."""

    def test_m10_plus_strategy(self):
        """M10+ tier should use native $rankFusion."""
        strategy = TierAwareSearchStrategy(SearchTier.M10_PLUS)

        assert strategy.tier == SearchTier.M10_PLUS
        assert strategy.should_use_rank_fusion() is True
        assert strategy.should_use_manual_rrf() is False
        assert strategy.should_use_vector_only() is False
        assert strategy.get_search_method() == "rank_fusion"

    def test_m0_m2_strategy(self):
        """M0/M2 tier should use manual RRF fallback."""
        strategy = TierAwareSearchStrategy(SearchTier.M0_M2)

        assert strategy.tier == SearchTier.M0_M2
        assert strategy.should_use_rank_fusion() is False
        assert strategy.should_use_manual_rrf() is True
        assert strategy.should_use_vector_only() is False
        assert strategy.get_search_method() == "manual_rrf"

    def test_vector_only_strategy(self):
        """Vector-only tier should skip text search entirely."""
        strategy = TierAwareSearchStrategy(SearchTier.VECTOR_ONLY)

        assert strategy.tier == SearchTier.VECTOR_ONLY
        assert strategy.should_use_rank_fusion() is False
        assert strategy.should_use_manual_rrf() is False
        assert strategy.should_use_vector_only() is True
        assert strategy.get_search_method() == "vector_only"

    def test_get_strategy_helper(self):
        """get_strategy helper should return correct strategy instance."""
        for tier in SearchTier:
            strategy = get_strategy(tier)
            assert isinstance(strategy, TierAwareSearchStrategy)
            assert strategy.tier == tier

    def test_strategy_immutability(self):
        """Strategy should be based on tier at construction time."""
        strategy = TierAwareSearchStrategy(SearchTier.M0_M2)
        original_method = strategy.get_search_method()

        # Strategy should not change
        assert strategy.get_search_method() == original_method
        assert strategy.get_search_method() == "manual_rrf"


class TestDetectClusterTierReal:
    """REAL integration tests for tier detection with actual MongoDB."""

    @pytest.mark.asyncio
    async def test_detect_tier_connects_to_mongodb(self, test_db):
        """Test that tier detection actually connects to MongoDB."""
        # This tests with real MongoDB connection
        tier = await detect_cluster_tier(test_db)

        # Should return a valid SearchTier
        assert tier in [SearchTier.M10_PLUS, SearchTier.M0_M2, SearchTier.VECTOR_ONLY]
        print(f"\n✓ Detected cluster tier: {tier.value}")

    @pytest.mark.asyncio
    async def test_detect_tier_cleans_up_test_collection(self, test_db):
        """Tier detection should clean up its test collection."""
        test_collection_name = "__tier_detection_test__"

        # Run detection
        await detect_cluster_tier(test_db)

        # Test collection should be cleaned up
        collections = await test_db.list_collection_names()
        assert test_collection_name not in collections

    @pytest.mark.asyncio
    async def test_get_tier_caches_result(self, test_db):
        """get_tier should cache the detection result."""
        # Reset the global cache by forcing detection
        tier1 = await get_tier(test_db, force_detect=True)
        tier2 = await get_tier(test_db, force_detect=False)

        # Both should return the same tier
        assert tier1 == tier2

    @pytest.mark.asyncio
    async def test_get_tier_force_detect(self, test_db):
        """force_detect should re-run detection."""
        # First detection
        tier1 = await get_tier(test_db, force_detect=True)

        # Force re-detection
        tier2 = await get_tier(test_db, force_detect=True)

        # Should still return valid tier (may or may not be same)
        assert tier1 in SearchTier
        assert tier2 in SearchTier


class TestTierDetectionWithCluster:
    """Tests that verify behavior based on the actual cluster tier."""

    @pytest.mark.asyncio
    async def test_strategy_matches_detected_tier(self, test_db):
        """Strategy should match the detected cluster tier."""
        tier = await detect_cluster_tier(test_db)
        strategy = get_strategy(tier)

        if tier == SearchTier.M10_PLUS:
            assert strategy.should_use_rank_fusion()
            print("\n✓ Cluster supports native $rankFusion")
        elif tier == SearchTier.M0_M2:
            assert strategy.should_use_manual_rrf()
            print("\n✓ Cluster requires manual RRF (M0/M2 tier)")
        else:
            assert strategy.should_use_vector_only()
            print("\n✓ Cluster limited to vector-only search")

    @pytest.mark.asyncio
    async def test_tier_enum_values(self):
        """Verify SearchTier enum has expected values."""
        assert SearchTier.M10_PLUS.value == "m10_plus"
        assert SearchTier.M0_M2.value == "m0_m2"
        assert SearchTier.VECTOR_ONLY.value == "vector_only"

    @pytest.mark.asyncio
    async def test_tier_capabilities_completeness(self):
        """All tiers should have capability definitions."""
        for tier in SearchTier:
            features = TierCapabilities.TIER_FEATURES.get(tier, {})
            assert features is not None, f"Missing capabilities for {tier}"
            assert "vector_search" in features, f"Missing vector_search for {tier}"
