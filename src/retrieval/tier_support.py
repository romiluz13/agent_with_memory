"""
MongoDB Atlas Tier Support.

Detects cluster tier capabilities and provides appropriate search strategies.

Tier capabilities:
- M10+: Full $rankFusion support, all features
- M0/M2: No $rankFusion, requires manual RRF
- Vector-only: Fallback when text search unavailable
"""

import logging

from motor.motor_asyncio import AsyncIOMotorDatabase

from .config import SearchTier

logger = logging.getLogger(__name__)


class TierCapabilities:
    """Capabilities available at each tier."""

    TIER_FEATURES = {
        SearchTier.M10_PLUS: {
            "rank_fusion": True,
            "vector_search": True,
            "text_search": True,
            "graph_lookup": True,
            "lexical_prefilters": True,  # MongoDB 8.2+
        },
        SearchTier.M0_M2: {
            "rank_fusion": False,
            "vector_search": True,
            "text_search": True,
            "graph_lookup": False,
            "lexical_prefilters": False,
        },
        SearchTier.VECTOR_ONLY: {
            "rank_fusion": False,
            "vector_search": True,
            "text_search": False,
            "graph_lookup": False,
            "lexical_prefilters": False,
        },
    }

    @classmethod
    def supports(cls, tier: SearchTier, feature: str) -> bool:
        """Check if a tier supports a feature."""
        tier_features = cls.TIER_FEATURES.get(tier, {})
        return tier_features.get(feature, False)


async def detect_cluster_tier(db: AsyncIOMotorDatabase) -> SearchTier:
    """
    Detect MongoDB Atlas cluster tier by testing capabilities.

    Strategy:
    1. Try $rankFusion (M10+ only)
    2. Try $search (M0/M2 have this)
    3. Fallback to vector-only

    Args:
        db: MongoDB database instance

    Returns:
        Detected SearchTier
    """
    # Test collection for capability detection
    test_collection = db["__tier_detection_test__"]

    try:
        # Test 1: Try $rankFusion (M10+ feature)
        try:
            # This will fail fast if $rankFusion not supported
            pipeline = [
                {"$rankFusion": {"input": {"pipelines": {}}, "combination": {"weights": {}}}},
                {"$limit": 0},
            ]
            await test_collection.aggregate(pipeline).to_list(length=0)
            logger.info("Detected M10+ tier (supports $rankFusion)")
            return SearchTier.M10_PLUS
        except Exception as e:
            if "rankFusion" in str(e) or "not supported" in str(e).lower():
                logger.debug(f"$rankFusion not supported: {e}")
            # Continue to next test

        # Test 2: Try $search (M0/M2 have Atlas Search)
        try:
            pipeline = [
                {"$search": {"index": "default", "text": {"query": "test", "path": "content"}}},
                {"$limit": 0},
            ]
            await test_collection.aggregate(pipeline).to_list(length=0)
            logger.info("Detected M0/M2 tier (supports $search but not $rankFusion)")
            return SearchTier.M0_M2
        except Exception as e:
            if "index" in str(e).lower() or "search" in str(e).lower():
                # Index may not exist but $search is supported
                logger.debug(f"$search available (index error): {e}")
                return SearchTier.M0_M2

        # Fallback to vector-only
        logger.info("Detected vector-only tier (no text search)")
        return SearchTier.VECTOR_ONLY

    except Exception as e:
        logger.warning(f"Tier detection failed, assuming M10+: {e}")
        return SearchTier.M10_PLUS

    finally:
        # Clean up test collection
        try:
            await db.drop_collection("__tier_detection_test__")
        except Exception:
            pass


class TierAwareSearchStrategy:
    """
    Provides search strategy based on detected tier.
    """

    def __init__(self, tier: SearchTier):
        self.tier = tier
        self.capabilities = TierCapabilities()

    def should_use_rank_fusion(self) -> bool:
        """Check if we should use native $rankFusion."""
        return self.capabilities.supports(self.tier, "rank_fusion")

    def should_use_manual_rrf(self) -> bool:
        """Check if we need manual RRF fallback."""
        return not self.capabilities.supports(
            self.tier, "rank_fusion"
        ) and self.capabilities.supports(self.tier, "text_search")

    def should_use_vector_only(self) -> bool:
        """Check if we're limited to vector-only search."""
        return self.tier == SearchTier.VECTOR_ONLY

    def get_search_method(self) -> str:
        """Get the recommended search method name."""
        if self.should_use_rank_fusion():
            return "rank_fusion"
        elif self.should_use_manual_rrf():
            return "manual_rrf"
        else:
            return "vector_only"


# Global tier cache (detected once per process)
_detected_tier: SearchTier | None = None


async def get_tier(db: AsyncIOMotorDatabase, force_detect: bool = False) -> SearchTier:
    """
    Get the detected tier, with caching.

    Args:
        db: MongoDB database
        force_detect: Force re-detection

    Returns:
        Cached or newly detected tier
    """
    global _detected_tier

    if _detected_tier is None or force_detect:
        _detected_tier = await detect_cluster_tier(db)

    return _detected_tier


def get_strategy(tier: SearchTier) -> TierAwareSearchStrategy:
    """Get search strategy for a tier."""
    return TierAwareSearchStrategy(tier)
