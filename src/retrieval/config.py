"""
MongoDB Search Configuration Classes.

Separates MQL operators (for $vectorSearch) from Atlas operators (for $search).
This is CRITICAL - mixing them causes runtime errors.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SearchTier(Enum):
    """MongoDB Atlas cluster tier capabilities."""

    M10_PLUS = "m10_plus"  # Full $rankFusion support
    M0_M2 = "m0_m2"  # Manual RRF required
    VECTOR_ONLY = "vector_only"  # Fallback


# Best practice multiplier for numCandidates
NUM_CANDIDATES_MULTIPLIER = 20


@dataclass
class VectorSearchFilterConfig:
    """
    Filters for $vectorSearch stage - uses MQL operators.

    IMPORTANT: These filters use standard MongoDB query operators:
    - $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin

    Example:
        {"agent_id": {"$eq": "agent_123"}, "timestamp": {"$gte": some_date}}
    """

    agent_id: str | None = None
    user_id: str | None = None
    memory_types: list[str] | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    importance_min: float | None = None
    importance_max: float | None = None
    tags: list[str] | None = None
    thread_id: str | None = None
    # Generic filters for extensibility
    equality_filters: dict[str, Any] = field(default_factory=dict)
    in_filters: dict[str, list[Any]] = field(default_factory=dict)
    comparison_filters: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class AtlasSearchFilterConfig:
    """
    Filters for $search stage - uses Atlas Search operators.

    IMPORTANT: These filters use Atlas-specific operators:
    - equals, range, text, phrase, wildcard, compound

    Example:
        {"equals": {"path": "agent_id", "value": "agent_123"}}
        {"range": {"path": "timestamp", "gte": some_date}}
    """

    agent_id: str | None = None
    user_id: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    timestamp_field: str = "timestamp"
    # Generic filters for extensibility
    equality_filters: dict[str, Any] = field(default_factory=dict)
    in_filters: dict[str, list[Any]] = field(default_factory=dict)
    range_filters: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class LexicalPrefilterConfig:
    """
    Filters for $search.vectorSearch (MongoDB 8.2+).

    Enables lexical prefiltering before vector search.
    Uses Atlas Search operators within vectorSearch.
    """

    text_filters: list[dict[str, Any]] = field(default_factory=list)
    phrase_filters: list[dict[str, Any]] = field(default_factory=list)
    wildcard_filters: list[dict[str, Any]] = field(default_factory=list)
    range_filters: dict[str, dict[str, Any]] = field(default_factory=dict)
    geo_filters: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class HybridSearchConfig:
    """Master configuration for hybrid search."""

    # Vector search settings
    vector_index_name: str = "vector_index"
    vector_path: str = "embedding"
    num_candidates_multiplier: int = NUM_CANDIDATES_MULTIPLIER

    # Text search settings
    text_index_name: str = "text_search_index"
    text_search_path: str = "content"
    fuzzy_max_edits: int = 2
    fuzzy_prefix_length: int = 3

    # Hybrid fusion settings
    vector_weight: float = 0.6
    text_weight: float = 0.4
    use_rank_fusion: bool = True
    rrf_constant: int = 60  # Standard RRF constant

    # Quality thresholds
    cosine_threshold: float = 0.3
    over_fetch_multiplier: int = 2

    # Tier support
    enable_tier_fallback: bool = True
    detected_tier: SearchTier = SearchTier.M10_PLUS
