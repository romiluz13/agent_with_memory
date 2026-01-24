"""
MQL Filter Builder for $vectorSearch.

$vectorSearch uses standard MongoDB query operators (MQL).
This module builds filters that are safe for the $vectorSearch.filter field.

CRITICAL: Do NOT use Atlas Search operators here - they will cause errors.
"""

from typing import Any

from ..config import VectorSearchFilterConfig


def build_vector_search_filters(
    config: VectorSearchFilterConfig | None = None, **kwargs: Any
) -> dict[str, Any]:
    """
    Build MQL-compatible filters for $vectorSearch stage.

    Args:
        config: Filter configuration object
        **kwargs: Override config values (agent_id, user_id, etc.)

    Returns:
        Dict of MQL filters safe for $vectorSearch.filter

    Example output:
        {
            "agent_id": {"$eq": "agent_123"},
            "timestamp": {"$gte": datetime(...), "$lte": datetime(...)},
            "memory_type": {"$in": ["episodic", "semantic"]}
        }
    """
    filters: dict[str, Any] = {}

    # Use config or create empty one
    if config is None:
        config = VectorSearchFilterConfig()

    # Override with kwargs
    agent_id = kwargs.get("agent_id", config.agent_id)
    user_id = kwargs.get("user_id", config.user_id)
    memory_types = kwargs.get("memory_types", config.memory_types)
    start_date = kwargs.get("start_date", config.start_date)
    end_date = kwargs.get("end_date", config.end_date)
    importance_min = kwargs.get("importance_min", config.importance_min)
    importance_max = kwargs.get("importance_max", config.importance_max)
    tags = kwargs.get("tags", config.tags)
    thread_id = kwargs.get("thread_id", config.thread_id)

    # Build filters using MQL operators
    if agent_id:
        filters["agent_id"] = {"$eq": agent_id}

    if user_id:
        filters["user_id"] = {"$eq": user_id}

    if thread_id:
        filters["thread_id"] = {"$eq": thread_id}

    if memory_types:
        filters["memory_type"] = {"$in": memory_types}

    if tags:
        filters["metadata.tags"] = {"$in": tags}

    # Date range filter
    if start_date or end_date:
        timestamp_filter = {}
        if start_date:
            timestamp_filter["$gte"] = start_date
        if end_date:
            timestamp_filter["$lte"] = end_date
        if timestamp_filter:
            filters["timestamp"] = timestamp_filter

    # Importance range filter
    if importance_min is not None or importance_max is not None:
        importance_filter = {}
        if importance_min is not None:
            importance_filter["$gte"] = importance_min
        if importance_max is not None:
            importance_filter["$lte"] = importance_max
        if importance_filter:
            filters["importance"] = importance_filter

    # Add generic equality filters
    for field, value in config.equality_filters.items():
        filters[field] = {"$eq": value}

    # Add generic $in filters
    for field, values in config.in_filters.items():
        filters[field] = {"$in": values}

    # Add generic comparison filters
    for field, comparisons in config.comparison_filters.items():
        filters[field] = comparisons  # e.g., {"$gte": 10, "$lt": 100}

    return filters


def simplify_filters_for_basic_search(filters: dict[str, Any]) -> dict[str, Any]:
    """
    Convert MQL filters to simple equality filters for basic search.

    Some MongoDB operations don't support full MQL in filters.
    This simplifies {"field": {"$eq": "value"}} to {"field": "value"}.

    Args:
        filters: MQL-style filters

    Returns:
        Simplified filters safe for basic filter parameter
    """
    simplified = {}

    for field, condition in filters.items():
        if isinstance(condition, dict):
            # Extract $eq value if that's the only operator
            if list(condition.keys()) == ["$eq"]:
                simplified[field] = condition["$eq"]
            else:
                # Keep complex filters as-is (may cause errors in some contexts)
                simplified[field] = condition
        else:
            simplified[field] = condition

    return simplified
