"""
Atlas Search Filter Builder for $search stage.

$search uses Atlas-specific operators that are DIFFERENT from MQL.
This module builds filters for compound.filter clauses in $search.

CRITICAL: Do NOT use MQL operators here - they will cause errors.
"""

from typing import Any

from ..config import AtlasSearchFilterConfig


def build_atlas_search_filters(
    config: AtlasSearchFilterConfig | None = None, **kwargs: Any
) -> list[dict[str, Any]]:
    """
    Build Atlas Search-compatible filters for $search.compound.filter.

    Args:
        config: Filter configuration object
        **kwargs: Override config values

    Returns:
        List of Atlas Search filter clauses

    Example output:
        [
            {"equals": {"path": "agent_id", "value": "agent_123"}},
            {"range": {"path": "timestamp", "gte": datetime(...)}}
        ]
    """
    filters: list[dict[str, Any]] = []

    # Use config or create empty one
    if config is None:
        config = AtlasSearchFilterConfig()

    # Override with kwargs
    agent_id = kwargs.get("agent_id", config.agent_id)
    user_id = kwargs.get("user_id", config.user_id)
    start_date = kwargs.get("start_date", config.start_date)
    end_date = kwargs.get("end_date", config.end_date)
    timestamp_field = kwargs.get("timestamp_field", config.timestamp_field)

    # Build filters using Atlas operators
    if agent_id:
        filters.append({"equals": {"path": "agent_id", "value": agent_id}})

    if user_id:
        filters.append({"equals": {"path": "user_id", "value": user_id}})

    # Date range filter
    if start_date or end_date:
        range_filter = {"path": timestamp_field}
        if start_date:
            range_filter["gte"] = start_date
        if end_date:
            range_filter["lte"] = end_date
        filters.append({"range": range_filter})

    # Add generic equality filters
    for field, value in config.equality_filters.items():
        filters.append({"equals": {"path": field, "value": value}})

    # Add generic $in filters (using compound.should with minimumShouldMatch)
    for field, values in config.in_filters.items():
        if values:
            should_clauses = [{"equals": {"path": field, "value": v}} for v in values]
            filters.append({"compound": {"should": should_clauses, "minimumShouldMatch": 1}})

    # Add generic range filters
    for field, range_spec in config.range_filters.items():
        range_filter = {"path": field}
        range_filter.update(range_spec)  # e.g., {"gte": 10, "lt": 100}
        filters.append({"range": range_filter})

    return filters


def wrap_in_compound_filter(
    filters: list[dict[str, Any]],
    must_clauses: list[dict[str, Any]] | None = None,
    should_clauses: list[dict[str, Any]] | None = None,
    must_not_clauses: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Wrap filter clauses in a compound operator.

    Args:
        filters: Filter clauses (applied as filter, not scoring)
        must_clauses: Must match (affects scoring)
        should_clauses: Should match (affects scoring)
        must_not_clauses: Must not match

    Returns:
        Compound operator dict for $search
    """
    compound: dict[str, Any] = {}

    if filters:
        compound["filter"] = filters

    if must_clauses:
        compound["must"] = must_clauses

    if should_clauses:
        compound["should"] = should_clauses

    if must_not_clauses:
        compound["mustNot"] = must_not_clauses

    return {"compound": compound}
