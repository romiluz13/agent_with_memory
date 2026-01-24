"""
Lexical Prefilter Builder for $search.vectorSearch (MongoDB 8.2+).

$search.vectorSearch allows applying Atlas Search filters BEFORE vector search,
enabling powerful lexical prefiltering that can dramatically reduce the search space.

NOTE: This feature requires MongoDB 8.2+ and will not work on older versions.
The system should gracefully fall back to standard $vectorSearch when unavailable.
"""

import logging
from typing import TYPE_CHECKING, Any

from ..config import LexicalPrefilterConfig

if TYPE_CHECKING:
    from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


def build_lexical_prefilters(
    config: LexicalPrefilterConfig | None = None,
) -> dict[str, Any]:
    """
    Build lexical prefilters for $search.vectorSearch.

    Args:
        config: Lexical prefilter configuration

    Returns:
        Filter dict for vectorSearch.filter field

    Example output:
        {
            "compound": {
                "filter": [
                    {"text": {"path": "content", "query": "python"}},
                    {"range": {"path": "timestamp", "gte": datetime(...)}}
                ]
            }
        }
    """
    if config is None:
        return {}

    filter_clauses: list[dict[str, Any]] = []

    # Text filters (fuzzy matching)
    for text_filter in config.text_filters:
        query = text_filter.get("query", "")
        if not query:
            continue

        clause: dict[str, Any] = {
            "text": {
                "path": text_filter.get("path", "content"),
                "query": query,
            }
        }
        # Add fuzzy options if specified
        if text_filter.get("fuzzy", False):
            clause["text"]["fuzzy"] = {
                "maxEdits": text_filter.get("max_edits", 2),
                "prefixLength": text_filter.get("prefix_length", 3),
            }
        filter_clauses.append(clause)

    # Phrase filters (exact phrase matching)
    for phrase_filter in config.phrase_filters:
        query = phrase_filter.get("query", "")
        if not query:
            continue

        clause = {
            "phrase": {
                "path": phrase_filter.get("path", "content"),
                "query": query,
            }
        }
        if "slop" in phrase_filter:
            clause["phrase"]["slop"] = phrase_filter["slop"]
        filter_clauses.append(clause)

    # Wildcard filters (pattern matching)
    for wildcard_filter in config.wildcard_filters:
        query = wildcard_filter.get("query", "")
        if not query:
            continue

        clause = {
            "wildcard": {
                "path": wildcard_filter.get("path", "content"),
                "query": query,
                "allowAnalyzedField": wildcard_filter.get("allow_analyzed", True),
            }
        }
        filter_clauses.append(clause)

    # Range filters
    for field, range_spec in config.range_filters.items():
        clause = {"range": {"path": field, **range_spec}}  # e.g., {"gte": 10, "lte": 100}
        filter_clauses.append(clause)

    # Geo filters (if any)
    for geo_filter in config.geo_filters:
        # Geo near filter
        if "center" in geo_filter and "radius" in geo_filter:
            clause = {
                "geoWithin": {
                    "path": geo_filter.get("path", "location"),
                    "circle": {
                        "center": geo_filter["center"],  # [lng, lat]
                        "radius": geo_filter["radius"],  # in meters
                    },
                }
            }
            filter_clauses.append(clause)

    # Wrap in compound if we have filters
    if not filter_clauses:
        return {}

    return {"compound": {"filter": filter_clauses}}


def build_search_vector_search_stage(
    index_name: str,
    query_vector: list[float],
    vector_path: str = "embedding",
    limit: int = 10,
    num_candidates_multiplier: int = 20,
    lexical_filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build complete $search.vectorSearch stage.

    This is the MongoDB 8.2+ syntax that combines Atlas Search filtering
    with vector search in a single stage.

    Args:
        index_name: Name of the search index
        query_vector: Query embedding vector
        vector_path: Path to vector field in documents
        limit: Number of results
        num_candidates_multiplier: Multiplier for numCandidates
        lexical_filters: Precomputed lexical filters

    Returns:
        $search stage dict
    """
    num_candidates = limit * num_candidates_multiplier

    vector_search: dict[str, Any] = {
        "queryVector": query_vector,
        "path": vector_path,
        "numCandidates": num_candidates,
        "limit": limit,
    }

    # Add lexical prefilters if provided
    if lexical_filters:
        vector_search["filter"] = lexical_filters

    return {
        "$search": {
            "index": index_name,
            "vectorSearch": vector_search,
        }
    }


async def check_lexical_prefilter_support(db: "AsyncIOMotorDatabase") -> bool:
    """
    Check if the MongoDB version supports $search.vectorSearch.

    Requires MongoDB 8.2+.

    Args:
        db: MongoDB database instance

    Returns:
        True if lexical prefilters are supported
    """
    try:
        # Try to get server version
        server_info = await db.client.server_info()
        version = server_info.get("version", "0.0.0")

        # Parse version
        parts = version.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0

        # Requires MongoDB 8.2+
        if major > 8 or (major == 8 and minor >= 2):
            logger.info(f"MongoDB {version} supports lexical prefilters")
            return True

        logger.info(f"MongoDB {version} does not support lexical prefilters (requires 8.2+)")
        return False

    except Exception as e:
        logger.warning(f"Could not determine MongoDB version: {e}")
        return False


# Cache for version check
_LEXICAL_PREFILTER_SUPPORTED: bool | None = None


async def get_lexical_prefilter_support(
    db: "AsyncIOMotorDatabase", force_check: bool = False
) -> bool:
    """
    Get cached lexical prefilter support status.

    Args:
        db: MongoDB database instance
        force_check: Force re-check even if cached

    Returns:
        True if lexical prefilters are supported
    """
    global _LEXICAL_PREFILTER_SUPPORTED

    if _LEXICAL_PREFILTER_SUPPORTED is None or force_check:
        _LEXICAL_PREFILTER_SUPPORTED = await check_lexical_prefilter_support(db)

    return _LEXICAL_PREFILTER_SUPPORTED
