"""
Reciprocal Rank Fusion (RRF) Algorithm.

Used as fallback when native $rankFusion is not available (M0/M2 tiers).
Standard RRF formula: score = sum(1 / (k + rank)) for each result list.

Reference: https://dl.acm.org/doi/10.1145/1571941.1572114
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vector_search import SearchResult

logger = logging.getLogger(__name__)

# Standard RRF constant - affects how quickly scores decay with rank
DEFAULT_RRF_CONSTANT = 60


def reciprocal_rank_fusion(
    result_lists: dict[str, list["SearchResult"]],
    k: int = DEFAULT_RRF_CONSTANT,
    top_k: int | None = None,
    weights: dict[str, float] | None = None,
) -> list["SearchResult"]:
    """
    Merge multiple result lists using Reciprocal Rank Fusion.

    RRF formula: score = sum(weight * 1 / (k + rank))

    Args:
        result_lists: Dict mapping source name to results list
                      e.g., {"vector": [...], "text": [...]}
        k: RRF constant (default 60). Higher = more emphasis on top ranks.
        top_k: Maximum results to return (None = all)
        weights: Optional weights per source (default equal weights)

    Returns:
        Merged and sorted list of SearchResults

    Example:
        vector_results = [SearchResult(id="a", ...), SearchResult(id="b", ...)]
        text_results = [SearchResult(id="b", ...), SearchResult(id="c", ...)]
        merged = reciprocal_rank_fusion(
            {"vector": vector_results, "text": text_results},
            weights={"vector": 0.6, "text": 0.4}
        )
        # Returns merged results with combined RRF scores
    """
    # Import at runtime to avoid circular dependency
    from .vector_search import SearchResult

    if not result_lists:
        return []

    # Default to equal weights
    if weights is None:
        weights = dict.fromkeys(result_lists.keys(), 1.0)

    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {name: w / total_weight for name, w in weights.items()}

    # Track RRF scores and best result for each ID
    rrf_scores: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}
    source_info: dict[str, dict] = {}  # Track per-source info

    for source_name, results in result_lists.items():
        source_weight = weights.get(source_name, 1.0)

        for rank, result in enumerate(results):
            result_id = result.id

            # RRF contribution for this source
            rrf_contribution = source_weight * (1.0 / (k + rank))

            if result_id in rrf_scores:
                # Accumulate RRF score
                rrf_scores[result_id] += rrf_contribution
                # Track source info
                source_info[result_id]["scores"][source_name] = result.score
                source_info[result_id]["ranks"][source_name] = rank
            else:
                # First occurrence - initialize
                rrf_scores[result_id] = rrf_contribution
                result_map[result_id] = result
                source_info[result_id] = {
                    "scores": {source_name: result.score},
                    "ranks": {source_name: rank},
                }

    # Sort by combined RRF score
    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Apply top_k limit
    if top_k is not None:
        sorted_ids = sorted_ids[:top_k]

    # Build final results with RRF score as the score
    merged_results = []
    for result_id, rrf_score in sorted_ids:
        original = result_map[result_id]

        # Create new result with RRF score
        merged_result = SearchResult(
            id=original.id,
            content=original.content,
            metadata={
                **original.metadata,
                "rrf_score": rrf_score,
                "source_scores": source_info[result_id]["scores"],
                "source_ranks": source_info[result_id]["ranks"],
                "fusion_method": "rrf",
            },
            score=rrf_score,
        )
        merged_results.append(merged_result)

    logger.debug(
        f"RRF merged {sum(len(r) for r in result_lists.values())} results "
        f"from {len(result_lists)} sources into {len(merged_results)} results"
    )

    return merged_results


def weighted_score_fusion(
    result_lists: dict[str, list["SearchResult"]],
    weights: dict[str, float] | None = None,
    top_k: int | None = None,
) -> list["SearchResult"]:
    """
    Simple weighted score fusion (alternative to RRF).

    Normalizes scores within each source, then combines with weights.
    Useful when original scores are meaningful.

    Args:
        result_lists: Dict mapping source name to results
        weights: Weights per source (default equal)
        top_k: Maximum results to return

    Returns:
        Merged results sorted by weighted score
    """
    # Import at runtime to avoid circular dependency
    from .vector_search import SearchResult

    if not result_lists:
        return []

    # Default equal weights
    if weights is None:
        weights = dict.fromkeys(result_lists.keys(), 1.0)

    # Normalize weights - guard against zero total
    total_weight = sum(weights.values())
    if total_weight == 0:
        raise ValueError("Total weight cannot be zero - at least one weight must be positive")
    weights = {name: w / total_weight for name, w in weights.items()}

    # Normalize scores within each source (0-1 range)
    normalized_results: dict[str, dict[str, float]] = {}  # id -> {source: norm_score}
    result_map: dict[str, SearchResult] = {}

    for source_name, results in result_lists.items():
        if not results:
            continue

        # Find min/max scores for normalization
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score if max_score > min_score else 1.0

        for result in results:
            result_id = result.id
            norm_score = (result.score - min_score) / score_range

            if result_id not in normalized_results:
                normalized_results[result_id] = {}
                result_map[result_id] = result

            normalized_results[result_id][source_name] = norm_score

    # Calculate weighted scores
    weighted_scores: dict[str, float] = {}
    for result_id, source_scores in normalized_results.items():
        total = sum(
            source_scores.get(source, 0) * weights.get(source, 0) for source in weights.keys()
        )
        weighted_scores[result_id] = total

    # Sort and return
    sorted_ids = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
    if top_k:
        sorted_ids = sorted_ids[:top_k]

    return [
        SearchResult(
            id=result_map[rid].id,
            content=result_map[rid].content,
            metadata={
                **result_map[rid].metadata,
                "weighted_score": score,
                "fusion_method": "weighted",
            },
            score=score,
        )
        for rid, score in sorted_ids
    ]
