"""
Score Details Parser for MongoDB $rankFusion.

Parses the scoreDetails metadata from $rankFusion to extract
per-pipeline scores for observability and debugging.
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PipelineScore:
    """Score from a single pipeline in $rankFusion."""

    pipeline_name: str
    score: float
    rank: int | None = None
    weight: float | None = None
    weighted_contribution: float | None = None


@dataclass
class ParsedScoreDetails:
    """Parsed score details from $rankFusion."""

    total_score: float
    pipeline_scores: list[PipelineScore]
    fusion_method: str = "rank_fusion"
    raw_details: dict | None = None


def parse_score_details(
    score_details: dict[str, Any] | None,
) -> ParsedScoreDetails | None:
    """
    Parse $rankFusion scoreDetails into structured format.

    The scoreDetails structure from MongoDB:
    {
        "value": 0.85,  # Combined score
        "description": "weighted combination",
        "details": [
            {
                "value": 0.9,
                "description": "vectorPipeline",
                "details": [...]
            },
            {
                "value": 0.7,
                "description": "textPipeline",
                "details": [...]
            }
        ]
    }

    Args:
        score_details: Raw scoreDetails from MongoDB
        pipeline_names: Expected pipeline names for validation

    Returns:
        ParsedScoreDetails or None if parsing fails
    """
    if not score_details:
        return None

    try:
        total_score = score_details.get("value", 0.0)
        details = score_details.get("details", [])

        pipeline_scores = []
        for detail in details:
            if isinstance(detail, dict):
                pipeline_name = detail.get("description", "unknown")
                pipeline_score = detail.get("value", 0.0)

                # Try to extract weight if available
                weight = None
                nested_details = detail.get("details", [])
                for nested in nested_details:
                    if isinstance(nested, dict) and "weight" in str(nested):
                        # Extract weight from description
                        desc = nested.get("description", "")
                        if "weight" in desc.lower():
                            try:
                                weight = float(desc.split("=")[-1].strip())
                            except (ValueError, IndexError):
                                pass

                weighted_contribution = pipeline_score * (weight or 1.0)

                pipeline_scores.append(
                    PipelineScore(
                        pipeline_name=pipeline_name,
                        score=pipeline_score,
                        weight=weight,
                        weighted_contribution=weighted_contribution,
                    )
                )

        return ParsedScoreDetails(
            total_score=total_score,
            pipeline_scores=pipeline_scores,
            fusion_method="rank_fusion",
            raw_details=score_details,
        )

    except Exception as e:
        logger.warning(f"Failed to parse score details: {e}")
        return None


def format_score_summary(parsed: ParsedScoreDetails) -> str:
    """
    Format parsed scores for logging/debugging.

    Returns:
        Human-readable score summary
    """
    lines = [f"Total Score: {parsed.total_score:.4f}"]

    for ps in parsed.pipeline_scores:
        weight_str = f" (weight={ps.weight:.2f})" if ps.weight else ""
        lines.append(f"  {ps.pipeline_name}: {ps.score:.4f}{weight_str}")

    return "\n".join(lines)


def extract_vector_score(parsed: ParsedScoreDetails) -> float | None:
    """Extract vector pipeline score."""
    for ps in parsed.pipeline_scores:
        if "vector" in ps.pipeline_name.lower():
            return ps.score
    return None


def extract_text_score(parsed: ParsedScoreDetails) -> float | None:
    """Extract text pipeline score."""
    for ps in parsed.pipeline_scores:
        if "text" in ps.pipeline_name.lower():
            return ps.score
    return None


def get_dominant_pipeline(parsed: ParsedScoreDetails) -> str | None:
    """
    Determine which pipeline contributed most to the final score.

    Returns:
        Name of the dominant pipeline, or None if no scores
    """
    if not parsed.pipeline_scores:
        return None

    max_score = -1.0
    dominant = None

    for ps in parsed.pipeline_scores:
        contribution = ps.weighted_contribution or ps.score
        if contribution > max_score:
            max_score = contribution
            dominant = ps.pipeline_name

    return dominant


def parse_rrf_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Extract RRF-specific information from result metadata.

    Args:
        metadata: Result metadata dict

    Returns:
        Dict with RRF info (source_scores, source_ranks, fusion_method)
    """
    rrf_info = {
        "fusion_method": metadata.get("fusion_method"),
        "rrf_score": metadata.get("rrf_score"),
        "source_scores": metadata.get("source_scores", {}),
        "source_ranks": metadata.get("source_ranks", {}),
    }

    # Calculate which source contributed most
    if rrf_info["source_scores"]:
        max_source = max(rrf_info["source_scores"].items(), key=lambda x: x[1])
        rrf_info["dominant_source"] = max_source[0]

    return rrf_info
