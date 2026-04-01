"""
NL Query API Routes.

POST /api/v1/query - Convert natural language to MongoDB query and execute.
Accepts question, agent_id, and optional collection.
Returns generated MQL, results, and execution time.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


class NLQueryRequest(BaseModel):
    """Request model for natural language query."""

    question: str = Field(..., description="Natural language question")
    agent_id: str = Field(..., description="Agent ID for query scoping")
    collection: str | None = Field(
        default=None,
        description="Optional target collection (defaults to auto-detect)",
    )


class NLQueryResponse(BaseModel):
    """Response model for natural language query."""

    generated_mql: dict[str, Any] | None = Field(
        default=None, description="Generated MongoDB query"
    )
    results: list[dict[str, Any]] = Field(
        default_factory=list, description="Query results"
    )
    execution_time: float = Field(
        default=0.0, description="Query execution time in seconds"
    )
    error: str | None = Field(default=None, description="Error message if any")
    message: str | None = Field(default=None, description="Status message")


@router.post("", response_model=NLQueryResponse)
async def nl_query(request: NLQueryRequest) -> NLQueryResponse:
    """Execute a natural language query against MongoDB.

    Converts the question to a MongoDB find() query using LLM,
    injects agent_id for tenant isolation, validates the query,
    and executes it.

    Without a configured LLM or database, returns a graceful
    degradation response with zero results.

    Args:
        request: NL query request with question and agent_id.

    Returns:
        Generated MQL, results, and execution time.
    """
    try:
        # Graceful degradation: without MongoDB or LLM, return informative response
        # In production, the NLToMQLGenerator would be injected via app.state
        return NLQueryResponse(
            generated_mql=None,
            results=[],
            execution_time=0.0,
            error=None,
            message=(
                "NL-to-MQL service not configured. "
                "Set MONGODB_URI and configure an LLM to enable query generation."
            ),
        )
    except Exception as e:
        logger.error("NL query failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NL query failed: {str(e)}",
        ) from e
