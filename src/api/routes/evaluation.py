"""
Evaluation API Routes.

POST /api/v1/evaluate - Evaluate RAG pipeline quality.
Accepts question, answer, contexts, and optional ground_truth.
Returns precision, recall, relevancy, faithfulness scores.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ...evaluation.evaluator import RAGEvaluator

logger = logging.getLogger(__name__)

router = APIRouter()


class EvaluateRequest(BaseModel):
    """Request model for RAG evaluation."""

    question: str = Field(..., description="The user's question")
    answer: str = Field(..., description="The agent's answer")
    contexts: list[str] = Field(..., description="Retrieved context strings")
    ground_truth: str | None = Field(
        default=None, description="Optional ground truth answer"
    )


class EvaluateResponse(BaseModel):
    """Response model for RAG evaluation."""

    context_precision: float = Field(description="Retrieval precision (0-1)")
    context_recall: float = Field(description="Retrieval recall (0-1)")
    answer_relevancy: float = Field(description="Answer relevancy (0-1)")
    faithfulness: float = Field(description="Answer faithfulness (0-1)")
    overall_score: float = Field(description="Weighted average score (0-1)")
    metadata: dict[str, Any] = Field(default_factory=dict)


@router.post("", response_model=EvaluateResponse)
async def evaluate_rag(request: EvaluateRequest) -> EvaluateResponse:
    """Evaluate a RAG response for quality metrics.

    Runs LLM-as-judge evaluation when an LLM is available.
    Returns zero scores when no LLM is configured (graceful degradation).

    Args:
        request: Evaluation request with question, answer, contexts.

    Returns:
        Evaluation scores for all four metrics.
    """
    try:
        evaluator = RAGEvaluator()
        result = await evaluator.evaluate(
            question=request.question,
            answer=request.answer,
            contexts=request.contexts,
            ground_truth=request.ground_truth,
        )
        return EvaluateResponse(
            context_precision=result.context_precision,
            context_recall=result.context_recall,
            answer_relevancy=result.answer_relevancy,
            faithfulness=result.faithfulness,
            overall_score=result.overall_score,
            metadata=result.metadata,
        )
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}",
        ) from e
