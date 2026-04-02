"""
NL Query API routes.
"""

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


class NLQueryRequest(BaseModel):
    """Request model for natural language query."""

    question: str = Field(..., description="Natural language question")
    agent_id: str = Field(..., description="Agent ID for query scoping")
    collection: str | None = Field(
        default=None,
        description="Optional target collection (defaults to safe bounded default)",
    )


class NLQueryResponse(BaseModel):
    """Response model for natural language query."""

    generated_mql: dict[str, Any] | None = Field(default=None)
    results: list[dict[str, Any]] = Field(default_factory=list)
    execution_time: float = Field(default=0.0)
    error: str | None = Field(default=None)
    message: str | None = Field(default=None)


def _build_query_llm(settings):
    provider = settings.default_agent_provider
    model_name = settings.default_agent_model

    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            temperature=0,
            max_tokens=settings.default_agent_max_tokens,
        )

    if provider == "google" and os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_output_tokens=settings.default_agent_max_tokens,
        )

    if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model_name,
            temperature=0,
            max_tokens=settings.default_agent_max_tokens,
        )

    return None


@router.post("", response_model=NLQueryResponse)
async def nl_query(request: NLQueryRequest, app_request: Request) -> NLQueryResponse:
    """Execute a natural language query against MongoDB."""
    runtime = getattr(app_request.app.state, "runtime", None)
    if runtime is None:
        return NLQueryResponse(
            generated_mql=None,
            results=[],
            execution_time=0.0,
            error=None,
            message="NL-to-MQL service is not configured",
        )

    generator = runtime.nl_query_generator
    if generator is None:
        return NLQueryResponse(
            generated_mql=None,
            results=[],
            execution_time=0.0,
            error=None,
            message=runtime.startup_error or "NL-to-MQL service is not configured",
        )

    llm = _build_query_llm(runtime.settings)
    if llm is None:
        return NLQueryResponse(
            generated_mql=None,
            results=[],
            execution_time=0.0,
            error=None,
            message=(
                "NL-to-MQL requires a configured LLM provider. "
                "Set an OpenAI, Google, or Anthropic API key."
            ),
        )

    try:
        result = await generator.generate_query(
            question=request.question,
            agent_id=request.agent_id,
            llm=llm,
            collection_name=request.collection,
        )
        return NLQueryResponse(**result)
    except Exception as exc:
        logger.error("NL query failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NL query failed: {str(exc)}",
        ) from exc
