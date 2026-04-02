"""
Agent management routes backed by the canonical LangGraph registry.
"""

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

router = APIRouter()


class CreateAgentRequest(BaseModel):
    """Request model for creating an agent."""

    name: str = Field(..., description="Agent name")
    description: str = Field(default="", description="Agent description")
    model_provider: str = Field(default="openai", description="LLM provider")
    model_name: str = Field(default="gpt-4o", description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=1, le=32000)
    system_prompt: str | None = Field(default=None, description="Custom system prompt")
    enable_streaming: bool = Field(default=True)
    database_name: str | None = Field(default=None, description="Optional database override")


class UpdateAgentRequest(BaseModel):
    """Update model for mutable agent config."""

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=32000)
    system_prompt: str | None = Field(default=None)
    enable_streaming: bool | None = Field(default=None)


class AgentResponse(BaseModel):
    """Response model for agent operations."""

    agent_id: str
    name: str
    description: str
    status: str
    created_at: str
    metadata: dict[str, Any]


def _to_response(agent) -> AgentResponse:
    return AgentResponse(
        agent_id=agent.agent_id,
        name=agent.name,
        description=agent.description,
        status=agent.status,
        created_at=agent.created_at.isoformat(),
        metadata={
            "model_provider": agent.model_provider,
            "model_name": agent.model_name,
            "temperature": agent.temperature,
            "max_tokens": agent.max_tokens,
            "enable_streaming": agent.enable_streaming,
            "database_name": agent.database_name,
            "updated_at": agent.updated_at.isoformat(),
        },
    )


@router.post("/", response_model=AgentResponse)
async def create_agent(request: CreateAgentRequest, app_request: Request) -> AgentResponse:
    """Create and persist a new agent definition."""
    registry = app_request.app.state.agent_registry
    metadata = await registry.create_agent(request.model_dump())
    return _to_response(metadata)


@router.get("/", response_model=list[AgentResponse])
async def list_agents(app_request: Request) -> list[AgentResponse]:
    """List all persisted agents."""
    registry = app_request.app.state.agent_registry
    return [_to_response(agent) for agent in await registry.list_agents()]


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str, app_request: Request) -> AgentResponse:
    """Get a persisted agent definition."""
    registry = app_request.app.state.agent_registry
    metadata = await registry.get_metadata(agent_id)
    if metadata is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )
    return _to_response(metadata)


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str, app_request: Request) -> dict[str, str]:
    """Delete an agent definition and any live runtime instance."""
    registry = app_request.app.state.agent_registry
    deleted = await registry.delete_agent(agent_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )

    return {"status": "deleted", "agent_id": agent_id, "timestamp": datetime.now(UTC).isoformat()}


@router.put("/{agent_id}/config", response_model=AgentResponse)
async def update_agent_config(
    agent_id: str,
    request: UpdateAgentRequest,
    app_request: Request,
) -> AgentResponse:
    """Update mutable configuration for an agent."""
    registry = app_request.app.state.agent_registry
    try:
        updated = await registry.update_agent(agent_id, request.model_dump(exclude_none=True))
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        ) from exc

    return _to_response(updated)
