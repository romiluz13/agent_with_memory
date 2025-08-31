"""
Agent Management Routes
Create, configure, and manage agents
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends, Request
from pydantic import BaseModel, Field

from ...core.agent import AgentConfig, BaseAgent
from ...memory.manager import MemoryManager

router = APIRouter()

# In-memory agent storage (in production, use database)
active_agents: Dict[str, BaseAgent] = {}


class CreateAgentRequest(BaseModel):
    """Request model for creating an agent."""
    name: str = Field(..., description="Agent name")
    description: str = Field(default="", description="Agent description")
    model_provider: str = Field(default="openai", description="LLM provider")
    model_name: str = Field(default="gpt-4o", description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=1, le=8000)
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt")
    enable_streaming: bool = Field(default=True)


class AgentResponse(BaseModel):
    """Response model for agent operations."""
    agent_id: str
    name: str
    description: str
    status: str
    created_at: str
    metadata: Dict[str, Any]


@router.post("/", response_model=AgentResponse)
async def create_agent(
    request: CreateAgentRequest,
    app_request: Request
) -> AgentResponse:
    """
    Create a new agent instance.
    
    Args:
        request: Agent creation parameters
        app_request: FastAPI request object
        
    Returns:
        Created agent details
    """
    try:
        # Get memory manager from app state
        memory_manager: MemoryManager = app_request.app.state.memory_manager
        
        # Create agent configuration
        config = AgentConfig(
            name=request.name,
            description=request.description,
            model_provider=request.model_provider,
            model_name=request.model_name,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_prompt=request.system_prompt,
            enable_streaming=request.enable_streaming
        )
        
        # Create agent instance
        agent = BaseAgent(
            config=config,
            memory_manager=memory_manager
        )
        
        # Store agent
        active_agents[agent.agent_id] = agent
        
        return AgentResponse(
            agent_id=agent.agent_id,
            name=agent.config.name,
            description=agent.config.description,
            status="active",
            created_at=agent.created_at.isoformat(),
            metadata={
                "model_provider": agent.config.model_provider,
                "model_name": agent.config.model_name,
                "temperature": agent.config.temperature
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create agent: {str(e)}"
        )


@router.get("/", response_model=List[AgentResponse])
async def list_agents() -> List[AgentResponse]:
    """
    List all active agents.
    
    Returns:
        List of active agents
    """
    agents = []
    
    for agent_id, agent in active_agents.items():
        agents.append(AgentResponse(
            agent_id=agent_id,
            name=agent.config.name,
            description=agent.config.description,
            status="active",
            created_at=agent.created_at.isoformat(),
            metadata={
                "model_provider": agent.config.model_provider,
                "model_name": agent.config.model_name,
                "conversation_count": agent.conversation_count
            }
        ))
    
    return agents


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str) -> AgentResponse:
    """
    Get details of a specific agent.
    
    Args:
        agent_id: Agent identifier
        
    Returns:
        Agent details
    """
    if agent_id not in active_agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )
    
    agent = active_agents[agent_id]
    
    return AgentResponse(
        agent_id=agent_id,
        name=agent.config.name,
        description=agent.config.description,
        status="active",
        created_at=agent.created_at.isoformat(),
        metadata={
            "model_provider": agent.config.model_provider,
            "model_name": agent.config.model_name,
            "temperature": agent.config.temperature,
            "max_tokens": agent.config.max_tokens,
            "conversation_count": agent.conversation_count,
            "enable_streaming": agent.config.enable_streaming
        }
    )


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str) -> Dict[str, str]:
    """
    Delete an agent.
    
    Args:
        agent_id: Agent identifier
        
    Returns:
        Deletion confirmation
    """
    if agent_id not in active_agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )
    
    # Remove agent
    del active_agents[agent_id]
    
    return {
        "status": "deleted",
        "agent_id": agent_id,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.put("/{agent_id}/config")
async def update_agent_config(
    agent_id: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None
) -> AgentResponse:
    """
    Update agent configuration.
    
    Args:
        agent_id: Agent identifier
        temperature: New temperature setting
        max_tokens: New max tokens setting
        system_prompt: New system prompt
        
    Returns:
        Updated agent details
    """
    if agent_id not in active_agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )
    
    agent = active_agents[agent_id]
    
    # Update configuration
    if temperature is not None:
        agent.config.temperature = temperature
        agent.llm.temperature = temperature
    
    if max_tokens is not None:
        agent.config.max_tokens = max_tokens
        agent.llm.max_tokens = max_tokens
    
    if system_prompt is not None:
        agent.config.system_prompt = system_prompt
    
    return AgentResponse(
        agent_id=agent_id,
        name=agent.config.name,
        description=agent.config.description,
        status="active",
        created_at=agent.created_at.isoformat(),
        metadata={
            "model_provider": agent.config.model_provider,
            "model_name": agent.config.model_name,
            "temperature": agent.config.temperature,
            "max_tokens": agent.config.max_tokens,
            "updated_at": datetime.utcnow().isoformat()
        }
    )
