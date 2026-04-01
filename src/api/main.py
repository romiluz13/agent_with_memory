"""
FastAPI Main Application
Production-ready API for AI Agent Boilerplate
"""

import importlib.util as _ilu
import os
import pathlib as _pathlib
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Load environment variables
load_dotenv()


# Request/Response models
class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict[str, str]


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    user_id: str = "anonymous"


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tokens_used: int = 0


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    print("🚀 Starting AI Agent Boilerplate API...")

    # Initialize services here if needed

    yield

    # Shutdown
    print("👋 Shutting down AI Agent Boilerplate API...")


# Create FastAPI app
app = FastAPI(
    title="AI Agent Boilerplate API",
    description="Production-ready AI agent with sophisticated memory system",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Route Registration (new feature routes) ---
# NOTE: Existing routes (/health, /chat, /api/v1/agents) remain inline below.
# New feature routes are registered via include_router() here.
# To register a new route file:
#   1. Create src/api/routes/<feature>.py with `router = APIRouter()`
#   2. Add include_router() call here with appropriate prefix and tags
#
# Registered dynamically by each phase when route files are created:

# Import feature routes directly from file to avoid triggering
# src.api.routes.__init__ which eagerly imports modules with uninstalled
# dependencies (langchain_mcp_adapters in agents.py).
_eval_spec = _ilu.spec_from_file_location(
    "src.api.routes.evaluation",
    _pathlib.Path(__file__).parent / "routes" / "evaluation.py",
)
_eval_mod = _ilu.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(_eval_mod)
app.include_router(_eval_mod.router, prefix="/api/v1/evaluate", tags=["evaluation"])

_nlq_spec = _ilu.spec_from_file_location(
    "src.api.routes.nl_query",
    _pathlib.Path(__file__).parent / "routes" / "nl_query.py",
)
_nlq_mod = _ilu.module_from_spec(_nlq_spec)
_nlq_spec.loader.exec_module(_nlq_mod)
app.include_router(_nlq_mod.router, prefix="/api/v1/query", tags=["nl-query"])

_hitl_spec = _ilu.spec_from_file_location(
    "src.api.routes.hitl",
    _pathlib.Path(__file__).parent / "routes" / "hitl.py",
)
_hitl_mod = _ilu.module_from_spec(_hitl_spec)
_hitl_spec.loader.exec_module(_hitl_mod)
app.include_router(_hitl_mod.router, prefix="/api/v1/hitl", tags=["hitl"])

_tt_spec = _ilu.spec_from_file_location(
    "src.api.routes.time_travel",
    _pathlib.Path(__file__).parent / "routes" / "time_travel.py",
)
_tt_mod = _ilu.module_from_spec(_tt_spec)
_tt_spec.loader.exec_module(_tt_mod)
app.include_router(_tt_mod.router, prefix="/api/v1/time-travel", tags=["time-travel"])


@app.get("/", response_model=dict[str, str])
async def root():
    """Root endpoint."""
    return {"name": "AI Agent Boilerplate", "version": "0.1.0", "status": "operational"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    # Check service status
    services = {}

    # MongoDB -- use public health_check() API
    # mongodb-connection: "Health checks should use public APIs, not private members"
    try:
        from src.storage.mongodb_client import mongodb_client

        result = await mongodb_client.health_check()
        if result.get("status") == "healthy":
            services["mongodb"] = "healthy"
        else:
            services["mongodb"] = "not_initialized"
    except Exception:
        services["mongodb"] = "unhealthy"

    # Voyage AI
    services["voyage_ai"] = "configured" if os.getenv("VOYAGE_API_KEY") else "not_configured"

    # OpenAI
    services["openai"] = "configured" if os.getenv("OPENAI_API_KEY") else "not_configured"

    # Galileo
    services["galileo"] = "configured" if os.getenv("GALILEO_API_KEY") else "not_configured"

    return HealthResponse(
        status=(
            "healthy"
            if all(v in ["healthy", "configured"] for v in services.values())
            else "degraded"
        ),
        version="0.1.0",
        services=services,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the AI agent."""
    try:
        # For now, return a simple echo response
        # In production, this would call the actual agent
        response = f"Echo: {request.message}"

        return ChatResponse(
            response=response,
            session_id=request.session_id,
            tokens_used=len(request.message.split()),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/agents", response_model=dict[str, Any])
async def list_agents():
    """List available agents."""
    return {
        "agents": [
            {
                "id": "assistant",
                "name": "General Assistant",
                "description": "General-purpose AI assistant with memory",
                "capabilities": ["chat", "memory", "tools"],
            },
            {
                "id": "research",
                "name": "Research Agent",
                "description": "Specialized in research and analysis",
                "capabilities": ["research", "analysis", "memory"],
            },
        ]
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(status_code=404, content={"error": "Not found", "path": str(request.url)})


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
