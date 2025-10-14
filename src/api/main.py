"""
FastAPI Main Application
Production-ready API for AI Agent Boilerplate
"""

import os
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# FIX 1: Import the health router
from .routes import health as health_router
# FIX 2: Import the MongoDB initialization function
from ..storage.mongodb_client import initialize_mongodb, mongodb_client


# Load environment variables
load_dotenv()


# Request/Response models
class HealthResponse(BaseModel):
    status: str
    version: str
    services: Dict[str, str]


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
    
    # FIX 2: Initialize the database connection on startup
    await initialize_mongodb(uri=os.getenv("MONGODB_URI"), database=os.getenv("MONGODB_DB_NAME"))
    
    yield
    
    # Shutdown
    print("👋 Shutting down AI Agent Boilerplate API...")
    await mongodb_client.close()


# Create FastAPI app
app = FastAPI(
    title="AI Agent Boilerplate API",
    description="Production-ready AI agent with sophisticated memory system",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FIX 1: Add the health check routes to the main application
app.include_router(health_router.router, prefix="/health", tags=["Health"])


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "name": "AI Agent Boilerplate",
        "version": "0.1.0",
        "status": "operational"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    # This endpoint is now technically overridden by the one in health.py,
    # but we'll leave it here as a fallback. The /health/ready is the important one.
    services = {}
    
    try:
        await mongodb_client.client.admin.command('ping')
        services["mongodb"] = "healthy"
    except Exception:
        services["mongodb"] = "unhealthy"
    
    services["voyage_ai"] = "configured" if os.getenv("VOYAGE_API_KEY") else "not_configured"
    services["openai"] = "configured" if os.getenv("OPENAI_API_KEY") else "not_configured"
    services["galileo"] = "configured" if os.getenv("GALILEO_API_KEY") else "not_configured"
    
    return HealthResponse(
        status="healthy" if all(v in ["healthy", "configured"] for v in services.values()) else "degraded",
        version="0.1.0",
        services=services
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the AI agent."""
    try:
        response = f"Echo: {request.message}"
        
        return ChatResponse(
            response=response,
            session_id=request.session_id,
            tokens_used=len(request.message.split())
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/agents", response_model=Dict[str, Any])
async def list_agents():
    """List available agents."""
    return {
        "agents": [
            {
                "id": "assistant",
                "name": "General Assistant",
                "description": "General-purpose AI assistant with memory",
                "capabilities": ["chat", "memory", "tools"]
            },
            {
                "id": "research",
                "name": "Research Agent",
                "description": "Specialized in research and analysis",
                "capabilities": ["research", "analysis", "memory"]
            }
        ]
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "path": str(request.url)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )