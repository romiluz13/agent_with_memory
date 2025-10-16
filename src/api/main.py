"""
FastAPI Main Application
Production-ready API for AI Agent Boilerplate - CORRECTED AND FULLY FUNCTIONAL
"""

import os
import traceback
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# --- FIX Part 1: Import the Agent we worked on ---
from src.core.agent_langgraph import MongoDBLangGraphAgent

# Load environment variables
load_dotenv()

# --- FIX Part 2: Create a global variable to hold our single agent instance ---
agent: MongoDBLangGraphAgent = None

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    # The agent_id is in the original model, so we keep it for compatibility
    agent_id: str = "assistant"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    tokens_used: int = 0

# Lifespan context manager to initialize the agent on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    global agent
    print("🚀 Starting AI Agent Boilerplate API...")
    print("🤖 Initializing MongoDB LangGraph Agent for the API...")
    
    agent = MongoDBLangGraphAgent(
        mongodb_uri=os.getenv("MONGODB_URI"),
        agent_name="api_assistant", # Give it a unique name for the API
        model_provider="openai",
        model_name="gpt-4o"
    )
    print("✅ Agent initialized and ready.")
    
    yield
    
    # Shutdown
    print("👋 Shutting down AI Agent Boilerplate API...")

# Create FastAPI app
app = FastAPI(
    title="AI Agent Boilerplate API",
    description="Production-ready AI agent with sophisticated memory system",
    version="1.0.0", # Version bump to reflect it's working
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "name": "AI Agent Boilerplate",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the AI agent."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent is not initialized or is warming up. Please try again in a moment.")
    
    try:
        # --- FIX Part 3: Call the actual agent instead of echoing ---
        # The agent's method is `aexecute` and it expects 'message' and 'thread_id'
        agent_response = await agent.aexecute(
            message=request.message,
            thread_id=request.session_id
        )
        
        return ChatResponse(
            response=agent_response,
            session_id=request.session_id,
            tokens_used=0  # This can be properly implemented later
        )
        
    except Exception as e:
        print(f"--- ERROR DURING CHAT ---")
        traceback.print_exc()
        print(f"--- END ERROR ---")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")