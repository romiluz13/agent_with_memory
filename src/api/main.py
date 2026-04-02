"""
FastAPI application entrypoint for AWM 2.0.
"""

import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.middleware import (
    AuthMiddleware,
    ErrorHandlingMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
)
from src.api.routes import agents, chat, evaluation, health, hitl, memories, nl_query, time_travel
from src.api.runtime import RuntimeSettings, initialize_runtime, shutdown_runtime

load_dotenv()

logger = logging.getLogger(__name__)
bootstrap_settings = RuntimeSettings.from_env()


class LegacyChatRequest(BaseModel):
    """Compatibility request model for the legacy /chat endpoint."""

    message: str
    session_id: str = "default"
    user_id: str = "anonymous"


class LegacyChatResponse(BaseModel):
    """Compatibility response model for the legacy /chat endpoint."""

    response: str
    session_id: str
    tokens_used: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    runtime = await initialize_runtime()
    app.state.runtime = runtime
    app.state.settings = runtime.settings
    app.state.mongodb_client = runtime.mongodb
    app.state.sync_mongo_client = runtime.sync_mongo_client
    app.state.memory_manager = runtime.memory_manager
    app.state.agent_registry = runtime.agent_registry
    app.state.websocket_manager = runtime.websocket_manager
    app.state.nl_query_generator = runtime.nl_query_generator
    app.state.started_at = datetime.now(UTC)

    logger.info(
        "Started AWM 2.0 API (lane=%s, db=%s, mongodb_ready=%s)",
        runtime.settings.validation_lane,
        runtime.settings.database_name,
        runtime.memory_manager is not None,
    )

    yield

    await shutdown_runtime(runtime)
    logger.info("Stopped AWM 2.0 API")


app = FastAPI(
    title="Agent With Memory 2.0 API",
    description="Production-oriented AI agent starter with multi-layer memory and MongoDB",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=bootstrap_settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(memories.router, prefix="/api/v1/memories", tags=["memories"])
app.include_router(nl_query.router, prefix="/api/v1/query", tags=["nl-query"])
app.include_router(hitl.router, prefix="/api/v1/hitl", tags=["hitl"])
app.include_router(time_travel.router, prefix="/api/v1/time-travel", tags=["time-travel"])
app.include_router(evaluation.router, prefix="/api/v1/evaluate", tags=["evaluation"])


@app.get("/", response_model=dict[str, str])
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "name": "Agent With Memory 2.0",
        "version": "0.2.0",
        "status": "operational",
    }


@app.get("/health", response_model=dict[str, str])
async def legacy_health() -> dict[str, str]:
    """Compatibility health endpoint without redirect semantics."""
    return {
        "status": "healthy",
        "service": "Agent With Memory 2.0",
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.post("/chat", response_model=LegacyChatResponse)
async def legacy_chat(request: LegacyChatRequest) -> LegacyChatResponse:
    """Compatibility endpoint backed by the real assistant runtime."""
    runtime = app.state.runtime
    if runtime.memory_manager is None:
        raise HTTPException(
            status_code=503,
            detail=runtime.startup_error or "MongoDB runtime is not configured",
        )

    agent = await runtime.agent_registry.ensure_agent("assistant")
    response = await agent.invoke(
        message=request.message,
        user_id=request.user_id,
        session_id=request.session_id,
        conversation_id=request.session_id,
    )

    return LegacyChatResponse(
        response=response,
        session_id=request.session_id,
        tokens_used=len(response.split()),
    )


@app.websocket("/api/v1/ws/{agent_id}")
async def websocket_chat(
    websocket: WebSocket,
    agent_id: str,
):
    """Bidirectional chat over WebSocket using the canonical agent runtime."""
    runtime = app.state.runtime
    manager = runtime.websocket_manager

    user_id = websocket.query_params.get("user_id") or f"ws-{datetime.now(UTC).timestamp()}"
    session_id = websocket.query_params.get("session_id")
    conversation_id = websocket.query_params.get("conversation_id")

    await manager.connect(websocket, agent_id=agent_id, user_id=user_id)

    try:
        while True:
            payload = await websocket.receive_json()
            message = payload.get("message")
            if not message:
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": "message is required",
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )
                continue

            try:
                agent = await runtime.agent_registry.ensure_agent(agent_id)
            except KeyError as exc:
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": f"Agent {agent_id} not found",
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )
                raise HTTPException(status_code=404, detail=str(exc)) from exc

            async for event in agent.stream_events(
                message=message,
                user_id=user_id,
                session_id=session_id or payload.get("session_id"),
                conversation_id=conversation_id or payload.get("conversation_id"),
            ):
                await websocket.send_json(event)

    except WebSocketDisconnect:
        manager.disconnect(agent_id, user_id)
    except Exception as exc:
        logger.exception("WebSocket chat failed")
        if websocket.client_state.name == "CONNECTED":
            await websocket.send_json(
                {
                    "type": "error",
                    "error": str(exc),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
        manager.disconnect(agent_id, user_id)


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
