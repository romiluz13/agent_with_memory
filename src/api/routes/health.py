"""
Health Check Routes
System health and readiness endpoints
"""

import os
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from ...storage.mongodb_client import mongodb_client

router = APIRouter()


@router.get("/")
async def health_check() -> dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "service": "AI Agent Boilerplate",
        "version": "0.1.0",
    }


@router.get("/live")
async def liveness_probe() -> dict[str, str]:
    """
    Kubernetes liveness probe.
    Returns 200 if the service is alive.
    """
    return {"status": "alive"}


@router.get("/ready")
async def readiness_probe(app_request: Request) -> dict[str, Any]:
    """
    Kubernetes readiness probe.
    Checks if all dependencies are ready.
    """
    checks = {"mongodb": False, "embeddings": False, "memory_system": False}

    try:
        # Check MongoDB
        # mongodb-connection: Use public APIs, not private members
        result = await mongodb_client.health_check()
        checks["mongodb"] = result.get("status") == "healthy"

        # Check embeddings service
        if os.getenv("VOYAGE_API_KEY"):
            checks["embeddings"] = True

        checks["memory_system"] = app_request.app.state.memory_manager is not None

        # Overall status
        all_ready = all(checks.values())

        if not all_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"status": "not ready", "checks": checks},
            )

        return {"status": "ready", "checks": checks, "timestamp": datetime.now(UTC).isoformat()}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not ready", "error": str(e), "checks": checks},
        ) from e


@router.get("/stats")
async def system_stats(app_request: Request) -> dict[str, Any]:
    """Get system statistics."""
    try:
        # Get MongoDB stats
        db_stats = await mongodb_client.health_check()

        runtime = app_request.app.state.runtime
        memory_stats = (
            await runtime.memory_manager.get_stats()
            if runtime.memory_manager is not None
            else {"total_memories": 0, "cache_hit_rate": 0.0}
        )

        connection_stats = {
            "active_agents": len(runtime.agent_registry._metadata_cache),
            **runtime.websocket_manager.get_connection_stats(),
        }

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "database": db_stats,
            "memory": memory_stats,
            "connections": connection_stats,
            "uptime_seconds": int(
                (datetime.now(UTC) - app_request.app.state.started_at).total_seconds()
            ),
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}",
        ) from e
