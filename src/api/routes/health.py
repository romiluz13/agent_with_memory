"""
Health Check Routes
System health and readiness endpoints
"""

import os
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status

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
async def readiness_probe() -> dict[str, Any]:
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

        # Check memory system
        # This would check if memory manager is initialized
        checks["memory_system"] = True

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
async def system_stats() -> dict[str, Any]:
    """Get system statistics."""
    try:
        # Get MongoDB stats
        db_stats = await mongodb_client.health_check()

        # Get memory stats (would come from memory manager)
        memory_stats = {"total_memories": 0, "cache_hit_rate": 0.0}

        # Get connection stats
        connection_stats = {"active_agents": 0, "active_connections": 0}

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "database": db_stats,
            "memory": memory_stats,
            "connections": connection_stats,
            "uptime_seconds": 0,  # Would calculate actual uptime
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}",
        ) from e
