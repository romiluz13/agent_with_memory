"""
Health Check Routes
System health and readiness endpoints
"""

import os
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from ...storage.mongodb_client import mongodb_client

router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "AI Agent Boilerplate",
        "version": "0.1.0"
    }


@router.get("/live")
async def liveness_probe() -> Dict[str, str]:
    """
    Kubernetes liveness probe.
    Returns 200 if the service is alive.
    """
    return {"status": "alive"}


@router.get("/ready")
async def readiness_probe() -> Dict[str, Any]:
    """
    Kubernetes readiness probe.
    Checks if all dependencies are ready.
    """
    checks = {
        "mongodb": False,
        "embeddings": False,
        "memory_system": False
    }
    
    try:
        # Check MongoDB
        if mongodb_client._db:
            await mongodb_client._client.admin.command('ping')
            checks["mongodb"] = True
        
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
                detail={
                    "status": "not ready",
                    "checks": checks
                }
            )
        
        return {
            "status": "ready",
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not ready",
                "error": str(e),
                "checks": checks
            }
        )


@router.get("/stats")
async def system_stats() -> Dict[str, Any]:
    """Get system statistics."""
    try:
        # Get MongoDB stats
        db_stats = await mongodb_client.health_check()
        
        # Get memory stats (would come from memory manager)
        memory_stats = {
            "total_memories": 0,
            "cache_hit_rate": 0.0
        }
        
        # Get connection stats
        connection_stats = {
            "active_agents": 0,
            "active_connections": 0
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "database": db_stats,
            "memory": memory_stats,
            "connections": connection_stats,
            "uptime_seconds": 0  # Would calculate actual uptime
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )
