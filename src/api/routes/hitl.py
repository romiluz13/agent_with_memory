"""
Human-in-the-Loop API endpoints.
Provides approval workflow for sensitive agent operations.

Endpoints:
    GET  /pending/{agent_id}  - List pending approval requests
    POST /approve/{request_id} - Approve a request
    POST /reject/{request_id}  - Reject a request
"""

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


class PendingRequest(BaseModel):
    """A pending approval request."""

    request_id: str
    tool_name: str
    tool_args: dict[str, Any]
    agent_id: str
    status: str
    created_at: str


class PendingRequestsList(BaseModel):
    """List of pending requests."""

    agent_id: str
    requests: list[PendingRequest]
    total_count: int


class ApproveRejectRequest(BaseModel):
    """Request body for approve/reject endpoints."""

    feedback: str | None = Field(default=None, description="Optional feedback from reviewer")


class ApproveRejectResponse(BaseModel):
    """Response from approve/reject."""

    request_id: str
    status: str
    feedback: str | None


@router.get("/pending/{agent_id}", response_model=PendingRequestsList)
async def get_pending_requests(
    agent_id: str,
    app_request: Request,
    limit: int = Query(default=50, ge=1, le=500),
):
    """Get pending approval requests for an agent.

    Args:
        agent_id: The agent ID to filter by (MUST filter by agent_id)
        limit: Maximum number of requests to return

    Returns:
        List of pending approval requests
    """
    try:
        mongodb = app_request.app.state.mongodb_client
        if mongodb is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="MongoDB not configured",
            )

        collection = mongodb.db["approval_requests"]

        # CRITICAL: Filter by agent_id for multi-tenant isolation
        cursor = (
            collection.find({"agent_id": agent_id, "status": "pending"})
            .sort("created_at", -1)
            .limit(limit)
        )

        requests = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            requests.append(
                PendingRequest(
                    request_id=doc.get("request_id", str(doc["_id"])),
                    tool_name=doc.get("tool_name", ""),
                    tool_args=doc.get("tool_args", {}),
                    agent_id=doc.get("agent_id", ""),
                    status=doc.get("status", "pending"),
                    created_at=doc.get("created_at", ""),
                )
            )
        return PendingRequestsList(
            agent_id=agent_id,
            requests=requests,
            total_count=len(requests),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pending requests: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pending requests: {str(e)}",
        ) from e


@router.post("/approve/{request_id}", response_model=ApproveRejectResponse)
async def approve_request(
    request_id: str,
    app_request: Request,
    body: ApproveRejectRequest | None = None,
):
    """Approve a pending request.

    Args:
        request_id: The approval request ID
        body: Optional feedback

    Returns:
        Updated request status
    """
    try:
        mongodb = app_request.app.state.mongodb_client
        if mongodb is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="MongoDB not configured",
            )

        collection = mongodb.db["approval_requests"]

        feedback = body.feedback if body else None

        result = await collection.update_one(
            {"request_id": request_id, "status": "pending"},
            {
                "$set": {
                    "status": "approved",
                    "feedback": feedback,
                    "resolved_at": datetime.now(UTC).isoformat(),
                }
            },
        )
        if result.matched_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pending request {request_id} not found",
            )

        return ApproveRejectResponse(
            request_id=request_id,
            status="approved",
            feedback=feedback,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to approve request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.post("/reject/{request_id}", response_model=ApproveRejectResponse)
async def reject_request(
    request_id: str,
    app_request: Request,
    body: ApproveRejectRequest | None = None,
):
    """Reject a pending request.

    Args:
        request_id: The approval request ID
        body: Optional feedback explaining rejection

    Returns:
        Updated request status
    """
    try:
        mongodb = app_request.app.state.mongodb_client
        if mongodb is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="MongoDB not configured",
            )

        collection = mongodb.db["approval_requests"]

        feedback = body.feedback if body else None

        result = await collection.update_one(
            {"request_id": request_id, "status": "pending"},
            {
                "$set": {
                    "status": "rejected",
                    "feedback": feedback,
                    "resolved_at": datetime.now(UTC).isoformat(),
                }
            },
        )
        if result.matched_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pending request {request_id} not found",
            )

        return ApproveRejectResponse(
            request_id=request_id,
            status="rejected",
            feedback=feedback,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reject request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
