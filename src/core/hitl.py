"""
Human-in-the-Loop support for LangGraph agents.
Uses LangGraph's built-in interrupt() mechanism with MongoDBSaver persistence.

Pattern from LangChain+MongoDB article: "human-in-the-loop approval workflows"
Pattern from Mem0 reference: 5 lifecycle hooks, progressive disclosure.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class ApprovalAction(StrEnum):
    """Actions that can trigger human approval."""

    TOOL_EXECUTION = "tool_execution"
    SENSITIVE_QUERY = "sensitive_query"
    MEMORY_DELETE = "memory_delete"
    EXTERNAL_API = "external_api"


@dataclass
class ApprovalRequest:
    """Request sent to human for approval."""

    action: ApprovalAction
    description: str
    agent_id: str
    thread_id: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ApprovalResponse:
    """Response from human."""

    approved: bool
    feedback: str | None = None
    modified_action: dict[str, Any] | None = None


# Tools that require human approval (configurable)
SENSITIVE_TOOLS = {"delete_memory", "clear_all", "consolidate_memories"}


class HITLConfig:
    """Configuration for human-in-the-loop approval workflows."""

    def __init__(
        self,
        sensitive_tools: set[str] | None = None,
        require_approval_for_writes: bool = False,
    ):
        """Initialize HITL configuration.

        Args:
            sensitive_tools: Set of tool names requiring approval.
                Defaults to SENSITIVE_TOOLS.
            require_approval_for_writes: If True, all write operations
                (tool names starting with 'store_', 'update_', 'delete_')
                require approval.
        """
        self.sensitive_tools = sensitive_tools if sensitive_tools is not None else SENSITIVE_TOOLS
        self.require_approval_for_writes = require_approval_for_writes


async def check_approval_needed(tool_name: str, config: HITLConfig) -> bool:
    """Check if a tool invocation requires human approval.

    Args:
        tool_name: Name of the tool being invoked
        config: HITL configuration

    Returns:
        True if approval is required
    """
    # Check explicit sensitive tools list
    if tool_name in config.sensitive_tools:
        return True

    # Check write operations if configured
    if config.require_approval_for_writes:
        write_prefixes = ("store_", "update_", "delete_", "clear_", "consolidate_")
        if any(tool_name.startswith(prefix) for prefix in write_prefixes):
            return True

    return False


async def create_approval_request(
    tool_name: str, tool_args: dict[str, Any], agent_id: str
) -> dict[str, Any]:
    """Create an approval request document for MongoDB storage.

    Args:
        tool_name: Name of the tool requiring approval
        tool_args: Arguments passed to the tool
        agent_id: Agent ID (MUST be in the document)

    Returns:
        Approval request document ready for MongoDB insertion
    """
    return {
        "request_id": str(uuid.uuid4()),
        "tool_name": tool_name,
        "tool_args": tool_args,
        "agent_id": agent_id,
        "status": "pending",
        "created_at": datetime.now(UTC).isoformat(),
    }


async def store_approval_request(
    collection: Any, request: dict[str, Any]
) -> str:
    """Store approval request in MongoDB for async approval.

    Args:
        collection: MongoDB collection (Motor async)
        request: Approval request document (agent_id MUST be present)

    Returns:
        The inserted document's request_id
    """
    await collection.insert_one(request)
    return request["request_id"]


def requires_approval(tool_name: str) -> bool:
    """Check if a tool requires human approval.

    Quick check using default SENSITIVE_TOOLS set.

    Args:
        tool_name: Name of the tool

    Returns:
        True if tool is in the sensitive tools set
    """
    return tool_name in SENSITIVE_TOOLS
