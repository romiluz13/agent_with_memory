"""
Tests for Human-in-the-Loop (HITL) approval workflows.
Phase 5: LangGraph interrupt-based HITL pattern.
"""


import pytest

from src.core.hitl import (
    ApprovalAction,
    ApprovalRequest,
    ApprovalResponse,
    HITLConfig,
    check_approval_needed,
    create_approval_request,
    requires_approval,
)


class TestHITLConfig:
    """Tests for HITL configuration."""

    def test_default_sensitive_tools(self):
        """Default config includes standard sensitive tools."""
        config = HITLConfig()
        assert "delete_memory" in config.sensitive_tools
        assert "clear_all" in config.sensitive_tools
        assert "consolidate_memories" in config.sensitive_tools

    def test_custom_sensitive_tools(self):
        """Config accepts custom sensitive tools."""
        config = HITLConfig(sensitive_tools={"my_tool", "other_tool"})
        assert "my_tool" in config.sensitive_tools
        assert "delete_memory" not in config.sensitive_tools


class TestCheckApprovalNeeded:
    """Tests for approval checking logic."""

    @pytest.mark.asyncio
    async def test_sensitive_tool_requires_approval(self):
        """Sensitive tool names trigger approval requirement."""
        config = HITLConfig()
        result = await check_approval_needed("delete_memory", config)
        assert result is True

    @pytest.mark.asyncio
    async def test_normal_tool_no_approval(self):
        """Non-sensitive tools pass through without approval."""
        config = HITLConfig()
        result = await check_approval_needed("retrieve_memories", config)
        assert result is False

    @pytest.mark.asyncio
    async def test_approval_for_writes_when_enabled(self):
        """Write operations require approval when require_approval_for_writes is True."""
        config = HITLConfig(require_approval_for_writes=True)
        result = await check_approval_needed("store_memory", config)
        # store_ prefix indicates write
        assert result is True

    @pytest.mark.asyncio
    async def test_no_approval_for_writes_when_disabled(self):
        """Write operations pass through when require_approval_for_writes is False."""
        config = HITLConfig(require_approval_for_writes=False)
        result = await check_approval_needed("store_memory", config)
        assert result is False


class TestCreateApprovalRequest:
    """Tests for approval request creation."""

    @pytest.mark.asyncio
    async def test_creates_valid_request(self):
        """Approval request contains all required fields."""
        request = await create_approval_request(
            tool_name="delete_memory",
            tool_args={"memory_id": "123"},
            agent_id="agent_1",
        )
        assert request["tool_name"] == "delete_memory"
        assert request["tool_args"] == {"memory_id": "123"}
        assert request["agent_id"] == "agent_1"
        assert request["status"] == "pending"
        assert "created_at" in request
        assert "request_id" in request

    @pytest.mark.asyncio
    async def test_agent_id_in_request(self):
        """Agent ID MUST be in the approval request document."""
        request = await create_approval_request(
            tool_name="clear_all",
            tool_args={},
            agent_id="agent_42",
        )
        assert request["agent_id"] == "agent_42"


class TestRequiresApproval:
    """Tests for the requires_approval helper."""

    def test_sensitive_tool_detected(self):
        """Known sensitive tools return True."""
        assert requires_approval("delete_memory") is True
        assert requires_approval("clear_all") is True

    def test_normal_tool_passes(self):
        """Normal tools return False."""
        assert requires_approval("retrieve_memories") is False
        assert requires_approval("store_memory") is False


class TestApprovalDataclasses:
    """Tests for data model structures."""

    def test_approval_action_enum(self):
        """ApprovalAction enum has expected values."""
        assert ApprovalAction.TOOL_EXECUTION == "tool_execution"
        assert ApprovalAction.MEMORY_DELETE == "memory_delete"
        assert ApprovalAction.SENSITIVE_QUERY == "sensitive_query"

    def test_approval_request_dataclass(self):
        """ApprovalRequest holds all required fields."""
        req = ApprovalRequest(
            action=ApprovalAction.TOOL_EXECUTION,
            description="Delete memory 123",
            agent_id="agent_1",
            thread_id="thread_1",
            details={"memory_id": "123"},
        )
        assert req.agent_id == "agent_1"
        assert req.action == ApprovalAction.TOOL_EXECUTION

    def test_approval_response_dataclass(self):
        """ApprovalResponse holds approval decision."""
        resp = ApprovalResponse(approved=True, feedback="Looks good")
        assert resp.approved is True
        assert resp.feedback == "Looks good"

    def test_approval_response_defaults(self):
        """ApprovalResponse defaults to no feedback."""
        resp = ApprovalResponse(approved=False)
        assert resp.feedback is None
        assert resp.modified_action is None
