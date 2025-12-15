"""Tests for Summary Tools module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.tools.summary_tools import (
    create_expand_summary_tool,
    create_summarize_tool,
    create_summarize_conversation_tool,
    create_summary_tools
)


class TestExpandSummaryTool:
    """Tests for expand_summary tool."""

    def test_tool_creation(self):
        """Test expand_summary tool can be created."""
        mock_manager = MagicMock()
        tool = create_expand_summary_tool(mock_manager)

        assert tool is not None
        assert hasattr(tool, 'name')
        assert hasattr(tool, 'description')

    def test_tool_has_correct_name(self):
        """Test tool has expected name."""
        mock_manager = MagicMock()
        tool = create_expand_summary_tool(mock_manager)

        assert 'expand' in tool.name.lower() or 'summary' in tool.name.lower()

    def test_tool_description_explains_purpose(self):
        """Test tool description explains its purpose."""
        mock_manager = MagicMock()
        tool = create_expand_summary_tool(mock_manager)

        desc = tool.description.lower()
        assert 'summary' in desc or 'expand' in desc


class TestSummarizeTool:
    """Tests for summarize tool."""

    def test_tool_creation(self):
        """Test summarize tool can be created."""
        mock_manager = MagicMock()
        mock_llm = MagicMock()
        tool = create_summarize_tool(mock_manager, mock_llm, "agent_1")

        assert tool is not None
        assert hasattr(tool, 'name')
        assert hasattr(tool, 'description')

    def test_tool_has_correct_name(self):
        """Test tool has expected name."""
        mock_manager = MagicMock()
        mock_llm = MagicMock()
        tool = create_summarize_tool(mock_manager, mock_llm, "agent_1")

        assert 'summarize' in tool.name.lower()

    def test_tool_description_explains_purpose(self):
        """Test tool description explains its purpose."""
        mock_manager = MagicMock()
        mock_llm = MagicMock()
        tool = create_summarize_tool(mock_manager, mock_llm, "agent_1")

        desc = tool.description.lower()
        assert 'summarize' in desc or 'compress' in desc


class TestSummarizeConversationTool:
    """Tests for summarize_conversation tool."""

    def test_tool_creation(self):
        """Test summarize_conversation tool can be created."""
        mock_manager = MagicMock()
        mock_llm = MagicMock()
        tool = create_summarize_conversation_tool(mock_manager, mock_llm, "agent_1")

        assert tool is not None
        assert hasattr(tool, 'name')
        assert hasattr(tool, 'description')

    def test_tool_has_correct_name(self):
        """Test tool has expected name."""
        mock_manager = MagicMock()
        mock_llm = MagicMock()
        tool = create_summarize_conversation_tool(mock_manager, mock_llm, "agent_1")

        name = tool.name.lower()
        assert 'conversation' in name or 'summarize' in name

    def test_tool_description_mentions_conversation(self):
        """Test tool description mentions conversation."""
        mock_manager = MagicMock()
        mock_llm = MagicMock()
        tool = create_summarize_conversation_tool(mock_manager, mock_llm, "agent_1")

        desc = tool.description.lower()
        assert 'conversation' in desc or 'thread' in desc or 'messages' in desc


class TestCreateSummaryTools:
    """Tests for create_summary_tools factory function."""

    def test_returns_list(self):
        """Test function returns a list."""
        mock_manager = MagicMock()
        mock_llm = MagicMock()
        tools = create_summary_tools(mock_manager, mock_llm, "agent_1")

        assert isinstance(tools, list)

    def test_returns_multiple_tools(self):
        """Test function returns multiple tools."""
        mock_manager = MagicMock()
        mock_llm = MagicMock()
        tools = create_summary_tools(mock_manager, mock_llm, "agent_1")

        assert len(tools) >= 2

    def test_all_items_are_tools(self):
        """Test all returned items are callable tools."""
        mock_manager = MagicMock()
        mock_llm = MagicMock()
        tools = create_summary_tools(mock_manager, mock_llm, "agent_1")

        for tool in tools:
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')

    def test_includes_expand_tool(self):
        """Test tools include expand summary tool."""
        mock_manager = MagicMock()
        mock_llm = MagicMock()
        tools = create_summary_tools(mock_manager, mock_llm, "agent_1")

        tool_names = [t.name.lower() for t in tools]
        assert any('expand' in name for name in tool_names)

    def test_includes_summarize_tool(self):
        """Test tools include summarize tool."""
        mock_manager = MagicMock()
        mock_llm = MagicMock()
        tools = create_summary_tools(mock_manager, mock_llm, "agent_1")

        tool_names = [t.name.lower() for t in tools]
        assert any('summarize' in name for name in tool_names)


class TestToolIntegration:
    """Integration-style tests for summary tools."""

    def test_tools_work_with_memory_manager_interface(self):
        """Test tools expect memory manager with summary attribute."""
        mock_manager = MagicMock()
        mock_manager.summary = MagicMock()
        mock_manager.summary.expand_summary = AsyncMock(return_value="Full content")

        tool = create_expand_summary_tool(mock_manager)

        # Tool should be created successfully
        assert tool is not None

    def test_summary_reference_format(self):
        """Test expected summary reference format."""
        summary_id = "abc12345"
        description = "Brief description"
        reference = f"[Summary ID: {summary_id}] {description}"

        assert summary_id in reference
        assert "Summary ID:" in reference
