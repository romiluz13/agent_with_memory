"""Tests for tracer integration with LangGraph agent.

Note: agent_langgraph.py has pre-existing import issues (langchain.agents.tool
renamed to langchain.agents.tools). Source inspection is used instead of
module import for verification.
"""

import ast


class TestAgentLangGraphTracingSource:
    """Verify tracer integration via source code inspection."""

    def _get_ast(self):
        """Parse agent_langgraph.py AST."""
        with open("src/core/agent_langgraph.py") as f:
            return ast.parse(f.read())

    def test_agent_langgraph_imports_get_tracer(self):
        """agent_langgraph.py should import get_tracer from observability."""
        tree = self._get_ast()
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "observability" in node.module:
                    for alias in node.names:
                        imports.append(alias.name)
        assert "get_tracer" in imports, (
            "agent_langgraph.py must import get_tracer from observability.tracer"
        )

    def test_agent_node_uses_get_langchain_handler(self):
        """agent node should call get_tracer().get_langchain_handler()."""
        with open("src/core/agent_langgraph.py") as f:
            source = f.read()
        assert "get_langchain_handler" in source, (
            "agent_langgraph.py must call get_tracer().get_langchain_handler()"
        )

    def test_agent_node_passes_callbacks_config(self):
        """agent node should pass callbacks in invoke config."""
        with open("src/core/agent_langgraph.py") as f:
            source = f.read()
        assert '"callbacks"' in source or "'callbacks'" in source, (
            "agent_langgraph.py must pass callbacks in invoke config"
        )
