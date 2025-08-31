"""
AI Agent Boilerplate - Production Ready Agent Framework
"""

__version__ = "0.1.0"
__author__ = "AI Agent Boilerplate Team"

# Core exports
try:
    from .core.agent_langgraph import MongoDBLangGraphAgent
    __all__ = ["MongoDBLangGraphAgent"]
except ImportError:
    # During development, some imports may fail
    __all__ = []