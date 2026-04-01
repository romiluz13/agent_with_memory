"""
Observability module for AI Agent Boilerplate.

Provides end-to-end tracing for agent operations with graceful degradation.
Zero overhead when not configured (no env vars, no import cost).
"""

from .tracer import ObservabilityTracer, get_tracer

__all__ = ["ObservabilityTracer", "get_tracer"]
