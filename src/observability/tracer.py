"""
Observability tracer with graceful degradation.

Pattern from HybridRAG: lazy init, env var checks, zero overhead when disabled.
Supports Langfuse (default) and LangSmith providers.

Usage:
    tracer = get_tracer()
    with tracer.trace("retrieval", metadata={"query": q}):
        results = await search(...)

    tracer.trace_memory_operation("store", "episodic", agent_id)
"""

import logging
import os
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


class ObservabilityTracer:
    """Unified tracer that supports Langfuse (default) or LangSmith.

    Gracefully degrades to no-op when not configured.
    Lazy initialization ensures zero cost if never used.

    Args:
        provider: Observability provider name. Defaults to OBSERVABILITY_PROVIDER
            env var or "langfuse".
    """

    def __init__(self, provider: str | None = None) -> None:
        self._provider = provider or os.getenv("OBSERVABILITY_PROVIDER", "langfuse")
        self._client: Any = None
        self._enabled: bool = False
        self._lazy_initialized: bool = False

    def _lazy_init(self) -> None:
        """Initialize on first use. Zero cost if never called."""
        if self._lazy_initialized:
            return
        self._lazy_initialized = True

        if self._provider == "langfuse":
            self._init_langfuse()
        elif self._provider == "langsmith":
            self._init_langsmith()

    def _init_langfuse(self) -> None:
        """Initialize Langfuse provider."""
        secret = os.getenv("LANGFUSE_SECRET_KEY")
        public = os.getenv("LANGFUSE_PUBLIC_KEY")
        if secret and public:
            try:
                from langfuse import Langfuse

                self._client = Langfuse()
                self._enabled = True
                logger.info("Langfuse observability enabled")
            except ImportError:
                logger.debug("Langfuse not installed, tracing disabled")
            except Exception:
                logger.debug("Langfuse initialization failed, tracing disabled", exc_info=True)
        else:
            logger.debug("Langfuse keys not set, tracing disabled")

    def _init_langsmith(self) -> None:
        """Initialize LangSmith provider."""
        api_key = os.getenv("LANGCHAIN_API_KEY")
        if api_key:
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
            self._enabled = True
            logger.info("LangSmith observability enabled")

    @property
    def enabled(self) -> bool:
        """Whether tracing is enabled.

        Returns:
            True if the provider is configured and available.
        """
        self._lazy_init()
        return self._enabled

    def get_langchain_handler(self) -> Any:
        """Get LangChain callback handler for tracing LLM calls.

        Returns:
            CallbackHandler instance for Langfuse, or None if disabled
            or using LangSmith (which uses env var auto-detection).
        """
        self._lazy_init()
        if not self._enabled:
            return None
        if self._provider == "langfuse":
            try:
                from langfuse.langchain import CallbackHandler

                return CallbackHandler()
            except Exception:
                logger.debug("Langfuse CallbackHandler unavailable", exc_info=True)
                return None
        return None  # LangSmith uses env var auto-detection

    @contextmanager
    def trace(
        self, name: str, metadata: dict[str, Any] | None = None
    ):  # type: ignore[return]
        """Context manager for tracing operations.

        Args:
            name: Operation name for the trace.
            metadata: Optional metadata to attach to the trace.

        Yields:
            Trace object if enabled, None otherwise.
        """
        self._lazy_init()
        if not self._enabled:
            yield None
            return

        if self._provider == "langfuse" and self._client:
            try:
                trace_obj = self._client.trace(name=name, metadata=metadata or {})
            except Exception:
                logger.debug("Failed to create trace, continuing without tracing", exc_info=True)
                yield None
                return
            try:
                yield trace_obj
            finally:
                try:
                    trace_obj.update(status_message="completed")
                except Exception:
                    logger.debug("Failed to update trace status", exc_info=True)
        else:
            yield None

    def trace_memory_operation(
        self,
        operation: str,
        memory_type: str,
        agent_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Fire-and-forget trace for memory operations.

        Args:
            operation: Operation name (e.g., "store", "retrieve").
            memory_type: Memory type being operated on.
            agent_id: Agent ID for the operation.
            metadata: Additional metadata.
        """
        self._lazy_init()
        if not self._enabled or not self._client:
            return
        try:
            if self._provider == "langfuse":
                self._client.trace(
                    name=f"memory.{operation}",
                    metadata={
                        "memory_type": memory_type,
                        "agent_id": agent_id,
                        **(metadata or {}),
                    },
                )
        except Exception:
            logger.debug("Tracing memory operation failed, continuing", exc_info=True)

    def flush(self) -> None:
        """Flush pending traces. No-op if no client."""
        if self._client and hasattr(self._client, "flush"):
            self._client.flush()


# Global singleton
_tracer: ObservabilityTracer | None = None


def get_tracer() -> ObservabilityTracer:
    """Get or create the global tracer singleton.

    Returns:
        The global ObservabilityTracer instance.
    """
    global _tracer
    if _tracer is None:
        _tracer = ObservabilityTracer()
    return _tracer
