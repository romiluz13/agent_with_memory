"""Tests for observability tracer with graceful degradation."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.observability.tracer import ObservabilityTracer, get_tracer


class TestObservabilityTracerDisabled:
    """Tests for tracer behavior when Langfuse is not configured."""

    def test_tracer_disabled_when_no_env_vars(self):
        """Tracer should be disabled when no LANGFUSE keys are set."""
        with patch.dict(os.environ, {}, clear=True):
            tracer = ObservabilityTracer()
            assert tracer.enabled is False

    def test_tracer_lazy_init_not_called_until_accessed(self):
        """Tracer should not initialize until first property/method access."""
        tracer = ObservabilityTracer()
        assert tracer._lazy_initialized is False

    def test_trace_context_manager_noop_when_disabled(self):
        """trace() should yield None when tracer is disabled."""
        with patch.dict(os.environ, {}, clear=True):
            tracer = ObservabilityTracer()
            with tracer.trace("test_op", metadata={"key": "val"}) as trace_obj:
                assert trace_obj is None

    def test_trace_memory_operation_noop_when_disabled(self):
        """trace_memory_operation() should not raise when disabled."""
        with patch.dict(os.environ, {}, clear=True):
            tracer = ObservabilityTracer()
            # Should not raise
            tracer.trace_memory_operation(
                operation="store",
                memory_type="episodic",
                agent_id="agent_123",
                metadata={"key": "val"},
            )

    def test_langchain_handler_returns_none_when_disabled(self):
        """get_langchain_handler() should return None when disabled."""
        with patch.dict(os.environ, {}, clear=True):
            tracer = ObservabilityTracer()
            handler = tracer.get_langchain_handler()
            assert handler is None

    def test_flush_noop_when_disabled(self):
        """flush() should not raise when no client exists."""
        with patch.dict(os.environ, {}, clear=True):
            tracer = ObservabilityTracer()
            tracer.flush()  # Should not raise


class TestGetTracerSingleton:
    """Tests for the global tracer singleton."""

    def test_get_tracer_returns_instance(self):
        """get_tracer() should return an ObservabilityTracer."""
        # Reset global state
        import src.observability.tracer as tracer_module

        tracer_module._tracer = None
        tracer = get_tracer()
        assert isinstance(tracer, ObservabilityTracer)

    def test_get_tracer_singleton(self):
        """get_tracer() should return the same instance on repeated calls."""
        import src.observability.tracer as tracer_module

        tracer_module._tracer = None
        tracer1 = get_tracer()
        tracer2 = get_tracer()
        assert tracer1 is tracer2


class TestObservabilityTracerLangsmith:
    """Tests for LangSmith provider path."""

    def test_langsmith_provider_enabled_with_api_key(self):
        """LangSmith provider should enable when LANGCHAIN_API_KEY is set."""
        with patch.dict(
            os.environ,
            {"LANGCHAIN_API_KEY": "test-key"},
            clear=True,
        ):
            tracer = ObservabilityTracer(provider="langsmith")
            assert tracer.enabled is True

    def test_langsmith_provider_disabled_without_api_key(self):
        """LangSmith provider should be disabled without LANGCHAIN_API_KEY."""
        with patch.dict(os.environ, {}, clear=True):
            tracer = ObservabilityTracer(provider="langsmith")
            assert tracer.enabled is False

    def test_langsmith_handler_returns_none(self):
        """LangSmith uses env var auto-detection, so handler returns None."""
        with patch.dict(
            os.environ,
            {"LANGCHAIN_API_KEY": "test-key"},
            clear=True,
        ):
            tracer = ObservabilityTracer(provider="langsmith")
            handler = tracer.get_langchain_handler()
            assert handler is None


class TestObservabilityTracerLangfuseEnabled:
    """Tests for tracer behavior when Langfuse IS configured and available."""

    def test_langfuse_enabled_with_mocked_client(self):
        """When Langfuse keys are set and import succeeds, tracer should be enabled."""
        mock_langfuse_cls = MagicMock()
        mock_client = MagicMock()
        mock_langfuse_cls.return_value = mock_client

        with patch.dict(
            os.environ,
            {"LANGFUSE_SECRET_KEY": "sk-test", "LANGFUSE_PUBLIC_KEY": "pk-test"},
            clear=True,
        ):
            with patch.dict("sys.modules", {"langfuse": MagicMock(Langfuse=mock_langfuse_cls)}):
                tracer = ObservabilityTracer()
                assert tracer.enabled is True

    def test_get_langchain_handler_uses_langfuse_langchain_module(self):
        """get_langchain_handler() should import from langfuse.langchain, not langfuse.callback."""
        mock_handler_instance = MagicMock()
        mock_handler_cls = MagicMock(return_value=mock_handler_instance)
        mock_langfuse_mod = MagicMock(Langfuse=MagicMock())
        mock_langchain_mod = MagicMock(CallbackHandler=mock_handler_cls)

        with patch.dict(
            os.environ,
            {"LANGFUSE_SECRET_KEY": "sk-test", "LANGFUSE_PUBLIC_KEY": "pk-test"},
            clear=True,
        ):
            with patch.dict(
                "sys.modules",
                {
                    "langfuse": mock_langfuse_mod,
                    "langfuse.langchain": mock_langchain_mod,
                },
            ):
                tracer = ObservabilityTracer()
                tracer._lazy_init()
                handler = tracer.get_langchain_handler()
                assert handler is mock_handler_instance
                mock_handler_cls.assert_called_once()

    def test_trace_yields_trace_object_when_enabled(self):
        """trace() should yield a real trace object when Langfuse is enabled."""
        mock_trace_obj = MagicMock()
        mock_client = MagicMock()
        mock_client.trace.return_value = mock_trace_obj
        mock_langfuse_cls = MagicMock(return_value=mock_client)

        with patch.dict(
            os.environ,
            {"LANGFUSE_SECRET_KEY": "sk-test", "LANGFUSE_PUBLIC_KEY": "pk-test"},
            clear=True,
        ):
            with patch.dict("sys.modules", {"langfuse": MagicMock(Langfuse=mock_langfuse_cls)}):
                tracer = ObservabilityTracer()
                tracer._lazy_init()
                with tracer.trace("test_op", metadata={"key": "val"}) as t:
                    assert t is mock_trace_obj
                mock_trace_obj.update.assert_called_once_with(status_message="completed")

    def test_trace_does_not_crash_on_client_trace_exception(self):
        """trace() should not propagate exceptions from self._client.trace()."""
        mock_client = MagicMock()
        mock_client.trace.side_effect = RuntimeError("Langfuse API error")
        mock_langfuse_cls = MagicMock(return_value=mock_client)

        with patch.dict(
            os.environ,
            {"LANGFUSE_SECRET_KEY": "sk-test", "LANGFUSE_PUBLIC_KEY": "pk-test"},
            clear=True,
        ):
            with patch.dict("sys.modules", {"langfuse": MagicMock(Langfuse=mock_langfuse_cls)}):
                tracer = ObservabilityTracer()
                tracer._lazy_init()
                # Should NOT raise - must be caught internally
                with tracer.trace("failing_op") as t:
                    assert t is None  # Fallback to None on error

    def test_trace_memory_operation_does_not_crash_on_exception(self):
        """trace_memory_operation() should swallow exceptions from client."""
        mock_client = MagicMock()
        mock_client.trace.side_effect = RuntimeError("Langfuse API error")
        mock_langfuse_cls = MagicMock(return_value=mock_client)

        with patch.dict(
            os.environ,
            {"LANGFUSE_SECRET_KEY": "sk-test", "LANGFUSE_PUBLIC_KEY": "pk-test"},
            clear=True,
        ):
            with patch.dict("sys.modules", {"langfuse": MagicMock(Langfuse=mock_langfuse_cls)}):
                tracer = ObservabilityTracer()
                tracer._lazy_init()
                # Should NOT raise
                tracer.trace_memory_operation(
                    "store", "episodic", "agent_123", metadata={"test": True}
                )

    def test_init_langfuse_catches_non_import_exceptions(self):
        """_init_langfuse() should catch Exception, not just ImportError."""
        mock_langfuse_mod = MagicMock()
        mock_langfuse_mod.Langfuse.side_effect = ConnectionError("Network error")

        with patch.dict(
            os.environ,
            {"LANGFUSE_SECRET_KEY": "sk-test", "LANGFUSE_PUBLIC_KEY": "pk-test"},
            clear=True,
        ):
            with patch.dict("sys.modules", {"langfuse": mock_langfuse_mod}):
                tracer = ObservabilityTracer()
                # Should NOT raise - must be caught by broad except Exception
                assert tracer.enabled is False


class TestTracerInManagerProtection:
    """Tests for manager.py wrapping tracer calls in try/except."""

    @pytest.mark.asyncio
    async def test_store_memory_succeeds_when_tracer_throws(self):
        """store_memory must succeed even if tracer.trace_memory_operation raises."""
        from unittest.mock import AsyncMock

        mock_embed = MagicMock()
        mock_embed.generate_embedding = AsyncMock(
            return_value=MagicMock(embedding=[0.1] * 1024)
        )

        mock_tracer = MagicMock()
        mock_tracer.trace_memory_operation.side_effect = RuntimeError("Tracer crash!")

        # Patch get_embedding_service everywhere it is imported
        with (
            patch("src.embeddings.voyage_client.get_embedding_service", return_value=mock_embed),
            patch("src.memory.episodic.get_embedding_service", return_value=mock_embed),
            patch("src.memory.procedural.get_embedding_service", return_value=mock_embed),
            patch("src.memory.semantic.get_embedding_service", return_value=mock_embed),
            patch("src.memory.working.get_embedding_service", return_value=mock_embed),
            patch("src.memory.cache.get_embedding_service", return_value=mock_embed),
            patch("src.memory.entity.get_embedding_service", return_value=mock_embed),
            patch("src.memory.summary.get_embedding_service", return_value=mock_embed),
            patch("src.memory.manager.get_embedding_service", return_value=mock_embed),
            patch("src.memory.manager.get_tracer", return_value=mock_tracer),
        ):
            from src.memory.base import MemoryType
            from src.memory.manager import MemoryManager

            mock_db = MagicMock()
            mock_collection = MagicMock()
            mock_db.__getitem__ = MagicMock(return_value=mock_collection)

            manager = MemoryManager(mock_db)
            manager.embedding_service = mock_embed

            # Mock the actual store to succeed
            mock_store = MagicMock()
            mock_store.store = AsyncMock(return_value="mem_123")
            manager.stores[MemoryType.SEMANTIC] = mock_store

            # Should NOT raise despite tracer throwing
            result = await manager.store_memory(
                content="Test content",
                memory_type=MemoryType.SEMANTIC,
                agent_id="agent_1",
            )
            assert result == "mem_123"

    @pytest.mark.asyncio
    async def test_retrieve_memories_succeeds_when_tracer_throws(self):
        """retrieve_memories must succeed even if tracer.trace_memory_operation raises."""
        from unittest.mock import AsyncMock

        mock_embed = MagicMock()
        mock_embed.generate_embedding = AsyncMock(
            return_value=MagicMock(embedding=[0.1] * 1024)
        )

        mock_tracer = MagicMock()
        mock_tracer.trace_memory_operation.side_effect = RuntimeError("Tracer crash!")

        with (
            patch("src.embeddings.voyage_client.get_embedding_service", return_value=mock_embed),
            patch("src.memory.episodic.get_embedding_service", return_value=mock_embed),
            patch("src.memory.procedural.get_embedding_service", return_value=mock_embed),
            patch("src.memory.semantic.get_embedding_service", return_value=mock_embed),
            patch("src.memory.working.get_embedding_service", return_value=mock_embed),
            patch("src.memory.cache.get_embedding_service", return_value=mock_embed),
            patch("src.memory.entity.get_embedding_service", return_value=mock_embed),
            patch("src.memory.summary.get_embedding_service", return_value=mock_embed),
            patch("src.memory.manager.get_embedding_service", return_value=mock_embed),
            patch("src.memory.manager.get_tracer", return_value=mock_tracer),
        ):
            from src.memory.base import MemoryType
            from src.memory.manager import MemoryManager

            mock_db = MagicMock()
            mock_collection = MagicMock()
            mock_db.__getitem__ = MagicMock(return_value=mock_collection)

            manager = MemoryManager(mock_db)
            manager.embedding_service = mock_embed

            # Mock episodic store to return empty
            mock_episodic = MagicMock()
            mock_episodic.retrieve = AsyncMock(return_value=[])
            manager.stores[MemoryType.EPISODIC] = mock_episodic

            # Mock semantic store to return empty
            mock_semantic = MagicMock()
            mock_semantic.retrieve = AsyncMock(return_value=[])
            manager.stores[MemoryType.SEMANTIC] = mock_semantic

            # Mock procedural store
            mock_proc = MagicMock()
            mock_proc.retrieve = AsyncMock(return_value=[])
            manager.stores[MemoryType.PROCEDURAL] = mock_proc

            # Should NOT raise despite tracer throwing
            results = await manager.retrieve_memories(
                query="test query",
                agent_id="agent_1",
                use_cache=False,
            )
            assert isinstance(results, list)
