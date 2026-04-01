"""
Tests for Time-Travel Debugging API.
Phase 6: MongoDBSaver state history endpoints.

Tests the time_travel route module directly via unit tests.
Route registration is verified via a separate TestClient test.
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


def _load_module(name, path):
    """Load a module by file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tt_module():
    """Load time_travel module."""
    mod_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "api", "routes", "time_travel.py"
    )
    return _load_module("src.api.routes.time_travel", mod_path)


class TestTimeTravelRouteRegistration:
    """Tests for time-travel route registration in the app."""

    def test_time_travel_routes_registered(self):
        """Verify time-travel routes are accessible (registered in main.py)."""
        from fastapi.testclient import TestClient

        main_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "api", "main.py"
        )
        main_mod = _load_module("_test_main_tt", main_path)
        client = TestClient(main_mod.app)

        response = client.get("/api/v1/time-travel/history/test-thread")
        assert response.status_code != 404, "Time-travel route not registered"

    def test_hitl_routes_registered(self):
        """Verify HITL routes are also registered in main.py."""
        from fastapi.testclient import TestClient

        main_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "api", "main.py"
        )
        main_mod = _load_module("_test_main_hitl", main_path)
        client = TestClient(main_mod.app)

        response = client.get("/api/v1/hitl/pending/test-agent")
        assert response.status_code != 404, "HITL route not registered"


class TestTimeTravelModels:
    """Tests for Pydantic models in time_travel module."""

    def test_state_snapshot_model(self, tt_module):
        """StateSnapshot model validates correctly."""
        snapshot = tt_module.StateSnapshot(
            checkpoint_id="cp_1",
            thread_id="thread_1",
            timestamp="2026-01-01T00:00:00",
            messages=[{"type": "human", "content": "Hello"}],
            metadata={"created_at": "2026-01-01T00:00:00"},
        )
        assert snapshot.checkpoint_id == "cp_1"
        assert snapshot.thread_id == "thread_1"
        assert len(snapshot.messages) == 1

    def test_state_history_model(self, tt_module):
        """StateHistory model validates correctly."""
        history = tt_module.StateHistory(
            thread_id="thread_1",
            snapshots=[],
            total_count=0,
        )
        assert history.thread_id == "thread_1"
        assert history.total_count == 0

    def test_replay_request_model(self, tt_module):
        """ReplayRequest requires message field."""
        req = tt_module.ReplayRequest(message="Try again")
        assert req.message == "Try again"

    def test_replay_response_model(self, tt_module):
        """ReplayResponse model validates correctly."""
        resp = tt_module.ReplayResponse(
            status="replay_ready",
            thread_id="thread_1",
            from_checkpoint="cp_1",
            new_thread_id="thread_1-replay-abc12345",
            message="Try again",
        )
        assert resp.status == "replay_ready"
        assert resp.new_thread_id == "thread_1-replay-abc12345"


class TestGetStateHistoryEndpoint:
    """Tests for GET /history/{thread_id} endpoint logic."""

    @pytest.mark.asyncio
    async def test_returns_503_when_no_mongodb(self, tt_module):
        """Return 503 when MONGODB_URI is not set."""
        from fastapi import HTTPException

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MONGODB_URI", None)
            with pytest.raises(HTTPException) as exc_info:
                await tt_module.get_state_history("test-thread", limit=50)
            assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_returns_snapshots(self, tt_module):
        """Return state history snapshots."""
        mock_saver = MagicMock()
        mock_state = MagicMock()
        mock_state.config = {"configurable": {"checkpoint_id": "cp_1"}}
        mock_state.metadata = {"created_at": "2026-01-01T00:00:00"}
        mock_msg = MagicMock()
        mock_msg.type = "human"
        mock_msg.content = "Hello"
        mock_state.values = {"messages": [mock_msg]}
        mock_saver.get_state_history.return_value = [mock_state]

        mock_saver_cls = MagicMock(return_value=mock_saver)
        mock_client_cls = MagicMock(return_value=MagicMock())

        tt_module.MongoDBSaver = mock_saver_cls
        tt_module.MongoClient = mock_client_cls

        try:
            with patch.dict(os.environ, {"MONGODB_URI": "mongodb://localhost:27017"}):
                result = await tt_module.get_state_history("test-thread", limit=50)

            assert result.thread_id == "test-thread"
            assert len(result.snapshots) == 1
            assert result.snapshots[0].checkpoint_id == "cp_1"
        finally:
            tt_module.MongoDBSaver = None
            tt_module.MongoClient = None


class TestGetSnapshotEndpoint:
    """Tests for GET /snapshot/{thread_id}/{checkpoint_id} endpoint logic."""

    @pytest.mark.asyncio
    async def test_returns_404_when_not_found(self, tt_module):
        """Return 404 when checkpoint doesn't exist."""
        from fastapi import HTTPException

        mock_saver = MagicMock()
        mock_saver.get.return_value = None

        mock_saver_cls = MagicMock(return_value=mock_saver)
        mock_client_cls = MagicMock(return_value=MagicMock())

        tt_module.MongoDBSaver = mock_saver_cls
        tt_module.MongoClient = mock_client_cls

        try:
            with patch.dict(os.environ, {"MONGODB_URI": "mongodb://localhost:27017"}):
                with pytest.raises(HTTPException) as exc_info:
                    await tt_module.get_state_snapshot("test-thread", "nonexistent-cp")
                assert exc_info.value.status_code == 404
        finally:
            tt_module.MongoDBSaver = None
            tt_module.MongoClient = None

    @pytest.mark.asyncio
    async def test_returns_snapshot_data(self, tt_module):
        """Return specific checkpoint state."""
        mock_saver = MagicMock()
        mock_state = MagicMock()
        mock_state.metadata = {"created_at": "2026-01-01T00:00:00"}
        mock_msg = MagicMock()
        mock_msg.type = "ai"
        mock_msg.content = "Hi there"
        mock_state.values = {"messages": [mock_msg]}
        mock_saver.get.return_value = mock_state

        mock_saver_cls = MagicMock(return_value=mock_saver)
        mock_client_cls = MagicMock(return_value=MagicMock())

        tt_module.MongoDBSaver = mock_saver_cls
        tt_module.MongoClient = mock_client_cls

        try:
            with patch.dict(os.environ, {"MONGODB_URI": "mongodb://localhost:27017"}):
                result = await tt_module.get_state_snapshot("test-thread", "cp_1")

            assert result.checkpoint_id == "cp_1"
            assert result.thread_id == "test-thread"
            assert len(result.messages) == 1
            assert result.messages[0]["content"] == "Hi there"
        finally:
            tt_module.MongoDBSaver = None
            tt_module.MongoClient = None


class TestReplayEndpoint:
    """Tests for POST /replay/{thread_id}/{checkpoint_id} endpoint logic."""

    @pytest.mark.asyncio
    async def test_returns_404_when_checkpoint_missing(self, tt_module):
        """Return 404 when checkpoint doesn't exist for replay."""
        from fastapi import HTTPException

        mock_saver = MagicMock()
        mock_saver.get.return_value = None

        mock_saver_cls = MagicMock(return_value=mock_saver)
        mock_client_cls = MagicMock(return_value=MagicMock())

        tt_module.MongoDBSaver = mock_saver_cls
        tt_module.MongoClient = mock_client_cls

        try:
            with patch.dict(os.environ, {"MONGODB_URI": "mongodb://localhost:27017"}):
                req = tt_module.ReplayRequest(message="Try again")
                with pytest.raises(HTTPException) as exc_info:
                    await tt_module.replay_from_checkpoint("test-thread", "bad-cp", req)
                assert exc_info.value.status_code == 404
        finally:
            tt_module.MongoDBSaver = None
            tt_module.MongoClient = None

    @pytest.mark.asyncio
    async def test_returns_replay_metadata(self, tt_module):
        """Return replay metadata when checkpoint exists."""
        mock_saver = MagicMock()
        mock_saver.get.return_value = MagicMock()

        mock_saver_cls = MagicMock(return_value=mock_saver)
        mock_client_cls = MagicMock(return_value=MagicMock())

        tt_module.MongoDBSaver = mock_saver_cls
        tt_module.MongoClient = mock_client_cls

        try:
            with patch.dict(os.environ, {"MONGODB_URI": "mongodb://localhost:27017"}):
                req = tt_module.ReplayRequest(message="Try again")
                result = await tt_module.replay_from_checkpoint("test-thread", "cp_1", req)

            assert result.status == "replay_ready"
            assert result.thread_id == "test-thread"
            assert result.from_checkpoint == "cp_1"
            assert "test-thread-replay-" in result.new_thread_id
            assert result.message == "Try again"
        finally:
            tt_module.MongoDBSaver = None
            tt_module.MongoClient = None
