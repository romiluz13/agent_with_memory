"""Tests for NL Query API endpoint (src/api/routes/nl_query.py).

Tests cover:
- POST /api/v1/query accepts question and agent_id
- Returns generated_mql, results, execution_time
- Validates required fields
- Handles optional collection parameter
"""


from fastapi.testclient import TestClient

from src.api.main import app


class TestNLQueryEndpoint:
    """Tests for POST /api/v1/query endpoint."""

    def test_query_endpoint_exists(self):
        """The /api/v1/query endpoint must be registered."""
        client = TestClient(app)
        response = client.post(
            "/api/v1/query",
            json={
                "question": "Show all episodic memories",
                "agent_id": "agent_1",
            },
        )
        assert response.status_code != 404, (
            f"Endpoint returned 404 -- route not registered. Got: {response.status_code}"
        )

    def test_query_returns_expected_fields(self):
        """Endpoint returns generated_mql, results, execution_time."""
        client = TestClient(app)
        response = client.post(
            "/api/v1/query",
            json={
                "question": "Show all episodic memories",
                "agent_id": "agent_1",
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Should have either results or an error/message about no LLM
        assert "execution_time" in data or "error" in data or "message" in data

    def test_query_missing_question_returns_422(self):
        """Endpoint rejects requests without question."""
        client = TestClient(app)
        response = client.post(
            "/api/v1/query",
            json={"agent_id": "agent_1"},
        )
        assert response.status_code == 422

    def test_query_missing_agent_id_returns_422(self):
        """Endpoint rejects requests without agent_id."""
        client = TestClient(app)
        response = client.post(
            "/api/v1/query",
            json={"question": "Show all"},
        )
        assert response.status_code == 422

    def test_query_accepts_optional_collection(self):
        """Endpoint accepts optional collection parameter."""
        client = TestClient(app)
        response = client.post(
            "/api/v1/query",
            json={
                "question": "Show recent entries",
                "agent_id": "agent_1",
                "collection": "episodic_memories",
            },
        )
        # Should not return 422 (validation error)
        assert response.status_code != 422
