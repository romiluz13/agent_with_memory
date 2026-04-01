"""Tests for Evaluation API endpoint (src/api/routes/evaluation.py).

Tests cover:
- POST /api/v1/evaluate accepts question/answer/contexts
- Returns scores as JSON with expected shape
- Handles missing fields gracefully
- Validates request body
"""


from fastapi.testclient import TestClient

from src.api.main import app


class TestEvaluationEndpoint:
    """Tests for POST /api/v1/evaluate endpoint."""

    def test_evaluate_endpoint_exists(self):
        """The /api/v1/evaluate endpoint must be registered."""
        client = TestClient(app)
        # POST with valid body should not return 404
        response = client.post(
            "/api/v1/evaluate",
            json={
                "question": "What is Python?",
                "answer": "A programming language.",
                "contexts": ["Python is a high-level language."],
            },
        )
        assert response.status_code != 404, (
            f"Endpoint returned 404 -- route not registered. Got: {response.status_code}"
        )

    def test_evaluate_returns_four_metrics(self):
        """Endpoint returns context_precision, context_recall, answer_relevancy, faithfulness."""
        client = TestClient(app)
        response = client.post(
            "/api/v1/evaluate",
            json={
                "question": "What is Python?",
                "answer": "A programming language.",
                "contexts": ["Python is a high-level language."],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "context_precision" in data
        assert "context_recall" in data
        assert "answer_relevancy" in data
        assert "faithfulness" in data
        assert "overall_score" in data

    def test_evaluate_with_ground_truth(self):
        """Endpoint accepts optional ground_truth."""
        client = TestClient(app)
        response = client.post(
            "/api/v1/evaluate",
            json={
                "question": "What is Python?",
                "answer": "A programming language.",
                "contexts": ["Python is a high-level language."],
                "ground_truth": "Python is a general-purpose programming language.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "context_precision" in data

    def test_evaluate_missing_question_returns_422(self):
        """Endpoint rejects requests without question field."""
        client = TestClient(app)
        response = client.post(
            "/api/v1/evaluate",
            json={
                "answer": "A programming language.",
                "contexts": ["Python is a high-level language."],
            },
        )
        assert response.status_code == 422

    def test_evaluate_missing_answer_returns_422(self):
        """Endpoint rejects requests without answer field."""
        client = TestClient(app)
        response = client.post(
            "/api/v1/evaluate",
            json={
                "question": "What is Python?",
                "contexts": ["Python is a high-level language."],
            },
        )
        assert response.status_code == 422

    def test_evaluate_missing_contexts_returns_422(self):
        """Endpoint rejects requests without contexts field."""
        client = TestClient(app)
        response = client.post(
            "/api/v1/evaluate",
            json={
                "question": "What is Python?",
                "answer": "A programming language.",
            },
        )
        assert response.status_code == 422
