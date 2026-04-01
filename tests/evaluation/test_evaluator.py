"""Tests for RAG Evaluation Pipeline (src/evaluation/evaluator.py).

Tests cover:
- EvalResult dataclass structure
- RAGEvaluator initialization and graceful degradation
- LLM-as-judge evaluation with mocked LLM
- Batch evaluation
- Error handling when no LLM configured
- Retrieval evaluation (precision/recall)
- Answer evaluation (relevancy/faithfulness)
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.evaluation.evaluator import EvalResult, RAGEvaluator


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_eval_result_has_four_metrics(self):
        """EvalResult must have context_precision, context_recall, answer_relevancy, faithfulness."""
        result = EvalResult(
            context_precision=0.8,
            context_recall=0.7,
            answer_relevancy=0.9,
            faithfulness=0.85,
            overall_score=0.8125,
            metadata={"evaluator": "test"},
        )
        assert result.context_precision == 0.8
        assert result.context_recall == 0.7
        assert result.answer_relevancy == 0.9
        assert result.faithfulness == 0.85

    def test_eval_result_overall_score(self):
        """EvalResult overall_score is computed externally."""
        result = EvalResult(
            context_precision=1.0,
            context_recall=1.0,
            answer_relevancy=1.0,
            faithfulness=1.0,
            overall_score=1.0,
            metadata={},
        )
        assert result.overall_score == 1.0

    def test_eval_result_metadata(self):
        """EvalResult carries evaluator metadata."""
        result = EvalResult(
            context_precision=0.0,
            context_recall=0.0,
            answer_relevancy=0.0,
            faithfulness=0.0,
            overall_score=0.0,
            metadata={"evaluator": "llm_judge", "model": "gpt-4"},
        )
        assert result.metadata["evaluator"] == "llm_judge"


class TestRAGEvaluatorInit:
    """Tests for RAGEvaluator initialization."""

    def test_evaluator_without_llm(self):
        """Evaluator can be created without an LLM."""
        evaluator = RAGEvaluator()
        assert evaluator is not None

    def test_evaluator_with_llm(self):
        """Evaluator accepts an LLM for judge-based evaluation."""
        mock_llm = MagicMock()
        evaluator = RAGEvaluator(llm=mock_llm)
        assert evaluator is not None


class TestEvaluateNoLLM:
    """Tests for evaluation when no LLM is provided."""

    @pytest.mark.asyncio
    async def test_no_llm_returns_zeros(self):
        """Without LLM, all scores should be 0.0."""
        evaluator = RAGEvaluator()
        result = await evaluator.evaluate(
            question="What is Python?",
            answer="A programming language.",
            contexts=["Python is a high-level programming language."],
        )
        assert isinstance(result, EvalResult)
        assert result.context_precision == 0.0
        assert result.context_recall == 0.0
        assert result.answer_relevancy == 0.0
        assert result.faithfulness == 0.0
        assert result.metadata.get("reason") == "No LLM configured"


class TestEvaluateWithLLM:
    """Tests for LLM-as-judge evaluation."""

    @pytest.mark.asyncio
    async def test_llm_judge_returns_scores(self):
        """LLM judge returns parsed scores from LLM response."""
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "context_precision": 0.9,
            "context_recall": 0.8,
            "answer_relevancy": 0.85,
            "faithfulness": 0.75,
        })
        mock_llm.ainvoke.return_value = mock_response

        evaluator = RAGEvaluator(llm=mock_llm)
        result = await evaluator.evaluate(
            question="What is MongoDB?",
            answer="A NoSQL database.",
            contexts=["MongoDB is a document-oriented NoSQL database."],
        )
        assert isinstance(result, EvalResult)
        assert result.context_precision == 0.9
        assert result.context_recall == 0.8
        assert result.answer_relevancy == 0.85
        assert result.faithfulness == 0.75
        assert result.metadata["evaluator"] == "llm_judge"

    @pytest.mark.asyncio
    async def test_llm_judge_with_ground_truth(self):
        """LLM judge uses ground_truth when provided."""
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "context_precision": 0.95,
            "context_recall": 0.9,
            "answer_relevancy": 0.88,
            "faithfulness": 0.92,
        })
        mock_llm.ainvoke.return_value = mock_response

        evaluator = RAGEvaluator(llm=mock_llm)
        result = await evaluator.evaluate(
            question="What is MongoDB?",
            answer="A NoSQL database.",
            contexts=["MongoDB is a document-oriented NoSQL database."],
            ground_truth="MongoDB is a document database.",
        )
        assert isinstance(result, EvalResult)
        # Verify ground_truth was included in prompt
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert "Ground Truth" in call_args
        assert "MongoDB is a document database." in call_args

    @pytest.mark.asyncio
    async def test_llm_judge_handles_malformed_json(self):
        """LLM judge gracefully handles non-JSON LLM responses."""
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "I cannot evaluate this properly."
        mock_llm.ainvoke.return_value = mock_response

        evaluator = RAGEvaluator(llm=mock_llm)
        result = await evaluator.evaluate(
            question="What is X?",
            answer="Y",
            contexts=["Z"],
        )
        assert isinstance(result, EvalResult)
        assert result.context_precision == 0.0
        assert "error" in result.metadata

    @pytest.mark.asyncio
    async def test_llm_judge_handles_llm_exception(self):
        """LLM judge gracefully handles LLM invocation failures."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = RuntimeError("LLM down")

        evaluator = RAGEvaluator(llm=mock_llm)
        result = await evaluator.evaluate(
            question="What is X?",
            answer="Y",
            contexts=["Z"],
        )
        assert isinstance(result, EvalResult)
        assert result.context_precision == 0.0
        assert "error" in result.metadata


class TestEvaluateRetrieval:
    """Tests for retrieval-specific evaluation."""

    @pytest.mark.asyncio
    async def test_evaluate_retrieval_precision_recall(self):
        """evaluate_retrieval computes precision and recall."""
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "precision": 0.8,
            "recall": 0.6,
        })
        mock_llm.ainvoke.return_value = mock_response

        evaluator = RAGEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_retrieval(
            query="What is Python?",
            retrieved_docs=["Python is a language.", "Java is a language."],
            ground_truth_docs=["Python is a high-level language.", "Python was created by Guido."],
        )
        assert "precision" in result
        assert "recall" in result
        assert isinstance(result["precision"], float)
        assert isinstance(result["recall"], float)


class TestEvaluateAnswer:
    """Tests for answer-specific evaluation."""

    @pytest.mark.asyncio
    async def test_evaluate_answer_relevancy_faithfulness(self):
        """evaluate_answer computes relevancy and faithfulness."""
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "relevancy": 0.9,
            "faithfulness": 0.85,
        })
        mock_llm.ainvoke.return_value = mock_response

        evaluator = RAGEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_answer(
            question="What is MongoDB?",
            answer="MongoDB is a NoSQL database.",
            contexts=["MongoDB is a document-oriented NoSQL database."],
        )
        assert "relevancy" in result
        assert "faithfulness" in result
        assert isinstance(result["relevancy"], float)
        assert isinstance(result["faithfulness"], float)


class TestEvaluateBatch:
    """Tests for batch evaluation."""

    @pytest.mark.asyncio
    async def test_evaluate_batch_multiple_cases(self):
        """evaluate_batch processes multiple test cases."""
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "context_precision": 0.8,
            "context_recall": 0.7,
            "answer_relevancy": 0.9,
            "faithfulness": 0.85,
        })
        mock_llm.ainvoke.return_value = mock_response

        evaluator = RAGEvaluator(llm=mock_llm)
        test_cases = [
            {
                "question": "Q1",
                "answer": "A1",
                "contexts": ["C1"],
            },
            {
                "question": "Q2",
                "answer": "A2",
                "contexts": ["C2"],
                "ground_truth": "GT2",
            },
        ]
        results = await evaluator.run_evaluation(test_cases)
        assert len(results) == 2
        assert all(isinstance(r, EvalResult) for r in results)
