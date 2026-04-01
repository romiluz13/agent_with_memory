"""
RAG Evaluation Pipeline.

Measures retrieval accuracy and answer quality using LLM-as-judge.
RAGAS-inspired but custom implementation to avoid dependency conflicts.
Gracefully degrades when no LLM is configured.

Pattern from HybridRAG: configurable LLM evaluator with graceful degradation.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of a RAG evaluation run.

    Attributes:
        context_precision: Are retrieved contexts relevant? (0-1)
        context_recall: Are all relevant contexts retrieved? (0-1)
        answer_relevancy: Is the answer relevant to the question? (0-1)
        faithfulness: Is the answer grounded in context? (0-1)
        overall_score: Weighted average of all metrics.
        metadata: Evaluator metadata (evaluator name, errors, etc).
    """

    context_precision: float
    context_recall: float
    answer_relevancy: float
    faithfulness: float
    overall_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGEvaluator:
    """Evaluates RAG pipeline quality using LLM-as-judge.

    Uses a custom LLM-as-judge approach inspired by RAGAS metrics.
    Gracefully degrades to zero scores when no LLM is provided.

    Args:
        llm: Optional LLM instance with async ainvoke method.
            If None, all evaluations return zero scores.
    """

    _JUDGE_PROMPT = """Evaluate this RAG response on a scale of 0.0 to 1.0.

Question: {question}
Answer: {answer}
Retrieved Contexts: {contexts}
{ground_truth_section}

Score each metric:
1. context_precision: Are the retrieved contexts relevant to the question? (0-1)
2. context_recall: Do the contexts contain enough info to answer? (0-1)
3. answer_relevancy: Is the answer relevant to the question? (0-1)
4. faithfulness: Is the answer grounded in the provided contexts? (0-1)

Return ONLY JSON: {{"context_precision": 0.X, "context_recall": 0.X, "answer_relevancy": 0.X, "faithfulness": 0.X}}"""

    _RETRIEVAL_PROMPT = """Evaluate retrieval quality on a scale of 0.0 to 1.0.

Query: {query}
Retrieved Documents: {retrieved_docs}
Ground Truth Documents: {ground_truth_docs}

Score:
1. precision: What fraction of retrieved documents are relevant? (0-1)
2. recall: What fraction of relevant documents were retrieved? (0-1)

Return ONLY JSON: {{"precision": 0.X, "recall": 0.X}}"""

    _ANSWER_PROMPT = """Evaluate answer quality on a scale of 0.0 to 1.0.

Question: {question}
Answer: {answer}
Retrieved Contexts: {contexts}

Score:
1. relevancy: Is the answer relevant to the question? (0-1)
2. faithfulness: Is the answer grounded in the provided contexts? (0-1)

Return ONLY JSON: {{"relevancy": 0.X, "faithfulness": 0.X}}"""

    def __init__(self, llm: Any = None) -> None:
        self._llm = llm

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> EvalResult:
        """Evaluate a single RAG response.

        Args:
            question: The user's question.
            answer: The agent's answer.
            contexts: Retrieved context strings.
            ground_truth: Optional ground truth answer.

        Returns:
            EvalResult with precision, recall, relevancy, faithfulness.
        """
        if not self._llm:
            return EvalResult(
                context_precision=0.0,
                context_recall=0.0,
                answer_relevancy=0.0,
                faithfulness=0.0,
                overall_score=0.0,
                metadata={"evaluator": "none", "reason": "No LLM configured"},
            )

        return await self._evaluate_with_llm_judge(
            question, answer, contexts, ground_truth
        )

    async def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: list[str],
        ground_truth_docs: list[str],
    ) -> dict[str, float]:
        """Measure precision and recall of retrieved documents.

        Args:
            query: The search query.
            retrieved_docs: Documents returned by retrieval.
            ground_truth_docs: Known relevant documents.

        Returns:
            Dict with 'precision' and 'recall' float values.
        """
        if not self._llm:
            return {"precision": 0.0, "recall": 0.0}

        prompt = self._RETRIEVAL_PROMPT.format(
            query=query,
            retrieved_docs=json.dumps(retrieved_docs[:5]),
            ground_truth_docs=json.dumps(ground_truth_docs[:5]),
        )

        try:
            response = await self._llm.ainvoke(prompt)
            scores = self._parse_json_response(response.content)
            return {
                "precision": float(scores.get("precision", 0.0)),
                "recall": float(scores.get("recall", 0.0)),
            }
        except Exception as e:
            logger.warning("Retrieval evaluation failed: %s", e)
            return {"precision": 0.0, "recall": 0.0}

    async def evaluate_answer(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> dict[str, float]:
        """Measure answer relevancy and faithfulness.

        Args:
            question: The user's question.
            answer: The agent's answer.
            contexts: Retrieved context strings.

        Returns:
            Dict with 'relevancy' and 'faithfulness' float values.
        """
        if not self._llm:
            return {"relevancy": 0.0, "faithfulness": 0.0}

        prompt = self._ANSWER_PROMPT.format(
            question=question,
            answer=answer,
            contexts=json.dumps(contexts[:5]),
        )

        try:
            response = await self._llm.ainvoke(prompt)
            scores = self._parse_json_response(response.content)
            return {
                "relevancy": float(scores.get("relevancy", 0.0)),
                "faithfulness": float(scores.get("faithfulness", 0.0)),
            }
        except Exception as e:
            logger.warning("Answer evaluation failed: %s", e)
            return {"relevancy": 0.0, "faithfulness": 0.0}

    async def run_evaluation(
        self,
        test_cases: list[dict[str, Any]],
    ) -> list[EvalResult]:
        """Run full evaluation suite on test cases.

        Args:
            test_cases: List of dicts with keys: question, answer, contexts,
                and optionally ground_truth.

        Returns:
            List of EvalResult for each test case.
        """
        results = []
        for case in test_cases:
            result = await self.evaluate(
                question=case["question"],
                answer=case["answer"],
                contexts=case.get("contexts", []),
                ground_truth=case.get("ground_truth"),
            )
            results.append(result)
        return results

    async def _evaluate_with_llm_judge(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None,
    ) -> EvalResult:
        """Use LLM-as-judge for evaluation.

        Args:
            question: The user's question.
            answer: The agent's answer.
            contexts: Retrieved context strings.
            ground_truth: Optional ground truth answer.

        Returns:
            EvalResult with parsed scores from LLM.
        """
        ground_truth_section = (
            f"Ground Truth: {ground_truth}" if ground_truth else ""
        )
        prompt = self._JUDGE_PROMPT.format(
            question=question,
            answer=answer,
            contexts=json.dumps(contexts[:3]),
            ground_truth_section=ground_truth_section,
        )

        try:
            response = await self._llm.ainvoke(prompt)
            scores = self._parse_json_response(response.content)

            cp = float(scores.get("context_precision", 0.0))
            cr = float(scores.get("context_recall", 0.0))
            ar = float(scores.get("answer_relevancy", 0.0))
            ff = float(scores.get("faithfulness", 0.0))

            return EvalResult(
                context_precision=cp,
                context_recall=cr,
                answer_relevancy=ar,
                faithfulness=ff,
                overall_score=(cp + cr + ar + ff) / 4.0,
                metadata={"evaluator": "llm_judge"},
            )
        except Exception as e:
            logger.warning("LLM judge evaluation failed: %s", e)
            return EvalResult(
                context_precision=0.0,
                context_recall=0.0,
                answer_relevancy=0.0,
                faithfulness=0.0,
                overall_score=0.0,
                metadata={"evaluator": "llm_judge", "error": str(e)},
            )

    @staticmethod
    def _parse_json_response(content: str) -> dict[str, Any]:
        """Parse JSON from LLM response, tolerating surrounding text.

        Args:
            content: Raw LLM response text.

        Returns:
            Parsed dict from the JSON portion.

        Raises:
            ValueError: If no valid JSON object found.
        """
        text = content.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in response")
        return json.loads(text[start : end + 1])
