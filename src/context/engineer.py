"""
Context Engineer
Monitors token usage and auto-summarizes at threshold.
Based on Oracle Memory Engineering pattern.
"""

import logging
import uuid
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Model token limits
MODEL_TOKEN_LIMITS = {
    # OpenAI Models
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    # Anthropic Models
    "claude-3-5-sonnet": 200000,
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-opus-4-5": 200000,
    "claude-sonnet-4": 200000,
    # Google Models
    "gemini-pro": 32000,
    "gemini-1.5-pro": 1000000,
}


@dataclass
class ContextUsage:
    """Context window usage statistics."""
    tokens: int
    max_tokens: int
    percent: float
    remaining: int


class ContextEngineer:
    """
    Manages context window efficiently.
    Monitors usage, triggers summarization, and manages references.
    """

    # Estimation parameters
    CHARS_PER_TOKEN = 4
    DEFAULT_THRESHOLD = 0.80
    BUFFER_PERCENTAGE = 0.20

    def __init__(
        self,
        threshold: float = 0.80,
        default_model: str = "gpt-4o-mini"
    ):
        """
        Initialize Context Engineer.

        Args:
            threshold: Percentage threshold to trigger compression (0.0-1.0)
            default_model: Default model for token limit lookup
        """
        self.threshold = threshold
        self.default_model = default_model
        self._summaries_created: List[Dict[str, Any]] = []

    def calculate_usage(
        self,
        context: str,
        model: Optional[str] = None
    ) -> ContextUsage:
        """
        Calculate context window usage.

        Args:
            context: Current context string
            model: Model name for token limit

        Returns:
            ContextUsage with tokens, max, percent, remaining
        """
        model = model or self.default_model
        estimated_tokens = len(context) // self.CHARS_PER_TOKEN
        max_tokens = self._get_max_tokens(model)
        percentage = (estimated_tokens / max_tokens) * 100
        remaining = max_tokens - estimated_tokens

        return ContextUsage(
            tokens=estimated_tokens,
            max_tokens=max_tokens,
            percent=round(percentage, 1),
            remaining=remaining
        )

    def should_compress(
        self,
        context: str,
        model: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> bool:
        """
        Check if context should be compressed.

        Args:
            context: Current context string
            model: Model name
            threshold: Override default threshold

        Returns:
            True if compression recommended
        """
        usage = self.calculate_usage(context, model)
        threshold_pct = (threshold or self.threshold) * 100
        return usage.percent > threshold_pct

    async def summarize_and_offload(
        self,
        context: str,
        memory_manager,
        llm,
        agent_id: str,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Summarize context and store in summary memory.

        Args:
            context: Context to summarize
            memory_manager: MemoryManager with summary store
            llm: LLM for summarization
            agent_id: Agent ID for memory ownership
            thread_id: Optional thread ID

        Returns:
            Dict with summary_id, description, reference, and new context
        """
        from .summarizer import ContextSummarizer

        summarizer = ContextSummarizer()

        # Generate summary
        summary_result = await summarizer.summarize(context, llm)

        # Generate unique ID
        summary_id = str(uuid.uuid4())[:8]

        # Store in summary memory
        await memory_manager.summary.store_summary(
            summary_id=summary_id,
            full_content=context,
            summary=summary_result["summary"],
            description=summary_result["description"],
            agent_id=agent_id,
            thread_id=thread_id
        )

        # Create reference for compact context
        reference = self.create_reference(summary_id, summary_result["description"])

        result = {
            "summary_id": summary_id,
            "summary": summary_result["summary"],
            "description": summary_result["description"],
            "reference": reference,
            "original_length": len(context),
            "compressed_length": len(reference)
        }

        self._summaries_created.append(result)
        logger.info(f"Created summary {summary_id}: {summary_result['description']}")

        return result

    def create_reference(self, summary_id: str, description: str) -> str:
        """
        Create a compact reference for a summary.

        Args:
            summary_id: Summary ID
            description: Brief description

        Returns:
            Formatted reference string
        """
        return f"[Summary ID: {summary_id}] {description}"

    def build_context_with_references(
        self,
        summaries: List[Dict[str, str]],
        current_context: str
    ) -> str:
        """
        Build context including summary references.

        Args:
            summaries: List of {summary_id, description} dicts
            current_context: Current (non-summarized) context

        Returns:
            Combined context with references
        """
        if not summaries:
            return current_context

        references_section = "## Summary Memory\nUse expand_summary(id) to get full content:\n"
        for s in summaries:
            references_section += f"  - [ID: {s['summary_id']}] {s['description']}\n"

        return f"{references_section}\n{current_context}"

    def get_summaries_created(self) -> List[Dict[str, Any]]:
        """Get list of summaries created in this session."""
        return self._summaries_created.copy()

    def clear_session_summaries(self):
        """Clear the session summary tracking."""
        self._summaries_created.clear()

    def _get_max_tokens(self, model: str) -> int:
        """Get max tokens for a model."""
        # Try exact match
        if model in MODEL_TOKEN_LIMITS:
            return MODEL_TOKEN_LIMITS[model]

        # Try partial match
        for key, value in MODEL_TOKEN_LIMITS.items():
            if key in model.lower():
                return value

        # Default fallback
        logger.warning(f"Unknown model '{model}', using default 128000 tokens")
        return 128000

    def format_usage_string(self, usage: ContextUsage) -> str:
        """Format usage for display."""
        return f"{usage.percent}% ({usage.tokens:,}/{usage.max_tokens:,} tokens)"
