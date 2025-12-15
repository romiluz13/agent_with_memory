"""
Context Summarizer
LLM-powered summarization with consistent quality.
Based on Oracle Memory Engineering pattern.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SummarizationConfig:
    """Configuration for summarization."""
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_input_length: int = 3000
    max_summary_tokens: int = 200
    max_label_tokens: int = 30


class ContextSummarizer:
    """
    Summarizes context using LLM with consistent quality.
    """

    SUMMARIZE_PROMPT = """Summarize the following content in 2-3 sentences.
Preserve key facts, decisions, and important details.
Be concise but comprehensive.

Content:
{content}

Summary:"""

    LABEL_PROMPT = """Write a 10-word (maximum) label/title for this summary.
Be descriptive and specific.

Summary: {summary}

Label:"""

    def __init__(self, config: Optional[SummarizationConfig] = None):
        """Initialize summarizer with config."""
        self.config = config or SummarizationConfig()

    async def summarize(
        self,
        content: str,
        llm,
        max_input_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Summarize content and generate a label.

        Args:
            content: Content to summarize
            llm: LLM instance for generation
            max_input_length: Override max input length

        Returns:
            Dict with 'summary' and 'description' keys
        """
        max_len = max_input_length or self.config.max_input_length

        # Truncate if needed
        truncated = content[:max_len] if len(content) > max_len else content

        # Generate summary
        summary_prompt = self.SUMMARIZE_PROMPT.format(content=truncated)
        summary_response = await llm.ainvoke(summary_prompt)
        summary = summary_response.content.strip()

        # Generate label/description
        label_prompt = self.LABEL_PROMPT.format(summary=summary)
        label_response = await llm.ainvoke(label_prompt)
        description = label_response.content.strip()

        # Clean up label (remove quotes, limit length)
        description = description.strip('"\'')
        if len(description.split()) > 12:
            description = " ".join(description.split()[:10]) + "..."

        return {
            "summary": summary,
            "description": description,
            "original_length": len(content),
            "truncated": len(content) > max_len
        }

    async def summarize_conversation(
        self,
        messages: list,
        llm
    ) -> Dict[str, Any]:
        """
        Summarize a conversation (list of messages).

        Args:
            messages: List of message dicts with 'role' and 'content'
            llm: LLM instance

        Returns:
            Summary result
        """
        # Format messages as text
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted.append(f"[{role}]: {content}")

        conversation_text = "\n".join(formatted)
        return await self.summarize(conversation_text, llm)

    async def batch_summarize(
        self,
        contents: list,
        llm
    ) -> list:
        """
        Summarize multiple contents.

        Args:
            contents: List of content strings
            llm: LLM instance

        Returns:
            List of summary results
        """
        results = []
        for content in contents:
            try:
                result = await self.summarize(content, llm)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to summarize content: {e}")
                results.append({
                    "summary": "Failed to summarize",
                    "description": "Error",
                    "error": str(e)
                })
        return results


async def summarize_context(
    content: str,
    memory_manager,
    llm,
    agent_id: str = "system"
) -> Dict[str, Any]:
    """
    Convenience function to summarize and store.

    Args:
        content: Content to summarize
        memory_manager: MemoryManager instance
        llm: LLM instance
        agent_id: Agent ID

    Returns:
        Result with summary_id, summary, description
    """
    import uuid

    summarizer = ContextSummarizer()
    result = await summarizer.summarize(content, llm)

    summary_id = str(uuid.uuid4())[:8]

    await memory_manager.summary.store_summary(
        summary_id=summary_id,
        full_content=content,
        summary=result["summary"],
        description=result["description"],
        agent_id=agent_id
    )

    return {
        "id": summary_id,
        "summary": result["summary"],
        "description": result["description"]
    }
