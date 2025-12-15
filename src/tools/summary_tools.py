"""
Summary Tools for Agent
LangChain tools for JIT summary expansion and context summarization.
Based on Oracle Memory Engineering pattern.
"""

import logging
from typing import Optional
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def create_expand_summary_tool(memory_manager):
    """
    Create expand_summary tool with bound memory_manager.

    Args:
        memory_manager: MemoryManager instance with summary store

    Returns:
        LangChain tool function
    """

    @tool
    async def expand_summary(summary_id: str) -> str:
        """
        Expand a summary reference to get the full original content.

        Use this when you see [Summary ID: xxx] and need more details
        about what was summarized. This is Just-In-Time (JIT) retrieval.

        Args:
            summary_id: The summary ID from the reference (e.g., "abc12345")

        Returns:
            Full original content that was summarized, or error message
        """
        try:
            full_content = await memory_manager.summary.expand_summary(summary_id)
            if full_content:
                return f"## Expanded Summary [{summary_id}]\n\n{full_content}"
            return f"Summary {summary_id} not found."
        except Exception as e:
            logger.error(f"Failed to expand summary {summary_id}: {e}")
            return f"Error expanding summary: {str(e)}"

    return expand_summary


def create_summarize_tool(memory_manager, llm, agent_id: str):
    """
    Create summarize_and_store tool with bound dependencies.

    Args:
        memory_manager: MemoryManager instance
        llm: LLM instance for summarization
        agent_id: Agent ID for memory ownership

    Returns:
        LangChain tool function
    """

    @tool
    async def summarize_and_store(
        text: str,
        thread_id: Optional[str] = None
    ) -> str:
        """
        Summarize long text and store it for later retrieval.

        Use this when:
        - Context is getting too long and you need to compress it
        - You want to save important information in a compact form
        - You need to create a reference for later expansion

        The summary will be stored and you'll receive a reference ID
        that can be expanded later using expand_summary.

        Args:
            text: The text to summarize
            thread_id: Optional thread ID to associate with the summary

        Returns:
            Summary reference in format: [Summary ID: xxx] Brief description
        """
        try:
            from ..context.summarizer import summarize_context

            result = await summarize_context(
                content=text,
                memory_manager=memory_manager,
                llm=llm,
                agent_id=agent_id
            )

            return f"[Summary ID: {result['id']}] {result['description']}"

        except Exception as e:
            logger.error(f"Failed to summarize and store: {e}")
            return f"Error creating summary: {str(e)}"

    return summarize_and_store


def create_summarize_conversation_tool(memory_manager, llm, agent_id: str):
    """
    Create tool to summarize and mark conversation as summarized.

    Args:
        memory_manager: MemoryManager instance
        llm: LLM instance
        agent_id: Agent ID

    Returns:
        LangChain tool function
    """

    @tool
    async def summarize_conversation(thread_id: str) -> str:
        """
        Summarize all unsummarized messages in a conversation thread.

        This will:
        1. Read all unsummarized messages in the thread
        2. Create a summary using the LLM
        3. Store the summary with full content for later expansion
        4. Mark the original messages as summarized (they won't be loaded again)

        Use this to compact long conversations while preserving access to details.

        Args:
            thread_id: The conversation thread ID to summarize

        Returns:
            Summary reference or status message
        """
        try:
            from ..context.engineer import ContextEngineer

            # Get unsummarized messages
            messages = await memory_manager.episodic.list_memories(
                filters={
                    "agent_id": agent_id,
                    "metadata.thread_id": thread_id,
                    "summary_id": None
                },
                limit=100
            )

            if not messages:
                return "No unsummarized messages found in this thread."

            # Format messages as text
            content = "\n".join([
                f"[{m.metadata.get('role', 'unknown')}]: {m.content}"
                for m in messages
            ])

            # Use context engineer to summarize and store
            engineer = ContextEngineer()
            result = await engineer.summarize_and_offload(
                context=content,
                memory_manager=memory_manager,
                llm=llm,
                agent_id=agent_id,
                thread_id=thread_id
            )

            # Mark messages as summarized
            marked_count = await memory_manager.episodic.mark_as_summarized(
                agent_id=agent_id,
                thread_id=thread_id,
                summary_id=result["summary_id"]
            )

            return (
                f"Summarized {len(messages)} messages.\n"
                f"Reference: {result['reference']}\n"
                f"Marked {marked_count} messages as summarized."
            )

        except Exception as e:
            logger.error(f"Failed to summarize conversation: {e}")
            return f"Error summarizing conversation: {str(e)}"

    return summarize_conversation


def create_summary_tools(
    memory_manager,
    llm,
    agent_id: str
) -> list:
    """
    Create all summary-related tools.

    Args:
        memory_manager: MemoryManager instance
        llm: LLM instance
        agent_id: Agent ID

    Returns:
        List of LangChain tools
    """
    return [
        create_expand_summary_tool(memory_manager),
        create_summarize_tool(memory_manager, llm, agent_id),
        create_summarize_conversation_tool(memory_manager, llm, agent_id)
    ]
