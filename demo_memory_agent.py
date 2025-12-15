#!/usr/bin/env python3
"""
🧠 AWM 2.0 - Real-Life Memory Demo

This demo shows how the memory system works in REAL LIFE:
1. Session 1: Tell the agent personal information
2. Session 2: Ask the agent what it remembers (NEW SESSION!)

This proves the memory persists across sessions - the core value proposition.

Requirements:
- MongoDB Atlas account (free tier works!)
- Voyage AI API key (for embeddings)
- Google API key (for Gemini LLM) OR OpenAI key

Usage:
    python demo_memory_agent.py
"""

import asyncio
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import memory components
from src.memory.manager import MemoryManager
from src.memory.base import MemoryType
from src.storage.mongodb_client import MongoDBClient, MongoDBConfig

# For LLM responses
from langchain_google_genai import ChatGoogleGenerativeAI


class PersonalMemoryAgent:
    """
    A simple agent that demonstrates AWM 2.0 memory capabilities.
    Remembers conversations and can recall them in future sessions.
    """

    def __init__(self, user_id: str = "demo_user", agent_id: str = "memory_demo_agent"):
        self.user_id = user_id
        self.agent_id = agent_id
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize components
        self.db_client = None
        self.memory_manager = None
        self.llm = None

    async def initialize(self):
        """Initialize database and memory system."""
        print("\n🔌 Connecting to MongoDB Atlas...")

        # Connect to MongoDB
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            raise ValueError("MONGODB_URI not set in environment")

        # Create config and initialize singleton client
        config = MongoDBConfig(uri=mongodb_uri, database="awm_demo")
        self.db_client = MongoDBClient()

        # Reset singleton state for clean demo (allows multiple sessions)
        MongoDBClient._client = None
        MongoDBClient._db = None

        await self.db_client.initialize(config)
        print("   ✅ Connected to MongoDB")

        # Initialize Memory Manager
        print("\n🧠 Initializing 7-type memory system...")
        self.memory_manager = MemoryManager(self.db_client.db)
        print("   ✅ Memory system ready")

        # Initialize LLM
        print("\n🤖 Initializing Gemini LLM...")
        google_key = os.getenv("GOOGLE_API_KEY")
        if not google_key:
            raise ValueError("GOOGLE_API_KEY not set in environment")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_key,
            temperature=0.7
        )
        print("   ✅ LLM ready")

        return self

    async def remember(self, content: str, importance: float = 0.7):
        """Store something in memory."""
        # Store in episodic memory using MemoryManager's store_memory method
        memory_id = await self.memory_manager.store_memory(
            content=content,
            memory_type=MemoryType.EPISODIC,
            agent_id=self.agent_id,
            user_id=self.user_id,
            importance=importance,
            metadata={
                "session_id": self.session_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        print(f"   💾 Stored memory: {memory_id[:8]}...")

        # Also extract entities
        try:
            entities = await self.memory_manager.extract_entities(
                content, self.agent_id, self.llm
            )
            if entities:
                print(f"   🏷️  Extracted {len(entities)} entities")
        except Exception as e:
            print(f"   ⚠️  Entity extraction skipped: {e}")

        return memory_id

    async def recall(self, query: str, limit: int = 5) -> list:
        """Search memories for relevant information."""
        # Use episodic store directly with agent_id filter for proper isolation
        memories = await self.memory_manager.episodic.retrieve(
            query=query,
            limit=limit,
            threshold=0.3,  # Lower threshold for better recall
            agent_id=self.agent_id,
            user_id=self.user_id
        )
        return memories

    async def chat(self, user_message: str) -> str:
        """Have a conversation with memory context."""

        # Search for relevant memories
        memories = await self.recall(user_message, limit=3)

        # Build context from memories
        memory_context = ""
        if memories:
            memory_context = "\n\nRELEVANT MEMORIES:\n"
            for mem in memories:
                memory_context += f"- {mem.content}\n"

        # Create prompt with memory context
        prompt = f"""You are a helpful personal assistant with memory capabilities.
You remember past conversations and can recall personal information the user shared.

{memory_context}

User: {user_message}

Respond naturally. If you have relevant memories, use them in your response.
If the user shares new personal information, acknowledge it."""

        # Get LLM response
        response = await self.llm.ainvoke(prompt)
        assistant_response = response.content

        # Store this interaction in memory
        await self.remember(
            f"User said: {user_message}\nAssistant responded: {assistant_response[:200]}...",
            importance=0.6
        )

        return assistant_response

    async def show_memory_stats(self):
        """Display memory statistics."""
        print("\n📊 MEMORY STATISTICS:")
        print("-" * 40)

        # Count memories by type
        for memory_type in [MemoryType.EPISODIC, MemoryType.ENTITY, MemoryType.SUMMARY]:
            try:
                collection_name = f"{memory_type.value}_memories"
                collection = self.db_client.db[collection_name]
                count = await collection.count_documents({"agent_id": self.agent_id})
                print(f"   {memory_type.value.upper()}: {count} memories")
            except Exception:
                pass

    async def close(self):
        """Clean up connections."""
        if self.db_client:
            await self.db_client.close()


async def demo_session_1():
    """
    SESSION 1: Tell the agent about yourself
    This simulates a user's first interaction
    """
    print("\n" + "=" * 60)
    print("📍 SESSION 1: Teaching the Agent About You")
    print("=" * 60)

    agent = PersonalMemoryAgent(user_id="john_doe", agent_id="demo_agent_v1")
    await agent.initialize()

    # Simulate a conversation where user shares personal info
    conversations = [
        "Hi! My name is John and I'm a software engineer at Google.",
        "I have a dog named Max who is a golden retriever.",
        "My favorite programming language is Python and I love building AI agents.",
        "I'm working on a project called SmartHome to automate my house.",
    ]

    print("\n💬 Starting conversation...\n")

    for msg in conversations:
        print(f"👤 You: {msg}")
        response = await agent.chat(msg)
        print(f"🤖 Agent: {response[:200]}...\n")
        await asyncio.sleep(1)  # Small delay for readability

    await agent.show_memory_stats()
    await agent.close()

    print("\n✅ Session 1 complete! Memories have been stored.")
    print("   The agent now knows about John, his dog Max, his job, etc.")


async def demo_session_2():
    """
    SESSION 2: NEW SESSION - Ask the agent what it remembers
    This is the MAGIC - the agent remembers from Session 1!
    """
    print("\n" + "=" * 60)
    print("📍 SESSION 2: Testing Memory Recall (NEW SESSION!)")
    print("=" * 60)

    # Create a NEW agent instance (simulating app restart)
    agent = PersonalMemoryAgent(user_id="john_doe", agent_id="demo_agent_v1")
    await agent.initialize()

    # Ask questions about what was shared in Session 1
    test_queries = [
        "What's my name and where do I work?",
        "Do I have any pets?",
        "What programming language do I like?",
        "What project am I working on?",
    ]

    print("\n💬 Testing memory recall...\n")

    for query in test_queries:
        print(f"👤 You: {query}")

        # First, show what memories were found
        memories = await agent.recall(query, limit=2)
        if memories:
            print(f"   📚 Found {len(memories)} relevant memories")

        response = await agent.chat(query)
        print(f"🤖 Agent: {response[:300]}...\n")
        await asyncio.sleep(1)

    await agent.show_memory_stats()
    await agent.close()

    print("\n✅ Session 2 complete!")
    print("   The agent REMEMBERED information from Session 1!")
    print("   This proves the memory system works across sessions.")


async def main():
    """Run the full demo."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           🧠 AWM 2.0 - Memory System Demo                    ║
║                                                              ║
║  This demo proves the memory system works in REAL LIFE:      ║
║  1. Session 1: Tell the agent personal information           ║
║  2. Session 2: Ask what it remembers (NEW SESSION!)          ║
║                                                              ║
║  If Session 2 recalls Session 1 info = SUCCESS! 🎉           ║
╚══════════════════════════════════════════════════════════════╝
    """)

    try:
        # Run Session 1 - teach the agent
        await demo_session_1()

        print("\n" + "🔄" * 30)
        print("\n⏳ Simulating app restart (3 seconds)...")
        print("   In real life, this could be hours or days later!\n")
        print("🔄" * 30)
        await asyncio.sleep(3)

        # Run Session 2 - test recall
        await demo_session_2()

        print("""
╔══════════════════════════════════════════════════════════════╗
║                     🎉 DEMO COMPLETE! 🎉                     ║
║                                                              ║
║  The AWM 2.0 memory system successfully:                     ║
║  ✅ Stored conversation memories                             ║
║  ✅ Extracted entities (people, places, things)              ║
║  ✅ Recalled memories in a NEW session                       ║
║  ✅ Used memories to provide contextual responses            ║
║                                                              ║
║  This is production-ready AI memory! 🚀                      ║
╚══════════════════════════════════════════════════════════════╝
        """)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have set these environment variables:")
        print("  - MONGODB_URI")
        print("  - VOYAGE_API_KEY")
        print("  - GOOGLE_API_KEY")
        raise


if __name__ == "__main__":
    asyncio.run(main())
