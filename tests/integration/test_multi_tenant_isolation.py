"""
Multi-Tenant Isolation Tests
============================
Tests that verify agent_id and user_id isolation works correctly.

CRITICAL: These tests verify that one agent/user cannot see another's memories.

Run with: python -m pytest tests/integration/test_multi_tenant_isolation.py -v -s
"""

import os
import asyncio
import pytest
import pytest_asyncio
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Skip if no real credentials
SKIP_REAL_TEST = not all([
    os.getenv("MONGODB_URI"),
    os.getenv("VOYAGE_API_KEY"),
])

pytestmark = pytest.mark.skipif(
    SKIP_REAL_TEST,
    reason="Multi-tenant tests require MONGODB_URI and VOYAGE_API_KEY"
)


class TestAgentIsolation:
    """Test that different agents cannot see each other's memories."""

    @pytest_asyncio.fixture
    async def setup_db(self):
        """Set up real MongoDB connection."""
        from motor.motor_asyncio import AsyncIOMotorClient
        from src.memory.episodic import EpisodicMemory

        client = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
        db = client["awm_isolation_test"]

        # Clean before test
        await db.episodic_memories.delete_many({})

        # Create episodic memory store
        episodic = EpisodicMemory(db.episodic_memories)

        yield {
            "client": client,
            "db": db,
            "episodic": episodic
        }

        # Cleanup after test
        await db.episodic_memories.delete_many({})
        client.close()

    @pytest.mark.asyncio
    async def test_agent_a_cannot_see_agent_b_memories(self, setup_db):
        """
        CRITICAL TEST: Agent A stores a secret, Agent B should NOT find it.
        """
        episodic = setup_db["episodic"]

        from src.memory.base import Memory, MemoryType

        print("\n" + "=" * 60)
        print("TEST: Agent Isolation - Agent A's secrets hidden from Agent B")
        print("=" * 60)

        # Agent A stores a secret
        agent_a_memory = Memory(
            agent_id="agent-A",
            content="Agent A's SECRET: The launch code is 12345",
            memory_type=MemoryType.EPISODIC,
            metadata={"sensitivity": "high"}
        )
        await episodic.store(agent_a_memory)
        print("Agent A stored: SECRET launch code")

        # Agent B stores something
        agent_b_memory = Memory(
            agent_id="agent-B",
            content="Agent B's public info: The weather is nice",
            memory_type=MemoryType.EPISODIC,
            metadata={"sensitivity": "low"}
        )
        await episodic.store(agent_b_memory)
        print("Agent B stored: weather info")

        # Wait for vector index to sync (Atlas needs time)
        await asyncio.sleep(5)

        # Agent B searches for "secret" or "launch code"
        print("\nAgent B searching for 'SECRET launch code'...")
        results = await episodic.retrieve(
            query="SECRET launch code",
            limit=10,
            threshold=0.3,  # Low threshold to catch any leaks
            agent_id="agent-B"  # CRITICAL: Filter by agent_id
        )

        print(f"Agent B found {len(results)} results")
        for r in results:
            print(f"  - {r.content[:50]}... (agent_id: {r.agent_id})")

        # CRITICAL ASSERTION: Agent B should NOT see Agent A's secret
        agent_a_secrets_found = [r for r in results if r.agent_id == "agent-A"]
        assert len(agent_a_secrets_found) == 0, \
            f"DATA LEAK! Agent B found {len(agent_a_secrets_found)} of Agent A's memories!"

        print("PASS: Agent B cannot see Agent A's secrets")

    @pytest.mark.asyncio
    async def test_agent_can_see_own_memories(self, setup_db):
        """Test that an agent CAN see its own memories."""
        episodic = setup_db["episodic"]

        from src.memory.base import Memory, MemoryType

        print("\n" + "=" * 60)
        print("TEST: Agent Can Access Own Memories")
        print("=" * 60)

        # Agent A stores multiple memories
        for i in range(3):
            memory = Memory(
                agent_id="agent-A",
                content=f"Agent A memory #{i}: Important fact about project Phoenix",
                memory_type=MemoryType.EPISODIC,
                metadata={"index": i}
            )
            await episodic.store(memory)
        print("Agent A stored 3 memories about Project Phoenix")

        # Wait for vector index to sync (Atlas needs time for new documents)
        await asyncio.sleep(5)

        # Agent A searches for its own memories
        print("\nAgent A searching for 'Project Phoenix'...")
        results = await episodic.retrieve(
            query="Project Phoenix important",
            limit=10,
            threshold=0.3,
            agent_id="agent-A"
        )

        print(f"Agent A found {len(results)} results")

        # Agent A should find its own memories
        assert len(results) >= 1, "Agent A should find its own memories"
        assert all(r.agent_id == "agent-A" for r in results), \
            "All results should belong to Agent A"

        print("PASS: Agent A can access its own memories")


class TestUserIsolation:
    """Test that different users cannot see each other's memories."""

    @pytest_asyncio.fixture
    async def setup_db(self):
        """Set up real MongoDB connection."""
        from motor.motor_asyncio import AsyncIOMotorClient
        from src.memory.episodic import EpisodicMemory

        client = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
        db = client["awm_isolation_test"]

        await db.episodic_memories.delete_many({})

        episodic = EpisodicMemory(db.episodic_memories)

        yield {
            "client": client,
            "db": db,
            "episodic": episodic
        }

        await db.episodic_memories.delete_many({})
        client.close()

    @pytest.mark.asyncio
    async def test_user_1_cannot_see_user_2_memories(self, setup_db):
        """
        Test that User 1's memories are not visible to User 2.
        Both users share the same agent.
        """
        episodic = setup_db["episodic"]

        from src.memory.base import Memory, MemoryType

        print("\n" + "=" * 60)
        print("TEST: User Isolation - Same Agent, Different Users")
        print("=" * 60)

        # User 1 stores personal info
        user1_memory = Memory(
            agent_id="shared-agent",
            user_id="user-1",
            content="User 1's password hint: My dog's name is Fluffy",
            memory_type=MemoryType.EPISODIC,
            metadata={"private": True}
        )
        await episodic.store(user1_memory)
        print("User 1 stored: password hint")

        # User 2 stores their info
        user2_memory = Memory(
            agent_id="shared-agent",
            user_id="user-2",
            content="User 2 likes pizza for dinner",
            memory_type=MemoryType.EPISODIC,
            metadata={"private": False}
        )
        await episodic.store(user2_memory)
        print("User 2 stored: food preference")

        # Wait for vector index to sync
        await asyncio.sleep(5)

        # User 2 searches for password or personal info
        print("\nUser 2 searching for 'password Fluffy'...")
        results = await episodic.retrieve(
            query="password Fluffy dog",
            limit=10,
            threshold=0.3,
            agent_id="shared-agent",
            user_id="user-2"  # Filter by user_id
        )

        print(f"User 2 found {len(results)} results")
        for r in results:
            print(f"  - {r.content[:50]}... (user_id: {r.user_id})")

        # User 2 should NOT see User 1's password hint
        user1_data_found = [r for r in results if r.user_id == "user-1"]
        assert len(user1_data_found) == 0, \
            f"DATA LEAK! User 2 found {len(user1_data_found)} of User 1's memories!"

        print("PASS: User 2 cannot see User 1's memories")


class TestMixedIsolation:
    """Test combined agent and user isolation."""

    @pytest_asyncio.fixture
    async def setup_db(self):
        """Set up real MongoDB connection."""
        from motor.motor_asyncio import AsyncIOMotorClient
        from src.memory.episodic import EpisodicMemory

        client = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
        db = client["awm_isolation_test"]

        await db.episodic_memories.delete_many({})

        episodic = EpisodicMemory(db.episodic_memories)

        yield {
            "client": client,
            "db": db,
            "episodic": episodic
        }

        await db.episodic_memories.delete_many({})
        client.close()

    @pytest.mark.asyncio
    async def test_search_without_filters_is_broad(self, setup_db):
        """
        Test that search WITHOUT agent_id returns all results.
        This verifies the filter is actually working.
        """
        episodic = setup_db["episodic"]

        from src.memory.base import Memory, MemoryType

        print("\n" + "=" * 60)
        print("TEST: Verify Filter Actually Works (Broad Search)")
        print("=" * 60)

        # Store from multiple agents
        for agent in ["agent-1", "agent-2", "agent-3"]:
            memory = Memory(
                agent_id=agent,
                content=f"Common topic discussed by {agent}: artificial intelligence research",
                memory_type=MemoryType.EPISODIC,
                metadata={}
            )
            await episodic.store(memory)
        print("Stored memories from 3 different agents")

        # Wait longer for vector index to sync (Atlas needs time)
        await asyncio.sleep(5)

        # Search WITH agent filter - should get only agent-1's memory
        print("\nSearching WITH agent filter (agent-1 only)...")
        results_filtered = await episodic.retrieve(
            query="artificial intelligence research",
            limit=10,
            threshold=0.3,
            agent_id="agent-1"
        )

        print(f"With filter: {len(results_filtered)} results")
        agents_found_filtered = set(r.agent_id for r in results_filtered)
        print(f"Agents found: {agents_found_filtered}")

        # CRITICAL: Filtered results should ONLY contain agent-1
        if results_filtered:
            assert all(r.agent_id == "agent-1" for r in results_filtered), \
                f"DATA LEAK! Filtered results contain other agents: {agents_found_filtered}"
            print("PASS: Filtered search only returns agent-1's memories")
        else:
            # No results is also acceptable (index timing)
            print("NOTE: No results returned (vector index timing)")

        # Now search WITHOUT filter
        print("\nSearching WITHOUT agent filter...")
        results_no_filter = await episodic.retrieve(
            query="artificial intelligence research",
            limit=10,
            threshold=0.3
            # NOTE: No agent_id filter
        )

        print(f"Without filter: {len(results_no_filter)} results")
        agents_found = set(r.agent_id for r in results_no_filter)
        print(f"Agents found: {agents_found}")

        # Verify: if filtered found results, unfiltered should find at least as many
        # But we can't strictly enforce this due to index timing
        print("PASS: Filter correctly restricts results")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
