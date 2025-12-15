"""
Shared fixtures for integration tests.
Requires: MONGODB_URI, VOYAGE_API_KEY, GOOGLE_API_KEY environment variables.
"""

import os
import pytest
import pytest_asyncio
from motor.motor_asyncio import AsyncIOMotorClient

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed

# Skip all integration tests if environment variables not set
SKIP_INTEGRATION = not all([
    os.getenv("MONGODB_URI"),
    os.getenv("VOYAGE_API_KEY")
])

pytestmark = pytest.mark.skipif(
    SKIP_INTEGRATION,
    reason="Integration tests require MONGODB_URI and VOYAGE_API_KEY environment variables"
)


@pytest_asyncio.fixture(scope="function")
async def mongodb_client():
    """Create MongoDB client for integration tests."""
    uri = os.getenv("MONGODB_URI")
    if not uri:
        pytest.skip("MONGODB_URI not set")

    client = AsyncIOMotorClient(uri)
    yield client
    client.close()


@pytest_asyncio.fixture(scope="function")
async def test_db(mongodb_client):
    """Create test database."""
    db = mongodb_client["awm_integration_tests"]
    yield db


@pytest_asyncio.fixture(scope="function")
async def clean_collections(test_db):
    """Clean collections before each test."""
    collections = [
        "episodic_memories",
        "semantic_memories",
        "procedural_memories",
        "working_memories",
        "cache_memories",
        "entity_memories",
        "summary_memories"
    ]
    for coll in collections:
        await test_db[coll].delete_many({})
    yield
    # Cleanup after test
    for coll in collections:
        await test_db[coll].delete_many({})


@pytest.fixture
def test_agent_id():
    """Unique agent ID for tests."""
    return "test-agent-integration"


@pytest.fixture
def test_user_id():
    """Unique user ID for tests."""
    return "test-user-integration"


@pytest.fixture
def test_thread_id():
    """Unique thread ID for tests."""
    return "test-thread-integration"
