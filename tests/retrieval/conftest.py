"""
Shared fixtures for retrieval integration tests.
Uses REAL MongoDB connections - no mocks!
"""

import os
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from motor.motor_asyncio import AsyncIOMotorClient

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# MongoDB connection string - MUST be set via environment variable
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    pytest.skip(
        "MONGODB_URI environment variable required for integration tests", allow_module_level=True
    )

# Test database name
TEST_DB_NAME = "awm_retrieval_tests"
TEST_COLLECTION_NAME = "test_vectors"


@pytest_asyncio.fixture(scope="function")
async def mongodb_client():
    """Create MongoDB client for each test."""
    client = AsyncIOMotorClient(MONGODB_URI)
    # Verify connection
    await client.admin.command("ping")
    yield client
    client.close()


@pytest_asyncio.fixture(scope="function")
async def test_db(mongodb_client):
    """Get or create test database."""
    db = mongodb_client[TEST_DB_NAME]
    yield db


@pytest_asyncio.fixture(scope="function")
async def test_collection(test_db):
    """Create fresh test collection for each test."""
    collection = test_db[TEST_COLLECTION_NAME]
    # Clean before test
    await collection.delete_many({})
    yield collection
    # Clean after test
    await collection.delete_many({})


@pytest_asyncio.fixture(scope="function")
async def seeded_collection(test_collection):
    """Collection with pre-seeded test documents with embeddings."""
    # Create test documents with mock embeddings (1024 dimensions for Voyage AI)
    base_embedding = [0.1] * 1024

    test_docs = [
        {
            "content": "Python is a great programming language for AI development",
            "embedding": [x + 0.01 for x in base_embedding],  # Slightly different
            "metadata": {"tags": ["python", "ai"], "category": "programming"},
            "agent_id": "test-agent",
            "memory_type": "semantic",
            "timestamp": datetime.now(UTC),
            "importance": 0.8,
        },
        {
            "content": "MongoDB provides excellent vector search capabilities",
            "embedding": [x + 0.02 for x in base_embedding],
            "metadata": {"tags": ["mongodb", "database"], "category": "database"},
            "agent_id": "test-agent",
            "memory_type": "semantic",
            "timestamp": datetime.now(UTC),
            "importance": 0.7,
        },
        {
            "content": "Machine learning models require large datasets for training",
            "embedding": [x + 0.03 for x in base_embedding],
            "metadata": {"tags": ["ml", "data"], "category": "ai"},
            "agent_id": "test-agent",
            "memory_type": "episodic",
            "timestamp": datetime.now(UTC),
            "importance": 0.9,
        },
        {
            "content": "Neural networks are the foundation of deep learning",
            "embedding": [x + 0.04 for x in base_embedding],
            "metadata": {"tags": ["neural", "deep-learning"], "category": "ai"},
            "agent_id": "other-agent",  # Different agent
            "memory_type": "semantic",
            "timestamp": datetime.now(UTC),
            "importance": 0.6,
        },
        {
            "content": "Vector embeddings capture semantic meaning of text",
            "embedding": [x + 0.05 for x in base_embedding],
            "metadata": {"tags": ["vectors", "embeddings"], "category": "nlp"},
            "agent_id": "test-agent",
            "memory_type": "procedural",
            "timestamp": datetime.now(UTC),
            "importance": 0.85,
        },
    ]

    await test_collection.insert_many(test_docs)
    print(f"✓ Seeded {len(test_docs)} test documents")
    yield test_collection


@pytest.fixture
def query_embedding():
    """Query embedding for testing (similar to base)."""
    return [0.1] * 1024


@pytest.fixture
def sample_search_results():
    """Sample SearchResult objects for RRF testing."""
    from src.retrieval.vector_search import SearchResult

    return {
        "vector": [
            SearchResult(id="doc1", content="Python programming", metadata={}, score=0.95),
            SearchResult(id="doc2", content="MongoDB database", metadata={}, score=0.85),
            SearchResult(id="doc3", content="Machine learning", metadata={}, score=0.75),
        ],
        "text": [
            SearchResult(id="doc2", content="MongoDB database", metadata={}, score=0.90),
            SearchResult(id="doc4", content="Deep learning", metadata={}, score=0.80),
            SearchResult(id="doc1", content="Python programming", metadata={}, score=0.70),
        ],
    }
