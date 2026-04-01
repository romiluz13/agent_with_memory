"""
Setup MongoDB indexes for GraphRAG $graphLookup traversal.

Creates B-tree indexes needed for efficient $graphLookup on entity_memories:
- metadata.entity_name + agent_id (connectToField)
- metadata.relationships.target + agent_id (connectFromField)

Usage:
    python scripts/setup_graph_indexes.py
"""

import asyncio
import logging
import os

from motor.motor_asyncio import AsyncIOMotorClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_graph_indexes(uri: str | None = None, db_name: str = "ai_agent_boilerplate"):
    """Create B-tree indexes for $graphLookup traversal.

    Args:
        uri: MongoDB connection URI. Defaults to MONGODB_URI env var.
        db_name: Database name.
    """
    connection_uri = uri or os.getenv("MONGODB_URI")
    if not connection_uri:
        logger.error("MONGODB_URI not set. Cannot create indexes.")
        return

    client = AsyncIOMotorClient(connection_uri)
    db = client[db_name]
    collection = db["entity_memories"]

    # Index for connectToField: metadata.entity_name
    # Used when $graphLookup needs to match target names to entities
    await collection.create_index(
        [("metadata.entity_name", 1), ("agent_id", 1)],
        name="graph_entity_name_agent",
    )
    logger.info("Created index: graph_entity_name_agent")

    # Index for connectFromField: metadata.relationships.target
    # Used when $graphLookup follows relationship targets
    await collection.create_index(
        [("metadata.relationships.target", 1), ("agent_id", 1)],
        name="graph_relationships_target_agent",
    )
    logger.info("Created index: graph_relationships_target_agent")

    client.close()
    logger.info("Graph indexes created successfully.")


if __name__ == "__main__":
    asyncio.run(setup_graph_indexes())
