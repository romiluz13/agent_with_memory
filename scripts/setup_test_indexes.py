#!/usr/bin/env python3
"""
Setup vector search indexes on Atlas for testing.
Run with: python scripts/setup_test_indexes.py
"""

import asyncio

from motor.motor_asyncio import AsyncIOMotorClient

# Direct credentials for testing
MONGODB_URI = "mongodb+srv://rom:05101994@cluster0.466n8j.mongodb.net/?appName=Cluster0"

MEMORY_COLLECTIONS = [
    "episodic_memories",
    "semantic_memories",
    "procedural_memories",
    "working_memories",
    "cache_memories",
    "entity_memories",
    "summary_memories",
]

DATABASES = ["awm_production_test", "awm_debug_test", "awm_test"]


async def create_vector_index(collection, index_name="vector_index"):
    """Create vector search index on a collection."""
    index_definition = {
        "name": index_name,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "similarity": "cosine",
                    "numDimensions": 1024,
                },
                {"type": "filter", "path": "agent_id"},
                {"type": "filter", "path": "user_id"},
                {"type": "filter", "path": "memory_type"},
                {"type": "filter", "path": "thread_id"},
                {"type": "filter", "path": "timestamp"},
                {"type": "filter", "path": "importance"},
                {"type": "filter", "path": "metadata.tags"},
                {"type": "filter", "path": "metadata.entity_type"},
            ]
        },
    }

    try:
        # Check if exists
        existing = await collection.list_search_indexes().to_list(length=None)
        if any(idx.get("name") == index_name for idx in existing):
            print(f"  ✓ {index_name} already exists")
            return "exists"

        # Create
        result = await collection.create_search_index(index_definition)
        print(f"  ✓ Created {index_name}: {result}")
        return "created"
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"  ✓ {index_name} already exists")
            return "exists"
        print(f"  ✗ Error creating {index_name}: {e}")
        return "error"


async def create_text_index(collection, index_name="text_search_index"):
    """Create text search index on a collection."""
    index_definition = {
        "name": index_name,
        "type": "search",
        "definition": {
            "mappings": {
                "dynamic": False,
                "fields": {
                    "content": {"type": "string", "analyzer": "lucene.standard"},
                    "agent_id": {"type": "string"},
                    "user_id": {"type": "string"},
                    "metadata.tags": {"type": "string", "analyzer": "lucene.standard"},
                    "metadata.entity_name": {"type": "string", "analyzer": "lucene.keyword"},
                    "metadata.entity_type": {"type": "string", "analyzer": "lucene.keyword"},
                    "timestamp": {"type": "date"},
                    "created_at": {"type": "date"},
                },
            }
        },
    }

    try:
        existing = await collection.list_search_indexes().to_list(length=None)
        if any(idx.get("name") == index_name for idx in existing):
            print(f"  ✓ {index_name} already exists")
            return "exists"

        result = await collection.create_search_index(index_definition)
        print(f"  ✓ Created {index_name}: {result}")
        return "created"
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"  ✓ {index_name} already exists")
            return "exists"
        print(f"  ✗ Error creating {index_name}: {e}")
        return "error"


async def main():
    print("=" * 60)
    print("Setting up Vector Search Indexes on MongoDB Atlas")
    print("=" * 60)

    client = AsyncIOMotorClient(MONGODB_URI)

    # Verify connection
    try:
        server_info = await client.admin.command("serverStatus")
        print("\n✓ Connected to MongoDB Atlas")
        print(f"  Version: {server_info.get('version', 'unknown')}")
    except Exception as e:
        print(f"\n✗ Connection failed: {e}")
        return

    # Create indexes for each database
    for db_name in DATABASES:
        print(f"\n{'='*40}")
        print(f"Database: {db_name}")
        print(f"{'='*40}")

        db = client[db_name]

        for coll_name in MEMORY_COLLECTIONS:
            print(f"\n  Collection: {coll_name}")
            collection = db[coll_name]

            # Insert a dummy doc to ensure collection exists
            await collection.update_one(
                {"_id": "__index_setup_marker__"},
                {"$set": {"_id": "__index_setup_marker__", "setup": True}},
                upsert=True,
            )

            await create_vector_index(collection)
            await create_text_index(collection)

    # Check final status
    print("\n" + "=" * 60)
    print("Index Status Check")
    print("=" * 60)

    for db_name in DATABASES:
        db = client[db_name]
        print(f"\n{db_name}:")
        for coll_name in MEMORY_COLLECTIONS:
            collection = db[coll_name]
            try:
                indexes = await collection.list_search_indexes().to_list(length=None)
                statuses = [(idx.get("name"), idx.get("status", "UNKNOWN")) for idx in indexes]
                if statuses:
                    status_str = ", ".join([f"{n}:{s}" for n, s in statuses])
                    print(f"  {coll_name}: {status_str}")
                else:
                    print(f"  {coll_name}: No indexes")
            except Exception as e:
                print(f"  {coll_name}: Error checking - {e}")

    print("\n" + "=" * 60)
    print("NOTE: Atlas indexes take 1-5 minutes to build.")
    print("Wait for status to change from PENDING to READY before testing.")
    print("=" * 60)

    client.close()


if __name__ == "__main__":
    asyncio.run(main())
