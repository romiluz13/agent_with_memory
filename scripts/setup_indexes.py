#!/usr/bin/env python3
"""
MongoDB Atlas Search Index Setup Script
Creates vector search and text search indexes for AWM 2.0 hybrid search.

Based on MongoDB's official GenAI-Showcase pattern.

Usage:
    python scripts/setup_indexes.py
"""

import os
import sys
import time

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

# Memory collections that need indexes
MEMORY_COLLECTIONS = [
    "episodic_memories",
    "semantic_memories",
    "procedural_memories",
    "working_memories",
    "cache_memories",
    "entity_memories",
    "summary_memories",
]


def create_ttl_index(collection, field="ttl"):
    """Create TTL index for automatic document expiration.

    mongodb-query-optimizer: TTL indexes automatically remove expired documents.
    expireAfterSeconds=0 means expire at the datetime value in the ttl field.

    Args:
        collection: MongoDB collection
        field: Field name containing the expiration datetime
    """
    print(f"  Creating TTL index on '{field}' for {collection.name}...")
    try:
        result = collection.create_index(field, expireAfterSeconds=0)
        print(f"    TTL index created: {result}")
        return True
    except Exception as e:
        error_msg = str(e)
        if "already exists" in error_msg or "IndexOptionsConflict" in error_msg:
            print(f"    TTL index already exists on '{field}'")
            return True
        print(f"    Error creating TTL index: {e}")
        return False


def create_vector_search_index(collection, index_name="vector_index", dimensions=1024):
    """Create vector search index if it doesn't exist."""
    print(f"  Creating vector search index '{index_name}' on {collection.name}...")

    try:
        # Check if index already exists
        existing_indexes = list(collection.list_search_indexes())
        for index in existing_indexes:
            if index["name"] == index_name:
                print(f"    ✅ Vector index '{index_name}' already exists")
                return True
    except Exception as e:
        print(f"    ⚠️ Could not list indexes: {e}")
        return False

    # Define vector search index with filter fields for multi-tenant isolation
    index_definition = {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": dimensions,
                "similarity": "cosine",
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
    }

    search_index_model = SearchIndexModel(
        definition=index_definition,
        name=index_name,
        type="vectorSearch",
    )

    try:
        result = collection.create_search_index(model=search_index_model)
        print(f"    ✅ Vector index '{result}' created and building...")
        return True
    except Exception as e:
        error_msg = str(e)
        if "already exists" in error_msg:
            print(f"    ✅ Vector index '{index_name}' already exists")
            return True
        else:
            print(f"    ❌ Error creating vector index: {e}")
            return False


def create_text_search_index(collection, index_name="text_search_index"):
    """Create text search index for hybrid search."""
    print(f"  Creating text search index '{index_name}' on {collection.name}...")

    try:
        # Check if index already exists
        existing_indexes = list(collection.list_search_indexes())
        for index in existing_indexes:
            if index["name"] == index_name:
                print(f"    ✅ Text index '{index_name}' already exists")
                return True
    except Exception as e:
        print(f"    ⚠️ Could not list indexes: {e}")
        return False

    # Define text search index (following GenAI-Showcase pattern)
    index_definition = {
        "mappings": {
            "dynamic": False,
            "fields": {
                "content": {
                    "type": "string",
                    "analyzer": "lucene.standard",
                },
                "metadata.tags": {"type": "string"},
                "agent_id": {"type": "string"},
                "user_id": {"type": "string"},
            },
        }
    }

    search_index_model = SearchIndexModel(
        definition=index_definition,
        name=index_name,
        type="search",
    )

    try:
        result = collection.create_search_index(model=search_index_model)
        print(f"    ✅ Text index '{result}' created and building...")
        return True
    except Exception as e:
        error_msg = str(e)
        if "already exists" in error_msg:
            print(f"    ✅ Text index '{index_name}' already exists")
            return True
        else:
            print(f"    ❌ Error creating text index: {e}")
            return False


def wait_for_index_ready(collection, index_name, timeout=120):
    """Wait for an index to become queryable."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            indices = list(collection.list_search_indexes(index_name))
            if indices and indices[0].get("queryable") is True:
                return True
            time.sleep(5)
        except Exception:
            time.sleep(5)

    return False


def setup_indexes():
    """Setup all MongoDB Atlas search indexes for AWM 2.0."""
    print()
    print("=" * 60)
    print("  AWM 2.0 - MongoDB Atlas Search Index Setup")
    print("=" * 60)
    print()

    # Load environment variables
    load_dotenv()

    # Get MongoDB connection details
    mongodb_uri = os.getenv("MONGODB_URI")
    database_name = os.getenv("MONGODB_DB_NAME", "awm_demo")
    embedding_dim = int(os.getenv("VOYAGE_EMBEDDING_DIMENSION", "1024"))

    if not mongodb_uri:
        print("❌ MONGODB_URI environment variable not set")
        return False

    try:
        # Connect to MongoDB
        print("🔌 Connecting to MongoDB...")
        client = MongoClient(mongodb_uri)
        db = client[database_name]

        # Test connection
        client.admin.command("ping")
        print(f"✅ Connected to database: {database_name}")
        print()

        # Create indexes for each memory collection
        print(f"📚 Setting up indexes for {len(MEMORY_COLLECTIONS)} collections...")
        print()

        vector_success = 0
        text_success = 0

        for collection_name in MEMORY_COLLECTIONS:
            print(f"📦 {collection_name}:")
            collection = db[collection_name]

            # Ensure collection exists
            if collection_name not in db.list_collection_names():
                collection.insert_one({"_temp": "init"})
                collection.delete_one({"_temp": "init"})
                print(f"    Created collection '{collection_name}'")

            # Create vector search index
            if create_vector_search_index(collection, dimensions=embedding_dim):
                vector_success += 1

            # Create text search index
            if create_text_search_index(collection):
                text_success += 1

            # Create TTL index for collections that use TTL
            # mongodb-query-optimizer: TTL indexes auto-remove expired documents
            if collection_name in ("working_memories", "cache_memories"):
                create_ttl_index(collection, field="ttl")

            print()

        # Summary
        print("=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        print(f"  Vector indexes: {vector_success}/{len(MEMORY_COLLECTIONS)}")
        print(f"  Text indexes:   {text_success}/{len(MEMORY_COLLECTIONS)}")
        print()
        print("⏳ Indexes are building in the background...")
        print("   This may take 1-5 minutes on MongoDB Atlas.")
        print()
        print("🎉 Once ready, AWM 2.0 will use HYBRID SEARCH by default!")
        print("   - Vector search for semantic similarity")
        print("   - Text search for keyword matching")
        print("   - $rankFusion combines both for best results")
        print()

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    finally:
        if "client" in locals():
            client.close()


def main():
    success = setup_indexes()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
