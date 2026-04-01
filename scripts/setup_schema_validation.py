#!/usr/bin/env python3
"""
MongoDB Schema Validation Setup Script
Applies $jsonSchema validation to all 7 memory collections.

mongodb-schema-design: "$jsonSchema validation prevents malformed documents"

Uses validationAction: "warn" initially (not "error") to avoid breaking existing inserts.
Switch to "error" after data migration is verified.

Usage:
    python scripts/setup_schema_validation.py
"""

import os
import sys

from dotenv import load_dotenv
from pymongo import MongoClient

# Base schema shared across all memory collections
BASE_SCHEMA = {
    "bsonType": "object",
    "required": ["content", "memory_type", "agent_id", "created_at"],
    "properties": {
        "content": {
            "bsonType": "string",
            "description": "Memory content text",
        },
        "memory_type": {
            "bsonType": "string",
            "description": "Type of memory (episodic, semantic, etc.)",
        },
        "agent_id": {
            "bsonType": "string",
            "description": "Agent that owns this memory",
        },
        "created_at": {
            "bsonType": "date",
            "description": "Creation timestamp",
        },
        "updated_at": {
            "bsonType": "date",
            "description": "Last update timestamp",
        },
        "embedding": {
            "bsonType": "array",
            "description": "Vector embedding (1024 dimensions)",
            "items": {"bsonType": "double"},
        },
        "importance": {
            "bsonType": "double",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Importance score",
        },
    },
}


# Collection-specific schema extensions
COLLECTION_SCHEMAS = {
    "episodic_memories": {
        # Episodic memories may include thread_id in metadata
    },
    "semantic_memories": {
        # Semantic memories may include domain in metadata
    },
    "procedural_memories": {
        # Procedural memories may include skill_level in metadata
    },
    "working_memories": {
        "properties": {
            "ttl": {
                "bsonType": "date",
                "description": "Time-to-live for automatic expiration",
            }
        }
    },
    "cache_memories": {
        "properties": {
            "ttl": {
                "bsonType": "date",
                "description": "Time-to-live for automatic expiration",
            }
        }
    },
    "entity_memories": {
        # Entity memories have entity_name and entity_type in metadata
    },
    "summary_memories": {
        # Summary memories have summary_id in metadata
    },
}


def apply_schema_validation(db, collection_name, extra_schema=None):
    """Apply $jsonSchema validation to a collection.

    Args:
        db: MongoDB database
        collection_name: Name of the collection
        extra_schema: Additional schema properties to merge
    """
    print(f"  Applying schema validation to {collection_name}...")

    # Merge base schema with collection-specific properties
    schema = dict(BASE_SCHEMA)
    if extra_schema and "properties" in extra_schema:
        merged_props = dict(schema.get("properties", {}))
        merged_props.update(extra_schema["properties"])
        schema["properties"] = merged_props

    try:
        # Use collMod to apply validation rules
        db.command(
            {
                "collMod": collection_name,
                "validator": {"$jsonSchema": schema},
                "validationLevel": "moderate",
                "validationAction": "warn",  # Start with warn, switch to error after migration
            }
        )
        print("    Schema validation applied (validationAction: warn)")
        return True

    except Exception as e:
        error_msg = str(e)
        if "ns not found" in error_msg.lower() or "doesn't exist" in error_msg.lower():
            # Collection does not exist yet — create it with validation
            try:
                db.create_collection(
                    collection_name,
                    validator={"$jsonSchema": schema},
                    validationLevel="moderate",
                    validationAction="warn",
                )
                print("    Created collection with schema validation (warn)")
                return True
            except Exception as create_err:
                print(f"    Error creating collection: {create_err}")
                return False
        else:
            print(f"    Error applying validation: {e}")
            return False


def setup_schema_validation():
    """Apply schema validation to all 7 memory collections."""
    print()
    print("=" * 60)
    print("  AWM 2.0 - MongoDB Schema Validation Setup")
    print("=" * 60)
    print()

    load_dotenv()

    mongodb_uri = os.getenv("MONGODB_URI")
    database_name = os.getenv("MONGODB_DB_NAME", "awm_demo")

    if not mongodb_uri:
        print("MONGODB_URI environment variable not set")
        return False

    try:
        print("Connecting to MongoDB...")
        client = MongoClient(mongodb_uri)
        db = client[database_name]

        client.admin.command("ping")
        print(f"Connected to database: {database_name}")
        print()

        success_count = 0
        for collection_name, extra_schema in COLLECTION_SCHEMAS.items():
            if apply_schema_validation(db, collection_name, extra_schema):
                success_count += 1

        print()
        print("=" * 60)
        print(f"  Schema validation applied: {success_count}/{len(COLLECTION_SCHEMAS)}")
        print()
        print("  NOTE: validationAction is 'warn' — documents that fail")
        print("  validation will be logged but NOT rejected.")
        print("  Switch to 'error' after verifying existing data complies.")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        if "client" in locals():
            client.close()


def main():
    success = setup_schema_validation()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
