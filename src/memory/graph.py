"""
GraphRAG module for knowledge graph retrieval.
Extends entity memory with relationships and $graphLookup traversal.

Pattern from HybridRAG: Entity extraction + relationship linking + graph traversal.
Pattern from Mem0 reference: Dual storage (vector + graph) per memory.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from ..embeddings.voyage_client import get_embedding_service
from ..retrieval.vector_search import SearchResult, VectorSearchEngine
from .entity import EntityMemory

logger = logging.getLogger(__name__)


@dataclass
class Relationship:
    """A relationship between two entities."""

    source_entity: str  # Entity name
    target_entity: str  # Entity name
    relationship_type: str  # e.g., "WORKS_AT", "LOCATED_IN", "RELATED_TO"
    agent_id: str  # Multi-tenant isolation
    metadata: dict[str, Any] | None = field(default=None)


class GraphMemory:
    """
    Knowledge graph layer on top of entity memory.

    Schema for entity_memories collection (extended):
    {
        "_id": ObjectId,
        "content": "Entity description",
        "metadata": {
            "entity_name": "John",
            "entity_type": "PERSON",
            "relationships": [
                {"target": "Acme Corp", "type": "WORKS_AT"},
                {"target": "New York", "type": "LOCATED_IN"}
            ]
        },
        "agent_id": "agent_123",
        "embedding": [...]
    }

    Indexes needed:
    - B-tree on metadata.entity_name + agent_id (for $graphLookup connectToField)
    - B-tree on metadata.relationships.target + agent_id (for connectFromField)
    """

    RELATIONSHIP_EXTRACTION_PROMPT = (
        "Extract relationships between entities from the following text.\n"
        'Return JSON array: [{{"source": "Entity1", "target": "Entity2", '
        '"type": "RELATIONSHIP_TYPE"}}]\n'
        "Valid relationship types: WORKS_AT, LOCATED_IN, PART_OF, RELATED_TO, "
        "CREATED_BY, MANAGES, USES\n"
        "If no relationships found, return: []\n\n"
        'Text: "{text}"'
    )

    def __init__(self, db: AsyncIOMotorDatabase, entity_memory: EntityMemory):
        """Initialize GraphMemory.

        Args:
            db: Motor database instance
            entity_memory: Existing EntityMemory store
        """
        self.db = db
        self.entity_memory = entity_memory
        self.collection = db["entity_memories"]
        self.search_engine = VectorSearchEngine(self.collection)

    async def add_relationship(self, relationship: Relationship) -> bool:
        """Add a relationship between two entities.

        Both entities must exist and belong to the same agent.

        Args:
            relationship: The relationship to add

        Returns:
            True if relationship was added
        """
        # Find source entity (scoped to agent)
        source = await self.collection.find_one(
            {
                "metadata.entity_name": relationship.source_entity,
                "agent_id": relationship.agent_id,
            }
        )
        if not source:
            logger.warning(f"Source entity not found: {relationship.source_entity}")
            return False

        # Add relationship to source entity's metadata using $addToSet
        result = await self.collection.update_one(
            {"_id": source["_id"]},
            {
                "$addToSet": {
                    "metadata.relationships": {
                        "target": relationship.target_entity,
                        "type": relationship.relationship_type,
                    }
                }
            },
        )
        return result.modified_count > 0

    async def extract_and_store_relationships(
        self, text: str, agent_id: str, llm: Any
    ) -> list[dict[str, Any]]:
        """Extract relationships from text using LLM and store them.

        Args:
            text: Text to extract relationships from
            agent_id: Agent ID for isolation
            llm: LLM for extraction

        Returns:
            List of extracted relationships
        """
        if not text or len(text.strip()) < 10:
            return []

        prompt = self.RELATIONSHIP_EXTRACTION_PROMPT.format(text=text[:1000])
        response = await llm.ainvoke(prompt)

        try:
            result = response.content.strip()
            start, end = result.find("["), result.rfind("]")
            if start == -1 or end == -1:
                return []

            relationships = json.loads(result[start : end + 1])
            stored = []

            for rel in relationships:
                if not isinstance(rel, dict):
                    continue

                relationship = Relationship(
                    source_entity=rel.get("source", ""),
                    target_entity=rel.get("target", ""),
                    relationship_type=rel.get("type", "RELATED_TO"),
                    agent_id=agent_id,
                )

                if relationship.source_entity and relationship.target_entity:
                    success = await self.add_relationship(relationship)
                    if success:
                        stored.append(rel)

            return stored

        except Exception as e:
            logger.error(f"Failed to extract relationships: {e}")
            return []

    async def graph_lookup(
        self,
        start_entity: str,
        agent_id: str,
        max_depth: int = 2,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Traverse entity relationships using MongoDB $graphLookup.

        Args:
            start_entity: Entity name to start from
            agent_id: Agent ID for isolation
            max_depth: Maximum traversal depth
            limit: Maximum results

        Returns:
            List of connected entities with relationship paths
        """
        pipeline = [
            # Start from the seed entity
            {
                "$match": {
                    "metadata.entity_name": start_entity,
                    "agent_id": agent_id,
                }
            },
            # Traverse relationships via $graphLookup
            {
                "$graphLookup": {
                    "from": "entity_memories",
                    "startWith": "$metadata.relationships.target",
                    "connectFromField": "metadata.relationships.target",
                    "connectToField": "metadata.entity_name",
                    "as": "connected_entities",
                    "maxDepth": max_depth,
                    "depthField": "depth",
                    "restrictSearchWithMatch": {
                        "agent_id": agent_id,
                    },
                }
            },
            # Limit results
            {"$limit": limit},
            # Project relevant fields
            {
                "$project": {
                    "entity_name": "$metadata.entity_name",
                    "entity_type": "$metadata.entity_type",
                    "relationships": "$metadata.relationships",
                    "connected_entities": {
                        "$map": {
                            "input": "$connected_entities",
                            "as": "e",
                            "in": {
                                "name": "$$e.metadata.entity_name",
                                "type": "$$e.metadata.entity_type",
                                "depth": "$$e.depth",
                            },
                        }
                    },
                }
            },
        ]

        results = []
        async for doc in self.collection.aggregate(pipeline):
            results.append(doc)

        return results

    async def entity_boosted_search(
        self,
        query: str,
        agent_id: str,
        limit: int = 10,
        entity_boost: float = 0.3,
    ) -> list[SearchResult]:
        """Hybrid search with entity boost.

        Combines vector similarity with entity relationship signal.

        Pattern from HybridRAG: cross-encoder + entity signal boost.

        Args:
            query: Search query
            agent_id: Agent ID
            limit: Max results
            entity_boost: Weight for entity signal (0-1)

        Returns:
            Reranked search results
        """
        embedding_service = get_embedding_service()
        embedding_result = await embedding_service.generate_embedding(
            query, input_type="query"
        )

        vector_results = await self.search_engine.hybrid_search(
            query_text=query,
            query_embedding=embedding_result.embedding,
            limit=limit * 2,  # Over-fetch for reranking
            filter_query={"agent_id": agent_id},
        )

        # Extract entities mentioned in the query
        query_entities: set[str] = set()
        async for doc in self.collection.find(
            {"agent_id": agent_id}, {"metadata.entity_name": 1}
        ):
            name = doc.get("metadata", {}).get("entity_name", "")
            if name and name.lower() in query.lower():
                query_entities.add(name)

        # Boost results that mention matched entities
        for result in vector_results:
            entity_score = 0.0
            content_lower = result.content.lower()
            for entity in query_entities:
                if entity.lower() in content_lower:
                    entity_score += entity_boost

            result.score = result.score * (1 - entity_boost) + entity_score

        # Re-sort by boosted score
        vector_results.sort(key=lambda r: r.score, reverse=True)
        return vector_results[:limit]
