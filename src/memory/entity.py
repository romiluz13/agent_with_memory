"""
Entity Memory Store
LLM-powered extraction of entities (PERSON, ORGANIZATION, LOCATION, SYSTEM, CONCEPT)
Based on Oracle Memory Engineering pattern.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection

from .base import Memory, MemoryStore, MemoryType
from ..retrieval.vector_search import VectorSearchEngine
from ..embeddings.voyage_client import get_embedding_service

logger = logging.getLogger(__name__)


# Entity types supported
ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION", "SYSTEM", "CONCEPT"]


@dataclass
class ExtractionConfig:
    """Configuration for entity extraction."""
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_tokens: int = 500
    provider: str = "google"


class EntityMemory(MemoryStore):
    """
    Entity memory for storing extracted entities.
    Uses LLM to extract PERSON, ORGANIZATION, LOCATION, SYSTEM, CONCEPT from text.
    """

    EXTRACTION_PROMPT = '''Extract entities from the following text.
Return JSON array: [{{"name": "X", "type": "PERSON|ORGANIZATION|LOCATION|SYSTEM|CONCEPT", "description": "brief description"}}]
If no entities found, return: []

Text: "{text}"'''

    def __init__(
        self,
        collection: AsyncIOMotorCollection,
        config: Optional[ExtractionConfig] = None
    ):
        """Initialize with MongoDB collection."""
        self.collection = collection
        self.search_engine = VectorSearchEngine(collection)
        self.embedding_service = get_embedding_service()
        self.config = config or ExtractionConfig()
        self._llm = None  # Set by agent at runtime

    def set_llm(self, llm):
        """Set the LLM for entity extraction."""
        self._llm = llm

    async def store(self, memory: Memory) -> str:
        """Store an entity memory."""
        try:
            memory.memory_type = MemoryType.ENTITY

            # Add entity-specific metadata
            memory.metadata["entity_type"] = memory.metadata.get("entity_type", "UNKNOWN")
            memory.metadata["entity_name"] = memory.metadata.get("entity_name", "")
            memory.metadata["mentions"] = memory.metadata.get("mentions", 1)

            # Generate embedding if not present
            if not memory.embedding:
                embedding_result = await self.embedding_service.generate_embedding(
                    memory.content, input_type="document"
                )
                memory.embedding = embedding_result.embedding

            # Convert to dict for MongoDB
            memory_dict = memory.model_dump(exclude={"id"})
            memory_dict["_id"] = ObjectId()

            # Check for existing similar entity
            existing = await self._find_existing_entity(
                memory.metadata.get("entity_name", ""),
                memory.metadata.get("entity_type", "")
            )
            if existing:
                return await self._merge_entities(existing, memory)

            result = await self.collection.insert_one(memory_dict)
            logger.debug(f"Stored entity memory: {result.inserted_id}")
            return str(result.inserted_id)

        except Exception as e:
            logger.error(f"Failed to store entity memory: {e}")
            raise

    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.7,
        search_mode: str = "hybrid"
    ) -> List[Memory]:
        """
        Retrieve entity memories by similarity.

        Uses hybrid search (vector + full-text) by default for best results.
        Based on MongoDB's official GenAI-Showcase pattern with $rankFusion.

        Args:
            query: Search query text
            limit: Maximum memories to return
            threshold: Minimum similarity threshold
            search_mode: Search strategy - "hybrid" (default), "semantic", or "text"

        Returns:
            List of relevant memories
        """
        try:
            embedding_result = await self.embedding_service.generate_embedding(
                query, input_type="query"
            )

            # Execute search based on mode
            filter_query = {"memory_type": "entity"}

            if search_mode == "hybrid":
                # Hybrid search: vector + full-text with $rankFusion (DEFAULT)
                results = await self.search_engine.hybrid_search(
                    query_text=query,
                    query_embedding=embedding_result.embedding,
                    limit=limit,
                    filter_query=filter_query
                )
            else:
                # Vector-only search (semantic mode)
                results = await self.search_engine.search(
                    query_embedding=embedding_result.embedding,
                    limit=limit,
                    filter_query=filter_query
                )

            memories = []
            for result in results:
                if result.score >= threshold:
                    doc = await self.collection.find_one({"_id": ObjectId(result.id)})
                    if doc:
                        doc["id"] = str(doc.pop("_id"))
                        memory = Memory(**doc)
                        memory.metadata["search_score"] = result.score
                        memories.append(memory)

            # Sort by mentions and search score
            memories.sort(
                key=lambda m: (
                    m.metadata.get("mentions", 1),
                    m.metadata.get("search_score", 0)
                ),
                reverse=True
            )

            return memories

        except Exception as e:
            logger.error(f"Failed to retrieve entity memories: {e}")
            return []

    async def get_by_id(self, memory_id: str) -> Optional[Memory]:
        """Get entity memory by ID."""
        try:
            doc = await self.collection.find_one({"_id": ObjectId(memory_id)})
            if doc:
                doc["id"] = str(doc.pop("_id"))
                return Memory(**doc)
            return None
        except Exception as e:
            logger.error(f"Failed to get entity memory {memory_id}: {e}")
            return None

    async def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update an entity memory."""
        try:
            updates["updated_at"] = datetime.utcnow()
            result = await self.collection.update_one(
                {"_id": ObjectId(memory_id)},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update entity memory {memory_id}: {e}")
            return False

    async def delete(self, memory_id: str) -> bool:
        """Delete an entity memory."""
        try:
            result = await self.collection.delete_one({"_id": ObjectId(memory_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to delete entity memory {memory_id}: {e}")
            return False

    async def list_memories(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Memory]:
        """List entity memories with filters."""
        try:
            query = filters or {}
            query["memory_type"] = "entity"

            cursor = self.collection.find(query).skip(offset).limit(limit)
            cursor = cursor.sort([
                ("metadata.mentions", -1),
                ("importance", -1)
            ])

            memories = []
            async for doc in cursor:
                doc["id"] = str(doc.pop("_id"))
                memories.append(Memory(**doc))

            return memories
        except Exception as e:
            logger.error(f"Failed to list entity memories: {e}")
            return []

    async def clear_all(self, confirm: bool = False) -> int:
        """Clear all entity memories."""
        if not confirm:
            raise ValueError("Must confirm deletion")

        try:
            result = await self.collection.delete_many({"memory_type": "entity"})
            return result.deleted_count
        except Exception as e:
            logger.error(f"Failed to clear entity memories: {e}")
            return 0

    async def extract_and_store(
        self,
        text: str,
        llm,
        agent_id: str,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from text using LLM and store them.

        Args:
            text: Text to extract entities from
            llm: LLM instance to use for extraction
            agent_id: Agent ID for memory ownership
            user_id: Optional user ID

        Returns:
            List of extracted entity dictionaries
        """
        if not text or len(text.strip()) < 5:
            return []

        try:
            # Truncate text if too long
            truncated_text = text[:1000] if len(text) > 1000 else text

            # Create extraction prompt
            prompt = self.EXTRACTION_PROMPT.format(text=truncated_text)

            # Call LLM for extraction
            response = await llm.ainvoke(prompt)
            result = response.content.strip()

            # Parse JSON response
            start, end = result.find("["), result.rfind("]")
            if start == -1 or end == -1:
                return []

            entities = json.loads(result[start:end+1])

            # Store each extracted entity
            stored_entities = []
            for entity in entities:
                if not isinstance(entity, dict) or not entity.get("name"):
                    continue

                entity_name = entity.get("name", "")
                entity_type = entity.get("type", "UNKNOWN")
                description = entity.get("description", "")

                # Validate entity type
                if entity_type not in ENTITY_TYPES:
                    entity_type = "CONCEPT"

                content = f"{entity_name} ({entity_type}): {description}"

                memory = Memory(
                    agent_id=agent_id,
                    user_id=user_id,
                    memory_type=MemoryType.ENTITY,
                    content=content,
                    metadata={
                        "entity_name": entity_name,
                        "entity_type": entity_type,
                        "description": description,
                        "mentions": 1,
                        "extracted_from": truncated_text[:100]
                    },
                    tags=[entity_type.lower()]
                )

                await self.store(memory)
                stored_entities.append(entity)

            logger.info(f"Extracted and stored {len(stored_entities)} entities")
            return stored_entities

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse entity extraction response: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
            return []

    async def _find_existing_entity(
        self,
        name: str,
        entity_type: str
    ) -> Optional[Memory]:
        """Find existing entity by name and type."""
        if not name:
            return None

        try:
            doc = await self.collection.find_one({
                "memory_type": "entity",
                "metadata.entity_name": {"$regex": f"^{name}$", "$options": "i"},
                "metadata.entity_type": entity_type
            })

            if doc:
                doc["id"] = str(doc.pop("_id"))
                return Memory(**doc)
            return None
        except Exception as e:
            logger.error(f"Failed to find existing entity: {e}")
            return None

    async def _merge_entities(self, existing: Memory, new: Memory) -> str:
        """Merge new entity info with existing."""
        # Increment mentions
        existing.metadata["mentions"] = existing.metadata.get("mentions", 1) + 1

        # Update importance based on mentions
        existing.importance = min(1.0, existing.importance + 0.1)

        await self.update(
            existing.id,
            {
                "metadata": existing.metadata,
                "importance": existing.importance
            }
        )

        return existing.id

    async def get_entities_by_type(
        self,
        entity_type: str,
        limit: int = 50
    ) -> List[Memory]:
        """Get entities of a specific type."""
        return await self.list_memories(
            filters={"metadata.entity_type": entity_type},
            limit=limit
        )

    async def increment_mentions(self, memory_id: str) -> bool:
        """Increment entity mention count."""
        try:
            result = await self.collection.update_one(
                {"_id": ObjectId(memory_id)},
                {
                    "$inc": {"metadata.mentions": 1},
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to increment mentions: {e}")
            return False
