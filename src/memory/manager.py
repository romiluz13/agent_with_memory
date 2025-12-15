"""
Memory Manager - Orchestrates all 5 memory types
Central hub for memory operations across the system
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorDatabase

from .base import Memory, MemoryType, MemoryImportance, MemoryStore
from .episodic import EpisodicMemory
from .procedural import ProceduralMemory
from .semantic import SemanticMemory
from .working import WorkingMemory
from .cache import SemanticCache
from .entity import EntityMemory
from .summary import SummaryMemory
from ..embeddings.voyage_client import get_embedding_service
from ..retrieval.vector_search import MultiCollectionSearch

logger = logging.getLogger(__name__)


class MemoryConfig:
    """Configuration for memory system."""
    
    def __init__(
        self,
        episodic_ttl_days: int = 90,
        working_ttl_minutes: int = 60,
        consolidation_interval_hours: int = 24,
        importance_threshold: float = 0.7,
        cache_ttl_seconds: int = 3600,
        auto_consolidate: bool = True
    ):
        self.episodic_ttl_days = episodic_ttl_days
        self.working_ttl_minutes = working_ttl_minutes
        self.consolidation_interval_hours = consolidation_interval_hours
        self.importance_threshold = importance_threshold
        self.cache_ttl_seconds = cache_ttl_seconds
        self.auto_consolidate = auto_consolidate


class MemoryManager:
    """
    Central manager for all 5 memory types.
    Coordinates memory operations and provides unified interface.
    """
    
    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        config: Optional[MemoryConfig] = None
    ):
        """
        Initialize memory manager with database.
        
        Args:
            db: MongoDB database instance
            config: Memory configuration
        """
        self.db = db
        self.config = config or MemoryConfig()
        
        # Initialize all memory stores
        self.episodic = EpisodicMemory(db["episodic_memories"])
        self.procedural = ProceduralMemory(db["procedural_memories"])
        self.semantic = SemanticMemory(db["semantic_memories"])
        self.working = WorkingMemory(db["working_memories"])
        self.cache = SemanticCache(db["semantic_cache"])
        self.entity = EntityMemory(db["entity_memories"])
        self.summary = SummaryMemory(db["summary_memories"])

        # Memory type mapping
        self.stores: Dict[MemoryType, MemoryStore] = {
            MemoryType.EPISODIC: self.episodic,
            MemoryType.PROCEDURAL: self.procedural,
            MemoryType.SEMANTIC: self.semantic,
            MemoryType.WORKING: self.working,
            MemoryType.CACHE: self.cache,
            MemoryType.ENTITY: self.entity,
            MemoryType.SUMMARY: self.summary
        }
        
        # Multi-collection search
        self.multi_search = MultiCollectionSearch({
            "episodic": db["episodic_memories"],
            "procedural": db["procedural_memories"],
            "semantic": db["semantic_memories"],
            "entity": db["entity_memories"],
            "summary": db["summary_memories"]
        })
        
        # Embedding service
        self.embedding_service = get_embedding_service()
        
        # Stats tracking
        self.stats = {
            "stores": 0,
            "retrievals": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        agent_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Store a memory in the appropriate store.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            agent_id: Agent ID
            user_id: Optional user ID
            metadata: Additional metadata
            importance: Importance score (0-1)
            tags: Optional tags
            
        Returns:
            Memory ID
        """
        # Generate embedding
        embedding_result = await self.embedding_service.generate_embedding(
            content,
            input_type="document"
        )
        
        # Determine importance level
        if importance >= 0.9:
            importance_level = MemoryImportance.CRITICAL
        elif importance >= 0.7:
            importance_level = MemoryImportance.HIGH
        elif importance >= 0.5:
            importance_level = MemoryImportance.MEDIUM
        elif importance >= 0.3:
            importance_level = MemoryImportance.LOW
        else:
            importance_level = MemoryImportance.TRIVIAL
        
        # Set TTL based on memory type
        ttl = None
        if memory_type == MemoryType.EPISODIC:
            ttl = datetime.utcnow() + timedelta(days=self.config.episodic_ttl_days)
        elif memory_type == MemoryType.WORKING:
            ttl = datetime.utcnow() + timedelta(minutes=self.config.working_ttl_minutes)
        elif memory_type == MemoryType.CACHE:
            ttl = datetime.utcnow() + timedelta(seconds=self.config.cache_ttl_seconds)
        
        # Create memory object
        memory = Memory(
            agent_id=agent_id,
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            embedding=embedding_result.embedding,
            metadata=metadata or {},
            importance=importance,
            importance_level=importance_level,
            ttl=ttl,
            tags=tags or []
        )
        
        # Store in appropriate store
        store = self.stores[memory_type]
        memory_id = await store.store(memory)
        
        self.stats["stores"] += 1
        logger.info(f"Stored {memory_type} memory: {memory_id}")
        
        return memory_id
    
    async def retrieve_memories(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10,
        threshold: float = 0.7,
        use_cache: bool = True,
        search_mode: str = "hybrid"
    ) -> List[Memory]:
        """
        Retrieve relevant memories across specified types.

        Uses hybrid search (vector + full-text) by default for best results.
        Based on MongoDB's official GenAI-Showcase pattern with $rankFusion.

        Args:
            query: Query text
            memory_types: Types to search (None = all)
            limit: Maximum memories to return
            threshold: Minimum similarity threshold
            use_cache: Whether to check cache first
            search_mode: Search strategy - "hybrid" (default), "semantic", or "text"

        Returns:
            List of relevant memories
        """
        # Check cache first if enabled
        if use_cache:
            cache_results = await self.cache.retrieve(
                query, limit=1, threshold=0.95, search_mode=search_mode
            )
            if cache_results:
                self.stats["cache_hits"] += 1
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cache_results
            self.stats["cache_misses"] += 1

        # Generate query embedding
        embedding_result = await self.embedding_service.generate_embedding(
            query,
            input_type="query"
        )

        # Determine which memory types to search
        if memory_types is None:
            memory_types = [
                MemoryType.EPISODIC,
                MemoryType.PROCEDURAL,
                MemoryType.SEMANTIC
            ]

        # Search across specified memory types
        all_results = []

        for memory_type in memory_types:
            if memory_type in self.stores:
                store = self.stores[memory_type]
                results = await store.retrieve(
                    query, limit=limit, threshold=threshold, search_mode=search_mode
                )
                all_results.extend(results)
        
        # Sort by relevance (score stored in metadata during retrieval)
        all_results.sort(
            key=lambda m: m.metadata.get("search_score", 0),
            reverse=True
        )
        
        # Limit total results
        final_results = all_results[:limit]
        
        # Store in cache for future use
        if use_cache and final_results:
            cache_content = f"Query: {query}\nResults: {[m.content for m in final_results]}"
            await self.store_memory(
                content=cache_content,
                memory_type=MemoryType.CACHE,
                agent_id="system",
                metadata={"query": query, "results_count": len(final_results)}
            )
        
        self.stats["retrievals"] += 1
        logger.info(f"Retrieved {len(final_results)} memories for query: {query[:50]}...")
        
        return final_results
    
    async def consolidate_memories(
        self,
        agent_id: str,
        memory_type: Optional[MemoryType] = None
    ) -> int:
        """
        Consolidate similar memories to reduce redundancy.
        
        Args:
            agent_id: Agent ID
            memory_type: Specific type to consolidate (None = all)
            
        Returns:
            Number of memories consolidated
        """
        total_consolidated = 0
        
        # Determine which types to consolidate
        types_to_consolidate = [memory_type] if memory_type else [
            MemoryType.EPISODIC,
            MemoryType.SEMANTIC
        ]
        
        for mem_type in types_to_consolidate:
            if mem_type in self.stores:
                store = self.stores[mem_type]
                consolidated = await store.consolidate()
                total_consolidated += consolidated
        
        logger.info(f"Consolidated {total_consolidated} memories for agent {agent_id}")
        return total_consolidated
    
    async def transfer_to_longterm(
        self,
        agent_id: str,
        threshold_hours: int = 24
    ) -> int:
        """
        Transfer important working memories to long-term storage.
        
        Args:
            agent_id: Agent ID
            threshold_hours: Age threshold for transfer
            
        Returns:
            Number of memories transferred
        """
        # Get old working memories
        cutoff = datetime.utcnow() - timedelta(hours=threshold_hours)
        old_memories = await self.working.list_memories(
            filters={
                "agent_id": agent_id,
                "created_at": {"$lt": cutoff},
                "importance": {"$gte": self.config.importance_threshold}
            }
        )
        
        transferred = 0
        for memory in old_memories:
            # Determine target memory type based on content
            if "procedure" in memory.tags or "workflow" in memory.tags:
                target_type = MemoryType.PROCEDURAL
            elif "fact" in memory.tags or "knowledge" in memory.tags:
                target_type = MemoryType.SEMANTIC
            else:
                target_type = MemoryType.EPISODIC
            
            # Store in long-term memory
            await self.store_memory(
                content=memory.content,
                memory_type=target_type,
                agent_id=agent_id,
                user_id=memory.user_id,
                metadata=memory.metadata,
                importance=memory.importance,
                tags=memory.tags
            )
            
            # Remove from working memory
            await self.working.delete(memory.id)
            transferred += 1
        
        logger.info(f"Transferred {transferred} memories to long-term storage")
        return transferred
    
    async def get_agent_context(
        self,
        agent_id: str,
        user_id: Optional[str] = None,
        include_working: bool = True
    ) -> Dict[str, Any]:
        """
        Get complete context for an agent.
        
        Args:
            agent_id: Agent ID
            user_id: Optional user ID
            include_working: Include working memory
            
        Returns:
            Complete context dictionary
        """
        context = {
            "agent_id": agent_id,
            "user_id": user_id,
            "memories": {},
            "stats": {}
        }
        
        # Get memories from each type
        for memory_type, store in self.stores.items():
            if memory_type == MemoryType.WORKING and not include_working:
                continue
            
            filters = {"agent_id": agent_id}
            if user_id:
                filters["user_id"] = user_id
            
            memories = await store.list_memories(filters=filters, limit=20)
            context["memories"][memory_type.value] = [
                {
                    "content": m.content,
                    "importance": m.importance,
                    "created_at": m.created_at.isoformat(),
                    "tags": m.tags
                }
                for m in memories
            ]
            context["stats"][memory_type.value] = len(memories)
        
        return context
    
    async def clear_agent_memories(
        self,
        agent_id: str,
        memory_types: Optional[List[MemoryType]] = None,
        confirm: bool = False
    ) -> int:
        """
        Clear all memories for an agent.
        
        Args:
            agent_id: Agent ID
            memory_types: Specific types to clear (None = all)
            confirm: Must be True to proceed
            
        Returns:
            Number of memories deleted
        """
        if not confirm:
            raise ValueError("Must confirm memory deletion")
        
        total_deleted = 0
        
        # Determine which types to clear
        types_to_clear = memory_types or list(self.stores.keys())
        
        for memory_type in types_to_clear:
            if memory_type in self.stores:
                store = self.stores[memory_type]
                memories = await store.list_memories(
                    filters={"agent_id": agent_id}
                )
                for memory in memories:
                    if await store.delete(memory.id):
                        total_deleted += 1
        
        logger.warning(f"Deleted {total_deleted} memories for agent {agent_id}")
        return total_deleted
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get memory system statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "operations": self.stats.copy(),
            "memory_counts": {},
            "cache_performance": {
                "hit_rate": 0.0
            }
        }
        
        # Get counts for each memory type
        for memory_type, store in self.stores.items():
            count = len(await store.list_memories(limit=1))
            stats["memory_counts"][memory_type.value] = count
        
        # Calculate cache hit rate
        total_cache_ops = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total_cache_ops > 0:
            stats["cache_performance"]["hit_rate"] = (
                self.stats["cache_hits"] / total_cache_ops
            )

        return stats

    async def extract_entities(
        self,
        text: str,
        agent_id: str,
        llm,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract and store entities from text using LLM.

        Args:
            text: Text to extract entities from
            agent_id: Agent ID for memory ownership
            llm: LLM instance for extraction
            user_id: Optional user ID

        Returns:
            List of extracted entity dictionaries
        """
        return await self.entity.extract_and_store(
            text=text,
            llm=llm,
            agent_id=agent_id,
            user_id=user_id
        )

    async def get_summary_references(
        self,
        agent_id: str,
        thread_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """
        Get summary references for context inclusion.

        Args:
            agent_id: Agent ID
            thread_id: Optional thread filter
            limit: Max references

        Returns:
            List of {summary_id, description} for context
        """
        return await self.summary.list_summary_references(
            agent_id=agent_id,
            thread_id=thread_id,
            limit=limit
        )
