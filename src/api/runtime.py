"""
Application runtime services for the FastAPI API surface.

MongoDB rule for this repository:
MongoDB skills plus official MongoDB documentation are the source of truth for
MongoDB behavior, deployment, and indexing decisions.
"""

import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, Field
from pymongo import MongoClient

from ..core.agent_langgraph import MongoDBLangGraphAgent
from ..memory.manager import MemoryManager
from ..storage.mongodb_client import MongoDBClient, initialize_mongodb, mongodb_client
from ..tools.nl_to_mql import NLToMQLGenerator
from .websocket import WebSocketManager

logger = logging.getLogger(__name__)

ValidationLane = Literal["developer_quickstart", "local_validation", "cloud_validation"]


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "agent"


class RuntimeSettings(BaseModel):
    """Settings used by the application runtime."""

    mongodb_uri: str | None = Field(default=None)
    validation_lane: ValidationLane = Field(default="developer_quickstart")
    developer_quickstart_db_name: str = Field(default="awm_quickstart")
    local_validation_db_name: str = Field(default="awm_local_validation")
    cloud_validation_db_name: str = Field(default="awm_cloud_validation")
    explicit_database_name: str | None = Field(default=None)
    cors_allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    auth_required: bool = Field(default=False)
    jwt_secret_key: str | None = Field(default=None)
    valid_api_keys: list[str] = Field(default_factory=list)
    default_agent_provider: str = Field(default="openai")
    default_agent_model: str = Field(default="gpt-4o")
    default_agent_temperature: float = Field(default=0.7)
    default_agent_max_tokens: int = Field(default=2000)
    default_agent_enable_streaming: bool = Field(default=True)
    default_agent_name: str = Field(default="assistant")
    default_agent_description: str = Field(default="General-purpose assistant with memory")
    default_agent_system_prompt: str | None = Field(default=None)

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        validation_lane = os.getenv("MONGODB_VALIDATION_LANE", "developer_quickstart")
        default_provider = os.getenv("DEFAULT_AGENT_PROVIDER")
        if not default_provider:
            if os.getenv("OPENAI_API_KEY"):
                default_provider = "openai"
            elif os.getenv("GOOGLE_API_KEY"):
                default_provider = "google"
            elif os.getenv("ANTHROPIC_API_KEY"):
                default_provider = "anthropic"
            else:
                default_provider = "openai"

        return cls(
            mongodb_uri=os.getenv("MONGODB_URI"),
            validation_lane=validation_lane,
            developer_quickstart_db_name=os.getenv(
                "MONGODB_DEVELOPER_QUICKSTART_DB_NAME", "awm_quickstart"
            ),
            local_validation_db_name=os.getenv(
                "MONGODB_LOCAL_VALIDATION_DB_NAME", "awm_local_validation"
            ),
            cloud_validation_db_name=os.getenv(
                "MONGODB_CLOUD_VALIDATION_DB_NAME", "awm_cloud_validation"
            ),
            explicit_database_name=os.getenv("MONGODB_DB_NAME"),
            cors_allowed_origins=_parse_csv(
                os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000")
            )
            or ["http://localhost:3000"],
            auth_required=_parse_bool(os.getenv("AUTH_REQUIRED"), default=False),
            jwt_secret_key=os.getenv("JWT_SECRET_KEY"),
            valid_api_keys=_parse_csv(os.getenv("VALID_API_KEYS")),
            default_agent_provider=default_provider,
            default_agent_model=os.getenv(
                "DEFAULT_AGENT_MODEL",
                os.getenv("OPENAI_MODEL", os.getenv("GOOGLE_MODEL", "gpt-4o")),
            ),
            default_agent_temperature=float(os.getenv("AGENT_DEFAULT_TEMPERATURE", "0.7")),
            default_agent_max_tokens=int(os.getenv("AGENT_DEFAULT_MAX_TOKENS", "2000")),
            default_agent_enable_streaming=_parse_bool(
                os.getenv("AGENT_ENABLE_STREAMING"), default=True
            ),
            default_agent_name=os.getenv("DEFAULT_AGENT_NAME", "assistant"),
            default_agent_description=os.getenv(
                "DEFAULT_AGENT_DESCRIPTION",
                "General-purpose assistant with memory",
            ),
            default_agent_system_prompt=os.getenv("DEFAULT_AGENT_SYSTEM_PROMPT"),
        )

    @property
    def database_name(self) -> str:
        if self.explicit_database_name:
            return self.explicit_database_name
        if self.validation_lane == "cloud_validation":
            return self.cloud_validation_db_name
        if self.validation_lane == "local_validation":
            return self.local_validation_db_name
        return self.developer_quickstart_db_name


class AgentMetadata(BaseModel):
    """Persisted metadata for a runtime agent."""

    agent_id: str
    name: str
    description: str = ""
    model_provider: str = "openai"
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 2000
    system_prompt: str | None = None
    enable_streaming: bool = True
    database_name: str | None = None
    status: str = "active"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentRegistry:
    """Tracks persisted agent metadata and live LangGraph agent instances."""

    def __init__(self, settings: RuntimeSettings):
        self.settings = settings
        self._metadata_cache: dict[str, AgentMetadata] = {}
        self._instances: dict[str, MongoDBLangGraphAgent] = {}
        self._db: AsyncIOMotorDatabase | None = None
        self._memory_manager: MemoryManager | None = None
        self._sync_client: MongoClient | None = None

    @property
    def collection(self):
        if self._db is None:
            return None
        return self._db["agents"]

    def bind_runtime(
        self,
        db: AsyncIOMotorDatabase | None,
        memory_manager: MemoryManager | None,
        sync_client: MongoClient | None,
    ) -> None:
        self._db = db
        self._memory_manager = memory_manager
        self._sync_client = sync_client

    async def initialize(self) -> None:
        if self.collection is None:
            return

        await self.collection.create_index("agent_id", unique=True)
        await self.collection.create_index("updated_at")

        async for doc in self.collection.find({}):
            doc.pop("_id", None)
            metadata = AgentMetadata(**doc)
            self._metadata_cache[metadata.agent_id] = metadata

        await self.ensure_default_agents()

    async def ensure_default_agents(self) -> None:
        default_specs = [
            {
                "agent_id": "assistant",
                "name": self.settings.default_agent_name,
                "description": self.settings.default_agent_description,
                "model_provider": self.settings.default_agent_provider,
                "model_name": self.settings.default_agent_model,
                "temperature": self.settings.default_agent_temperature,
                "max_tokens": self.settings.default_agent_max_tokens,
                "system_prompt": self.settings.default_agent_system_prompt,
                "enable_streaming": self.settings.default_agent_enable_streaming,
                "database_name": self.settings.database_name,
            },
            {
                "agent_id": "research",
                "name": "research",
                "description": "Research-oriented assistant with memory",
                "model_provider": self.settings.default_agent_provider,
                "model_name": self.settings.default_agent_model,
                "temperature": 0.2,
                "max_tokens": self.settings.default_agent_max_tokens,
                "system_prompt": (
                    "You are a rigorous research assistant. Distinguish evidence from "
                    "speculation, cite retrieved context, and preserve important facts."
                ),
                "enable_streaming": self.settings.default_agent_enable_streaming,
                "database_name": self.settings.database_name,
            },
        ]

        for spec in default_specs:
            if spec["agent_id"] not in self._metadata_cache:
                await self.create_agent(spec, persist_only=not self.is_runtime_available)

    @property
    def is_runtime_available(self) -> bool:
        return bool(self.settings.mongodb_uri and self._sync_client is not None)

    async def create_agent(self, data: dict[str, Any], persist_only: bool = False) -> AgentMetadata:
        now = datetime.now(UTC)
        agent_id = (
            data.get("agent_id") or f"{_slugify(data['name'])}-{now.strftime('%Y%m%d%H%M%S')}"
        )
        metadata = AgentMetadata(
            agent_id=agent_id,
            name=data["name"],
            description=data.get("description", ""),
            model_provider=data.get("model_provider", self.settings.default_agent_provider),
            model_name=data.get("model_name", self.settings.default_agent_model),
            temperature=float(data.get("temperature", self.settings.default_agent_temperature)),
            max_tokens=int(data.get("max_tokens", self.settings.default_agent_max_tokens)),
            system_prompt=data.get("system_prompt"),
            enable_streaming=bool(
                data.get("enable_streaming", self.settings.default_agent_enable_streaming)
            ),
            database_name=data.get("database_name") or self.settings.database_name,
            created_at=now,
            updated_at=now,
        )

        self._metadata_cache[metadata.agent_id] = metadata

        if self.collection is not None:
            await self.collection.replace_one(
                {"agent_id": metadata.agent_id},
                metadata.model_dump(mode="json"),
                upsert=True,
            )

        if not persist_only and self.is_runtime_available:
            self._instances[metadata.agent_id] = self._build_agent(metadata)

        return metadata

    async def list_agents(self) -> list[AgentMetadata]:
        if self.collection is not None and not self._metadata_cache:
            await self.initialize()
        return sorted(
            self._metadata_cache.values(),
            key=lambda item: (
                item.created_at
                if item.created_at.tzinfo is not None
                else item.created_at.replace(tzinfo=UTC)
            ),
        )

    async def get_metadata(self, agent_id: str) -> AgentMetadata | None:
        metadata = self._metadata_cache.get(agent_id)
        if metadata or self.collection is None:
            return metadata

        doc = await self.collection.find_one({"agent_id": agent_id})
        if not doc:
            return None

        doc.pop("_id", None)
        metadata = AgentMetadata(**doc)
        self._metadata_cache[agent_id] = metadata
        return metadata

    async def ensure_agent(self, agent_id: str) -> MongoDBLangGraphAgent:
        if agent_id in self._instances:
            return self._instances[agent_id]

        metadata = await self.get_metadata(agent_id)
        if metadata is None:
            raise KeyError(agent_id)
        if not self.is_runtime_available:
            raise RuntimeError("MongoDB runtime is not configured; cannot load live agent")

        agent = self._build_agent(metadata)
        self._instances[agent_id] = agent
        return agent

    async def update_agent(self, agent_id: str, updates: dict[str, Any]) -> AgentMetadata:
        metadata = await self.get_metadata(agent_id)
        if metadata is None:
            raise KeyError(agent_id)

        payload = metadata.model_dump()
        payload.update({key: value for key, value in updates.items() if value is not None})
        payload["updated_at"] = datetime.now(UTC)
        updated = AgentMetadata(**payload)
        self._metadata_cache[agent_id] = updated

        if self.collection is not None:
            await self.collection.replace_one(
                {"agent_id": agent_id},
                updated.model_dump(mode="json"),
                upsert=True,
            )

        existing = self._instances.pop(agent_id, None)
        if existing is not None:
            await existing.close()

        return updated

    async def delete_agent(self, agent_id: str) -> bool:
        metadata = await self.get_metadata(agent_id)
        if metadata is None:
            return False

        live = self._instances.pop(agent_id, None)
        if live is not None:
            await live.close()

        self._metadata_cache.pop(agent_id, None)

        if self.collection is not None:
            await self.collection.delete_one({"agent_id": agent_id})

        return True

    async def close(self) -> None:
        for agent in list(self._instances.values()):
            await agent.close()
        self._instances.clear()

    def _build_agent(self, metadata: AgentMetadata) -> MongoDBLangGraphAgent:
        if not self.settings.mongodb_uri:
            raise RuntimeError("MONGODB_URI is required to build an agent runtime")

        return MongoDBLangGraphAgent(
            mongodb_uri=self.settings.mongodb_uri,
            mongo_client=self._sync_client,
            agent_id=metadata.agent_id,
            agent_name=metadata.name,
            model_provider=metadata.model_provider,
            model_name=metadata.model_name,
            temperature=metadata.temperature,
            max_tokens=metadata.max_tokens,
            database_name=metadata.database_name or self.settings.database_name,
            system_prompt=metadata.system_prompt,
            user_tools=[],
            enable_streaming=metadata.enable_streaming,
        )


@dataclass
class ApplicationRuntime:
    """Shared application runtime state."""

    settings: RuntimeSettings
    mongodb: MongoDBClient | None
    sync_mongo_client: MongoClient | None
    memory_manager: MemoryManager | None
    agent_registry: AgentRegistry
    websocket_manager: WebSocketManager
    nl_query_generator: NLToMQLGenerator | None
    startup_error: str | None = None


async def initialize_runtime() -> ApplicationRuntime:
    """Create and initialize the app runtime."""
    settings = RuntimeSettings.from_env()
    websocket_manager = WebSocketManager()
    agent_registry = AgentRegistry(settings)

    mongodb: MongoDBClient | None = None
    sync_mongo_client: MongoClient | None = None
    memory_manager: MemoryManager | None = None
    nl_query_generator: NLToMQLGenerator | None = None
    startup_error: str | None = None

    try:
        if settings.mongodb_uri:
            mongodb = await initialize_mongodb(
                uri=settings.mongodb_uri,
                database=settings.database_name,
                max_pool_size=int(os.getenv("MONGODB_MAX_POOL_SIZE", "100")),
                min_pool_size=int(os.getenv("MONGODB_MIN_POOL_SIZE", "10")),
            )
            memory_manager = MemoryManager(mongodb.db)
            nl_query_generator = NLToMQLGenerator(mongodb.db)
            sync_mongo_client = MongoClient(settings.mongodb_uri)
            agent_registry.bind_runtime(mongodb.db, memory_manager, sync_mongo_client)
            await agent_registry.initialize()
            await _ensure_runtime_indexes(mongodb.db)
        else:
            startup_error = "MONGODB_URI is not configured"
            logger.warning(startup_error)
    except Exception as exc:
        startup_error = str(exc)
        logger.exception("Runtime initialization failed")

    return ApplicationRuntime(
        settings=settings,
        mongodb=mongodb,
        sync_mongo_client=sync_mongo_client,
        memory_manager=memory_manager,
        agent_registry=agent_registry,
        websocket_manager=websocket_manager,
        nl_query_generator=nl_query_generator,
        startup_error=startup_error,
    )


async def shutdown_runtime(runtime: ApplicationRuntime) -> None:
    """Close runtime resources cleanly."""
    await runtime.agent_registry.close()
    await runtime.websocket_manager.disconnect_all()

    if runtime.mongodb is not None:
        await runtime.mongodb.close()

    if runtime.sync_mongo_client is not None:
        runtime.sync_mongo_client.close()

    mongodb_client.reset()


async def _ensure_runtime_indexes(db: AsyncIOMotorDatabase) -> None:
    """Create runtime indexes needed by the HTTP API."""
    approval_requests = db["approval_requests"]
    await approval_requests.create_index(
        [("agent_id", 1), ("status", 1), ("created_at", -1)],
        name="approval_requests_agent_status_created_at",
    )
