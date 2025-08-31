"""
MongoDB Connection Manager
Handles async MongoDB connections with connection pooling
"""

import logging
from typing import Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class MongoDBConfig(BaseModel):
    """MongoDB connection configuration."""
    uri: str = Field(..., description="MongoDB connection URI")
    database: str = Field(default="ai_agent_boilerplate", description="Database name")
    max_pool_size: int = Field(default=100, description="Maximum connection pool size")
    min_pool_size: int = Field(default=10, description="Minimum connection pool size")
    max_idle_time_ms: int = Field(default=30000, description="Max idle time for connections")
    server_selection_timeout_ms: int = Field(default=5000, description="Server selection timeout")


class MongoDBClient:
    """Manages MongoDB connections with connection pooling."""
    
    _instance: Optional['MongoDBClient'] = None
    _client: Optional[AsyncIOMotorClient] = None
    _db: Optional[AsyncIOMotorDatabase] = None
    
    def __new__(cls):
        """Singleton pattern for connection management."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the client (only once due to singleton)."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.config: Optional[MongoDBConfig] = None
    
    async def initialize(self, config: MongoDBConfig):
        """
        Initialize the MongoDB connection.
        
        Args:
            config: MongoDB configuration
        """
        if self._client is not None:
            logger.warning("MongoDB client already initialized")
            return
        
        self.config = config
        
        try:
            # Create client with connection pooling
            self._client = AsyncIOMotorClient(
                config.uri,
                maxPoolSize=config.max_pool_size,
                minPoolSize=config.min_pool_size,
                maxIdleTimeMS=config.max_idle_time_ms,
                serverSelectionTimeoutMS=config.server_selection_timeout_ms,
            )
            
            # Get database reference
            self._db = self._client[config.database]
            
            # Test connection
            await self._client.admin.command('ping')
            logger.info(f"Connected to MongoDB database: {config.database}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    @property
    def db(self) -> AsyncIOMotorDatabase:
        """Get the database instance."""
        if self._db is None:
            raise RuntimeError("MongoDB client not initialized. Call initialize() first.")
        return self._db
    
    @property
    def client(self) -> AsyncIOMotorClient:
        """Get the client instance."""
        if self._client is None:
            raise RuntimeError("MongoDB client not initialized. Call initialize() first.")
        return self._client
    
    async def close(self):
        """Close the MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("MongoDB connection closed")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the database connection.
        
        Returns:
            Health status information
        """
        try:
            # Ping the server
            result = await self._client.admin.command('ping')
            
            # Get server info
            server_info = await self._client.server_info()
            
            # Get database stats
            db_stats = await self._db.command("dbStats")
            
            return {
                "status": "healthy",
                "ping": result,
                "version": server_info.get("version"),
                "database": self.config.database,
                "collections": db_stats.get("collections"),
                "dataSize": db_stats.get("dataSize"),
                "indexes": db_stats.get("indexes"),
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def get_collection(self, name: str):
        """
        Get a collection by name.
        
        Args:
            name: Collection name
            
        Returns:
            Collection instance
        """
        return self.db[name]
    
    async def list_collections(self) -> list:
        """
        List all collections in the database.
        
        Returns:
            List of collection names
        """
        return await self.db.list_collection_names()
    
    async def create_collection(self, name: str, **kwargs) -> bool:
        """
        Create a new collection.
        
        Args:
            name: Collection name
            **kwargs: Additional collection options
            
        Returns:
            True if created successfully
        """
        try:
            await self.db.create_collection(name, **kwargs)
            logger.info(f"Created collection: {name}")
            return True
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Collection {name} already exists")
                return True
            logger.error(f"Failed to create collection {name}: {e}")
            return False


# Global instance
mongodb_client = MongoDBClient()


@asynccontextmanager
async def get_mongodb():
    """
    Async context manager for MongoDB access.
    
    Usage:
        async with get_mongodb() as db:
            collection = db["memories"]
            # Use collection...
    """
    if mongodb_client._db is None:
        raise RuntimeError("MongoDB not initialized")
    
    yield mongodb_client.db


async def initialize_mongodb(
    uri: str,
    database: str = "ai_agent_boilerplate",
    **kwargs
) -> MongoDBClient:
    """
    Initialize the global MongoDB client.
    
    Args:
        uri: MongoDB connection URI
        database: Database name
        **kwargs: Additional configuration options
        
    Returns:
        Initialized MongoDB client
    """
    config = MongoDBConfig(
        uri=uri,
        database=database,
        **kwargs
    )
    
    await mongodb_client.initialize(config)
    return mongodb_client


# Example usage
if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def main():
        # Initialize
        client = await initialize_mongodb(
            uri=os.getenv("MONGODB_URI"),
            database=os.getenv("MONGODB_DB_NAME", "ai_agent_boilerplate")
        )
        
        # Health check
        health = await client.health_check()
        print(f"Health: {health}")
        
        # List collections
        collections = await client.list_collections()
        print(f"Collections: {collections}")
        
        # Close
        await client.close()
    
    asyncio.run(main())
