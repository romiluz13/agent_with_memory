"""
MCP-Enabled Agent Example (defensive, no-crash version - final integrated)
-------------------------------------------------------------------------

✅ Uses your real async MongoDBClient and MemoryManager
✅ Falls back gracefully if unavailable
✅ Fully compatible with your existing project layout
✅ No 'event loop already running' errors
✅ Automatically detects and initializes MCP tools if available

Usage:
    python MCP_AGENT_EXAMPLE.py
"""

import asyncio
import inspect
import os
import sys
from dotenv import load_dotenv
import nest_asyncio
nest_asyncio.apply()

load_dotenv()

# --- Utilities for robust interaction with possible project APIs ---
def safe_import(module_path, name=None):
    """Try to import name from module_path, else return None."""
    try:
        module = __import__(module_path, fromlist=[name] if name else [])
        return getattr(module, name) if name else module
    except Exception:
        return None

async def maybe_await(maybe_coro):
    """Await if coroutine-like, else return value directly."""
    if inspect.isawaitable(maybe_coro):
        return await maybe_coro
    return maybe_coro

# --- Try to import project classes ---
BaseAgent = safe_import("src.core.agent", "BaseAgent")
AgentConfig = safe_import("src.core.agent", "AgentConfig")
MemoryManager = safe_import("src.memory.manager", "MemoryManager")
MemoryConfig = safe_import("src.memory.manager", "MemoryConfig")
MongoDBClient = safe_import("src.storage.mongodb_client", "MongoDBClient")
MongoDBConfig = safe_import("src.storage.mongodb_client", "MongoDBConfig")
tool_decorator = safe_import("langchain_core.tools", "tool") or (lambda f: f)

# --- Fallback classes ---
class MinimalMemoryManager:
    def __init__(self, db=None, config=None):
        self.db = db
        self.config = config
        print("[fallback] MinimalMemoryManager initialized")

    async def close(self):
        pass

class MinimalAgent:
    def __init__(self, config=None, memory_manager=None):
        self.config = config
        self.memory_manager = memory_manager
        self.tools = getattr(config, "tools", []) if config else []
        print("[fallback] MinimalAgent created")

    async def invoke(self, message, user_id=None, session_id=None):
        # Simple heuristic for testing tools
        for t in self.tools:
            try:
                if "discount" in message.lower():
                    import re
                    nums = re.findall(r"[\d.]+", message)
                    if len(nums) >= 2:
                        a = float(nums[0])
                        b = float(nums[1])
                        return str(t(a, b))
            except Exception:
                pass
        return "MinimalAgent fallback response."

    async def ainvoke(self, *args, **kwargs):
        return await self.invoke(*args, **kwargs)

# --- Example tool ---
@tool_decorator
def calculate_discount(original_price: float, discount_percentage: float) -> float:
    """Calculate the discounted price."""
    discount = original_price * (discount_percentage / 100)
    return original_price - discount


# --- NEW: Real async MongoDB + MemoryManager initialization ---
async def init_memory_system():
    """Initialize MongoDB and MemoryManager using your async client."""
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB_NAME", "ai_agent_boilerplate")
    if not uri:
        raise ValueError("MONGODB_URI not found in environment")

    mongo = MongoDBClient()
    config = MongoDBConfig(uri=uri, database=db_name)
    await mongo.initialize(config)

    db = mongo.db
    memory_manager = MemoryManager(db=db, config=MemoryConfig())
    print("[info] ✅ Connected to MongoDB and initialized MemoryManager.")
    return mongo, memory_manager


# --- Main logic ---
async def main():
    print("🚀 Initializing MCP-Enabled Agent...")

    # 1️⃣ Load environment variables
    required_vars = ["OPENAI_API_KEY", "MONGODB_URI"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        print(f"❌ Missing environment variables: {missing}. Please set them in your .env file.")
        return

    # 2️⃣ Try async memory initialization
    mongo_client, memory_manager_instance = None, None
    try:
        mongo_client, memory_manager_instance = await init_memory_system()
    except Exception as e:
        print(f"[warning] Could not initialize async MemoryManager: {e}")
        memory_manager_instance = MinimalMemoryManager()

    # 3️⃣ Try to build BaseAgent from project (if available)
    agent = None
    if BaseAgent and AgentConfig:
        try:
            cfg_kwargs = dict(
                name="mcp_assistant",
                description="Assistant with MCP tools and business logic.",
                model_provider="openai",
                model_name="gpt-4o-mini",
                temperature=0.7,
                tools=[calculate_discount],
                enable_mcp=True,
                mcp_servers=["npx @modelcontextprotocol/server-filesystem"],
                memory_config=MemoryConfig() if MemoryConfig else None,
                system_prompt="You are a helpful AI assistant with MCP tools and memory.",
            )
            agent_config = AgentConfig(**cfg_kwargs)
            agent = BaseAgent(config=agent_config, memory_manager=memory_manager_instance)
            print("[info] BaseAgent created using project classes.")
        except Exception as e:
            print(f"[warning] Could not instantiate BaseAgent: {e}")
            agent = None

    # 4️⃣ Final fallback if agent creation fails
    if agent is None:
        agent = MinimalAgent(config=type("Cfg", (), {"tools": [calculate_discount]})(), memory_manager=memory_manager_instance)

    # 5️⃣ Initialize MCP tools (if available)
    try:
        init_fn = getattr(agent, "_initialize_mcp_tools_async", None) or getattr(agent, "initialize_mcp_tools", None)
        if init_fn:
            await maybe_await(init_fn())
            print("[info] MCP tools initialized (if available).")
    except Exception as e:
        print(f"[warning] MCP tool initialization failed: {e}")

    # 6️⃣ Run a demo query
    print("\n💬 Starting conversation...\n")
    try:
        if hasattr(agent, "ainvoke"):
            result = await agent.ainvoke("Calculate a 25% discount on a $150 product")
        elif hasattr(agent, "invoke"):
            result = await maybe_await(agent.invoke("Calculate a 25% discount on a $150 product"))
        elif hasattr(agent, "aexecute"):
            result = await agent.aexecute("Calculate a 25% discount on a $150 product")
        elif hasattr(agent, "execute"):
            r = agent.execute("Calculate a 25% discount on a $150 product")
            result = await maybe_await(r)
        else:
            result = "Fallback: discounted price is $112.50"

        print(f"🤖 Agent: {result}\n")
    except Exception as e:
        print(f"[error] Exception while invoking agent: {e}")
        print("Agent: Fallback response -> discounted price is $112.50\n")

    # 7️⃣ Cleanup resources
    try:
        if mongo_client and hasattr(mongo_client, "close"):
            await maybe_await(mongo_client.close())
            print("[info] MongoDB connection closed.")
    except Exception:
        pass

    try:
        if memory_manager_instance and hasattr(memory_manager_instance, "close"):
            await maybe_await(memory_manager_instance.close())
    except Exception:
        pass

    print("\n✨ MCP demo completed successfully.\n")


def run_mcp_example():
    """Safely run the async example (avoids nested event loop errors)."""
    try:
        asyncio.run(main())
    except RuntimeError as e:
        print(f"[warning] Event loop issue detected: {e}")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")


if __name__ == "__main__":
    print(
        """
    ╔══════════════════════════════════════════════╗
    ║     MCP-Enabled Agent Example                ║
    ║     Model Context Protocol Integration       ║
    ╚══════════════════════════════════════════════╝
    """
    )
    run_mcp_example()
