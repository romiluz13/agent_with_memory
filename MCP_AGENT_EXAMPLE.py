"""
MCP-Enabled Agent Example
Demonstrates how to use Model Context Protocol tools with the AI Agent Boilerplate.
"""

import asyncio
import os
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our agent components
from src.core.agent import BaseAgent, AgentConfig
from src.memory.manager import MemoryManager, MemoryConfig
from src.storage.mongodb_client import MongoDBClient
from langchain_core.tools import tool

# Custom business tool example
@tool
def calculate_discount(original_price: float, discount_percentage: float) -> float:
    """Calculate the discounted price."""
    discount = original_price * (discount_percentage / 100)
    return original_price - discount


async def main():
    """
    Example of creating an agent with MCP tools.
    """
    
    print("🚀 Initializing MCP-Enabled Agent...")
    
    # Initialize MongoDB
    mongo_client = MongoDBClient()
    # await mongo_client.connect()
    
    # Configure memory
    memory_config = MemoryConfig(
        enable_episodic=True,
        enable_semantic=True,
        enable_procedural=True,
        enable_working=True,
        enable_cache=True
    )
    
    # Initialize memory manager
    memory_manager = MemoryManager(
        config=memory_config,
        mongo_client=mongo_client,
        user_id="mcp_demo_user"
    )
    
    # Configure agent with MCP
    agent_config = AgentConfig(
        name="mcp_assistant",
        description="An assistant with MCP tools for file system and GitHub access",
        model_provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.7,
        
        # Custom tools
        tools=[calculate_discount],
        
        # MCP Configuration
        enable_mcp=True,
        mcp_servers=[
            "npx @modelcontextprotocol/server-filesystem",
            # Add more servers as needed:
            # "npx @modelcontextprotocol/server-github",
            # "npx @modelcontextprotocol/server-brave-search",
        ],
        
        # Memory configuration
        memory_config=memory_config,
        
        # System prompt
        system_prompt="""You are a helpful AI assistant with access to:
        1. File system operations through MCP
        2. Memory of past conversations
        3. Business tools for calculations
        
        Always be helpful, accurate, and remember context from previous interactions."""
    )
    
    # Create the agent
    agent = BaseAgent(
        config=agent_config,
        memory_manager=memory_manager
    )
    
    # Optional: Load MCP tools asynchronously if needed
    await agent._initialize_mcp_tools_async()
    
    print(f"✅ Agent initialized with {len(agent.tools)} tools")
    
    # List available tools
    print("\n📦 Available Tools:")
    for tool in agent.tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Example conversations
    print("\n💬 Starting conversation...\n")
    
    # Example 1: Using file system MCP tool
    response1 = await agent.ainvoke(
        message="Can you list the files in the current directory?",
        user_id="mcp_demo_user",
        session_id="demo_session_1"
    )
    print(f"Agent: {response1}\n")
    
    # Example 2: Using custom tool
    response2 = await agent.ainvoke(
        message="Calculate a 25% discount on a $150 product",
        user_id="mcp_demo_user",
        session_id="demo_session_1"
    )
    print(f"Agent: {response2}\n")
    
    # Example 3: Using memory
    response3 = await agent.ainvoke(
        message="What files did I ask about earlier?",
        user_id="mcp_demo_user",
        session_id="demo_session_1"
    )
    print(f"Agent: {response3}\n")
    
    # Check MCP toolkit status
    if agent.mcp_toolkit:
        status = agent.mcp_toolkit.get_status()
        print("\n📊 MCP Toolkit Status:")
        print(f"  - Total MCP tools loaded: {status['total_tools']}")
        print(f"  - Loaded servers: {status['loaded_servers']}")
        print(f"  - Tool names: {status['tool_names'][:5]}...")  # Show first 5
    
    # Cleanup
    await mongo_client.close()
    if agent.mcp_toolkit:
        await agent.mcp_toolkit.cleanup()
    
    print("\n✨ Demo completed successfully!")


def run_mcp_example():
    """
    Convenience function to run the example.
    """
    # Check required environment variables
    required_vars = ["OPENAI_API_KEY", "MONGODB_URI"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("Please set them in your .env file")
        return
    
    # Run the async main function
    asyncio.run(main())


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════╗
    ║     MCP-Enabled Agent Example                ║
    ║     Model Context Protocol Integration       ║
    ╚══════════════════════════════════════════════╝
    """)
    
    run_mcp_example()
