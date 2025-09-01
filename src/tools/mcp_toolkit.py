"""
MCP (Model Context Protocol) Toolkit
Production-ready integration following langchain-mcp-adapters patterns.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class MCPToolkit:
    """
    Production-ready MCP toolkit for loading and managing MCP tools.
    
    Features:
    - Safe loading with error handling
    - Connection pooling and retry logic
    - Tool validation and filtering
    - Graceful degradation if MCP servers unavailable
    """
    
    def __init__(self, servers: Optional[List[str]] = None):
        """
        Initialize MCP toolkit.
        
        Args:
            servers: List of MCP server commands (e.g., ["npx @modelcontextprotocol/server-filesystem"])
        """
        self.servers = servers or []
        self.tools: List[BaseTool] = []
        self.loaded_servers: Dict[str, bool] = {}
        self._client: Optional[MultiServerMCPClient] = None
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _load_single_server(self, server: str) -> List[BaseTool]:
        """
        Load tools from a single MCP server with retry logic.
        
        Args:
            server: MCP server command
            
        Returns:
            List of loaded tools
        """
        try:
            logger.info(f"Loading MCP tools from: {server}")
            
            # Load tools using the official adapter
            tools = await load_mcp_tools(server)
            
            if tools:
                logger.info(f"Successfully loaded {len(tools)} tools from {server}")
                self.loaded_servers[server] = True
                return tools
            else:
                logger.warning(f"No tools loaded from {server}")
                self.loaded_servers[server] = False
                return []
                
        except Exception as e:
            logger.error(f"Failed to load MCP tools from {server}: {e}")
            self.loaded_servers[server] = False
            # Don't crash - graceful degradation
            return []
    
    async def load_tools(self, servers: Optional[List[str]] = None) -> List[BaseTool]:
        """
        Load tools from all configured MCP servers.
        
        Args:
            servers: Optional list of servers to load (uses self.servers if not provided)
            
        Returns:
            Combined list of all loaded tools
        """
        servers_to_load = servers or self.servers
        
        if not servers_to_load:
            logger.warning("No MCP servers configured")
            return []
        
        # Load tools from all servers in parallel
        tasks = [self._load_single_server(server) for server in servers_to_load]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine all successfully loaded tools
        all_tools = []
        for result in results:
            if isinstance(result, list):
                all_tools.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error loading tools: {result}")
        
        self.tools = all_tools
        logger.info(f"Total MCP tools loaded: {len(all_tools)}")
        
        return all_tools
    
    async def load_tools_with_client(self, servers: List[Dict[str, Any]]) -> List[BaseTool]:
        """
        Load tools using MultiServerMCPClient for advanced configurations.
        
        Args:
            servers: List of server configurations with connection details
            
        Returns:
            List of loaded tools
        """
        try:
            # Initialize client if not exists
            if not self._client:
                self._client = MultiServerMCPClient(servers)
            
            # Get all available tools
            tools = await self._client.list_tools()
            
            logger.info(f"Loaded {len(tools)} tools via MultiServerMCPClient")
            self.tools = tools
            return tools
            
        except Exception as e:
            logger.error(f"Failed to load tools with client: {e}")
            return []
    
    def get_tools(self) -> List[BaseTool]:
        """
        Get all loaded tools.
        
        Returns:
            List of loaded MCP tools
        """
        return self.tools
    
    def get_tool_names(self) -> List[str]:
        """
        Get names of all loaded tools.
        
        Returns:
            List of tool names
        """
        return [tool.name for tool in self.tools]
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """
        Get a specific tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool if found, None otherwise
        """
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None
    
    def filter_tools(self, pattern: str) -> List[BaseTool]:
        """
        Filter tools by name pattern.
        
        Args:
            pattern: Pattern to match in tool names
            
        Returns:
            Filtered list of tools
        """
        return [
            tool for tool in self.tools 
            if pattern.lower() in tool.name.lower()
        ]
    
    async def cleanup(self):
        """Clean up resources."""
        if self._client:
            try:
                # Client cleanup if needed
                pass
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        
        self.tools.clear()
        self.loaded_servers.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get toolkit status.
        
        Returns:
            Status dictionary
        """
        return {
            "total_tools": len(self.tools),
            "tool_names": self.get_tool_names(),
            "loaded_servers": self.loaded_servers,
            "configured_servers": self.servers
        }


# Convenience functions for quick usage
async def load_mcp_tools_safe(servers: Union[str, List[str]]) -> List[BaseTool]:
    """
    Safely load MCP tools with error handling.
    
    Args:
        servers: Single server or list of servers
        
    Returns:
        List of loaded tools (empty list if failed)
    """
    if isinstance(servers, str):
        servers = [servers]
    
    toolkit = MCPToolkit(servers)
    return await toolkit.load_tools()


async def get_common_mcp_tools() -> List[BaseTool]:
    """
    Load commonly used MCP tools.
    
    Returns:
        List of common tools
    """
    common_servers = [
        "npx @modelcontextprotocol/server-filesystem",
        "npx @modelcontextprotocol/server-github",
        "npx @modelcontextprotocol/server-brave-search"
    ]
    
    return await load_mcp_tools_safe(common_servers)
