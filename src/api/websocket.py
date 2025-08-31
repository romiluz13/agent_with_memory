"""
WebSocket Manager
Handles real-time bidirectional communication
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for a single agent."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept and store a new connection."""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.connection_metadata[user_id] = {
            "connected_at": datetime.utcnow(),
            "message_count": 0
        }
        logger.info(f"WebSocket connected: agent={self.agent_id}, user={user_id}")
    
    def disconnect(self, user_id: str):
        """Remove a connection."""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            del self.connection_metadata[user_id]
            logger.info(f"WebSocket disconnected: agent={self.agent_id}, user={user_id}")
    
    async def send_personal_message(self, message: str, user_id: str):
        """Send a message to a specific user."""
        if user_id in self.active_connections:
            websocket = self.active_connections[user_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(message)
                self.connection_metadata[user_id]["message_count"] += 1
    
    async def send_personal_json(self, data: Dict[str, Any], user_id: str):
        """Send JSON data to a specific user."""
        if user_id in self.active_connections:
            websocket = self.active_connections[user_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(data)
                self.connection_metadata[user_id]["message_count"] += 1
    
    async def broadcast(self, message: str):
        """Broadcast a message to all connected users."""
        disconnected = []
        for user_id, connection in self.active_connections.items():
            try:
                if connection.client_state == WebSocketState.CONNECTED:
                    await connection.send_text(message)
                else:
                    disconnected.append(user_id)
            except Exception as e:
                logger.error(f"Failed to send to {user_id}: {e}")
                disconnected.append(user_id)
        
        # Clean up disconnected clients
        for user_id in disconnected:
            self.disconnect(user_id)
    
    async def broadcast_json(self, data: Dict[str, Any]):
        """Broadcast JSON data to all connected users."""
        await self.broadcast(json.dumps(data))


class WebSocketManager:
    """Global WebSocket manager for all agents."""
    
    def __init__(self):
        self.agent_managers: Dict[str, ConnectionManager] = {}
    
    async def connect(self, websocket: WebSocket, agent_id: str, user_id: Optional[str] = None):
        """Connect a user to an agent."""
        if user_id is None:
            user_id = f"anonymous_{datetime.utcnow().timestamp()}"
        
        if agent_id not in self.agent_managers:
            self.agent_managers[agent_id] = ConnectionManager(agent_id)
        
        await self.agent_managers[agent_id].connect(websocket, user_id)
    
    def disconnect(self, agent_id: str, user_id: str):
        """Disconnect a user from an agent."""
        if agent_id in self.agent_managers:
            self.agent_managers[agent_id].disconnect(user_id)
    
    async def disconnect_all(self):
        """Disconnect all connections (for shutdown)."""
        for agent_id, manager in self.agent_managers.items():
            for user_id in list(manager.active_connections.keys()):
                try:
                    websocket = manager.active_connections[user_id]
                    await websocket.close()
                except Exception as e:
                    logger.error(f"Error closing websocket: {e}")
                manager.disconnect(user_id)
    
    async def process_message(
        self,
        agent_id: str,
        user_id: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a message from a WebSocket client.
        
        Args:
            agent_id: Agent identifier
            user_id: User identifier
            message: User message
            metadata: Optional metadata
            
        Returns:
            Response data
        """
        try:
            # Here you would integrate with the actual agent
            # For now, return a mock response
            response = {
                "type": "message",
                "content": f"Echo from {agent_id}: {message}",
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            return {
                "type": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def send_to_user(
        self,
        agent_id: str,
        user_id: str,
        data: Dict[str, Any]
    ):
        """Send data to a specific user."""
        if agent_id in self.agent_managers:
            await self.agent_managers[agent_id].send_personal_json(data, user_id)
    
    async def broadcast_to_agent_users(
        self,
        agent_id: str,
        data: Dict[str, Any]
    ):
        """Broadcast data to all users connected to an agent."""
        if agent_id in self.agent_managers:
            await self.agent_managers[agent_id].broadcast_json(data)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about active connections."""
        stats = {
            "total_agents": len(self.agent_managers),
            "total_connections": sum(
                len(m.active_connections) for m in self.agent_managers.values()
            ),
            "agents": {}
        }
        
        for agent_id, manager in self.agent_managers.items():
            stats["agents"][agent_id] = {
                "connections": len(manager.active_connections),
                "users": list(manager.active_connections.keys()),
                "metadata": manager.connection_metadata
            }
        
        return stats


class StreamingResponse:
    """Helper for streaming responses over WebSocket."""
    
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.chunks_sent = 0
    
    async def send_chunk(self, chunk: str, finished: bool = False):
        """Send a streaming chunk."""
        await self.websocket.send_json({
            "type": "stream",
            "chunk": chunk,
            "finished": finished,
            "chunk_number": self.chunks_sent
        })
        self.chunks_sent += 1
    
    async def send_error(self, error: str):
        """Send an error message."""
        await self.websocket.send_json({
            "type": "error",
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def send_metadata(self, metadata: Dict[str, Any]):
        """Send metadata about the stream."""
        await self.websocket.send_json({
            "type": "metadata",
            "metadata": metadata,
            "timestamp": datetime.utcnow().isoformat()
        })
