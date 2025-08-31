"""
Base Agent Class with LangGraph Orchestration
Core agent implementation with memory integration
"""

import logging
from typing import TypedDict, List, Dict, Any, Optional, Literal
from datetime import datetime
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.base import BaseCheckpointer  # Import issue - will be fixed
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from ..memory.manager import MemoryManager, MemoryConfig
from ..memory.base import MemoryType

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the agent graph."""
    messages: List[BaseMessage]
    memories: List[Dict[str, Any]]
    context: Dict[str, Any]
    next_action: Optional[str]
    tool_calls: List[Dict[str, Any]]
    final_answer: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    description: str = ""
    model_provider: Literal["openai", "anthropic", "google"] = "openai"
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 2000
    system_prompt: Optional[str] = None
    tools: List[BaseTool] = None
    memory_config: Optional[MemoryConfig] = None
    enable_streaming: bool = True
    timeout_seconds: int = 30
    max_iterations: int = 10


class BaseAgent:
    """
    Base agent class with LangGraph orchestration and memory.
    Provides foundation for all specialized agents.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        memory_manager: MemoryManager,
        checkpointer: Optional[Any] = None
    ):
        """
        Initialize the agent.
        
        Args:
            config: Agent configuration
            memory_manager: Memory management system
            checkpointer: Optional state checkpointer
        """
        self.config = config
        self.memory_manager = memory_manager
        self.checkpointer = checkpointer
        
        # Initialize LLM
        self.llm = self._create_llm()
        
        # Initialize tools
        self.tools = config.tools or []
        self.tool_executor = ToolExecutor(self.tools) if self.tools else None
        
        # Build the graph
        self.graph = self._build_graph()
        
        # Agent metadata
        self.agent_id = f"{config.name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.created_at = datetime.utcnow()
        self.conversation_count = 0
        
        logger.info(f"Initialized agent: {self.agent_id}")
    
    def _create_llm(self) -> BaseChatModel:
        """Create the language model based on configuration."""
        if self.config.model_provider == "openai":
            return ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                streaming=self.config.enable_streaming
            )
        elif self.config.model_provider == "anthropic":
            return ChatAnthropic(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
        elif self.config.model_provider == "google":
            return ChatGoogleGenerativeAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens_to_sample=self.config.max_tokens
            )
        else:
            raise ValueError(f"Unknown model provider: {self.config.model_provider}")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retrieve_memory", self.retrieve_memory_node)
        workflow.add_node("process_input", self.process_input_node)
        workflow.add_node("execute_tools", self.execute_tools_node)
        workflow.add_node("generate_response", self.generate_response_node)
        workflow.add_node("store_memory", self.store_memory_node)
        
        # Set entry point
        workflow.set_entry_point("retrieve_memory")
        
        # Add edges
        workflow.add_edge("retrieve_memory", "process_input")
        
        # Conditional routing after processing
        workflow.add_conditional_edges(
            "process_input",
            self.route_after_processing,
            {
                "tools": "execute_tools",
                "respond": "generate_response",
                "end": END
            }
        )
        
        # After tool execution, generate response
        workflow.add_edge("execute_tools", "generate_response")
        
        # After generating response, store memory
        workflow.add_edge("generate_response", "store_memory")
        
        # End after storing memory
        workflow.add_edge("store_memory", END)
        
        # Compile the graph
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def retrieve_memory_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant memories for the conversation."""
        try:
            # Get the last user message
            last_message = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    last_message = msg.content
                    break
            
            if last_message:
                # Retrieve relevant memories
                memories = await self.memory_manager.retrieve_memories(
                    query=last_message,
                    memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL],
                    limit=5,
                    use_cache=True
                )
                
                # Convert to dict format
                state["memories"] = [
                    {
                        "type": mem.memory_type.value,
                        "content": mem.content,
                        "importance": mem.importance,
                        "metadata": mem.metadata
                    }
                    for mem in memories
                ]
                
                logger.debug(f"Retrieved {len(memories)} relevant memories")
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            state["memories"] = []
        
        return state
    
    async def process_input_node(self, state: AgentState) -> AgentState:
        """Process input and decide next action."""
        try:
            # Build context from memories
            memory_context = self._build_memory_context(state["memories"])
            
            # Create system message with context
            system_msg = self._create_system_message(memory_context)
            
            # Prepare messages for LLM
            messages = [system_msg] + state["messages"]
            
            # Get LLM response with tool consideration
            if self.tools:
                response = await self.llm.apredict_messages(
                    messages,
                    functions=[tool.as_dict() for tool in self.tools]
                )
                
                # Check if tools should be used
                if hasattr(response, 'additional_kwargs') and 'function_call' in response.additional_kwargs:
                    state["next_action"] = "tools"
                    state["tool_calls"] = [response.additional_kwargs['function_call']]
                else:
                    state["next_action"] = "respond"
                    state["context"]["llm_response"] = response.content
            else:
                response = await self.llm.apredict_messages(messages)
                state["next_action"] = "respond"
                state["context"]["llm_response"] = response.content
            
        except Exception as e:
            logger.error(f"Failed to process input: {e}")
            state["next_action"] = "respond"
            state["context"]["error"] = str(e)
        
        return state
    
    async def execute_tools_node(self, state: AgentState) -> AgentState:
        """Execute requested tools."""
        try:
            results = []
            for tool_call in state["tool_calls"]:
                # Create tool invocation
                tool_invocation = ToolInvocation(
                    tool=tool_call["name"],
                    tool_input=tool_call["arguments"]
                )
                
                # Execute tool
                result = await self.tool_executor.ainvoke(tool_invocation)
                results.append(result)
            
            state["context"]["tool_results"] = results
            logger.debug(f"Executed {len(results)} tools")
            
        except Exception as e:
            logger.error(f"Failed to execute tools: {e}")
            state["context"]["tool_error"] = str(e)
        
        return state
    
    async def generate_response_node(self, state: AgentState) -> AgentState:
        """Generate final response."""
        try:
            # Check if we have tool results to incorporate
            if "tool_results" in state["context"]:
                # Generate response incorporating tool results
                tool_context = f"Tool results: {state['context']['tool_results']}"
                messages = state["messages"] + [SystemMessage(content=tool_context)]
                response = await self.llm.apredict_messages(messages)
                state["final_answer"] = response.content
            elif "llm_response" in state["context"]:
                # Use the already generated response
                state["final_answer"] = state["context"]["llm_response"]
            else:
                # Fallback response
                state["final_answer"] = "I apologize, but I encountered an issue processing your request."
            
            # Add AI message to conversation
            state["messages"].append(AIMessage(content=state["final_answer"]))
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            state["final_answer"] = f"Error generating response: {str(e)}"
        
        return state
    
    async def store_memory_node(self, state: AgentState) -> AgentState:
        """Store conversation in memory."""
        try:
            # Store episodic memory of the conversation
            last_human_msg = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    last_human_msg = msg.content
                    break
            
            if last_human_msg and state["final_answer"]:
                # Create conversation memory
                conversation = f"User: {last_human_msg}\nAssistant: {state['final_answer']}"
                
                await self.memory_manager.store_memory(
                    content=conversation,
                    memory_type=MemoryType.EPISODIC,
                    agent_id=self.agent_id,
                    user_id=state.get("metadata", {}).get("user_id"),
                    metadata={
                        "timestamp": datetime.utcnow().isoformat(),
                        "conversation_id": state.get("metadata", {}).get("conversation_id"),
                        "turn_count": len(state["messages"]) // 2
                    },
                    importance=0.7
                )
                
                # Store in working memory for session context
                await self.memory_manager.store_memory(
                    content=last_human_msg,
                    memory_type=MemoryType.WORKING,
                    agent_id=self.agent_id,
                    user_id=state.get("metadata", {}).get("user_id"),
                    metadata={
                        "session_id": state.get("metadata", {}).get("session_id", "default"),
                        "response": state["final_answer"]
                    },
                    importance=0.5
                )
                
                logger.debug("Stored conversation in memory")
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
        
        return state
    
    def route_after_processing(self, state: AgentState) -> str:
        """Determine next step after processing."""
        return state.get("next_action", "respond")
    
    def _build_memory_context(self, memories: List[Dict[str, Any]]) -> str:
        """Build context string from memories."""
        if not memories:
            return ""
        
        context_parts = ["Relevant context from memory:"]
        for mem in memories:
            context_parts.append(f"- [{mem['type']}] {mem['content']}")
        
        return "\n".join(context_parts)
    
    def _create_system_message(self, memory_context: str) -> SystemMessage:
        """Create system message with memory context."""
        base_prompt = self.config.system_prompt or f"You are {self.config.name}, a helpful AI assistant."
        
        if memory_context:
            full_prompt = f"{base_prompt}\n\n{memory_context}"
        else:
            full_prompt = base_prompt
        
        return SystemMessage(content=full_prompt)
    
    async def invoke(
        self,
        message: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Process a message and return response.
        
        Args:
            message: User message
            user_id: Optional user ID
            session_id: Optional session ID
            conversation_id: Optional conversation ID
            
        Returns:
            Agent response
        """
        # Create initial state
        state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "memories": [],
            "context": {},
            "next_action": None,
            "tool_calls": [],
            "final_answer": None,
            "metadata": {
                "user_id": user_id,
                "session_id": session_id or f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "conversation_id": conversation_id or f"conv_{self.conversation_count}",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        # Run the graph
        result = await self.graph.ainvoke(state)
        
        self.conversation_count += 1
        
        return result["final_answer"]
    
    async def stream(
        self,
        message: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Stream response for a message.
        
        Args:
            message: User message
            user_id: Optional user ID
            session_id: Optional session ID
            
        Yields:
            Response chunks
        """
        # Similar to invoke but with streaming
        # Implementation depends on specific streaming requirements
        pass
