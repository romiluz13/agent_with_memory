"""
MongoDB + LangGraph Agent Implementation.

Based on official MongoDB LangGraph integration patterns for short-term
checkpointing plus long-term MongoDB-backed memory.
"""

import logging
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.mongodb import MongoDBStore, create_vector_index_config
from pymongo import MongoClient

from ..embeddings.langchain_voyage import VoyageEmbeddingsAdapter
from ..observability.tracer import get_tracer

logger = logging.getLogger(__name__)


# Define the graph state (from MongoDB notebook)
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]


class MongoDBLangGraphAgent:
    """
    Agent implementation following MongoDB's official LangGraph integration patterns.
    Combines short-term memory (checkpointer) and long-term memory (store).
    """

    def __init__(
        self,
        mongodb_uri: str,
        mongo_client: MongoClient | None = None,
        agent_id: str | None = None,
        agent_name: str = "assistant",
        model_provider: str = "openai",
        model_name: str = "gpt-4o",
        embedding_model: str = "voyage-3-large",
        database_name: str = "ai_agent_boilerplate",
        system_prompt: str | None = None,
        user_tools: list | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        enable_streaming: bool = True,
    ):
        """
        Initialize the agent with MongoDB connection.

        Args:
            mongodb_uri: MongoDB connection string
            agent_name: Name of the agent
            model_provider: LLM provider (openai, anthropic, google)
            model_name: Model name
            embedding_model: Voyage AI embedding model
            database_name: MongoDB database name
            system_prompt: Custom system prompt for the agent's persona
            user_tools: List of custom tools for the agent to use
        """
        self.mongodb_uri = mongodb_uri
        self.agent_id = agent_id or f"{agent_name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        self.agent_name = agent_name
        self.model_provider = model_provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_streaming = enable_streaming
        self.database_name = database_name
        self.created_at = datetime.now(UTC)
        self.conversation_count = 0
        self.system_prompt = system_prompt or (
            "You are a helpful AI assistant with memory capabilities."
            " You can save important information and retrieve past memories."
            " Think step-by-step and use your tools when needed."
            " You have access to the following tools: {tool_names}."
        )

        # Initialize MongoDB client
        self.client = mongo_client or MongoClient(mongodb_uri)
        self._owns_client = mongo_client is None
        self.db = self.client[database_name]

        # Initialize embeddings (from MongoDB notebook)
        self.embedding_dimensions = 1024  # Standard dimension for all embeddings
        self.embedding_model = VoyageEmbeddingsAdapter(
            model=embedding_model,
            output_dimension=self.embedding_dimensions,
        )

        # Initialize LLM
        self.llm = self._create_llm(model_provider, model_name)

        # Initialize checkpointer for short-term memory
        self.checkpointer = MongoDBSaver(self.client)

        # Initialize store for long-term memory
        self._memory_store_cm = None
        self.memory_store = self._create_memory_store()

        # Define tools, including user-provided ones
        self.tools = self._create_tools(user_tools or [])

        # Build the graph
        self.graph = self._build_graph()

        logger.info("Initialized MongoDB LangGraph Agent: %s (%s)", agent_name, self.agent_id)

    def _create_llm(self, provider: str, model_name: str):
        """Create LLM based on provider."""
        if provider == "openai":
            return ChatOpenAI(
                model=model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                streaming=self.enable_streaming,
            )
        elif provider == "anthropic":
            return ChatAnthropic(
                model=model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif provider == "google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _create_memory_store(self) -> MongoDBStore:
        """Create MongoDB store for long-term memory (from notebook)."""
        try:
            index_config = create_vector_index_config(
                embed=self.embedding_model,
                dims=self.embedding_dimensions,
                relevance_score_fn="cosine",
                fields=["content"],
            )
            self._memory_store_cm = MongoDBStore.from_conn_string(
                conn_string=self.mongodb_uri,
                db_name=self.database_name,
                collection_name="agent_memories",
                index_config=index_config,
                auto_index_timeout=60,
            )
            return self._memory_store_cm.__enter__()
        except Exception as exc:
            logger.warning(
                "Falling back to non-indexed MongoDBStore for %s due to index setup issue: %s",
                self.agent_id,
                exc,
            )
            self._memory_store_cm = MongoDBStore.from_conn_string(
                conn_string=self.mongodb_uri,
                db_name=self.database_name,
                collection_name="agent_memories",
            )
            return self._memory_store_cm.__enter__()

    def _create_tools(self, user_tools: list) -> list:
        """Create agent tools, combining built-in and user-provided tools."""

        # Your boilerplate's built-in memory tools
        built_in_tools = []

        # Tool to save important interactions to memory
        @tool
        def save_memory(content: str) -> str:
            """Save important information to memory."""
            self.memory_store.put(
                namespace=("agent", self.agent_id),
                key=f"memory_{hash((self.agent_id, content))}",
                value={
                    "content": content,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "agent_id": self.agent_id,
                },
            )
            return f"Memory saved: {content}"

        # Tool to retrieve memories using vector search
        @tool
        def retrieve_memories(query: str) -> str:
            """Retrieve relevant memories based on a query."""
            try:
                results = self.memory_store.search(("agent", self.agent_id), query=query, limit=3)
            except Exception as exc:
                logger.warning("Memory store search unavailable for %s: %s", self.agent_id, exc)
                return "Long-term memory search is temporarily unavailable."

            if results:
                memories = [result.value["content"] for result in results]
                return "Retrieved memories:\n" + "\n".join(memories)
            return "No relevant memories found."

        # Vector search tool for documents
        @tool
        def vector_search(user_query: str) -> str:
            """
            Retrieve information using vector search to answer a user query.
            Based on MongoDB's retrieve-documents.js pattern.
            """
            # Initialize vector store
            vector_store = MongoDBAtlasVectorSearch.from_connection_string(
                connection_string=self.mongodb_uri,
                namespace=f"{self.database_name}.documents",
                embedding=self.embedding_model,
                text_key="text",
                embedding_key="embedding",
                relevance_score_fn="cosine",
            )

            retriever = vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 5}  # Retrieve top 5
            )

            results = retriever.invoke(user_query)

            # Concatenate results
            context = "\n\n".join(
                [f"{doc.metadata.get('title', 'Doc')}: {doc.page_content}" for doc in results]
            )
            return context

        built_in_tools.extend([save_memory, retrieve_memories, vector_search])

        # Add user-provided tools
        all_tools = built_in_tools + user_tools
        return all_tools

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow (from MongoDB notebook)."""
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        # Provide tool names to prompt
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in self.tools]))

        # Bind tools to LLM
        llm_with_tools = prompt | self.llm.bind_tools(self.tools)

        # Create tools map
        tools_by_name = {tool.name: tool for tool in self.tools}

        # Define agent node (with optional observability tracing)
        def agent(state: GraphState) -> dict[str, list]:
            messages = state["messages"]
            # Pass LangChain callback handler when tracing is enabled
            handler = get_tracer().get_langchain_handler()
            invoke_config = {"callbacks": [handler]} if handler else {}
            result = llm_with_tools.invoke(messages, config=invoke_config)
            return {"messages": [result]}

        # Define tools node
        def tools_node(state: GraphState) -> dict[str, list]:
            result = []
            tool_calls = state["messages"][-1].tool_calls

            for tool_call in tool_calls:
                tool = tools_by_name[tool_call["name"]]
                observation = tool.invoke(tool_call["args"])
                result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

            return {"messages": result}

        # Define routing function
        def route_tools(state: GraphState):
            messages = state.get("messages", [])
            if len(messages) > 0:
                ai_message = messages[-1]
            else:
                raise ValueError(f"No messages found in state: {state}")

            if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
                return "tools"
            return END

        # Build the graph
        graph = StateGraph(GraphState)

        # Add nodes
        graph.add_node("agent", agent)
        graph.add_node("tools", tools_node)

        # Add edges
        graph.add_edge(START, "agent")
        graph.add_edge("tools", "agent")

        # Add conditional edge
        graph.add_conditional_edges("agent", route_tools, {"tools": "tools", END: END})

        # Compile with checkpointer for short-term memory
        return graph.compile(checkpointer=self.checkpointer)

    def execute(self, user_input: str, thread_id: str | None = None) -> str:
        """
        Execute the graph with user input.

        Args:
            user_input: User's message
            thread_id: Thread ID for conversation persistence

        Returns:
            Agent's response
        """
        # Configure thread for persistence
        config = {"configurable": {"thread_id": thread_id or "default"}}

        # Prepare input
        input_state = {"messages": [HumanMessage(content=user_input)]}

        # Execute graph
        result = None
        for output in self.graph.stream(input_state, config):
            for key, value in output.items():
                logger.debug(f"Node {key}: {value}")
                result = value

        # Extract final answer
        if result and "messages" in result:
            return self._extract_final_message(result)

        return "I couldn't generate a response."

    async def aexecute(self, user_input: str, thread_id: str | None = None) -> str:
        """
        Async execute the graph with user input.

        Args:
            user_input: User's message
            thread_id: Thread ID for conversation persistence

        Returns:
            Agent's response
        """
        # Configure thread for persistence
        config = {"configurable": {"thread_id": thread_id or "default"}}

        # Prepare input
        input_state = {"messages": [HumanMessage(content=user_input)]}

        # Execute graph
        result = None
        async for output in self.graph.astream(input_state, config):
            for key, value in output.items():
                logger.debug(f"Node {key}: {value}")
                result = value

        # Extract final answer
        if result and "messages" in result:
            return self._extract_final_message(result)

        return "I couldn't generate a response."

    async def invoke(
        self,
        message: str,
        user_id: str | None = None,
        session_id: str | None = None,
        conversation_id: str | None = None,
    ) -> str:
        """Invoke the canonical LangGraph runtime for a chat turn."""
        thread_id = self.build_thread_id(
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
        )
        response = await self.aexecute(message, thread_id=thread_id)
        self.conversation_count += 1
        return response

    async def stream_events(
        self,
        message: str,
        user_id: str | None = None,
        session_id: str | None = None,
        conversation_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream actual graph execution events for a chat turn."""
        thread_id = self.build_thread_id(
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
        )
        config = {"configurable": {"thread_id": thread_id}}
        input_state = {"messages": [HumanMessage(content=message)]}

        yield {
            "type": "run_started",
            "agent_id": self.agent_id,
            "thread_id": thread_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        async for update in self.graph.astream(input_state, config, stream_mode="updates"):
            for node_name, payload in update.items():
                event: dict[str, Any] = {
                    "type": "node_completed",
                    "agent_id": self.agent_id,
                    "thread_id": thread_id,
                    "node": node_name,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                message_text = self._extract_final_message(payload)
                if message_text:
                    event["content"] = message_text
                yield event

        final_state = await self.graph.aget_state(config)
        final_response = self._extract_final_message(final_state.values)
        self.conversation_count += 1
        yield {
            "type": "completed",
            "agent_id": self.agent_id,
            "thread_id": thread_id,
            "conversation_id": self.extract_conversation_id(thread_id),
            "content": final_response,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def build_thread_id(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        conversation_id: str | None = None,
    ) -> str:
        """Create a stable persisted thread ID scoped to the agent."""
        base = conversation_id or session_id or user_id or "default"
        return f"{self.agent_id}:{base}"

    @staticmethod
    def extract_conversation_id(thread_id: str) -> str:
        """Extract the human conversation identifier from a persisted thread ID."""
        if ":" not in thread_id:
            return thread_id
        return thread_id.split(":", 1)[1]

    async def close(self) -> None:
        """Close agent-owned resources."""
        if self._memory_store_cm is not None:
            self._memory_store_cm.__exit__(None, None, None)
            self._memory_store_cm = None
        if self._owns_client:
            self.client.close()

    def create_vector_indexes(self):
        """
        Create vector search indexes for collections.
        Based on MongoDB Developer examples with proper error handling.
        """
        import time

        collections_to_index = ["documents", "agent_memories"]

        for collection_name in collections_to_index:
            collection = self.db[collection_name]

            # MongoDB Atlas vector index definition (mongodb-developer pattern)
            # NOTE: This creates a minimal index for LangGraph's own collections
            # (documents, agent_memories). For the 7 memory store collections,
            # use scripts/setup_indexes.py which is the canonical index setup
            # with full filter fields (agent_id, user_id, etc.).
            index_definition = {
                "name": "vector_index",
                "type": "vectorSearch",
                "definition": {
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "similarity": "cosine",
                            "numDimensions": self.embedding_dimensions,
                        }
                    ]
                },
            }

            try:
                # Check if index already exists (mongodb-developer pattern)
                existing_indexes = list(collection.list_search_indexes())
                index_exists = any(idx.get("name") == "vector_index" for idx in existing_indexes)

                if not index_exists:
                    logger.info(f"Creating vector index for {collection_name}...")
                    result = collection.create_search_index(index_definition)
                    logger.info(f"Index creation initiated: {result}")

                    # Wait for index to be ready (optional - for production)
                    # Note: In production, this is usually done async
                    for _i in range(15):  # Wait up to 15 seconds
                        time.sleep(1)
                        try:
                            indexes = list(collection.list_search_indexes())
                            vector_index = next(
                                (idx for idx in indexes if idx.get("name") == "vector_index"), None
                            )
                            if vector_index and vector_index.get("status") == "READY":
                                logger.info(f"✅ Vector index ready for {collection_name}")
                                break
                        except Exception:
                            pass  # Index might still be creating
                    else:
                        logger.info(
                            f"⏳ Vector index for {collection_name} still creating (this is normal)"
                        )
                else:
                    logger.info(f"✅ Vector index already exists for {collection_name}")

            except Exception as e:
                # More specific error handling
                if "already exists" in str(e) or "IndexAlreadyExists" in str(e):
                    logger.info(f"✅ Vector index already exists for {collection_name}")
                elif "NamespaceNotFound" in str(e):
                    logger.info(
                        f"📝 Collection {collection_name} will be created when first document is added"
                    )
                else:
                    logger.warning(f"⚠️ Index creation issue for {collection_name}: {e}")
                    logger.info("This is usually fine - indexes can be created later")

    @staticmethod
    def _extract_final_message(state: dict[str, Any] | Any) -> str:
        """Extract the last meaningful AI/tool message from graph state."""
        if not isinstance(state, dict):
            return ""

        messages = state.get("messages", [])
        if not messages:
            return ""

        last_message = messages[-1]
        if isinstance(last_message, (AIMessage, ToolMessage, HumanMessage)):
            return str(getattr(last_message, "content", "") or "")
        if hasattr(last_message, "content"):
            return str(last_message.content or "")
        if isinstance(last_message, dict):
            return str(last_message.get("content", "") or "")
        return str(last_message)


# Example usage matching MongoDB notebook patterns
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # Initialize agent
    agent = MongoDBLangGraphAgent(
        mongodb_uri=os.getenv("MONGODB_URI"),
        agent_name="assistant",
        model_provider="openai",
        model_name="gpt-4o",
    )

    # Create indexes
    agent.create_vector_indexes()

    # Test execution
    response = agent.execute(
        "What are some movies that take place in the ocean?", thread_id="test_session"
    )
    print(f"Response: {response}")

    # Test memory
    response = agent.execute("Remember that I prefer funny movies.", thread_id="test_session")
    print(f"Response: {response}")

    # Test memory retrieval
    response = agent.execute("What do you know about me?", thread_id="test_session")
    print(f"Response: {response}")
