"""
MongoDB + LangGraph Agent Implementation
Based on official MongoDB docs-notebooks/ai-integrations/langgraph.ipynb
Integrates both short-term (checkpointer) and long-term (store) memory
"""

import os
import logging
from typing import Annotated, TypedDict, Dict, List, Optional
from typing_extensions import TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.store.mongodb import MongoDBStore, create_vector_index_config
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers.full_text_search import MongoDBAtlasFullTextSearchRetriever
from langchain_voyageai import VoyageAIEmbeddings
from pymongo import MongoClient

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
        agent_name: str = "assistant",
        model_provider: str = "openai",
        model_name: str = "gpt-4o",
        embedding_model: str = "voyage-3-large",
        embedding_dimensions: int = 1024,
        database_name: str = "ai_agent_boilerplate"
    ):
        """
        Initialize the agent with MongoDB connection.
        
        Args:
            mongodb_uri: MongoDB connection string
            agent_name: Name of the agent
            model_provider: LLM provider (openai, anthropic, google)
            model_name: Model name
            embedding_model: Voyage AI embedding model
            embedding_dimensions: Embedding vector dimensions
            database_name: MongoDB database name
        """
        self.mongodb_uri = mongodb_uri
        self.agent_name = agent_name
        self.database_name = database_name
        
        # Initialize MongoDB client
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[database_name]
        
        # Initialize embeddings (from MongoDB notebook)
        self.embedding_model = VoyageAIEmbeddings(
            model=embedding_model
            # Note: VoyageAI embeddings have fixed dimensions per model
        )
        
        # Standardize all Voyage models to 1024 dimensions for consistency
        # This ensures compatibility across all vector operations
        self.embedding_dimensions = 1024
        
        # Initialize LLM
        self.llm = self._create_llm(model_provider, model_name)
        
        # Initialize checkpointer for short-term memory
        self.checkpointer = MongoDBSaver(self.client)
        
        # Initialize store for long-term memory
        self.memory_store = self._create_memory_store()
        
        # Define tools
        self.tools = self._create_tools()
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info(f"Initialized MongoDB LangGraph Agent: {agent_name}")
    
    def _create_llm(self, provider: str, model_name: str):
        """Create LLM based on provider."""
        if provider == "openai":
            return ChatOpenAI(model=model_name)
        elif provider == "anthropic":
            return ChatAnthropic(model=model_name)
        elif provider == "google":
            return ChatGoogleGenerativeAI(model=model_name)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _create_memory_store(self) -> MongoDBStore:
        """Create MongoDB store for long-term memory (from notebook)."""
        # Vector search index configuration for memory collection
        index_config = create_vector_index_config(
            embed=self.embedding_model,
            dims=self.embedding_dimensions,
            relevance_score_fn="dotProduct",
            fields=["content"]
        )
        
        # Create store with auto-indexing
        store = MongoDBStore.from_conn_string(
            conn_string=self.mongodb_uri,
            db_name=self.database_name,
            collection_name="agent_memories",
            index_config=index_config,
            auto_index_timeout=60  # Wait for index creation
        )
        
        return store
    
    def _create_tools(self) -> List:
        """Create agent tools (from MongoDB notebook)."""
        tools = []
        
        # Tool to save important interactions to memory
        @tool
        def save_memory(content: str) -> str:
            """Save important information to memory."""
            with MongoDBStore.from_conn_string(
                conn_string=self.mongodb_uri,
                db_name=self.database_name,
                collection_name="agent_memories",
                index_config=create_vector_index_config(
                    embed=self.embedding_model,
                    dims=self.embedding_dimensions,
                    relevance_score_fn="dotProduct",
                    fields=["content"]
                )
            ) as store:
                store.put(
                    namespace=("agent", self.agent_name),
                    key=f"memory_{hash(content)}",
                    value={"content": content, "timestamp": datetime.utcnow().isoformat()}
                )
            return f"Memory saved: {content}"
        
        # Tool to retrieve memories using vector search
        @tool
        def retrieve_memories(query: str) -> str:
            """Retrieve relevant memories based on a query."""
            with MongoDBStore.from_conn_string(
                conn_string=self.mongodb_uri,
                db_name=self.database_name,
                collection_name="agent_memories",
                index_config=create_vector_index_config(
                    embed=self.embedding_model,
                    dims=self.embedding_dimensions,
                    relevance_score_fn="dotProduct",
                    fields=["content"]
                )
            ) as store:
                results = store.search(("agent", self.agent_name), query=query, limit=3)
                
                if results:
                    memories = [result.value["content"] for result in results]
                    return f"Retrieved memories:\n" + "\n".join(memories)
                else:
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
                embedding_key="vector_embeddings",
                relevance_score_fn="dotProduct"
            )
            
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # Retrieve top 5
            )
            
            results = retriever.invoke(user_query)
            
            # Concatenate results
            context = "\n\n".join([f"{doc.metadata.get('title', 'Doc')}: {doc.page_content}" for doc in results])
            return context
        
        tools.extend([save_memory, retrieve_memories, vector_search])
        return tools
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow (from MongoDB notebook)."""
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant with memory capabilities."
                    " You can save important information and retrieve past memories."
                    " Think step-by-step and use your tools when needed."
                    " You have access to the following tools: {tool_names}."
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        # Provide tool names to prompt
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in self.tools]))
        
        # Bind tools to LLM
        llm_with_tools = prompt | self.llm.bind_tools(self.tools)
        
        # Create tools map
        tools_by_name = {tool.name: tool for tool in self.tools}
        
        # Define agent node
        def agent(state: GraphState) -> Dict[str, List]:
            messages = state["messages"]
            result = llm_with_tools.invoke(messages)
            return {"messages": [result]}
        
        # Define tools node
        def tools_node(state: GraphState) -> Dict[str, List]:
            result = []
            tool_calls = state["messages"][-1].tool_calls
            
            for tool_call in tool_calls:
                tool = tools_by_name[tool_call["name"]]
                observation = tool.invoke(tool_call["args"])
                result.append(ToolMessage(
                    content=observation,
                    tool_call_id=tool_call["id"]
                ))
            
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
        graph.add_conditional_edges(
            "agent",
            route_tools,
            {"tools": "tools", END: END}
        )
        
        # Compile with checkpointer for short-term memory
        return graph.compile(checkpointer=self.checkpointer)
    
    def execute(self, user_input: str, thread_id: Optional[str] = None) -> str:
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
        input_state = {
            "messages": [
                HumanMessage(content=user_input)
            ]
        }
        
        # Execute graph
        result = None
        for output in self.graph.stream(input_state, config):
            for key, value in output.items():
                logger.debug(f"Node {key}: {value}")
                result = value
        
        # Extract final answer
        if result and "messages" in result:
            final_message = result["messages"][-1]
            if hasattr(final_message, "content"):
                return final_message.content
        
        return "I couldn't generate a response."
    
    async def aexecute(self, user_input: str, thread_id: Optional[str] = None) -> str:
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
        input_state = {
            "messages": [
                HumanMessage(content=user_input)
            ]
        }
        
        # Execute graph
        result = None
        async for output in self.graph.astream(input_state, config):
            for key, value in output.items():
                logger.debug(f"Node {key}: {value}")
                result = value
        
        # Extract final answer
        if result and "messages" in result:
            final_message = result["messages"][-1]
            if hasattr(final_message, "content"):
                return final_message.content
        
        return "I couldn't generate a response."
    
    def create_vector_indexes(self):
        """
        Create vector search indexes for collections.
        Based on MongoDB Developer examples with proper error handling.
        """
        import time
        
        collections_to_index = [
            "documents", 
            "agent_memories"
        ]
        
        for collection_name in collections_to_index:
            collection = self.db[collection_name]
            
            # MongoDB Atlas vector index definition (mongodb-developer pattern)
            index_definition = {
                "name": "vector_index",
                "type": "vectorSearch",
                "definition": {
                    "fields": [
                        {
                            "type": "vector",
                            "path": "vector_embeddings", 
                            "similarity": "cosine",
                            "numDimensions": self.embedding_dimensions
                        }
                    ]
                }
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
                    for i in range(15):  # Wait up to 15 seconds
                        time.sleep(1)
                        try:
                            indexes = list(collection.list_search_indexes())
                            vector_index = next((idx for idx in indexes if idx.get("name") == "vector_index"), None)
                            if vector_index and vector_index.get("status") == "READY":
                                logger.info(f"✅ Vector index ready for {collection_name}")
                                break
                        except:
                            pass  # Index might still be creating
                    else:
                        logger.info(f"⏳ Vector index for {collection_name} still creating (this is normal)")
                else:
                    logger.info(f"✅ Vector index already exists for {collection_name}")
                    
            except Exception as e:
                # More specific error handling
                if "already exists" in str(e) or "IndexAlreadyExists" in str(e):
                    logger.info(f"✅ Vector index already exists for {collection_name}")
                elif "NamespaceNotFound" in str(e):
                    logger.info(f"📝 Collection {collection_name} will be created when first document is added")
                else:
                    logger.warning(f"⚠️ Index creation issue for {collection_name}: {e}")
                    logger.info("This is usually fine - indexes can be created later")


# Example usage matching MongoDB notebook patterns
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize agent
    agent = MongoDBLangGraphAgent(
        mongodb_uri=os.getenv("MONGODB_URI"),
        agent_name="assistant",
        model_provider="openai",
        model_name="gpt-4o"
    )
    
    # Create indexes
    agent.create_vector_indexes()
    
    # Test execution
    response = agent.execute(
        "What are some movies that take place in the ocean?",
        thread_id="test_session"
    )
    print(f"Response: {response}")
    
    # Test memory
    response = agent.execute(
        "Remember that I prefer funny movies.",
        thread_id="test_session"
    )
    print(f"Response: {response}")
    
    # Test memory retrieval
    response = agent.execute(
        "What do you know about me?",
        thread_id="test_session"
    )
    print(f"Response: {response}")
