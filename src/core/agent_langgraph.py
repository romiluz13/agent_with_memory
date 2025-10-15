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
# <<< FIX: Import OperationFailure to catch the specific MongoDB error that occurs on M0 clusters >>>
from pymongo.errors import OperationFailure

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
        embedding_model: str = "voyage-2", # Using a valid model name from .env.example
        database_name: str = "ai_agent_boilerplate",
        system_prompt: Optional[str] = None,
        user_tools: Optional[List] = None
    ):
        """
        Initialize the agent with MongoDB connection.
        """
        self.mongodb_uri = mongodb_uri
        self.agent_name = agent_name
        self.database_name = database_name
        self.system_prompt = system_prompt or (
            "You are a helpful AI assistant with memory capabilities."
            " You can save important information and retrieve past memories."
            " Think step-by-step and use your tools when needed."
            " You have access to the following tools: {tool_names}."
        )
        
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[database_name]
        self.embedding_model = VoyageAIEmbeddings(model=embedding_model)
        self.embedding_dimensions = 1024
        self.llm = self._create_llm(model_provider, model_name)
        
        # <<< FIX: The original file had 'MongoDBSaver(self.client)', which is incorrect. >>>
        # The correct way to initialize is with from_conn_string, which also handles db selection.
        self.checkpointer = MongoDBSaver.from_conn_string(mongodb_uri, db_name=database_name)
        
        # This line was in your original code but is unused later.
        # It's safe to keep, but the tools re-initialize the store themselves.
        # self.memory_store = self._create_memory_store()
        
        self.tools = self._create_tools(user_tools or [])
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
        # This function is not actively used by the tools but is here for completeness.
        index_config = create_vector_index_config(
            embed=self.embedding_model,
            dims=self.embedding_dimensions,
            relevance_score_fn="dotProduct",
            fields=["content"]
        )
        store = MongoDBStore.from_conn_string(
            conn_string=self.mongodb_uri,
            db_name=self.database_name,
            collection_name="agent_memories",
            index_config=index_config,
            auto_index_timeout=60
        )
        return store
    
    def _create_tools(self, user_tools: List) -> List:
        """Create agent tools, combining built-in and user-provided tools."""
        
        built_in_tools = []
        
        @tool
        def save_memory(content: str) -> str:
            """Save important information to memory for later retrieval."""
            # <<< FIX: Wrap the entire operation in a try...except block to prevent crashes on M0 tier >>>
            try:
                with MongoDBStore.from_conn_string(
                    conn_string=self.mongodb_uri,
                    db_name=self.database_name,
                    collection_name="agent_memories",
                    index_config=create_vector_index_config(
                        embed=self.embedding_model, dims=self.embedding_dimensions,
                        relevance_score_fn="dotProduct", fields=["content"]
                    ),
                    auto_index_timeout=5 # Use a shorter timeout
                ) as store:
                    store.put(
                        namespace=("agent", self.agent_name),
                        key=f"memory_{hash(content)}",
                        value={"content": content, "timestamp": datetime.utcnow().isoformat()}
                    )
                return f"Memory saved successfully: '{content}'"
            except (TimeoutError, OperationFailure) as e:
                logger.warning(f"Could not save to vector memory (likely using M0 Free Tier): {e}")
                return "Note: Vector search is not available on this database tier. Memory was not saved for semantic search."

        @tool
        def retrieve_memories(query: str) -> str:
            """Retrieve relevant memories based on a query using semantic search."""
            # <<< FIX: Wrap the entire operation in a try...except block to prevent crashes on M0 tier >>>
            try:
                with MongoDBStore.from_conn_string(
                    conn_string=self.mongodb_uri,
                    db_name=self.database_name,
                    collection_name="agent_memories",
                     index_config=create_vector_index_config(
                        embed=self.embedding_model, dims=self.embedding_dimensions,
                        relevance_score_fn="dotProduct", fields=["content"]
                    ),
                    auto_index_timeout=5
                ) as store:
                    results = store.search(("agent", self.agent_name), query=query, limit=3)
                    if results:
                        memories = [result.value["content"] for result in results]
                        return f"Retrieved memories:\n" + "\n".join(f"- {mem}" for mem in memories)
                    else:
                        return "No relevant memories found."
            except (TimeoutError, OperationFailure) as e:
                logger.warning(f"Vector search failed (likely using M0 Free Tier): {e}")
                return "Vector search is not available on this database tier. Cannot retrieve memories."
        
        @tool
        def vector_search(user_query: str) -> str:
            """Retrieve information from ingested documents using vector search to answer a user query."""
            # <<< FIX: Wrap the entire operation in a try...except block to prevent crashes on M0 tier >>>
            try:
                vector_store = MongoDBAtlasVectorSearch.from_connection_string(
                    connection_string=self.mongodb_uri,
                    namespace=f"{self.database_name}.documents",
                    embedding=self.embedding_model
                )
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                results = retriever.invoke(user_query)
                if not results:
                    return "No relevant documents found via vector search."
                context = "\n\n".join([f"Source: {doc.metadata.get('source', 'N/A')}\nContent: {doc.page_content}" for doc in results])
                return f"Found the following information:\n{context}"
            except (TimeoutError, OperationFailure) as e:
                logger.warning(f"Vector search failed (likely using M0 Free Tier): {e}")
                return "Vector search is not available on this database tier. Cannot search documents."

        built_in_tools.extend([save_memory, retrieve_memories, vector_search])
        
        all_tools = built_in_tools + user_tools
        return all_tools
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in self.tools]))
        llm_with_tools = prompt | self.llm.bind_tools(self.tools)
        tools_by_name = {tool.name: tool for tool in self.tools}

        def agent_node(state: GraphState) -> Dict[str, List]:
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        def tool_node(state: GraphState) -> Dict[str, List]:
            result = []
            for tool_call in state["messages"][-1].tool_calls:
                tool_to_call = tools_by_name[tool_call["name"]]
                observation = tool_to_call.invoke(tool_call["args"])
                # <<< FIX: Ensure the observation content is always a string before creating a ToolMessage >>>
                result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
            return {"messages": result}

        def router(state: GraphState):
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
                return "tools"
            return END

        graph = StateGraph(GraphState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", router, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")
        
        return graph.compile(checkpointer=self.checkpointer)
    
    def execute(self, user_input: str, thread_id: Optional[str] = None) -> str:
        """Execute the graph with user input."""
        config = {"configurable": {"thread_id": thread_id or "default_thread"}}
        input_state = {"messages": [HumanMessage(content=user_input)]}
        
        final_message_content = "I couldn't generate a response."
        for output in self.graph.stream(input_state, config):
            if output:
                last_node_key = list(output.keys())[-1]
                if "messages" in output[last_node_key]:
                    final_message = output[last_node_key]['messages'][-1]
                    if hasattr(final_message, "content"):
                        final_message_content = final_message.content

        return final_message_content
    
    async def aexecute(self, user_input: str, thread_id: Optional[str] = None) -> str:
        """Async execute the graph with user input."""
        config = {"configurable": {"thread_id": thread_id or "default_thread"}}
        input_state = {"messages": [HumanMessage(content=user_input)]}
        
        final_message_content = "I couldn't generate a response."
        async for output in self.graph.astream(input_state, config):
             if output:
                last_node_key = list(output.keys())[-1]
                if "messages" in output[last_node_key]:
                    final_message = output[last_node_key]['messages'][-1]
                    if hasattr(final_message, "content"):
                        final_message_content = final_message.content
                    
        return final_message_content
    
    # The rest of your original file remains unchanged...
    def create_vector_indexes(self):
        """
        Create vector search indexes for collections.
        """
        import time
        
        collections_to_index = [
            "documents", 
            "agent_memories"
        ]
        
        for collection_name in collections_to_index:
            collection = self.db[collection_name]
            
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
                existing_indexes = list(collection.list_search_indexes())
                index_exists = any(idx.get("name") == "vector_index" for idx in existing_indexes)
                
                if not index_exists:
                    logger.info(f"Creating vector index for {collection_name}...")
                    collection.create_search_index(index_definition)
                else:
                    logger.info(f"✅ Vector index already exists for {collection_name}")
                    
            except OperationFailure as e:
                # This will catch the error on M0 Free Tiers
                logger.warning(f"⚠️ Could not create vector index for {collection_name} (this is expected on M0 clusters): {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred during index creation for {collection_name}: {e}")

# Example usage from your original file
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    load_dotenv()
    
    agent = MongoDBLangGraphAgent(
        mongodb_uri=os.getenv("MONGODB_URI"),
        agent_name="assistant",
        model_provider="openai",
        model_name="gpt-4o"
    )
    
    agent.create_vector_indexes()
    
    response = agent.execute(
        "What are some movies that take place in the ocean?",
        thread_id="test_session"
    )
    print(f"Response: {response}")