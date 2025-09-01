"""
Example: MongoDB + LangGraph Agent with Galileo Monitoring
Demonstrates patterns from official MongoDB repositories
"""

import os
import asyncio
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.agent_langgraph import MongoDBLangGraphAgent
from src.ingestion.mongodb_ingestion import MongoDBDocumentIngestion, MongoDBRAGGenerator
from src.observability.galileo_monitor import get_galileo_monitor


async def main():
    """
    Complete example showing MongoDB + LangGraph patterns.
    Based on:
    - mongodb/docs-notebooks/ai-integrations/langgraph.ipynb
    - mongodb-developer/atlas-search-playground-chatbot-starter
    """
    
    # Load environment variables
    load_dotenv()
    
    # Initialize Galileo monitoring
    monitor = get_galileo_monitor()
    print("✅ Galileo monitoring initialized")
    
    # 1. Initialize MongoDB Document Ingestion (from atlas-search-playground-chatbot-starter)
    print("\n📚 Setting up document ingestion...")
    ingestion = MongoDBDocumentIngestion(
        mongodb_uri=os.getenv("MONGODB_URI"),
        database_name="ai_agent_boilerplate",
        collection_name="documents"
    )
    
    # Create vector index (MongoDB pattern)
    ingestion.create_vector_index()
    print("✅ Vector index created")
    
    # 2. Ingest sample documents
    print("\n📄 Ingesting sample documents...")
    sample_text = """
    AI agents are autonomous software programs that can perceive their environment,
    make decisions, and take actions to achieve specific goals. They use various
    techniques including machine learning, natural language processing, and reasoning.
    
    Memory systems in AI agents allow them to:
    - Remember past conversations (episodic memory)
    - Learn and improve workflows (procedural memory)
    - Accumulate domain knowledge (semantic memory)
    - Maintain current context (working memory)
    - Cache frequent queries for performance (semantic cache)
    
    LangGraph is a framework for building stateful, multi-actor applications with LLMs.
    It provides graph-based orchestration of agent workflows with built-in state management.
    
    MongoDB Atlas Vector Search enables semantic search capabilities by storing and
    querying high-dimensional vector embeddings alongside your data.
    """
    
    result = await ingestion.ingest_text(
        text_content=sample_text,
        source_name="ai_agents_intro",
        metadata={"category": "documentation", "version": "1.0"}
    )
    print(f"✅ Ingested {result['chunks_created']} chunks")
    
    # 3. Initialize MongoDB LangGraph Agent (from MongoDB notebook)
    print("\n🤖 Initializing MongoDB LangGraph Agent...")
    agent = MongoDBLangGraphAgent(
        mongodb_uri=os.getenv("MONGODB_URI"),
        agent_name="research_assistant",
        model_provider="openai",
        model_name="gpt-4o",
        embedding_model="voyage-3-large",
        embedding_dimensions=1024
    )
    
    # Create vector indexes for agent
    agent.create_vector_indexes()
    print("✅ Agent initialized with MongoDB memory")
    
    # 4. Test Agent with Vector Search (MongoDB pattern)
    print("\n🔍 Testing vector search...")
    
    with monitor.trace_generation("vector_search_test", "gpt-4o") as trace_id:
        response = await agent.aexecute(
            "What are AI agents and how do they use memory?",
            thread_id="demo_session_1"
        )
        print(f"Agent Response: {response}")
        
        # Log to Galileo
        monitor.log_generation(
            prompt="What are AI agents and how do they use memory?",
            response=response,
            model="gpt-4o",
            user_id="demo_user",
            session_id="demo_session_1"
        )
    
    # 5. Test Memory Storage (MongoDB Store pattern)
    print("\n💾 Testing memory storage...")
    
    response = await agent.aexecute(
        "Remember that I'm interested in learning about LangGraph and MongoDB integration.",
        thread_id="demo_session_1"
    )
    print(f"Memory Storage Response: {response}")
    
    # 6. Test Memory Retrieval (MongoDB Store pattern)
    print("\n🧠 Testing memory retrieval...")
    
    response = await agent.aexecute(
        "What am I interested in learning about?",
        thread_id="demo_session_1"
    )
    print(f"Memory Retrieval Response: {response}")
    
    # 7. Test Session Persistence (MongoDB Checkpointer pattern)
    print("\n📌 Testing session persistence...")
    
    # Continue conversation in same thread
    response = await agent.aexecute(
        "Can you explain more about how they work together?",
        thread_id="demo_session_1"
    )
    print(f"Continued Conversation: {response}")
    
    # 8. Test Cross-Session Memory (Long-term memory)
    print("\n🔄 Testing cross-session memory...")
    
    # New session but should remember preferences
    response = await agent.aexecute(
        "Based on what you know about me, what topics should I explore?",
        thread_id="demo_session_2"  # Different session
    )
    print(f"Cross-Session Response: {response}")
    
    # 9. Test RAG with Retrieved Documents
    print("\n📖 Testing RAG with document retrieval...")
    
    # Initialize RAG generator
    rag_generator = MongoDBRAGGenerator(
        ingestion_system=ingestion,
        llm_provider="openai",
        model_name="gpt-4o"
    )
    
    # Generate RAG response (MongoDB generate-response.js pattern)
    with monitor.trace_generation("rag_test", "gpt-4o") as trace_id:
        rag_response = await rag_generator.generate_response(
            question="How does MongoDB Atlas Vector Search work?",
            num_candidates=100,  # MongoDB's recommended value
            limit=5
        )
        print(f"RAG Response: {rag_response}")
        
        # Log retrieval to Galileo
        retrieved_docs = await ingestion.retrieve_documents(
            "How does MongoDB Atlas Vector Search work?",
            num_candidates=100,
            limit=5
        )
        
        monitor.log_retrieval(
            query="How does MongoDB Atlas Vector Search work?",
            retrieved_documents=retrieved_docs,
            user_id="demo_user",
            session_id="demo_session_1"
        )
    
    # 10. Show Galileo Metrics Summary
    print("\n📊 Galileo Monitoring Summary:")
    metrics = monitor.get_metrics_summary()
    print(f"Monitoring Status: {metrics}")
    
    print("\n✨ Example completed successfully!")
    print("Check your Galileo dashboard for detailed analytics")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
