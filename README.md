# 🧠 AI Agent Boilerplate with Memory

Production-ready AI agent framework with sophisticated 5-component memory system, built on MongoDB Atlas and LangGraph.

## ⚡ Quick Start (5 Minutes)

```bash
# 1. Clone and setup
git clone https://github.com/romiluz13/agent_with_memory.git
cd agent_with_memory
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Run example
python examples/mongodb_langgraph_example.py
```

## 🏗️ Architecture

Built on **MongoDB's Official LangGraph Integration** patterns:

```
┌─────────────────────────────────────────┐
│           Client (Web/API)              │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│            FastAPI + WebSocket          │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│     LangGraph with MongoDB Memory       │
│   • MongoDBSaver (Short-term/Sessions)  │
│   • MongoDBStore (Long-term/Persistent) │
└───┬─────────────────────────────┬───────┘
    │                             │
┌───▼──────────┐       ┌─────────▼────────┐
│ 5-Component  │       │   Tool Framework  │
│   Memory     │       │  (Extensible)     │
│   System     │       └──────────────────┘
└──────────────┘
    │
┌───▼──────────────────────────────────────┐
│     MongoDB Atlas + Vector Search        │
│   • Voyage AI Embeddings (1024 dims)    │
│   • Cosine Similarity                   │
└──────────────────────────────────────────┘
```

## 🧠 5-Component Memory System

### 1. **Episodic Memory** - Conversation History
```python
# Stores: User interactions, conversation context, temporal sequences
agent.memory.episodic.store("User asked about Python tutorials")
```

### 2. **Procedural Memory** - Learned Workflows  
```python
# Stores: Step-by-step processes, successful patterns, automation sequences
agent.memory.procedural.store("To debug: 1. Check logs 2. Verify config 3. Test connections")
```

### 3. **Semantic Memory** - Domain Knowledge
```python
# Stores: Facts, concepts, relationships, domain expertise
agent.memory.semantic.store("MongoDB Atlas supports vector search with cosine similarity")
```

### 4. **Working Memory** - Current Context
```python
# Stores: Active session data, temporary context, current focus
agent.memory.working.store("Currently helping user with database optimization")
```

### 5. **Semantic Cache** - Performance Optimization
```python
# Stores: Frequently accessed information, query results, computed responses
agent.memory.cache.get("common_database_questions")
```

## 🚀 Features

- **🔥 MongoDB + LangGraph**: Official integration patterns
- **🧠 Sophisticated Memory**: 5-component memory system  
- **⚡ Vector Search**: MongoDB Atlas with Voyage AI embeddings
- **🔍 Observability**: Galileo AI for LLM monitoring
- **🛠️ Production Ready**: FastAPI, Docker, comprehensive testing
- **📚 Document Ingestion**: PDF processing with smart chunking
- **🔧 Extensible**: Easy to add custom tools and agents

## 📦 Project Structure

```
agent_with_memory/
├── src/
│   ├── core/              # Agent implementations
│   ├── memory/            # 5-component memory system
│   ├── ingestion/         # Document processing
│   ├── embeddings/        # Voyage AI integration
│   ├── storage/           # MongoDB utilities
│   ├── observability/     # Galileo monitoring
│   └── api/               # FastAPI backend
├── examples/              # Ready-to-run examples
├── tests/                 # Comprehensive test suite
├── infrastructure/        # Docker & deployment configs
└── requirements.txt       # Python dependencies
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# MongoDB Atlas
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/
MONGODB_DB_NAME=agent_memory_db

# AI Services
VOYAGE_API_KEY=your_voyage_key
OPENAI_API_KEY=your_openai_key

# Observability (Optional)
GALILEO_API_KEY=your_galileo_key
GALILEO_PROJECT_NAME=agent_with_memory
```

## 💻 Usage Examples

### Basic Agent
```python
from src import MongoDBLangGraphAgent

# Initialize agent
agent = MongoDBLangGraphAgent(
    mongodb_uri=os.getenv("MONGODB_URI"),
    agent_name="assistant",
    model_provider="openai",
    model_name="gpt-4o"
)

# Chat with memory
response = await agent.aexecute(
    "What did we discuss about Python yesterday?",
    thread_id="user_123"
)
```

### Document Ingestion
```python
from src.ingestion import MongoDBDocumentIngestion

# Process documents
ingestion = MongoDBDocumentIngestion(
    mongodb_uri=os.getenv("MONGODB_URI")
)

# Ingest PDF
result = await ingestion.ingest_pdf("documents/manual.pdf")
```

### Memory Operations
```python
# Store different types of memories
await agent.memory.episodic.store("User prefers technical explanations")
await agent.memory.semantic.store("FastAPI is a modern Python web framework")
await agent.memory.procedural.store("To deploy: 1. Build image 2. Push to registry 3. Update config")

# Retrieve relevant memories
memories = await agent.memory.retrieve("web framework deployment")
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with coverage
pytest --cov=src --cov-report=html
```

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up -d

# Scale API instances
docker-compose up -d --scale api=3

# Production deployment
docker-compose -f docker-compose.prod.yml up -d
```

## 📊 Monitoring with Galileo AI

Track LLM performance, RAG quality, and user interactions:

- **Generation Metrics**: Response time, token usage, model performance
- **Retrieval Quality**: Search relevance scores, document quality  
- **Memory Analytics**: Usage patterns, effectiveness metrics
- **Error Tracking**: Automatic error detection and alerting

## 🛠️ Development

### Adding Custom Tools
```python
from langchain.agents import tool

@tool
def custom_search(query: str) -> str:
    """Custom search implementation."""
    return f"Results for: {query}"

# Add to agent
agent.tools.append(custom_search)
```

### Creating Specialized Agents
```python
class ResearchAgent(MongoDBLangGraphAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add research-specific tools and prompts
```

## 📚 Documentation

- **API Reference**: `/docs` when running FastAPI server
- **Architecture Guide**: `examples/` directory
- **Memory System**: `src/memory/` implementation
- **MongoDB Integration**: Based on official [docs-notebooks](https://github.com/mongodb/docs-notebooks/)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MongoDB**: Official LangGraph integration patterns
- **LangChain/LangGraph**: Agent orchestration framework
- **Voyage AI**: High-quality embeddings
- **Galileo AI**: LLM observability platform

## 🆘 Support

- 📧 Issues: [GitHub Issues](https://github.com/romiluz13/agent_with_memory/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/romiluz13/agent_with_memory/discussions)
- 📖 Documentation: [Wiki](https://github.com/romiluz13/agent_with_memory/wiki)

---

**Built with ❤️ for the AI community**

*Clone → Configure → Deploy in 5 minutes* 🚀