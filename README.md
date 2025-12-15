# 🧠 Agent With Memory (AWM 2.0)

**Production-ready AI memory system for any agent.** Built on MongoDB Atlas Vector Search.

## 🎯 What This Is

A **plug-and-play memory layer** that gives any AI agent persistent memory across sessions. Your agent remembers users, learns from interactions, and builds knowledge over time.

**Without memory**: ChatGPT that forgets everything.
**With memory**: A true AI agent that learns and grows.

## ⚡ Quick Start (5 Minutes)

```bash
# 1. Clone
git clone https://github.com/romiluz13/agent_with_memory.git
cd agent_with_memory

# 2. Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure (create .env file)
MONGODB_URI=mongodb+srv://...
VOYAGE_API_KEY=pa-...
GOOGLE_API_KEY=AIza...

# 4. Run the demo
python demo_memory_agent.py
```

The demo proves memory works across sessions:
- **Session 1**: Tell the agent your name, job, pet, etc.
- **Session 2**: NEW session asks what it remembers → It recalls everything!

## 🔌 Plug & Play - Add Memory to ANY Agent

```python
from src.memory.manager import MemoryManager
from src.memory.base import MemoryType
from src.storage.mongodb_client import MongoDBClient, MongoDBConfig

# 1. Connect to MongoDB
config = MongoDBConfig(uri="mongodb+srv://...", database="my_app")
db_client = MongoDBClient()
await db_client.initialize(config)

# 2. Create Memory Manager
memory = MemoryManager(db_client.db)

# 3. Store memories (from your agent's conversations)
await memory.store_memory(
    content="User said they love Python and work at Google",
    memory_type=MemoryType.EPISODIC,
    agent_id="my_agent",
    user_id="user_123"
)

# 4. Retrieve relevant memories (before responding)
memories = await memory.episodic.retrieve(
    query="What programming language does the user like?",
    agent_id="my_agent",
    user_id="user_123",
    limit=5
)

# 5. Use memories in your agent's context
for mem in memories:
    print(f"Remembered: {mem.content}")
```

**That's it!** 5 lines to add persistent memory to any agent.

## 🧠 7-Type Memory System

| Type | Purpose | Example |
|------|---------|---------|
| **EPISODIC** | Conversation history | "User asked about Python tutorials" |
| **SEMANTIC** | Facts & knowledge | "MongoDB supports vector search" |
| **PROCEDURAL** | How-to workflows | "To deploy: build → push → update" |
| **WORKING** | Current session context | "Currently helping with optimization" |
| **CACHE** | Fast retrieval | Frequently accessed data |
| **ENTITY** | People, places, things | "John works at Google" |
| **SUMMARY** | Compressed context | Condensed conversation summaries |

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         Your Agent (Any Framework)       │
│   LangChain, LangGraph, Custom, etc.    │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│           MemoryManager                  │
│   • store_memory()                       │
│   • retrieve_memories()                  │
│   • extract_entities()                   │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│     MongoDB Atlas + Vector Search        │
│   • Voyage AI Embeddings (1024 dims)    │
│   • Cosine Similarity Search            │
│   • Multi-tenant Isolation              │
└──────────────────────────────────────────┘
```

## 📁 Project Structure

```
agent_with_memory/
├── demo_memory_agent.py    # 🚀 START HERE - Real-life demo
├── src/
│   ├── memory/             # 7-type memory system
│   │   ├── manager.py      # Main orchestrator
│   │   ├── episodic.py     # Conversation history
│   │   ├── semantic.py     # Facts & knowledge
│   │   ├── procedural.py   # Workflows
│   │   ├── working.py      # Session context
│   │   ├── cache.py        # Fast retrieval
│   │   ├── entity.py       # Entity extraction
│   │   └── summary.py      # Context compression
│   ├── context/            # Token management
│   │   ├── engineer.py     # Auto-compression at 80%
│   │   └── summarizer.py   # LLM summarization
│   ├── storage/            # MongoDB integration
│   │   ├── mongodb_client.py
│   │   └── vector_index.py
│   ├── embeddings/         # Voyage AI
│   │   └── voyage_client.py
│   └── retrieval/          # Vector search
│       └── vector_search.py
├── tests/                  # Comprehensive test suite
└── CLAUDE.md               # Project documentation
```

## 🔧 Key Features

### Multi-Tenant Isolation
Each agent and user has isolated memories:
```python
# Agent A's memories are separate from Agent B's
await memory.store_memory(..., agent_id="agent_a", user_id="user_1")
await memory.store_memory(..., agent_id="agent_b", user_id="user_1")
```

### Entity Extraction
Automatically extract people, organizations, locations:
```python
entities = await memory.extract_entities(
    text="I'm John, a software engineer at Google",
    agent_id="my_agent",
    llm=your_llm
)
# Returns: [{"name": "John", "type": "PERSON"}, {"name": "Google", "type": "ORGANIZATION"}]
```

### Context Compression
Auto-compress when context gets too long:
```python
from src.context.engineer import ContextEngineer

engineer = ContextEngineer()
if engineer.should_compress(context, model="gpt-4"):
    compressed = await engineer.compress(context, llm)
```

### Vector Search with Filters
Find relevant memories with semantic search:
```python
memories = await memory.episodic.retrieve(
    query="What does the user like?",
    agent_id="my_agent",
    user_id="user_123",
    limit=5,
    threshold=0.5  # Similarity threshold
)
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific tests
python -m pytest tests/integration/test_multi_tenant_isolation.py -v
python -m pytest tests/integration/test_entity_extraction.py -v
```

## 📝 Environment Variables

```bash
# Required
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/
VOYAGE_API_KEY=pa-...          # For embeddings
GOOGLE_API_KEY=AIza...         # For LLM (or use OpenAI)

# Optional
OPENAI_API_KEY=sk-...          # Alternative LLM
ANTHROPIC_API_KEY=sk-ant-...   # Alternative LLM
```

## 🚀 MongoDB Atlas Setup

1. Create a [MongoDB Atlas](https://www.mongodb.com/atlas) account (free tier works!)
2. Create a cluster
3. Get your connection string
4. The system auto-creates collections and vector indexes

### Required Vector Index

The system creates this automatically, but for reference:
```json
{
  "name": "vector_index",
  "type": "vectorSearch",
  "definition": {
    "fields": [
      {"type": "vector", "path": "embedding", "similarity": "cosine", "numDimensions": 1024},
      {"type": "filter", "path": "agent_id"},
      {"type": "filter", "path": "user_id"}
    ]
  }
}
```

## 🤝 Integration Examples

### With LangChain
```python
from langchain_openai import ChatOpenAI
from src.memory.manager import MemoryManager

llm = ChatOpenAI(model="gpt-4")
memory = MemoryManager(db)

# Before each LLM call, retrieve relevant memories
memories = await memory.episodic.retrieve(query=user_input, agent_id="my_agent")
context = "\n".join([m.content for m in memories])

response = llm.invoke(f"Context:\n{context}\n\nUser: {user_input}")

# After LLM response, store the interaction
await memory.store_memory(
    content=f"User: {user_input}\nAssistant: {response}",
    memory_type=MemoryType.EPISODIC,
    agent_id="my_agent"
)
```

### With LangGraph
```python
from langgraph.graph import StateGraph
from src.memory.manager import MemoryManager

memory = MemoryManager(db)

def memory_node(state):
    # Retrieve memories before processing
    memories = await memory.episodic.retrieve(
        query=state["input"],
        agent_id=state["agent_id"]
    )
    state["context"] = memories
    return state

# Add to your graph
workflow.add_node("memory", memory_node)
```

## 📄 License

MIT License - Use it, modify it, ship it!

## 🙏 Acknowledgments

- **MongoDB**: Atlas Vector Search
- **Voyage AI**: High-quality embeddings
- **LangChain/LangGraph**: Agent frameworks

---

**Built for the AI community** 🚀

*Clone → Configure → Remember Everything*
