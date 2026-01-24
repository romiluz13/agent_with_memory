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
│   • Hybrid Search (vector + full-text)  │
│   • Multi-tenant Isolation              │
└──────────────────────────────────────────┘
```

## 📁 Project Structure

```
agent_with_memory/
├── demo_memory_agent.py    # START HERE - Real-life demo
├── scripts/
│   ├── setup_indexes.py    # Create vector + text indexes
│   └── setup_test_indexes.py # Create indexes for test DBs
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
│   └── retrieval/          # Advanced search system
│       ├── vector_search.py    # Main search engine
│       ├── config.py           # Search configuration
│       ├── rrf.py              # Reciprocal Rank Fusion
│       ├── tier_support.py     # Atlas tier detection
│       ├── score_parser.py     # Score extraction
│       └── filters/            # Filter builders
│           ├── vector_search_filters.py  # MQL filters
│           ├── atlas_search_filters.py   # Atlas Search filters
│           └── lexical_prefilters.py     # MongoDB 8.2+ prefilters
├── tests/
│   ├── retrieval/          # 148 unit tests
│   └── integration/        # E2E production tests
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

### Hybrid Search (Vector + Full-Text)
Find relevant memories using MongoDB's `$rankFusion` for best results:
```python
# Hybrid search is the DEFAULT - combines semantic + keyword matching
memories = await memory.episodic.retrieve(
    query="What does the user like?",
    agent_id="my_agent",
    user_id="user_123",
    limit=5,
    threshold=0.5,
    search_mode="hybrid"  # Default - can also use "semantic" or "text"
)
```

**Why hybrid?**
- "John" → Exact keyword match finds the person
- "software developer" → Semantic similarity finds "engineer"
- Combined → Best of both worlds

### Atlas Tier Support
Works on ALL MongoDB Atlas tiers with automatic fallback:

| Tier | $rankFusion | Text Search | Fallback |
|------|-------------|-------------|----------|
| **M10+** | Native | Full | - |
| **M0/M2** | Manual RRF | Full | Reciprocal Rank Fusion |
| **Vector-only** | - | - | Vector search only |

The system auto-detects your cluster tier and uses the best available strategy.

## 🧪 Testing

```bash
# Run all tests (154+ tests)
python -m pytest tests/ -v

# Unit tests for retrieval system (148 tests)
python -m pytest tests/retrieval/ -v

# Production-realistic E2E tests (6 scenarios)
python -m pytest tests/integration/test_production_realistic.py -v
```

### Production Test Scenarios
| Test | What it Validates |
|------|-------------------|
| Customer Support Conversation | Full memory store/retrieve flow |
| Multi-Tenant Isolation | Agent isolation via agent_id |
| Long Conversation Memory | Memory over 20+ turns |
| Cross-Session Persistence | Data persists across sessions |
| Concurrent Users Stress | 100 users, 1000 operations |
| Hybrid Search Quality | Vector + text fusion |

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
4. Run the setup script to create indexes:

```bash
# Create all required indexes (vector + text for hybrid search)
python scripts/setup_indexes.py
```

### Search Indexes

The setup script creates **14 indexes** (7 vector + 7 text) for hybrid search:

**Vector Index** (for semantic similarity):
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

**Text Index** (for keyword matching):
```json
{
  "name": "text_search_index",
  "type": "search",
  "definition": {
    "mappings": {
      "fields": {
        "content": {"type": "string", "analyzer": "lucene.standard"},
        "agent_id": {"type": "string"},
        "user_id": {"type": "string"}
      }
    }
  }
}
```

> **Note**: Indexes take 1-5 minutes to build after creation. The system gracefully falls back to vector-only search until text indexes are ready.

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
