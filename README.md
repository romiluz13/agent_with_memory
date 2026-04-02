# Agent With Memory (AWM 2.0)

AWM 2.0 is a MongoDB-first starter for building AI applications with persistent memory, retrieval, and agent runtime infrastructure.

It is designed for developers who want a serious reference implementation, not a toy chatbot and not a one-command production platform.

## What You Get

- A real FastAPI API surface for agents, chat, memories, evaluation, HITL, time-travel, and WebSocket chat
- A 7-type memory system backed by MongoDB
- Hybrid retrieval with vector search, text search, and fallbacks when cluster capabilities are limited
- A LangGraph-first runtime with persisted checkpoints and long-term memory
- Seeded validation scripts for realistic data, Atlas cloud validation, and Atlas Local Preview validation
- Test-only external LLM smoke tooling that stays out of runtime code paths

## What This Repo Is

This repo is a strong starting point if you are building:

- an AI assistant with cross-session memory
- a memory-rich RAG application
- a LangGraph-based agent on MongoDB
- a reference architecture for MongoDB-powered AI systems

This repo is not:

- a promise that one `git clone` gives you a production app
- a hosted SaaS template with auth, billing, frontend, and deployment all finished
- a benchmark-backed claim of being the single best boilerplate on earth

## Why Developers Actually Use It

The point is not “look how many features fit in one README.”

The point is that you can start with a codebase that already solves the annoying parts teams usually rebuild badly:

- memory types with clear ownership and isolation rules
- retrieval that can work across cluster capability differences
- long-running chat state and replayable checkpoints
- seeded validation with real data instead of fake “hello world” strings
- API routes that match the runtime instead of demo-only placeholders

If you are building your own agent app, you should be able to copy patterns from here without first reverse-engineering a bunch of undocumented decisions.

## Feature Map

### Memory

| Type | Purpose |
|---|---|
| `episodic` | conversation history and past events |
| `semantic` | facts and knowledge |
| `procedural` | workflows and learned procedures |
| `working` | active session context |
| `cache` | fast semantic cache |
| `entity` | extracted people, systems, orgs, concepts |
| `summary` | compressed context with JIT expansion |

Key behaviors implemented in code:

- multi-tenant isolation via `agent_id` and optional `user_id`
- summary offload instead of destructive history deletion
- entity extraction and graph-style relationship traversal
- bounded graph relationship arrays to avoid unbounded growth

### Retrieval

- MongoDB Atlas Vector Search
- MongoDB Atlas Search text indexes
- hybrid search with vector + text fusion where supported
- fallback to vector-only when cluster capabilities do not allow full hybrid execution
- retrieval projections that avoid extra document fetches on the hot path

### Runtime

- LangGraph-first orchestration
- persisted checkpoints via MongoDB
- HTTP chat
- SSE streaming
- WebSocket chat
- HITL approval flow
- time-travel history, snapshot, and replay routes

### Validation

- deterministic realistic seeding
- Atlas cloud validation lane
- Atlas Local Preview validation lane
- test-only external LLM smoke lane for live model verification

## Why MongoDB Here

Most RAG demos get complicated in the wrong place.

They start with one database for app state, one store for vectors, one search system, and a pile of glue code. That can look sophisticated in a diagram, but it is a pain to debug when you are still trying to answer simple questions like:

- what did this agent remember
- why did retrieval return this result
- where did this checkpoint come from
- which indexes exist in this environment

This repo is opinionated about MongoDB because it keeps those concerns close together:

- application data
- long-term memory
- text search
- vector search
- graph-style traversal
- replay and checkpoint-adjacent debugging

For developers, the value is practical:

- fewer systems to bootstrap before you can test real behavior
- one mental model for data while the product is still changing fast
- simpler seeded validation and local repro
- less “impressive” architecture that collapses the moment memory starts acting like real application state

This is not a claim that MongoDB is the right answer for every stack. It is a claim that for a memory-heavy AI starter, it gives you a cleaner path from prototype to serious system.

## Architecture

```text
Client / UI / Agent Caller
        |
        v
FastAPI API
  - /api/v1/agents
  - /api/v1/chat
  - /api/v1/memories
  - /api/v1/query
  - /api/v1/hitl
  - /api/v1/time-travel
  - /api/v1/evaluate
  - /api/v1/ws/{agent_id}
        |
        v
LangGraph Runtime + Agent Registry
        |
        +--> MongoDBSaver checkpoints
        +--> MongoDB long-term memory store
        +--> MemoryManager
                 |
                 +--> episodic
                 +--> semantic
                 +--> procedural
                 +--> working
                 +--> cache
                 +--> entity
                 +--> summary
                 +--> graph
```

## Quick Start

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,test]"
cp env.example .env
```

Minimum useful env:

```bash
MONGODB_URI=...
VOYAGE_API_KEY=pa-...
OPENAI_API_KEY=...
```

You can also use `GOOGLE_API_KEY` or `ANTHROPIC_API_KEY` instead of OpenAI.

### 2. Run the demo

```bash
python demo_memory_agent.py
```

### 3. Run the API

```bash
uvicorn src.api.main:app --reload
```

Important routes:

| Route | Purpose |
|---|---|
| `GET /health` | compatibility health check |
| `GET /health/ready` | readiness probe |
| `POST /chat` | compatibility chat route backed by the real runtime |
| `POST /api/v1/chat/` | canonical chat route |
| `POST /api/v1/chat/stream` | SSE streaming |
| `GET /api/v1/agents` | list agents |
| `POST /api/v1/memories/` | store memory |
| `POST /api/v1/memories/search` | search memories |
| `GET /api/v1/memories/stats/summary` | memory stats |
| `POST /api/v1/query` | natural language to MongoDB query |
| `GET /api/v1/hitl/pending/{agent_id}` | pending approvals |
| `GET /api/v1/time-travel/history/{thread_id}` | checkpoint history |
| `POST /api/v1/evaluate` | RAG evaluation |
| `WS /api/v1/ws/{agent_id}` | WebSocket chat |

## Use It as a Library

### Memory Manager

```python
from src.memory.base import MemoryType
from src.memory.manager import MemoryManager
from src.storage.mongodb_client import initialize_mongodb

mongodb = await initialize_mongodb(
    uri="mongodb://localhost:27018/?directConnection=true",
    database="my_app",
)

memory = MemoryManager(mongodb.db)

await memory.store_memory(
    content="User prefers concise answers and works on retrieval systems",
    memory_type=MemoryType.EPISODIC,
    agent_id="assistant",
    user_id="user-123",
)

results = await memory.retrieve_memories(
    query="What style does the user prefer?",
    agent_id="assistant",
    user_id="user-123",
    limit=5,
)
```

### LangGraph Agent

```python
from src.core.agent_langgraph import MongoDBLangGraphAgent

agent = MongoDBLangGraphAgent(
    mongodb_uri="mongodb://localhost:27018/?directConnection=true",
    agent_name="assistant",
    model_provider="openai",
    model_name="gpt-4o",
    database_name="my_app",
)

response = await agent.invoke(
    message="Remember that I prefer short answers.",
    user_id="user-123",
    conversation_id="thread-1",
)
```

## Local Validation Lane: Atlas Local Preview

This repo now supports a real Atlas Local Preview workflow for search and vector validation.

### Reuse an existing local preview container or create one

```bash
python scripts/bootstrap_local_deployment.py
```

On this machine, the script detects and reuses a running preview container and prints a host-safe URI such as:

```bash
mongodb://localhost:27018/?directConnection=true
```

### Validate local search support

```bash
python scripts/validate_atlas_local_preview.py
```

This script:

- creates a real document
- creates a real vector index
- creates a real text index
- waits for both to become ready

### Point the app at Atlas Local Preview

```bash
export MONGODB_URI="mongodb://localhost:27018/?directConnection=true"
export MONGODB_VALIDATION_LANE=local_validation
python scripts/setup_indexes.py
uvicorn src.api.main:app --reload
```

## Cloud Validation Lane

For Atlas cloud validation with realistic data:

```bash
export MONGODB_URI="mongodb+srv://..."
export MONGODB_VALIDATION_LANE=cloud_validation
export VOYAGE_API_KEY="pa-..."

python scripts/seed_realistic_data.py --reset
python scripts/run_seeded_validation.py
```

The seeded validation script checks:

- collection population
- retrieval sanity
- API surface health
- live LLM availability for chat

## Test-Only External LLM Lane

The repo includes a test-only smoke script for an external LLM API or gateway. It is not wired into product runtime code.

Use it when you want to validate that an external model endpoint is alive before blaming the app:

```bash
export LLM_TEST_API_KEY="..."
export LLM_TEST_API_BASE_URL="https://your-endpoint.example/v1/chat/completions"
python scripts/test_llm_gateway.py --prompt "Return exactly the text: smoke ok"
```

This is useful for:

- OpenAI-compatible chat endpoints
- internal AI gateways
- provider proxies that normalize multiple models behind one interface

If your endpoint uses a non-Bearer auth header, set:

```bash
export LLM_TEST_AUTH_HEADER="api-key"
export LLM_TEST_AUTH_SCHEME=""
```

If you want to test provider-native APIs directly, use the provider's own shape and auth model:

- OpenAI uses the Responses API as the current primary path
- Anthropic uses the Messages API
- Gemini uses `generateContent` and `streamGenerateContent`

This lane is intentionally separate from the app runtime so provider checks stay lightweight and do not leak test-specific assumptions into product code.

## Natural Language to MongoDB Queries

The repo includes a read-only NL-to-MQL tool with safety controls:

- collection allowlist
- read-only enforcement
- agent scoping
- blocked write operators such as `$merge` and `$out`

Route:

```bash
POST /api/v1/query
```

## HITL and Time Travel

Two parts of the repo that are easy to miss but useful in real systems:

- HITL approval routes for sensitive actions
- replayable thread state through MongoDB-backed checkpoints

These are available through:

- [`src/api/routes/hitl.py`](src/api/routes/hitl.py)
- [`src/api/routes/time_travel.py`](src/api/routes/time_travel.py)

## Project Layout

```text
demo_memory_agent.py
scripts/
  bootstrap_local_deployment.py
  setup_indexes.py
  setup_schema_validation.py
  setup_graph_indexes.py
  seed_realistic_data.py
  run_seeded_validation.py
  validate_atlas_local_preview.py
  test_llm_gateway.py
src/
  api/
  context/
  core/
  embeddings/
  evaluation/
  memory/
  observability/
  retrieval/
  storage/
  tools/
tests/
```

## Development Notes

### MongoDB rule for this repo

For MongoDB-specific behavior, this repo treats MongoDB documentation and MongoDB tooling behavior as the source of truth.

### Python version

Use Python `3.11` to `3.13`.

### Install extras

```bash
pip install -e ".[dev,test]"
```

### Useful commands

```bash
ruff check src scripts tests
python -m pytest tests -q
python -m compileall src scripts tests
```

## Honest Status

This repo is strong as a MongoDB-first AI starter, but you should position it honestly:

- it is a serious reference implementation
- it is not magic production in a box
- some capabilities still depend on your Atlas tier, provider keys, and deployment choices

That is normal for real AI infrastructure.

## License

MIT
