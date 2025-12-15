# Agent With Memory 2.0 (AWM 2.0)

> Production-ready AI agent framework with advanced multi-layer memory architecture.

## Project Overview

AWM 2.0 is a sophisticated AI agent framework that implements a 7-type memory system inspired by cognitive science and Oracle AI Developer Hub innovations. The system uses MongoDB for persistent storage with vector search capabilities.

## Architecture

```
src/
├── core/           # Agent core (BaseAgent, AgentConfig, AgentState)
├── memory/         # 7-type memory system
│   ├── base.py         # Memory model, MemoryType enum
│   ├── manager.py      # MemoryManager orchestration
│   ├── episodic.py     # Conversation history
│   ├── semantic.py     # Knowledge/facts
│   ├── procedural.py   # How-to knowledge
│   ├── working.py      # Short-term context
│   ├── cache.py        # Fast retrieval cache
│   ├── entity.py       # LLM-extracted entities (NEW)
│   └── summary.py      # Compressed context (NEW)
├── context/        # Context engineering (NEW)
│   ├── engineer.py     # Token monitoring, auto-compress
│   └── summarizer.py   # LLM summarization
├── tools/          # Agent tools
│   ├── mcp_toolkit.py  # MCP integration
│   └── summary_tools.py # Summary expansion tools (NEW)
├── services/       # External services
│   ├── embeddings.py   # Voyage AI embeddings
│   └── vector_search.py # MongoDB vector search
└── utils/          # Utilities and config
```

## Memory Types

| Type | Purpose | Store |
|------|---------|-------|
| EPISODIC | Conversation history | episodic_memories |
| SEMANTIC | Facts and knowledge | semantic_memories |
| PROCEDURAL | How-to knowledge | procedural_memories |
| WORKING | Short-term context | working_memories |
| CACHE | Fast retrieval | cache_memories |
| ENTITY | Extracted entities | entity_memories |
| SUMMARY | Compressed context | summary_memories |

## Key Patterns

### Mark-as-Summarized (Not Delete)
Messages are marked with `summary_id` when compressed, preserving audit trail:
```python
await episodic.mark_as_summarized(agent_id, thread_id, summary_id)
```

### Context Compression at 80%
Auto-compress when token usage exceeds threshold:
```python
if context_engineer.should_compress(context, model):
    await context_engineer.summarize_and_offload(context, memory_manager, llm)
```

### Entity Extraction
LLM extracts PERSON, ORGANIZATION, LOCATION, SYSTEM, CONCEPT:
```python
entities = await memory_manager.extract_entities(text, agent_id, llm)
```

## Commands

- `/test` - Run pytest on tests/
- `/lint` - Run linting checks
- `/typecheck` - Run mypy type checking
- `/memory-status` - Show memory layer statistics

## Style Guidelines

- **Python**: 3.11+, async/await, type hints
- **Imports**: Group stdlib, third-party, local
- **Docstrings**: Google style
- **Line Length**: 100 chars max
- **Naming**: snake_case for functions/vars, PascalCase for classes

## Environment Variables

```bash
MONGODB_URI=mongodb+srv://...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=pa-...
```

## Key Dependencies

- `motor` - Async MongoDB driver
- `langchain` / `langchain-core` - LLM orchestration
- `langgraph` - Agent state graphs
- `voyageai` - Embeddings (1024 dimensions)
- `pydantic` - Data validation

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/memory/test_entity.py -v
```

## Common Tasks

### Adding a New Memory Type
1. Add to `MemoryType` enum in `base.py`
2. Create store class in `memory/`
3. Register in `MemoryManager.__init__`
4. Create MongoDB collection with vector index

### Modifying Entity Extraction
Edit `src/memory/entity.py`:
- `ENTITY_TYPES` - Add/remove entity categories
- `EXTRACTION_PROMPT` - Modify LLM prompt
- `ExtractionConfig` - Change model/provider

### Context Compression Settings
Edit `src/context/engineer.py`:
- `MODEL_TOKEN_LIMITS` - Add model limits
- `DEFAULT_THRESHOLD` - Change compression trigger (default 0.80)
