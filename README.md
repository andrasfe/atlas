# Atlas

Orchestration core for chunked program analysis workflow.

## Overview

Atlas is a Python orchestration system that coordinates analysis of large legacy assets (COBOL programs, copybooks, JCL, etc.) using:

- A ticket/work-item system (any vendor or custom)
- Multiple workers/agents ("Scribes") for chunk analysis
- A merge agent ("Aggregator") for hierarchical merging
- A review agent ("Challenger") for quality review and follow-up questions
- Tight per-agent context limits (e.g., 4k tokens)

## Key Design Principles

1. **Artifacts are the source of truth** - Tickets are pointers, not data stores
2. **Deterministic chunking** - Same source + profile = same chunks
3. **Bounded context** - No task exceeds configured context budget
4. **Structured outputs** - JSON/YAML for reliable aggregation
5. **Explicit uncertainty** - Workers record unknowns, never guess
6. **Reconcile loop** - Controller observes state and advances work

## Installation

```bash
# Install in development mode
pip install -e ".[dev]"
```

## Project Structure

```
atlas/
├── src/atlas/
│   ├── models/          # Pydantic data models
│   │   ├── enums.py     # Status, type, and kind enumerations
│   │   ├── artifact.py  # Artifact and ArtifactRef models
│   │   ├── work_item.py # WorkItem and payload models
│   │   ├── manifest.py  # Manifest and planning models
│   │   └── results.py   # Output artifact models
│   ├── adapters/        # Abstract interfaces for external systems
│   │   ├── ticket_system.py  # Ticket system adapter
│   │   ├── artifact_store.py # Artifact storage adapter
│   │   └── llm.py            # LLM provider adapter
│   ├── splitter/        # Source code chunking
│   ├── planner/         # Workflow planning and DAG construction
│   ├── controller/      # Workflow orchestration
│   ├── workers/         # Analysis agents
│   │   ├── scribe.py    # Chunk analysis worker
│   │   ├── aggregator.py # Merge worker
│   │   └── challenger.py # Review worker
│   ├── config/          # Configuration management
│   └── utils/           # Shared utilities
└── tests/
    ├── unit/            # Unit tests
    └── integration/     # Integration tests
```

## Workflow Phases

1. **Request & Plan** - Create manifest with chunks and merge DAG
2. **Chunk Analysis** - Scribes analyze individual chunks
3. **Hierarchical Merge** - Aggregators merge results bottom-up
4. **Challenger Review** - Challenger reviews and raises issues
5. **Follow-up Dispatch** - Controller routes issues to follow-ups
6. **Re-challenge Loop** - Optional iteration until acceptance

## Development

```bash
# Run tests
pytest

# Type checking
mypy src/atlas

# Linting
ruff check src/atlas

# Format code
ruff format src/atlas
```

## Configuration

Configuration is managed through environment variables with the `ATLAS_` prefix:

```bash
ATLAS_LOG_LEVEL=INFO
ATLAS_CONTEXT_BUDGET=4000
ATLAS_MAX_CHUNK_TOKENS=3500
ATLAS_LLM_PROVIDER=openai
ATLAS_LLM_MODEL=gpt-4
```

## License

MIT
