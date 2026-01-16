# Atlas

Orchestration core for chunked program analysis workflow.

## Overview

Atlas is a Python orchestration framework that coordinates analysis of large legacy assets (COBOL programs, copybooks, JCL, etc.) that exceed LLM context limits. It provides:

- **Deterministic chunking** of large source files
- **Hierarchical merge DAG** for consolidating chunk analyses
- **Reconciliation loop** for advancing work through phases
- **Abstract interfaces** for integration with external systems

Atlas is designed to be integrated into existing agent-based systems, not to run standalone. It provides orchestration machinery; the integrating system provides the actual analysis agents and LLM access.

## When to Use Atlas

Atlas adds value when source files exceed the context budget of your LLM:

| File Size | Approach |
|-----------|----------|
| Small (fits in context) | Direct agent analysis - Atlas not needed |
| Large (exceeds context) | Atlas chunking → hierarchical merge → aggregation |

The integrating system decides when to route files through Atlas based on token count.

## Architecture

Atlas follows the **ports and adapters** (hexagonal) architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                         Atlas                                │
│            (Orchestration Framework)                         │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Controller   │  │ Planner      │  │ Splitter     │       │
│  │ Reconciler   │  │ DAG Builder  │  │ COBOL/Line   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
│  ┌─────────────────────────────────────────────────┐        │
│  │         Abstract Interfaces (Ports)             │        │
│  │  • TicketSystemAdapter                          │        │
│  │  • ArtifactStoreAdapter                         │        │
│  │  • Worker (abstract base)                       │        │
│  └─────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ implements
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Integrating System                         │
│           (Your agent framework)                             │
│                                                              │
│  ┌─────────────────────────────────────────────────┐        │
│  │         Concrete Adapters                        │        │
│  │  • YourTicketAdapter : TicketSystemAdapter      │        │
│  │  • YourStoreAdapter : ArtifactStoreAdapter      │        │
│  │  • YourScribeWorker : Worker                    │        │
│  │  • YourChallengerWorker : Worker                │        │
│  └─────────────────────────────────────────────────┘        │
│                                                              │
│  ┌──────────────────────────────────────────────┐           │
│  │     Your Agents (with LLM access)            │           │
│  │  • Scribe, Challenger, Reviewer, etc.        │           │
│  └──────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

**Atlas does NOT call LLMs directly.** The integrating system's agents handle all LLM interactions. Atlas workers are thin wrappers that delegate to your agents.

## Integration Requirements

To integrate Atlas, your system must implement:

### 1. TicketSystemAdapter

Wraps your ticket/work-item system (Jira, Linear, custom, etc.):

```python
from atlas.adapters.ticket_system import TicketSystemAdapter

class YourTicketAdapter(TicketSystemAdapter):
    def __init__(self, your_client):
        self.client = your_client

    async def create_work_item(self, work_item: WorkItem) -> str:
        # Create ticket in your system
        ...

    async def update_status(self, work_id: str, status: WorkItemStatus) -> None:
        # Update ticket status
        ...

    # ... implement other abstract methods
```

### 2. ArtifactStoreAdapter

Wraps your artifact storage (filesystem, S3, database, etc.):

```python
from atlas.adapters.artifact_store import ArtifactStoreAdapter

class YourStoreAdapter(ArtifactStoreAdapter):
    async def store(self, artifact_id: str, content: bytes) -> str:
        # Store artifact, return URI
        ...

    async def retrieve(self, uri: str) -> bytes:
        # Retrieve artifact by URI
        ...

    # ... implement other abstract methods
```

### 3. Worker Implementations

Wrap your existing agents as Atlas workers:

```python
from atlas.workers.base import Worker

class YourScribeWorker(Worker):
    def __init__(self, your_scribe_agent):
        self.scribe = your_scribe_agent  # Your agent handles LLM

    async def analyze_chunk(self, content, payload, manifest) -> ChunkResult:
        # Delegate to your agent
        result = await self.scribe.analyze(content)
        # Convert to Atlas model
        return ChunkResult(...)
```

### 4. Aggregator as a Scribe Task

The Aggregator role (merging chunk results) is **not a separate agent type**. It's a different task for your documentation agent:

| Task | Prompt | Agent |
|------|--------|-------|
| Analyze chunk | "Document this code section" | Scribe |
| Merge results | "Consolidate these analyses into unified documentation" | Scribe (different prompt) |

```python
class YourAggregatorWorker(Worker):
    def __init__(self, your_scribe_agent):
        self.scribe = your_scribe_agent  # Same agent, different prompt

    async def merge_results(self, chunk_results, payload, manifest) -> MergeResult:
        # Use scribe with merge prompt
        return await self.scribe.merge(chunk_results)
```

### 5. Routing Logic

Your system decides when to use Atlas:

```python
async def process_file(file_path: str):
    content = read_file(file_path)
    token_count = estimate_tokens(content)

    if token_count <= CONTEXT_BUDGET:
        # Small file - direct to your agent
        return await your_scribe.analyze(content)
    else:
        # Large file - use Atlas for chunking/merge
        return await atlas_controller.process(file_path)
```

## Key Design Principles

1. **Artifacts are the source of truth** - Tickets are pointers, not data stores
2. **Deterministic chunking** - Same source + profile = same chunks
3. **Bounded context** - No task exceeds configured context budget
4. **Structured outputs** - JSON/YAML for reliable aggregation
5. **Explicit uncertainty** - Workers record unknowns, never guess
6. **Reconcile loop** - Controller observes state and advances work
7. **Integrator owns LLM** - Atlas orchestrates; your agents call LLMs

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
│   ├── adapters/        # Abstract interfaces (ports)
│   │   ├── ticket_system.py  # Ticket system port
│   │   └── artifact_store.py # Artifact storage port
│   ├── splitter/        # Source code chunking
│   │   ├── base.py      # Abstract splitter interface
│   │   ├── registry.py  # Splitter registry
│   │   ├── cobol.py     # COBOL-aware splitter
│   │   └── line_based.py # Fallback line-based splitter
│   ├── planner/         # Workflow planning and DAG construction
│   ├── controller/      # Workflow orchestration
│   │   ├── base.py      # Abstract controller
│   │   └── reconciler.py # Reconciliation loop
│   ├── workers/         # Abstract worker interfaces
│   ├── config/          # Configuration management
│   └── observability/   # Logging, metrics, events
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
7. **Finalize** - Generate final deliverables and trace report

## Installation

```bash
pip install -e ".[dev]"
```

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
```

## License

MIT
