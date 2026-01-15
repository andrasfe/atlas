"""Atlas - Orchestration core for chunked program analysis workflow.

This package provides a Python orchestration core ("Controller") that coordinates
analysis of large legacy assets (COBOL programs, copybooks, JCL, etc.) using:

- A ticket/work-item system (any vendor or custom)
- Multiple workers/agents ("Scribes")
- A merge agent ("Aggregator")
- A review agent ("Challenger") that can request clarifications after the first merge
- Tight per-agent context limits (e.g., 4k tokens)

Key Design Principles:
- Artifacts are the source of truth; tickets are pointers
- Deterministic chunking per artifact version
- Bounded context per task
- Structured outputs (JSON/YAML)
- Explicit uncertainty handling
- Controller reconciles desired state

See spec.md for the full specification.
"""

__version__ = "0.1.0"
