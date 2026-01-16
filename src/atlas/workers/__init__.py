"""Abstract worker interfaces for analysis agents.

Workers are the agents that perform the actual analysis work:
- Scribe: Analyzes individual chunks
- Aggregator: Merges chunk results
- Challenger: Reviews documentation and raises issues

These are abstract interfaces (ports). The integrating system implements
concrete workers that wrap their agents. Atlas does NOT provide worker
implementations - the integrating system's agents handle all LLM interactions.

Design Principles:
- Workers claim READY work items
- Workers write structured output artifacts
- Workers record open questions when context is insufficient
- Integrator owns the LLM - Atlas just orchestrates
"""

from atlas.workers.base import Worker
from atlas.workers.scribe import Scribe
from atlas.workers.aggregator import Aggregator
from atlas.workers.challenger import Challenger

__all__ = [
    "Worker",
    "Scribe",
    "Aggregator",
    "Challenger",
]
