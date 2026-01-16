"""Worker interfaces for analysis agents.

Workers are the agents that perform the actual analysis work:
- Scribe: Analyzes individual chunks
- Aggregator: Merges chunk results
- Challenger: Reviews documentation and raises issues
- PatchMerge: Applies follow-up answers to documentation
- Finalize: Produces final deliverables and marks job complete

Design Principles:
- Workers claim READY work items
- Workers write structured output artifacts
- Workers record open questions when context is insufficient
"""

from atlas.workers.base import Worker
from atlas.workers.scribe import Scribe
from atlas.workers.aggregator import Aggregator
from atlas.workers.challenger import Challenger
from atlas.workers.patch_merge_impl import PatchMergeWorker
from atlas.workers.finalize_impl import FinalizeWorker

__all__ = [
    "Worker",
    "Scribe",
    "Aggregator",
    "Challenger",
    "PatchMergeWorker",
    "FinalizeWorker",
]
