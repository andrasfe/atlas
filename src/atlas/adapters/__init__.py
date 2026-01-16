"""Adapter interfaces for external systems.

This module defines abstract base classes for integrating with:
- Ticket systems (any vendor or custom)
- Artifact stores (S3, local filesystem, etc.)

All adapters follow the principle that the core system should be
agnostic to the specific implementations. Integrating systems implement
these interfaces to connect Atlas to their infrastructure.

Note: Atlas does NOT include an LLM adapter. The integrating system's
agents handle all LLM interactions directly.

Design Principle:
    Works with any ticket system via adapters; no hard-coded ticket schema.
"""

from atlas.adapters.ticket_system import TicketSystemAdapter
from atlas.adapters.artifact_store import ArtifactStoreAdapter
from atlas.adapters.memory_ticket_system import (
    MemoryTicketSystem,
    WorkItemNotFoundError,
    InvalidStatusTransitionError,
    DuplicateWorkItemError,
)
from atlas.adapters.filesystem_store import (
    FilesystemArtifactStore,
    ArtifactNotFoundError,
    ArtifactWriteError,
    ArtifactReadError,
)

__all__ = [
    # Abstract adapters (ports)
    "TicketSystemAdapter",
    "ArtifactStoreAdapter",
    # Reference implementations
    "MemoryTicketSystem",
    "FilesystemArtifactStore",
    # Exceptions
    "WorkItemNotFoundError",
    "InvalidStatusTransitionError",
    "DuplicateWorkItemError",
    "ArtifactNotFoundError",
    "ArtifactWriteError",
    "ArtifactReadError",
]
