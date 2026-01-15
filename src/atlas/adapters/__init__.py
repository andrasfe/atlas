"""Adapter interfaces for external systems.

This module defines abstract base classes for integrating with:
- Ticket systems (any vendor or custom)
- Artifact stores (S3, local filesystem, etc.)
- LLM providers (OpenAI, Anthropic, etc.)

All adapters follow the principle that the core system should be
agnostic to the specific implementations.

Design Principle:
    Works with any ticket system via adapters; no hard-coded ticket schema.
"""

from atlas.adapters.ticket_system import TicketSystemAdapter
from atlas.adapters.artifact_store import ArtifactStoreAdapter
from atlas.adapters.llm import LLMAdapter, LLMResponse, LLMMessage
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
    # Abstract adapters
    "TicketSystemAdapter",
    "ArtifactStoreAdapter",
    "LLMAdapter",
    "LLMResponse",
    "LLMMessage",
    # Concrete implementations
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
