"""Splitter interfaces for chunking source artifacts.

The splitter module provides a plugin architecture for breaking source
artifacts into analyzable chunks that fit within context budgets while
preserving semantic boundaries where possible.

Components:
- Splitter: Abstract base class defining the splitter interface
- SplitResult: Result of a splitting operation
- SplitterRegistry: Registry for mapping artifact types to splitters
- COBOLSplitter: COBOL-aware splitter with semantic boundary detection
- LineBasedSplitter: Fallback splitter using simple line-count chunking

Design Principles:
- Deterministic chunking: Same source + profile = same chunks
- Plugin architecture: Register splitters by artifact type
- Fallback support: Unknown types use LineBasedSplitter
- Prefer semantic boundaries (divisions/sections/paragraphs)
- Ensure chunks fit within context budget
- Create chunk kinds to support targeted follow-ups

Usage:
    >>> from atlas.splitter import get_default_registry
    >>> registry = get_default_registry()
    >>> splitter = registry.get_splitter("cobol")
    >>> result = splitter.split(source, profile, "program.cbl")

Or directly:
    >>> from atlas.splitter import COBOLSplitter, LineBasedSplitter
    >>> splitter = COBOLSplitter()
    >>> result = splitter.split(source, profile, "program.cbl")
"""

from atlas.splitter.base import Splitter, SplitResult
from atlas.splitter.cobol import COBOLSplitter, COBOLStructure, SemanticBoundary
from atlas.splitter.line_based import LineBasedSplitter
from atlas.splitter.registry import (
    SplitterRegistry,
    SplitterNotFoundError,
    get_default_registry,
    reset_default_registry,
)

__all__ = [
    # Base classes
    "Splitter",
    "SplitResult",
    # Registry
    "SplitterRegistry",
    "SplitterNotFoundError",
    "get_default_registry",
    "reset_default_registry",
    # Concrete implementations
    "COBOLSplitter",
    "COBOLStructure",
    "SemanticBoundary",
    "LineBasedSplitter",
]
