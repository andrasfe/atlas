"""COBOL splitter interfaces for chunking source artifacts.

The splitter is responsible for breaking source artifacts into
analyzable chunks that fit within context budgets while preserving
semantic boundaries where possible.

Design Principles:
- Deterministic chunking: Same source + profile = same chunks
- Prefer semantic boundaries (divisions/sections/paragraphs)
- Ensure chunks fit within context budget
- Create chunk kinds to support targeted follow-ups
"""

from atlas.splitter.base import Splitter, SplitResult
from atlas.splitter.cobol import COBOLSplitter, COBOLStructure, SemanticBoundary

__all__ = [
    "Splitter",
    "SplitResult",
    "COBOLSplitter",
    "COBOLStructure",
    "SemanticBoundary",
]
