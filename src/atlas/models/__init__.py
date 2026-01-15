"""Core data models for Atlas.

This module contains all Pydantic models for:
- Artifacts (versioned input/output objects)
- Work Items (tickets representing units of work)
- Manifests (workflow plans and relationships)
- Result artifacts (chunk results, merge results, etc.)

All models follow the canonical entity definitions from the spec.
"""

from atlas.models.enums import (
    ArtifactType,
    ChunkKind,
    IssueSeverity,
    WorkItemStatus,
    WorkItemType,
)
from atlas.models.artifact import Artifact, ArtifactRef
from atlas.models.work_item import WorkItem, WorkItemPayload
from atlas.models.manifest import (
    Manifest,
    ChunkSpec,
    MergeNode,
    ReviewPolicy,
    SplitterProfile,
    AnalysisProfile,
)
from atlas.models.results import (
    ChunkResult,
    MergeResult,
    ChallengeResult,
    FollowupAnswer,
    DocumentationModel,
    Evidence,
    SymbolDef,
    IOOperation,
    ErrorHandlingPattern,
    Issue,
    ResolutionPlan,
    Section,
)

__all__ = [
    # Enums
    "ArtifactType",
    "ChunkKind",
    "IssueSeverity",
    "WorkItemStatus",
    "WorkItemType",
    # Core entities
    "Artifact",
    "ArtifactRef",
    "WorkItem",
    "WorkItemPayload",
    "Manifest",
    "ChunkSpec",
    "MergeNode",
    "ReviewPolicy",
    "SplitterProfile",
    "AnalysisProfile",
    # Results
    "ChunkResult",
    "MergeResult",
    "ChallengeResult",
    "FollowupAnswer",
    "DocumentationModel",
    "Evidence",
    "SymbolDef",
    "IOOperation",
    "ErrorHandlingPattern",
    "Issue",
    "ResolutionPlan",
    "Section",
]
