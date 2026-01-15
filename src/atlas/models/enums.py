"""Canonical enums for the Atlas workflow system.

This module defines all enumerations used throughout the system,
including status models, work types, artifact types, and severity levels.

The status model follows the canonical transitions defined in the spec:
- NEW -> READY (when eligible to claim)
- READY -> IN_PROGRESS (when claimed by worker)
- IN_PROGRESS -> DONE | FAILED (on completion)
- BLOCKED -> READY (when dependencies met)
- Any -> CANCELED (optional, when no longer needed)
"""

from enum import Enum, auto


class WorkItemStatus(str, Enum):
    """Canonical status model for work items.

    Ticketing systems vary. Map your internal statuses to this canonical model.

    Status Transitions:
        NEW -> READY: When work item becomes eligible to claim
        READY -> IN_PROGRESS: When claimed/leased by a worker
        IN_PROGRESS -> DONE: Completed successfully with valid outputs
        IN_PROGRESS -> FAILED: Completed with error; may be retried
        BLOCKED -> READY: When all dependencies are DONE
        Any -> CANCELED: When no longer needed (optional)

    Requirements:
        - A worker MUST NOT start work on BLOCKED items unless it can
          unblock them deterministically.
        - Merge and patch merge items MUST be BLOCKED until all required
          inputs are DONE.
    """

    NEW = "new"
    """Created, not eligible to claim."""

    READY = "ready"
    """Eligible to claim by a worker."""

    IN_PROGRESS = "in_progress"
    """Claimed/leased by a worker."""

    BLOCKED = "blocked"
    """Waiting for dependencies/inputs."""

    DONE = "done"
    """Completed successfully with valid outputs."""

    FAILED = "failed"
    """Completed with error; may be retried."""

    CANCELED = "canceled"
    """No longer needed (optional)."""

    def can_transition_to(self, target: "WorkItemStatus") -> bool:
        """Check if transition to target status is valid.

        Args:
            target: The target status to transition to.

        Returns:
            True if transition is valid, False otherwise.
        """
        valid_transitions: dict[WorkItemStatus, set[WorkItemStatus]] = {
            WorkItemStatus.NEW: {WorkItemStatus.READY, WorkItemStatus.CANCELED},
            WorkItemStatus.READY: {WorkItemStatus.IN_PROGRESS, WorkItemStatus.BLOCKED, WorkItemStatus.CANCELED},
            WorkItemStatus.IN_PROGRESS: {WorkItemStatus.DONE, WorkItemStatus.FAILED, WorkItemStatus.CANCELED},
            WorkItemStatus.BLOCKED: {WorkItemStatus.READY, WorkItemStatus.CANCELED},
            WorkItemStatus.DONE: {WorkItemStatus.CANCELED},  # Rare, but possible
            WorkItemStatus.FAILED: {WorkItemStatus.READY, WorkItemStatus.CANCELED},  # Retry
            WorkItemStatus.CANCELED: set(),  # Terminal state
        }
        return target in valid_transitions.get(self, set())


class WorkItemType(str, Enum):
    """Canonical work types (ticket types) for the workflow.

    Systems may rename these; payload schemas MUST be preserved.
    """

    DOC_REQUEST = "doc_request"
    """Root request: Generate documentation for an artifact."""

    DOC_PLAN = "doc_plan"
    """Optional separate planning work to produce a manifest."""

    DOC_CHUNK = "doc_chunk"
    """Analyze a specific chunk of the artifact."""

    DOC_MERGE = "doc_merge"
    """Merge chunk results or child merges into higher-level summary."""

    DOC_CHALLENGE = "doc_challenge"
    """Review documentation and raise questions/issues."""

    DOC_FOLLOWUP = "doc_followup"
    """Answer a specific challenger question with scoped analysis."""

    DOC_PATCH_MERGE = "doc_patch_merge"
    """Apply follow-up answers to documentation and update doc model."""

    DOC_FINALIZE = "doc_finalize"
    """Optional: Produce final deliverables and mark job accepted."""


class ArtifactType(str, Enum):
    """Types of source artifacts that can be analyzed.

    These represent the primary input types for the analysis workflow.
    """

    COBOL = "cobol"
    """COBOL source program."""

    COPYBOOK = "copybook"
    """COBOL copybook (included code)."""

    JCL = "jcl"
    """Job Control Language script."""

    OTHER = "other"
    """Other artifact type."""


class ChunkKind(str, Enum):
    """Classification of chunk content types.

    Used to support targeted follow-ups and semantic chunking.
    The planner creates chunk kinds to enable the challenger
    to route questions to appropriate scopes.
    """

    # COBOL-specific chunk kinds
    IDENTIFICATION_DIVISION = "identification_division"
    """COBOL IDENTIFICATION DIVISION."""

    ENVIRONMENT_DIVISION = "environment_division"
    """COBOL ENVIRONMENT DIVISION."""

    DATA_DIVISION = "data_division"
    """COBOL DATA DIVISION."""

    PROCEDURE_DIVISION = "procedure_division"
    """COBOL PROCEDURE DIVISION (may be split into parts)."""

    PROCEDURE_PART = "procedure_part"
    """Subset of PROCEDURE DIVISION paragraphs."""

    WORKING_STORAGE = "working_storage"
    """WORKING-STORAGE SECTION of DATA DIVISION."""

    FILE_SECTION = "file_section"
    """FILE SECTION of DATA DIVISION."""

    LINKAGE_SECTION = "linkage_section"
    """LINKAGE SECTION of DATA DIVISION."""

    # Generic chunk kinds
    GENERIC = "generic"
    """Generic chunk (line-based split)."""

    MIXED = "mixed"
    """Mixed content from multiple divisions/sections."""


class IssueSeverity(str, Enum):
    """Severity levels for challenger issues.

    Used to prioritize issues and determine workflow actions.
    """

    BLOCKER = "blocker"
    """Critical issue that must be resolved before acceptance."""

    MAJOR = "major"
    """Significant issue that should be addressed."""

    MINOR = "minor"
    """Minor issue or improvement suggestion."""

    QUESTION = "question"
    """Clarification question, not necessarily an issue."""
