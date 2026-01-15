"""Work Item (Ticket) models for tracking units of work.

A WorkItem represents a unit of work tracked in a ticketing system.
Work items are lightweight pointers that reference artifacts and manifests,
keeping the ticket payload small and bounded.

This module defines:
- WorkItemPayload: Type-specific payload schemas
- WorkItem: Full work item structure
"""

from typing import Any

from pydantic import BaseModel, Field

from atlas.models.enums import WorkItemStatus, WorkItemType
from atlas.models.artifact import ArtifactRef


class ChunkLocator(BaseModel):
    """Locator for a chunk within an artifact.

    Can specify either line ranges or semantic locators.

    Attributes:
        start_line: Starting line number (1-indexed).
        end_line: Ending line number (inclusive).
        division: COBOL division name (if semantic).
        section: COBOL section name (if semantic).
        paragraphs: List of paragraph names (if semantic).
    """

    start_line: int | None = Field(default=None, description="Starting line (1-indexed)")
    end_line: int | None = Field(default=None, description="Ending line (inclusive)")
    division: str | None = Field(default=None, description="COBOL division name")
    section: str | None = Field(default=None, description="COBOL section name")
    paragraphs: list[str] = Field(default_factory=list, description="Paragraph names")


class WorkItemPayload(BaseModel):
    """Base payload for work items.

    Each work type has specific required fields. This base class
    contains common fields; type-specific payloads extend this.

    Design Principle:
        Payloads MUST include references to artifacts/manifests.
        Payloads MUST NOT embed large objects.

    Attributes:
        job_id: Unique identifier for the analysis run.
        artifact_ref: Reference to the artifact being analyzed.
        manifest_uri: URI to the manifest (optional for DOC_REQUEST).
    """

    job_id: str = Field(..., description="Unique identifier for the analysis run")
    artifact_ref: ArtifactRef | None = Field(
        default=None,
        description="Reference to the artifact being analyzed",
    )
    manifest_uri: str | None = Field(
        default=None,
        description="URI to the manifest document",
    )


class DocRequestPayload(WorkItemPayload):
    """Payload for DOC_REQUEST work items.

    Root request to generate documentation for an artifact.

    Attributes:
        analysis_profile: Documentation template / extraction profile.
        context_budget: Maximum tokens per task.
        splitter_profile: Configuration for the chunking strategy.
    """

    analysis_profile: str = Field(..., description="Documentation template/profile name")
    context_budget: int = Field(default=4000, description="Maximum tokens per task")
    splitter_profile: str = Field(default="default", description="Chunking strategy name")


class DocChunkPayload(WorkItemPayload):
    """Payload for DOC_CHUNK work items.

    Request to analyze a specific chunk of an artifact.

    Attributes:
        chunk_id: Unique identifier for this chunk.
        chunk_locator: Line range or semantic locator.
        result_uri: Where to write the chunk result.
    """

    chunk_id: str = Field(..., description="Unique chunk identifier")
    chunk_locator: ChunkLocator = Field(..., description="Chunk location specification")
    result_uri: str = Field(..., description="URI for output chunk result")


class DocMergePayload(WorkItemPayload):
    """Payload for DOC_MERGE work items.

    Request to merge chunk results or child merges.

    Attributes:
        merge_node_id: Identifier for this merge node in the DAG.
        input_uris: URIs of inputs to merge.
        output_uri: Where to write the merge result.
    """

    merge_node_id: str = Field(..., description="Merge node identifier")
    input_uris: list[str] = Field(..., description="URIs of inputs to merge")
    output_uri: str = Field(..., description="URI for output merge result")


class DocChallengePayload(WorkItemPayload):
    """Payload for DOC_CHALLENGE work items.

    Request to review documentation and raise issues.

    Attributes:
        doc_uri: URI of the merged documentation to review.
        doc_model_uri: URI of the machine-readable doc model.
        challenge_profile: What to look for (error handling, I/O, etc.).
        output_uri: Where to write the challenge result.
    """

    doc_uri: str = Field(..., description="URI of documentation to review")
    doc_model_uri: str = Field(..., description="URI of documentation model")
    challenge_profile: str = Field(..., description="Challenge focus areas")
    output_uri: str = Field(..., description="URI for challenge result output")


class DocFollowupPayload(WorkItemPayload):
    """Payload for DOC_FOLLOWUP work items.

    Request to answer a specific challenger question.

    Attributes:
        issue_id: Identifier of the issue being addressed.
        scope: Chunk IDs, paragraph list, or query plan.
        inputs: URIs of relevant chunk results or source slices.
        output_uri: Where to write the follow-up answer.
    """

    issue_id: str = Field(..., description="Issue being addressed")
    scope: dict[str, Any] = Field(..., description="Scope specification")
    inputs: list[str] = Field(..., description="URIs of relevant inputs")
    output_uri: str = Field(..., description="URI for follow-up answer output")


class DocPatchMergePayload(WorkItemPayload):
    """Payload for DOC_PATCH_MERGE work items.

    Request to apply follow-up answers and update documentation.

    Attributes:
        base_doc_uri: URI of the base documentation.
        base_doc_model_uri: URI of the base doc model.
        inputs: URIs of follow-up answers to apply.
        output_doc_uri: Where to write updated documentation.
        output_doc_model_uri: Where to write updated doc model.
    """

    base_doc_uri: str = Field(..., description="URI of base documentation")
    base_doc_model_uri: str = Field(..., description="URI of base doc model")
    inputs: list[str] = Field(..., description="URIs of follow-up answers")
    output_doc_uri: str = Field(..., description="URI for updated documentation")
    output_doc_model_uri: str = Field(..., description="URI for updated doc model")


class WorkItem(BaseModel):
    """A unit of work tracked in a ticketing system.

    Work items are lightweight containers that track work status and
    reference artifacts through URIs. They should not contain large
    embedded objects.

    Design Principle:
        Tickets are pointers to artifacts. Keep payloads small.
        Use manifest_uri and artifact URIs instead of embedding data.

    Attributes:
        work_id: Unique identifier for this work item.
        work_type: Type of work (DOC_REQUEST, DOC_CHUNK, etc.).
        status: Current status in the workflow.
        payload: Type-specific payload with references.
        parent_work_id: ID of parent work item (if hierarchical).
        depends_on: List of work IDs this item depends on.
        cycle_number: Iteration number for challenger loops.
        idempotency_key: Stable key for deduplication.
        created_at: ISO timestamp of creation.
        updated_at: ISO timestamp of last update.
        metadata: Additional system-specific metadata.

    Example:
        >>> work_item = WorkItem(
        ...     work_id="chunk-001",
        ...     work_type=WorkItemType.DOC_CHUNK,
        ...     status=WorkItemStatus.READY,
        ...     payload=DocChunkPayload(
        ...         job_id="job-123",
        ...         chunk_id="chunk-001",
        ...         chunk_locator=ChunkLocator(start_line=1, end_line=100),
        ...         result_uri="s3://results/chunk-001.json"
        ...     )
        ... )
    """

    work_id: str = Field(..., description="Unique identifier for this work item")
    work_type: WorkItemType = Field(..., description="Type of work")
    status: WorkItemStatus = Field(
        default=WorkItemStatus.NEW,
        description="Current status in the workflow",
    )
    payload: WorkItemPayload = Field(..., description="Type-specific payload")
    parent_work_id: str | None = Field(
        default=None,
        description="Parent work item ID",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Work IDs this item depends on",
    )
    cycle_number: int = Field(
        default=1,
        description="Iteration number for challenger loops",
    )
    idempotency_key: str | None = Field(
        default=None,
        description="Stable key for deduplication",
    )
    created_at: str | None = Field(default=None, description="ISO timestamp of creation")
    updated_at: str | None = Field(default=None, description="ISO timestamp of last update")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional system-specific metadata",
    )

    def can_start(self) -> bool:
        """Check if this work item can be started.

        Returns:
            True if status is READY, False otherwise.
        """
        return self.status == WorkItemStatus.READY

    def compute_idempotency_key(self) -> str:
        """Compute a stable idempotency key for this work item.

        The key is derived from job_id, work_type, artifact_version,
        and type-specific identifiers (chunk_id, merge_node_id, issue_id).

        Returns:
            Stable idempotency key string.
        """
        parts = [
            self.payload.job_id,
            self.work_type.value,
        ]

        if self.payload.artifact_ref:
            parts.append(self.payload.artifact_ref.artifact_version)

        # Add type-specific identifiers
        if isinstance(self.payload, DocChunkPayload):
            parts.append(self.payload.chunk_id)
        elif isinstance(self.payload, DocMergePayload):
            parts.append(self.payload.merge_node_id)
        elif isinstance(self.payload, DocFollowupPayload):
            parts.append(self.payload.issue_id)

        return ":".join(parts)
