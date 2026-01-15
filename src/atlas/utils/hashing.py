"""Hashing utilities for content addressing and idempotency.

Provides consistent hashing for:
- Content-addressable storage
- Idempotency keys for work items
- Artifact versioning

The idempotency key system ensures ticket creation is idempotent
as required by the spec (Section 12.1):

    For any planned work item, controller MUST compute a stable
    idempotency key from:
    - job_id
    - work_type
    - artifact_version
    - (chunk_id or merge_node_id or issue_id)
"""

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atlas.models.work_item import WorkItem
    from atlas.models.enums import WorkItemType


def compute_content_hash(content: bytes | str) -> str:
    """Compute SHA-256 hash of content.

    Used for:
    - Artifact versioning (content-addressable storage)
    - Detecting content changes
    - Generating deterministic URIs

    Args:
        content: Content to hash (bytes or string).

    Returns:
        Hex-encoded SHA-256 hash (64 characters).

    Example:
        >>> compute_content_hash("hello world")
        'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def compute_idempotency_key(*parts: str) -> str:
    """Compute idempotency key from string parts.

    Creates a stable, deterministic key for work item deduplication.
    Parts are joined with ':' separator before hashing.

    Args:
        *parts: String parts to combine.

    Returns:
        Hex-encoded SHA-256 hash (64 characters).

    Example:
        >>> compute_idempotency_key("job-123", "doc_chunk", "abc123", "chunk-001")
        '...'  # deterministic hash

    Note:
        Use compute_work_item_idempotency_key() for work items as it
        handles type-specific identifier extraction automatically.
    """
    # Filter out None and empty values, convert all to string
    filtered_parts = [str(p) for p in parts if p]
    combined = ":".join(filtered_parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def compute_work_item_idempotency_key(
    job_id: str,
    work_type: "WorkItemType | str",
    artifact_version: str,
    identifier: str | None = None,
    *,
    chunk_id: str | None = None,
    merge_node_id: str | None = None,
    issue_id: str | None = None,
    cycle_number: int | None = None,
) -> str:
    """Compute idempotency key for a work item.

    Implements the idempotency key specification from Section 12.1:
    - job_id
    - work_type
    - artifact_version
    - type-specific identifier (chunk_id, merge_node_id, or issue_id)
    - optional cycle_number for challenger loop iterations

    Args:
        job_id: Unique job identifier.
        work_type: Work type (DOC_CHUNK, DOC_MERGE, etc.).
        artifact_version: Source artifact version/hash.
        identifier: Generic identifier (used if specific ids not provided).
        chunk_id: Chunk identifier (for DOC_CHUNK).
        merge_node_id: Merge node identifier (for DOC_MERGE).
        issue_id: Issue identifier (for DOC_FOLLOWUP, DOC_CHALLENGE).
        cycle_number: Iteration number for challenger loops.

    Returns:
        Hex-encoded SHA-256 hash (64 characters).

    Example:
        >>> compute_work_item_idempotency_key(
        ...     job_id="job-123",
        ...     work_type="doc_chunk",
        ...     artifact_version="abc123def456",
        ...     chunk_id="procedure_part_001"
        ... )
        '...'  # deterministic hash

    Note:
        The key is stable and deterministic - the same inputs will
        always produce the same key, enabling idempotent creation.
    """
    # Convert work_type to string if it's an enum
    if hasattr(work_type, "value"):
        work_type_str = work_type.value
    else:
        work_type_str = str(work_type)

    # Determine the type-specific identifier
    specific_id = chunk_id or merge_node_id or issue_id or identifier

    # Build parts list
    parts = [job_id, work_type_str, artifact_version]

    if specific_id:
        parts.append(specific_id)

    if cycle_number is not None:
        parts.append(f"cycle:{cycle_number}")

    return compute_idempotency_key(*parts)


def compute_work_item_key_from_work_item(work_item: "WorkItem") -> str:
    """Compute idempotency key from a WorkItem instance.

    Extracts the relevant fields from the work item and computes
    the idempotency key automatically.

    Args:
        work_item: The work item to generate a key for.

    Returns:
        Hex-encoded SHA-256 hash (64 characters).

    Example:
        >>> from atlas.models.work_item import WorkItem
        >>> key = compute_work_item_key_from_work_item(work_item)
    """
    # Import here to avoid circular dependency
    from atlas.models.work_item import (
        DocChunkPayload,
        DocMergePayload,
        DocFollowupPayload,
        DocChallengePayload,
    )

    payload = work_item.payload

    # Extract artifact version
    artifact_version = ""
    if payload.artifact_ref:
        artifact_version = payload.artifact_ref.artifact_version

    # Extract type-specific identifier
    chunk_id = None
    merge_node_id = None
    issue_id = None

    if isinstance(payload, DocChunkPayload):
        chunk_id = payload.chunk_id
    elif isinstance(payload, DocMergePayload):
        merge_node_id = payload.merge_node_id
    elif isinstance(payload, DocFollowupPayload):
        issue_id = payload.issue_id
    elif isinstance(payload, DocChallengePayload):
        # Challenge tickets use output_uri as identifier since they're per-doc
        issue_id = payload.output_uri

    return compute_work_item_idempotency_key(
        job_id=payload.job_id,
        work_type=work_item.work_type,
        artifact_version=artifact_version,
        chunk_id=chunk_id,
        merge_node_id=merge_node_id,
        issue_id=issue_id,
        cycle_number=work_item.cycle_number,
    )


def compute_artifact_uri(
    base_uri: str,
    artifact_id: str,
    version: str,
    suffix: str = "",
) -> str:
    """Compute deterministic artifact URI.

    Generates a predictable URI for artifact storage based on
    the artifact identity and version.

    Args:
        base_uri: Base storage URI (e.g., "s3://bucket/artifacts").
        artifact_id: Artifact identifier.
        version: Version hash or identifier.
        suffix: Optional suffix (e.g., ".json").

    Returns:
        Full artifact URI.

    Example:
        >>> compute_artifact_uri(
        ...     "s3://bucket",
        ...     "DRKBM100.cbl",
        ...     "abc123",
        ...     ".json"
        ... )
        's3://bucket/DRKBM100.cbl@abc123.json'
    """
    # Sanitize artifact_id for URI use
    safe_id = artifact_id.replace(" ", "_")
    return f"{base_uri.rstrip('/')}/{safe_id}@{version}{suffix}"


def compute_result_uri(
    base_uri: str,
    job_id: str,
    result_type: str,
    identifier: str,
    cycle_number: int = 1,
    suffix: str = ".json",
) -> str:
    """Compute deterministic result URI.

    Generates a predictable URI for workflow results (chunk results,
    merge results, etc.).

    Args:
        base_uri: Base storage URI.
        job_id: Job identifier.
        result_type: Type of result (chunks, merges, etc.).
        identifier: Result identifier (chunk_id, merge_node_id).
        cycle_number: Iteration number for challenger loops.
        suffix: File suffix.

    Returns:
        Full result URI.

    Example:
        >>> compute_result_uri(
        ...     "s3://results",
        ...     "job-123",
        ...     "chunks",
        ...     "procedure_part_001"
        ... )
        's3://results/job-123/cycle-1/chunks/procedure_part_001.json'
    """
    return (
        f"{base_uri.rstrip('/')}/{job_id}/cycle-{cycle_number}/"
        f"{result_type}/{identifier}{suffix}"
    )


def content_hash_short(content: bytes | str, length: int = 8) -> str:
    """Compute a shortened content hash.

    Useful for display purposes or when a full hash is too long.

    Args:
        content: Content to hash.
        length: Number of characters to return (max 64).

    Returns:
        Truncated hex-encoded SHA-256 hash.

    Example:
        >>> content_hash_short("hello world")
        'b94d27b9'
    """
    full_hash = compute_content_hash(content)
    return full_hash[:min(length, 64)]
