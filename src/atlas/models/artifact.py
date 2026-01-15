"""Artifact models for versioned input/output objects.

Artifacts are the source of truth in the Atlas system. Tickets are merely
pointers to artifacts stored in an artifact store.

This module defines:
- ArtifactRef: Lightweight reference to an artifact
- Artifact: Full artifact metadata
"""

from typing import Any

from pydantic import BaseModel, Field


class ArtifactRef(BaseModel):
    """Lightweight reference to an artifact.

    Used in payloads and manifests to reference artifacts without
    embedding the full metadata. This keeps tickets small.

    Attributes:
        artifact_id: Stable logical name (e.g., "DRKBM100.cbl").
        artifact_type: Type of artifact (COBOL, COPYBOOK, JCL, OTHER).
        artifact_version: Hash/commit/content hash for version pinning.
        artifact_uri: URI to fetch the artifact content.

    Example:
        >>> ref = ArtifactRef(
        ...     artifact_id="DRKBM100.cbl",
        ...     artifact_type="cobol",
        ...     artifact_version="abc123",
        ...     artifact_uri="s3://bucket/sources/DRKBM100.cbl@abc123"
        ... )
    """

    artifact_id: str = Field(
        ...,
        description="Stable logical name (e.g., DRKBM100.cbl)",
    )
    artifact_type: str = Field(
        ...,
        description="Type of artifact: cobol, copybook, jcl, other",
    )
    artifact_version: str = Field(
        ...,
        description="Hash/commit/content hash for version pinning",
    )
    artifact_uri: str = Field(
        ...,
        description="URI to fetch the artifact content",
    )


class Artifact(BaseModel):
    """A versioned input or output object stored in an artifact store.

    Artifacts represent both source inputs (COBOL programs, copybooks, JCL)
    and workflow outputs (chunk results, merge results, documentation).

    Design Principle:
        Artifacts are the source of truth; tickets are pointers.
        Tickets MUST NOT embed large objects. They should reference
        manifest + artifact URIs instead.

    Attributes:
        artifact_id: Stable logical name (e.g., "DRKBM100.cbl").
        artifact_type: Type of artifact (COBOL, COPYBOOK, JCL, OTHER).
        artifact_version: Hash/commit/content hash for version pinning.
        artifact_uri: URI to fetch the artifact content.
        metadata: Optional additional metadata (size, repo path, etc.).

    Example:
        >>> artifact = Artifact(
        ...     artifact_id="DRKBM100.cbl",
        ...     artifact_type="cobol",
        ...     artifact_version="abc123def456",
        ...     artifact_uri="s3://artifacts/DRKBM100.cbl@abc123def456",
        ...     metadata={"size_bytes": 45000, "repo_path": "src/cobol/DRKBM100.cbl"}
        ... )
    """

    artifact_id: str = Field(
        ...,
        description="Stable logical name (e.g., DRKBM100.cbl)",
    )
    artifact_type: str = Field(
        ...,
        description="Type of artifact: cobol, copybook, jcl, other",
    )
    artifact_version: str = Field(
        ...,
        description="Hash/commit/content hash for version pinning",
    )
    artifact_uri: str = Field(
        ...,
        description="URI to fetch the artifact content",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata (size, repo path, etc.)",
    )

    def to_ref(self) -> ArtifactRef:
        """Convert to a lightweight reference.

        Returns:
            ArtifactRef with the core identifying fields.
        """
        return ArtifactRef(
            artifact_id=self.artifact_id,
            artifact_type=self.artifact_type,
            artifact_version=self.artifact_version,
            artifact_uri=self.artifact_uri,
        )
