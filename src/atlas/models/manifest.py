"""Manifest models for workflow plans and relationships.

A Manifest is a JSON/YAML document stored in the artifact store that
describes the workflow plan and relationships. It includes chunk specs,
merge DAG, and review policies.

This module defines:
- SplitterProfile: Configuration for chunking strategy
- AnalysisProfile: Documentation template / extraction profile
- ChunkSpec: Specification for a single chunk
- MergeNode: Node in the merge DAG
- ReviewPolicy: Challenger configuration
- Manifest: Complete workflow plan
"""

from typing import Any

from pydantic import BaseModel, Field

from atlas.models.artifact import ArtifactRef
from atlas.models.enums import ChunkKind


class SplitterProfile(BaseModel):
    """Configuration for the chunking strategy.

    Controls how source artifacts are split into analyzable chunks.

    Design Principle:
        For the same source snapshot + splitter profile, chunk boundaries
        and chunk_ids MUST be stable (deterministic chunking).

    Attributes:
        name: Profile name identifier.
        prefer_semantic: Prefer semantic boundaries (divisions/sections/paragraphs).
        max_chunk_tokens: Maximum tokens per chunk (within context budget).
        overlap_lines: Number of overlapping lines between chunks.
        chunk_kinds: Allowed chunk kinds for this profile.
        custom_config: Additional splitter-specific configuration.

    Example:
        >>> profile = SplitterProfile(
        ...     name="cobol_semantic",
        ...     prefer_semantic=True,
        ...     max_chunk_tokens=3500,
        ...     overlap_lines=10,
        ... )
    """

    name: str = Field(default="default", description="Profile name identifier")
    prefer_semantic: bool = Field(
        default=True,
        description="Prefer semantic boundaries (divisions/sections)",
    )
    max_chunk_tokens: int = Field(
        default=3500,
        description="Maximum tokens per chunk",
    )
    overlap_lines: int = Field(
        default=10,
        description="Overlapping lines between chunks",
    )
    chunk_kinds: list[ChunkKind] = Field(
        default_factory=list,
        description="Allowed chunk kinds",
    )
    custom_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional splitter-specific configuration",
    )


class AnalysisProfile(BaseModel):
    """Documentation template / extraction profile.

    Defines what information to extract and how to format output.

    Attributes:
        name: Profile name identifier.
        template: Documentation template name.
        extract_symbols: Extract symbol definitions.
        extract_calls: Extract call relationships.
        extract_io: Extract I/O operations.
        extract_error_handling: Extract error handling patterns.
        custom_extractors: Additional extraction rules.

    Example:
        >>> profile = AnalysisProfile(
        ...     name="full_analysis",
        ...     extract_symbols=True,
        ...     extract_calls=True,
        ...     extract_io=True,
        ...     extract_error_handling=True,
        ... )
    """

    name: str = Field(default="default", description="Profile name identifier")
    template: str = Field(default="standard", description="Documentation template")
    extract_symbols: bool = Field(default=True, description="Extract symbol definitions")
    extract_calls: bool = Field(default=True, description="Extract call relationships")
    extract_io: bool = Field(default=True, description="Extract I/O operations")
    extract_error_handling: bool = Field(default=True, description="Extract error handling")
    custom_extractors: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional extraction rules",
    )


class ChunkSpec(BaseModel):
    """Specification for a single chunk in the manifest.

    Defines the boundaries and metadata for an analyzable chunk.

    Design Principle:
        Chunks MUST fit within context budget after overhead
        (prompt + required context refs).

    Attributes:
        chunk_id: Unique identifier for this chunk (stable).
        chunk_kind: Classification of chunk content.
        start_line: Starting line number (1-indexed).
        end_line: Ending line number (inclusive).
        division: COBOL division (if semantic chunking).
        section: COBOL section (if semantic chunking).
        paragraphs: List of paragraph names in this chunk.
        estimated_tokens: Estimated token count.
        result_uri: Where chunk result will be written.
        metadata: Additional chunk metadata.

    Example:
        >>> chunk = ChunkSpec(
        ...     chunk_id="procedure_part1",
        ...     chunk_kind=ChunkKind.PROCEDURE_PART,
        ...     start_line=500,
        ...     end_line=750,
        ...     division="PROCEDURE",
        ...     paragraphs=["MAIN-LOGIC", "PROCESS-RECORD"],
        ...     estimated_tokens=2500,
        ... )
    """

    chunk_id: str = Field(..., description="Unique chunk identifier")
    chunk_kind: ChunkKind = Field(
        default=ChunkKind.GENERIC,
        description="Classification of chunk content",
    )
    start_line: int = Field(..., description="Starting line (1-indexed)")
    end_line: int = Field(..., description="Ending line (inclusive)")
    division: str | None = Field(default=None, description="COBOL division name")
    section: str | None = Field(default=None, description="COBOL section name")
    paragraphs: list[str] = Field(default_factory=list, description="Paragraph names")
    estimated_tokens: int = Field(default=0, description="Estimated token count")
    result_uri: str | None = Field(default=None, description="URI for chunk result")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional chunk metadata",
    )


class MergeNode(BaseModel):
    """Node in the merge DAG.

    Defines how chunk results or child merges are combined.

    Design Principle:
        - Merge nodes MUST have bounded fan-in (suggested: 8-20 inputs).
        - Merge DAG MUST be a DAG (no cycles).
        - Root merge node produces the final doc + doc model.

    Attributes:
        merge_node_id: Unique identifier for this merge node.
        input_ids: IDs of inputs (chunk_ids or child merge_node_ids).
        is_root: True if this is the root merge node.
        level: Depth in merge hierarchy (0 = leaf merges).
        output_uri: Where merge result will be written.
        metadata: Additional merge node metadata.

    Example:
        >>> node = MergeNode(
        ...     merge_node_id="merge_procedure",
        ...     input_ids=["procedure_part1", "procedure_part2", "procedure_part3"],
        ...     level=1,
        ... )
    """

    merge_node_id: str = Field(..., description="Unique merge node identifier")
    input_ids: list[str] = Field(..., description="IDs of inputs to merge")
    is_root: bool = Field(default=False, description="True if root merge node")
    level: int = Field(default=0, description="Depth in merge hierarchy")
    output_uri: str | None = Field(default=None, description="URI for merge result")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional merge node metadata",
    )


class ReviewPolicy(BaseModel):
    """Configuration for the challenger review process.

    Defines what the challenger should look for and how to handle issues.

    Attributes:
        challenge_profile: Focus areas (error handling, I/O, restartability).
        max_iterations: Maximum challenger loop iterations.
        blocker_threshold: Stop if this many blockers found.
        auto_accept_minor_only: Accept if only minor issues remain.
        required_checks: List of required verification checks.

    Example:
        >>> policy = ReviewPolicy(
        ...     challenge_profile="comprehensive",
        ...     max_iterations=3,
        ...     auto_accept_minor_only=True,
        ...     required_checks=["error_handling", "io_operations"],
        ... )
    """

    challenge_profile: str = Field(
        default="standard",
        description="Challenge focus areas",
    )
    max_iterations: int = Field(
        default=3,
        description="Maximum challenger loop iterations",
    )
    blocker_threshold: int = Field(
        default=5,
        description="Stop if this many blockers found",
    )
    auto_accept_minor_only: bool = Field(
        default=True,
        description="Accept if only minor issues remain",
    )
    required_checks: list[str] = Field(
        default_factory=list,
        description="Required verification checks",
    )


class ArtifactOutputConfig(BaseModel):
    """Configuration for where to write output artifacts.

    Attributes:
        base_uri: Base URI for all outputs.
        chunk_results_path: Path pattern for chunk results.
        merge_results_path: Path pattern for merge results.
        doc_path: Path for final documentation.
        doc_model_path: Path for documentation model.
    """

    base_uri: str = Field(..., description="Base URI for outputs")
    chunk_results_path: str = Field(
        default="chunks/{chunk_id}.json",
        description="Path pattern for chunk results",
    )
    merge_results_path: str = Field(
        default="merges/{merge_node_id}.json",
        description="Path pattern for merge results",
    )
    doc_path: str = Field(
        default="doc/documentation.md",
        description="Path for final documentation",
    )
    doc_model_path: str = Field(
        default="doc/doc_model.json",
        description="Path for documentation model",
    )


class Manifest(BaseModel):
    """Complete workflow plan stored in the artifact store.

    A Manifest describes the entire analysis workflow including chunk
    specifications, merge DAG, and review policies. It is the source
    of truth for the controller's reconciliation loop.

    Design Principle:
        - Tickets reference manifest_uri; they do not embed manifest data.
        - Manifest enables deterministic, idempotent workflow execution.

    Attributes:
        job_id: Unique identifier for this analysis run.
        artifact_ref: Reference to the source artifact.
        analysis_profile: Documentation extraction profile.
        splitter_profile: Chunking configuration.
        context_budget: Maximum tokens per task.
        chunks: List of chunk specifications.
        merge_dag: List of merge nodes.
        review_policy: Challenger configuration.
        artifacts: Output artifact configuration.
        cycle_number: Current iteration number.
        created_at: ISO timestamp of creation.
        metadata: Additional manifest metadata.

    Example:
        >>> manifest = Manifest(
        ...     job_id="job-20240115-001",
        ...     artifact_ref=ArtifactRef(...),
        ...     analysis_profile=AnalysisProfile(name="full"),
        ...     splitter_profile=SplitterProfile(name="cobol_semantic"),
        ...     context_budget=4000,
        ...     chunks=[...],
        ...     merge_dag=[...],
        ... )
    """

    job_id: str = Field(..., description="Unique identifier for this analysis run")
    artifact_ref: ArtifactRef = Field(..., description="Reference to source artifact")
    analysis_profile: AnalysisProfile = Field(
        default_factory=AnalysisProfile,
        description="Documentation extraction profile",
    )
    splitter_profile: SplitterProfile = Field(
        default_factory=SplitterProfile,
        description="Chunking configuration",
    )
    context_budget: int = Field(default=4000, description="Maximum tokens per task")
    chunks: list[ChunkSpec] = Field(default_factory=list, description="Chunk specifications")
    merge_dag: list[MergeNode] = Field(default_factory=list, description="Merge DAG nodes")
    review_policy: ReviewPolicy = Field(
        default_factory=ReviewPolicy,
        description="Challenger configuration",
    )
    artifacts: ArtifactOutputConfig | None = Field(
        default=None,
        description="Output artifact configuration",
    )
    cycle_number: int = Field(default=1, description="Current iteration number")
    created_at: str | None = Field(default=None, description="ISO timestamp of creation")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional manifest metadata",
    )

    def get_chunk(self, chunk_id: str) -> ChunkSpec | None:
        """Get a chunk specification by ID.

        Args:
            chunk_id: The chunk identifier.

        Returns:
            ChunkSpec if found, None otherwise.
        """
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def get_merge_node(self, merge_node_id: str) -> MergeNode | None:
        """Get a merge node by ID.

        Args:
            merge_node_id: The merge node identifier.

        Returns:
            MergeNode if found, None otherwise.
        """
        for node in self.merge_dag:
            if node.merge_node_id == merge_node_id:
                return node
        return None

    def get_root_merge_node(self) -> MergeNode | None:
        """Get the root merge node.

        Returns:
            The root MergeNode if found, None otherwise.
        """
        for node in self.merge_dag:
            if node.is_root:
                return node
        return None

    def validate_dag(self) -> list[str]:
        """Validate the merge DAG for cycles and invalid references.

        Returns:
            List of validation error messages (empty if valid).

        TODO: Implement cycle detection and reference validation.
        """
        errors: list[str] = []

        # Check for root node
        root_nodes = [n for n in self.merge_dag if n.is_root]
        if len(root_nodes) == 0:
            errors.append("No root merge node found")
        elif len(root_nodes) > 1:
            errors.append(f"Multiple root merge nodes found: {[n.merge_node_id for n in root_nodes]}")

        # TODO: Implement cycle detection
        # TODO: Validate all input_ids reference valid chunks or merge nodes

        return errors
