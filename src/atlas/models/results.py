"""Result artifact models for workflow outputs.

All outputs MUST be structured and include provenance.
This module defines the normative schemas for:
- ChunkResult: Output of DOC_CHUNK
- MergeResult: Output of DOC_MERGE
- DocumentationModel: Required for challenger routing
- ChallengeResult: Output of DOC_CHALLENGE
- FollowupAnswer: Output of DOC_FOLLOWUP
"""

from typing import Any

from pydantic import BaseModel, Field

from atlas.models.enums import IssueSeverity


class Evidence(BaseModel):
    """Evidence reference for traceability.

    Links findings to source locations.

    Attributes:
        evidence_type: Type of evidence (line_range, symbol_ref, etc.).
        start_line: Starting line number.
        end_line: Ending line number.
        note: Explanatory note.
        source_ref: Additional source reference.
    """

    evidence_type: str = Field(default="line_range", description="Type of evidence")
    start_line: int | None = Field(default=None, description="Starting line")
    end_line: int | None = Field(default=None, description="Ending line")
    note: str | None = Field(default=None, description="Explanatory note")
    source_ref: str | None = Field(default=None, description="Additional source reference")


class SymbolDef(BaseModel):
    """Symbol definition extracted from source.

    Represents variables, paragraphs, sections, etc.

    Attributes:
        name: Symbol name.
        kind: Symbol kind (variable, paragraph, section, copybook, etc.).
        attributes: Additional attributes (level, picture, value, etc.).
        line_number: Definition line number.
    """

    name: str = Field(..., description="Symbol name")
    kind: str = Field(..., description="Symbol kind")
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional attributes",
    )
    line_number: int | None = Field(default=None, description="Definition line")


class IOOperation(BaseModel):
    """I/O operation extracted from source.

    Represents READ, WRITE, REWRITE, DELETE operations.

    Attributes:
        operation: Operation type (READ, WRITE, REWRITE, DELETE).
        file_name: File name if known.
        record_name: Record name if known.
        line_number: Operation line number.
        status_check: Whether FILE STATUS is checked.
    """

    operation: str = Field(..., description="Operation type")
    file_name: str | None = Field(default=None, description="File name")
    record_name: str | None = Field(default=None, description="Record name")
    line_number: int | None = Field(default=None, description="Line number")
    status_check: bool = Field(default=False, description="FILE STATUS checked")


class ErrorHandlingPattern(BaseModel):
    """Error handling pattern extracted from source.

    Represents FILE STATUS checks, ON ERROR, EVALUATE, ABEND, etc.

    Attributes:
        pattern_type: Type of pattern (file_status, on_error, evaluate, abend).
        description: Description of the pattern.
        line_numbers: Associated line numbers.
        related_symbols: Related symbols (files, variables).
    """

    pattern_type: str = Field(..., description="Pattern type")
    description: str = Field(default="", description="Pattern description")
    line_numbers: list[int] = Field(default_factory=list, description="Line numbers")
    related_symbols: list[str] = Field(default_factory=list, description="Related symbols")


class CallTarget(BaseModel):
    """Call target extracted from source.

    Represents PERFORM or CALL statements.

    Attributes:
        target: Target name (paragraph or program).
        call_type: Type of call (perform, call).
        is_external: True if external program call.
        line_number: Call line number.
    """

    target: str = Field(..., description="Target name")
    call_type: str = Field(default="perform", description="Call type")
    is_external: bool = Field(default=False, description="External call")
    line_number: int | None = Field(default=None, description="Line number")


class OpenQuestion(BaseModel):
    """Open question recorded when context is insufficient.

    Design Principle:
        If a worker lacks context, it MUST record "unknowns" / "open questions"
        rather than guessing.

    Attributes:
        question: The question text.
        context_needed: What additional context would help.
        suspected_location: Where the answer might be found.
    """

    question: str = Field(..., description="Question text")
    context_needed: str | None = Field(default=None, description="Context needed")
    suspected_location: str | None = Field(default=None, description="Suspected location")


class ChunkFacts(BaseModel):
    """Structured, mergeable facts extracted from a chunk.

    These facts are machine-oriented for reliable aggregation.

    Attributes:
        symbols_defined: Symbols defined in this chunk.
        symbols_used: Symbol names used (referenced).
        entrypoints: Entry points / paragraphs defined.
        paragraphs_defined: Paragraph names defined.
        calls: Call targets (internal and external).
        io_operations: I/O operations.
        error_handling: Error handling patterns.
    """

    symbols_defined: list[SymbolDef] = Field(
        default_factory=list,
        description="Symbols defined",
    )
    symbols_used: list[str] = Field(
        default_factory=list,
        description="Symbol names used",
    )
    entrypoints: list[str] = Field(
        default_factory=list,
        description="Entry points",
    )
    paragraphs_defined: list[str] = Field(
        default_factory=list,
        description="Paragraphs defined",
    )
    calls: list[CallTarget] = Field(
        default_factory=list,
        description="Call targets",
    )
    io_operations: list[IOOperation] = Field(
        default_factory=list,
        description="I/O operations",
    )
    error_handling: list[ErrorHandlingPattern] = Field(
        default_factory=list,
        description="Error handling patterns",
    )


class ChunkResult(BaseModel):
    """Output artifact for DOC_CHUNK work items.

    Contains the analysis results for a single chunk.

    Design Principle:
        All outputs MUST be structured and include provenance.
        Facts are machine-oriented for reliable aggregation.

    Attributes:
        job_id: Job identifier.
        artifact_id: Source artifact identifier.
        artifact_version: Source artifact version.
        chunk_id: Chunk identifier.
        chunk_locator: Chunk location (line range or semantic).
        chunk_kind: Classification of chunk content.
        summary: Short narrative summary.
        facts: Structured, mergeable facts.
        evidence: Source references for traceability.
        open_questions: Explicit unknowns.
        confidence: Confidence score (0.0 to 1.0).
        metadata: Additional metadata.

    Example:
        >>> result = ChunkResult(
        ...     job_id="job-123",
        ...     artifact_id="DRKBM100.cbl",
        ...     artifact_version="abc123",
        ...     chunk_id="procedure_part1",
        ...     chunk_kind="procedure_part",
        ...     summary="This chunk handles main processing logic...",
        ...     facts=ChunkFacts(...),
        ...     confidence=0.85,
        ... )
    """

    job_id: str = Field(..., description="Job identifier")
    artifact_id: str = Field(..., description="Source artifact ID")
    artifact_version: str = Field(..., description="Source artifact version")
    chunk_id: str = Field(..., description="Chunk identifier")
    chunk_locator: dict[str, Any] = Field(..., description="Chunk location")
    chunk_kind: str = Field(..., description="Chunk content classification")
    summary: str = Field(default="", description="Short narrative summary")
    facts: ChunkFacts = Field(default_factory=ChunkFacts, description="Structured facts")
    evidence: list[Evidence] = Field(default_factory=list, description="Source references")
    open_questions: list[OpenQuestion] = Field(
        default_factory=list,
        description="Explicit unknowns",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ConsolidatedFacts(BaseModel):
    """Consolidated facts from multiple chunk results.

    Merged call graph, IO map, and error handling behaviors.

    Attributes:
        call_graph_edges: Merged call graph edges.
        io_map: Merged I/O operations map.
        error_handling_behaviors: Consolidated error handling.
        symbols: All symbols from merged inputs.
    """

    call_graph_edges: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Merged call graph edges",
    )
    io_map: list[IOOperation] = Field(
        default_factory=list,
        description="Merged I/O operations",
    )
    error_handling_behaviors: list[ErrorHandlingPattern] = Field(
        default_factory=list,
        description="Consolidated error handling",
    )
    symbols: list[SymbolDef] = Field(
        default_factory=list,
        description="All symbols",
    )


class MergeConflict(BaseModel):
    """Conflict identified during merge.

    Records disagreements between inputs.

    Attributes:
        description: What disagreed.
        input_ids: Which inputs conflict.
        suggested_followup_scope: Scope for follow-up resolution.
    """

    description: str = Field(..., description="Conflict description")
    input_ids: list[str] = Field(..., description="Conflicting input IDs")
    suggested_followup_scope: str | None = Field(
        default=None,
        description="Suggested follow-up scope",
    )


class MergeCoverage(BaseModel):
    """Coverage information for a merge result.

    Tracks what was included and what was missing.

    Attributes:
        included_input_ids: IDs of included inputs.
        missing_input_ids: IDs of missing/failed inputs.
    """

    included_input_ids: list[str] = Field(default_factory=list, description="Included inputs")
    missing_input_ids: list[str] = Field(default_factory=list, description="Missing inputs")


class MergeResult(BaseModel):
    """Output artifact for DOC_MERGE work items.

    Contains the merged results from multiple chunk results or child merges.

    Attributes:
        job_id: Job identifier.
        artifact_id: Source artifact identifier.
        artifact_version: Source artifact version.
        merge_node_id: Merge node identifier.
        coverage: What inputs were included/missing.
        consolidated_facts: Merged facts.
        conflicts: Conflicts identified during merge.
        narrative_sections: Optional narrative fragments.
        doc_fragment_uri: Optional rendered fragment URI.
        metadata: Additional metadata.
    """

    job_id: str = Field(..., description="Job identifier")
    artifact_id: str = Field(..., description="Source artifact ID")
    artifact_version: str = Field(..., description="Source artifact version")
    merge_node_id: str = Field(..., description="Merge node identifier")
    coverage: MergeCoverage = Field(
        default_factory=MergeCoverage,
        description="Coverage information",
    )
    consolidated_facts: ConsolidatedFacts = Field(
        default_factory=ConsolidatedFacts,
        description="Merged facts",
    )
    conflicts: list[MergeConflict] = Field(default_factory=list, description="Merge conflicts")
    narrative_sections: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Narrative fragments",
    )
    doc_fragment_uri: str | None = Field(default=None, description="Rendered fragment URI")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Section(BaseModel):
    """A section in the documentation model.

    Provides traceability from documentation to source.

    Attributes:
        section_id: Unique section identifier.
        title: Section title.
        content: Section content (or URI pointer).
        source_refs: Chunk IDs and evidence references used.
    """

    section_id: str = Field(..., description="Section identifier")
    title: str = Field(..., description="Section title")
    content: str | None = Field(default=None, description="Section content")
    content_uri: str | None = Field(default=None, description="Content URI")
    source_refs: list[str] = Field(default_factory=list, description="Source references")


class DocIndex(BaseModel):
    """Index for routing challenger issues to chunks.

    Enables efficient lookup of which chunks contain specific elements.

    Attributes:
        symbol_to_chunks: Symbol name to chunk IDs.
        paragraph_to_chunk: Paragraph name to chunk ID.
        file_to_chunks: File name to chunk IDs.
    """

    symbol_to_chunks: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Symbol to chunk mapping",
    )
    paragraph_to_chunk: dict[str, str] = Field(
        default_factory=dict,
        description="Paragraph to chunk mapping",
    )
    file_to_chunks: dict[str, list[str]] = Field(
        default_factory=dict,
        description="File to chunk mapping",
    )


class DocumentationModel(BaseModel):
    """Machine-readable documentation structure for challenger routing.

    The final doc needs traceability so challenger issues can be routed
    to the appropriate chunks for follow-up.

    Design Principle:
        Required for challenger routing. Enables routing issues back to
        source chunks for targeted follow-up work.

    Attributes:
        doc_uri: URI of the rendered human documentation.
        sections: Documentation sections with source refs.
        index: Indexes for efficient routing.
        metadata: Additional metadata.
    """

    doc_uri: str = Field(..., description="URI of rendered documentation")
    sections: list[Section] = Field(default_factory=list, description="Documentation sections")
    index: DocIndex = Field(default_factory=DocIndex, description="Routing indexes")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Issue(BaseModel):
    """A challenger issue identified during review.

    Attributes:
        issue_id: Deterministic or generated identifier.
        severity: Issue severity level.
        question: Problem statement or question.
        doc_section_refs: Which doc sections are unclear.
        suspected_scopes: Chunk IDs, paragraph names, or "unknown".
        routing_hints: Symbols/paragraphs/files mentioned.
    """

    issue_id: str = Field(..., description="Issue identifier")
    severity: IssueSeverity = Field(..., description="Issue severity")
    question: str = Field(..., description="Problem statement or question")
    doc_section_refs: list[str] = Field(default_factory=list, description="Doc section refs")
    suspected_scopes: list[str] = Field(default_factory=list, description="Suspected scopes")
    routing_hints: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Routing hints",
    )


class FollowupTask(BaseModel):
    """A recommended follow-up task in the resolution plan.

    Attributes:
        issue_id: Issue being addressed.
        scope: Bounded scope for the follow-up.
        description: What needs to be investigated.
    """

    issue_id: str = Field(..., description="Issue being addressed")
    scope: dict[str, Any] = Field(..., description="Bounded scope")
    description: str = Field(default="", description="Investigation description")


class ResolutionPlan(BaseModel):
    """Plan for resolving challenger issues.

    Lists recommended follow-up tasks and whether patch merge is required.

    Attributes:
        followup_tasks: List of recommended follow-up tasks.
        requires_patch_merge: Whether patch merge is needed.
    """

    followup_tasks: list[FollowupTask] = Field(
        default_factory=list,
        description="Recommended follow-up tasks",
    )
    requires_patch_merge: bool = Field(default=False, description="Patch merge required")


class ChallengeResult(BaseModel):
    """Output artifact for DOC_CHALLENGE work items.

    Contains issues identified during documentation review and
    a resolution plan for addressing them.

    Attributes:
        job_id: Job identifier.
        artifact_id: Source artifact identifier.
        artifact_version: Source artifact version.
        issues: List of identified issues.
        resolution_plan: Plan for resolving issues.
        metadata: Additional metadata.
    """

    job_id: str = Field(..., description="Job identifier")
    artifact_id: str = Field(..., description="Source artifact ID")
    artifact_version: str = Field(..., description="Source artifact version")
    issues: list[Issue] = Field(default_factory=list, description="Identified issues")
    resolution_plan: ResolutionPlan = Field(
        default_factory=ResolutionPlan,
        description="Resolution plan",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def has_blockers(self) -> bool:
        """Check if any blocker issues exist.

        Returns:
            True if any issue has BLOCKER severity.
        """
        return any(issue.severity == IssueSeverity.BLOCKER for issue in self.issues)

    def issues_by_severity(self) -> dict[IssueSeverity, list[Issue]]:
        """Group issues by severity.

        Returns:
            Dictionary mapping severity to list of issues.
        """
        result: dict[IssueSeverity, list[Issue]] = {}
        for issue in self.issues:
            if issue.severity not in result:
                result[issue.severity] = []
            result[issue.severity].append(issue)
        return result


class FollowupAnswer(BaseModel):
    """Output artifact for DOC_FOLLOWUP work items.

    Contains the answer to a specific challenger question.

    Attributes:
        issue_id: Issue being addressed.
        scope: What was analyzed.
        answer: Clear text answer.
        facts: Structured, mergeable facts.
        evidence: Line ranges / refs.
        confidence: Confidence score (0.0 to 1.0).
        metadata: Additional metadata.
    """

    issue_id: str = Field(..., description="Issue being addressed")
    scope: dict[str, Any] = Field(..., description="Analyzed scope")
    answer: str = Field(..., description="Clear text answer")
    facts: ChunkFacts = Field(default_factory=ChunkFacts, description="Structured facts")
    evidence: list[Evidence] = Field(default_factory=list, description="Source references")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ChunkContribution(BaseModel):
    """Contribution of a chunk to a documentation section.

    Used in trace reports to track provenance.

    Attributes:
        chunk_id: The contributing chunk identifier.
        confidence: Confidence score from chunk analysis.
        open_questions: Any unresolved questions from this chunk.
        line_range: Source line range (start, end).
    """

    chunk_id: str = Field(..., description="Contributing chunk identifier")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Chunk confidence")
    open_questions: list[str] = Field(default_factory=list, description="Unresolved questions")
    line_range: tuple[int, int] | None = Field(default=None, description="Source line range")


class SectionTrace(BaseModel):
    """Trace information for a documentation section.

    Maps a section to its source chunks and tracks the analysis history.

    Attributes:
        section_id: The documentation section identifier.
        section_title: Section title for readability.
        chunk_contributions: Chunks that contributed to this section.
        challenger_iterations: Number of challenger review cycles.
        issues_raised: Issue IDs raised for this section.
        issues_resolved: Issue IDs resolved for this section.
    """

    section_id: str = Field(..., description="Documentation section identifier")
    section_title: str = Field(default="", description="Section title")
    chunk_contributions: list[ChunkContribution] = Field(
        default_factory=list,
        description="Contributing chunks",
    )
    challenger_iterations: int = Field(default=0, description="Review cycle count")
    issues_raised: list[str] = Field(default_factory=list, description="Issues raised")
    issues_resolved: list[str] = Field(default_factory=list, description="Issues resolved")


class TraceReport(BaseModel):
    """Trace report mapping documentation to source chunks.

    Per spec section 6.8, the trace report provides:
    - Per-section chunk contributions
    - Per-chunk confidence and open questions
    - Challenger iteration history
    - Issues raised and resolved

    Attributes:
        job_id: Job identifier.
        artifact_id: Source artifact identifier.
        artifact_version: Source artifact version.
        section_traces: Per-section trace information.
        total_chunks: Total number of chunks analyzed.
        total_sections: Total documentation sections.
        total_issues_raised: Total issues identified by challenger.
        total_issues_resolved: Total issues resolved.
        final_cycle: Final challenger iteration number.
        generated_at: ISO timestamp of report generation.
    """

    job_id: str = Field(..., description="Job identifier")
    artifact_id: str = Field(..., description="Source artifact identifier")
    artifact_version: str = Field(..., description="Source artifact version")
    section_traces: list[SectionTrace] = Field(
        default_factory=list,
        description="Per-section traces",
    )
    total_chunks: int = Field(default=0, description="Total chunks analyzed")
    total_sections: int = Field(default=0, description="Total documentation sections")
    total_issues_raised: int = Field(default=0, description="Total issues raised")
    total_issues_resolved: int = Field(default=0, description="Total issues resolved")
    final_cycle: int = Field(default=1, description="Final challenger cycle")
    generated_at: str = Field(default="", description="Report generation timestamp")


class JobStatistics(BaseModel):
    """Summary statistics for a completed documentation job.

    Provides high-level metrics about the analysis run.

    Attributes:
        job_id: Job identifier.
        artifact_id: Source artifact identifier.
        status: Final job status (accepted, completed_with_issues, etc.).
        total_chunks: Number of chunks analyzed.
        total_merges: Number of merge operations.
        challenger_cycles: Number of challenger iterations.
        issues_raised: Total issues identified.
        issues_resolved: Total issues resolved.
        blockers_remaining: Unresolved blocker count.
        average_confidence: Average chunk confidence score.
        total_lines_analyzed: Total source lines processed.
        duration_seconds: Total processing time if available.
        final_doc_uri: URI of final documentation.
        final_trace_uri: URI of trace report.
    """

    job_id: str = Field(..., description="Job identifier")
    artifact_id: str = Field(..., description="Source artifact identifier")
    status: str = Field(default="completed", description="Final job status")
    total_chunks: int = Field(default=0, description="Chunks analyzed")
    total_merges: int = Field(default=0, description="Merge operations")
    challenger_cycles: int = Field(default=0, description="Challenger iterations")
    issues_raised: int = Field(default=0, description="Issues identified")
    issues_resolved: int = Field(default=0, description="Issues resolved")
    blockers_remaining: int = Field(default=0, description="Unresolved blockers")
    average_confidence: float = Field(default=0.0, description="Average confidence")
    total_lines_analyzed: int = Field(default=0, description="Source lines processed")
    duration_seconds: float | None = Field(default=None, description="Processing time")
    final_doc_uri: str = Field(default="", description="Final documentation URI")
    final_trace_uri: str = Field(default="", description="Trace report URI")


class FinalizeResult(BaseModel):
    """Output artifact for DOC_FINALIZE work items.

    Contains references to all final deliverables produced.

    Attributes:
        job_id: Job identifier.
        artifact_id: Source artifact identifier.
        artifact_version: Source artifact version.
        status: Finalization status (accepted, completed_with_warnings).
        doc_uri: URI of final Markdown documentation.
        pdf_uri: Optional URI of PDF documentation.
        trace_uri: URI of trace report.
        summary_uri: URI of job statistics.
        warnings: Any warnings from finalization.
        metadata: Additional metadata.
    """

    job_id: str = Field(..., description="Job identifier")
    artifact_id: str = Field(..., description="Source artifact identifier")
    artifact_version: str = Field(..., description="Source artifact version")
    status: str = Field(default="accepted", description="Finalization status")
    doc_uri: str = Field(..., description="Final documentation URI")
    pdf_uri: str | None = Field(default=None, description="PDF documentation URI")
    trace_uri: str = Field(..., description="Trace report URI")
    summary_uri: str = Field(..., description="Job statistics URI")
    warnings: list[str] = Field(default_factory=list, description="Finalization warnings")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
