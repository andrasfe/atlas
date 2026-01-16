"""Finalize worker implementation for producing final deliverables.

The Finalize worker handles DOC_FINALIZE work items, producing:
- Final Markdown documentation
- Optional PDF output
- Trace report (chunk to doc section mapping)
- Summary statistics
- Job completion marking

Per spec section 6.8, this is the final phase that:
- Marks DOC_REQUEST as fully complete
- Produces final deliverables and status summary
- Optionally cleans up intermediate artifacts
"""

import logging
from datetime import datetime
from typing import Any

from atlas.adapters.artifact_store import ArtifactStoreAdapter
from atlas.adapters.llm import LLMAdapter
from atlas.adapters.ticket_system import TicketSystemAdapter
from atlas.models.enums import IssueSeverity, WorkItemStatus, WorkItemType
from atlas.models.results import (
    ChunkContribution,
    ChunkResult,
    DocumentationModel,
    FinalizeResult,
    JobStatistics,
    SectionTrace,
    TraceReport,
    ChallengeResult,
)
from atlas.models.work_item import (
    DocFinalizePayload,
    DocChunkPayload,
    DocChallengePayload,
    WorkItem,
)
from atlas.workers.base import Worker

logger = logging.getLogger(__name__)


class FinalizeWorker(Worker):
    """Concrete implementation of the Finalize worker.

    Produces final deliverables and marks the job as accepted.

    Design Principles:
        - Generates comprehensive trace report
        - Preserves provenance information
        - Produces clean final documentation
        - Supports optional PDF generation
        - Handles cleanup of intermediate artifacts

    Example:
        >>> worker = FinalizeWorker(
        ...     worker_id="finalize-1",
        ...     ticket_system=ticket_system,
        ...     artifact_store=artifact_store,
        ...     llm=llm,
        ... )
        >>> result = await worker.process(work_item)
    """

    def __init__(
        self,
        worker_id: str,
        ticket_system: TicketSystemAdapter,
        artifact_store: ArtifactStoreAdapter,
        llm: LLMAdapter,
    ):
        """Initialize the finalize worker.

        Args:
            worker_id: Unique identifier for this worker instance.
            ticket_system: Ticket system adapter.
            artifact_store: Artifact store adapter.
            llm: LLM adapter (not typically used for finalization).
        """
        super().__init__(worker_id, ticket_system, artifact_store, llm)

    @property
    def supported_work_types(self) -> list[WorkItemType]:
        """Finalize worker handles DOC_FINALIZE items."""
        return [WorkItemType.DOC_FINALIZE]

    async def process(self, work_item: WorkItem) -> FinalizeResult:
        """Process a finalize work item.

        Generates trace report, final documentation, summary statistics,
        and optionally cleans up intermediate artifacts.

        Args:
            work_item: The finalize work item.

        Returns:
            FinalizeResult with URIs to all deliverables.

        Raises:
            ValueError: If payload is not DocFinalizePayload.
        """
        payload = work_item.payload
        if not isinstance(payload, DocFinalizePayload):
            raise ValueError("Expected DocFinalizePayload")

        logger.info(f"Processing finalize for job {payload.job_id}")

        try:
            # Check if outputs already exist (idempotency)
            if await self._output_exists(work_item):
                logger.info(f"Finalize outputs already exist for {work_item.work_id}")
                return await self._load_existing_result(payload)

            # Load the documentation and model
            doc_content = await self._load_doc(payload.doc_uri)
            doc_model = await self._load_doc_model(payload.doc_model_uri)

            # Gather job data for trace and statistics
            job_data = await self._gather_job_data(payload.job_id)

            # Generate trace report
            trace_report = await self._generate_trace_report(
                payload, doc_model, job_data
            )
            await self.artifact_store.write_json(
                payload.output_trace_uri,
                trace_report.model_dump(),
            )
            logger.info(f"Wrote trace report to {payload.output_trace_uri}")

            # Generate summary statistics
            statistics = self._generate_statistics(payload, job_data, trace_report)
            await self.artifact_store.write_json(
                payload.output_summary_uri,
                statistics.model_dump(),
            )
            logger.info(f"Wrote statistics to {payload.output_summary_uri}")

            # Generate final Markdown documentation
            final_doc = self._generate_final_doc(doc_content, doc_model, statistics)
            await self.artifact_store.write_text(
                payload.output_doc_uri,
                final_doc,
            )
            logger.info(f"Wrote final documentation to {payload.output_doc_uri}")

            # Generate PDF if requested
            pdf_uri = None
            if payload.generate_pdf and payload.output_pdf_uri:
                pdf_uri = await self._generate_pdf(
                    final_doc, payload.output_pdf_uri
                )
                if pdf_uri:
                    logger.info(f"Wrote PDF to {pdf_uri}")

            # Handle cleanup if requested
            if payload.cleanup_intermediates:
                await self._cleanup_intermediates(payload.job_id)

            # Determine final status
            warnings: list[str] = []
            status = "accepted"
            if statistics.blockers_remaining > 0:
                status = "completed_with_blockers"
                warnings.append(
                    f"{statistics.blockers_remaining} blocker issues remain unresolved"
                )
            elif statistics.issues_raised > statistics.issues_resolved:
                status = "completed_with_warnings"
                warnings.append(
                    f"{statistics.issues_raised - statistics.issues_resolved} "
                    f"minor issues remain unresolved"
                )

            result = FinalizeResult(
                job_id=payload.job_id,
                artifact_id=payload.artifact_ref.artifact_id if payload.artifact_ref else "",
                artifact_version=payload.artifact_ref.artifact_version if payload.artifact_ref else "",
                status=status,
                doc_uri=payload.output_doc_uri,
                pdf_uri=pdf_uri,
                trace_uri=payload.output_trace_uri,
                summary_uri=payload.output_summary_uri,
                warnings=warnings,
                metadata={
                    "finalized_at": datetime.utcnow().isoformat(),
                    "worker_id": self.worker_id,
                },
            )

            logger.info(
                f"Finalize complete for {work_item.work_id}: status={status}"
            )

            return result

        except Exception as e:
            logger.exception(f"Finalize failed for {work_item.work_id}")
            raise

    async def _load_doc(self, uri: str) -> str:
        """Load documentation text.

        Args:
            uri: URI of the documentation.

        Returns:
            Documentation text content.
        """
        try:
            return await self.artifact_store.read_text(uri)
        except Exception as e:
            logger.error(f"Could not load doc from {uri}: {e}")
            raise

    async def _load_doc_model(self, uri: str) -> DocumentationModel:
        """Load documentation model.

        Args:
            uri: URI of the documentation model.

        Returns:
            DocumentationModel object.
        """
        try:
            data = await self.artifact_store.read_json(uri)
            return DocumentationModel(**data)
        except Exception as e:
            logger.warning(f"Could not load doc model from {uri}: {e}")
            return DocumentationModel(doc_uri=uri)

    async def _gather_job_data(self, job_id: str) -> dict[str, Any]:
        """Gather all job data needed for trace and statistics.

        Collects work items, chunk results, challenge results, etc.

        Args:
            job_id: The job identifier.

        Returns:
            Dictionary with categorized job data.
        """
        data: dict[str, Any] = {
            "chunks": [],
            "chunk_results": [],
            "challenges": [],
            "challenge_results": [],
            "followups": [],
            "patch_merges": [],
            "merges": [],
        }

        # Query all work items for job
        work_items = await self.ticket_system.query_by_job(job_id)

        for item in work_items:
            if item.work_type == WorkItemType.DOC_CHUNK:
                data["chunks"].append(item)
                # Try to load chunk result
                if isinstance(item.payload, DocChunkPayload):
                    try:
                        result_data = await self.artifact_store.read_json(
                            item.payload.result_uri
                        )
                        data["chunk_results"].append(ChunkResult(**result_data))
                    except Exception:
                        pass

            elif item.work_type == WorkItemType.DOC_CHALLENGE:
                data["challenges"].append(item)
                # Try to load challenge result
                if isinstance(item.payload, DocChallengePayload):
                    try:
                        result_data = await self.artifact_store.read_json(
                            item.payload.output_uri
                        )
                        data["challenge_results"].append(ChallengeResult(**result_data))
                    except Exception:
                        pass

            elif item.work_type == WorkItemType.DOC_FOLLOWUP:
                data["followups"].append(item)

            elif item.work_type == WorkItemType.DOC_PATCH_MERGE:
                data["patch_merges"].append(item)

            elif item.work_type == WorkItemType.DOC_MERGE:
                data["merges"].append(item)

        return data

    async def _generate_trace_report(
        self,
        payload: DocFinalizePayload,
        doc_model: DocumentationModel,
        job_data: dict[str, Any],
    ) -> TraceReport:
        """Generate the trace report mapping doc sections to chunks.

        Per spec section 6.8, includes:
        - Per-section chunk contributions
        - Per-chunk confidence and open questions
        - Challenger iteration history
        - Issues raised and resolved

        Args:
            payload: The finalize payload.
            doc_model: Documentation model with sections.
            job_data: Gathered job data.

        Returns:
            TraceReport with full provenance information.
        """
        # Build chunk result lookup
        chunk_result_map: dict[str, ChunkResult] = {}
        for result in job_data["chunk_results"]:
            chunk_result_map[result.chunk_id] = result

        # Build issue tracking
        all_issues_raised: set[str] = set()
        all_issues_resolved: set[str] = set()
        section_issues: dict[str, tuple[set[str], set[str]]] = {}  # raised, resolved

        for challenge in job_data["challenge_results"]:
            for issue in challenge.issues:
                all_issues_raised.add(issue.issue_id)
                # Track which sections have issues
                for section_ref in issue.doc_section_refs:
                    if section_ref not in section_issues:
                        section_issues[section_ref] = (set(), set())
                    section_issues[section_ref][0].add(issue.issue_id)

        # Issues are considered resolved if follow-ups addressed them
        for followup in job_data["followups"]:
            if followup.status == WorkItemStatus.DONE:
                if hasattr(followup.payload, "issue_id"):
                    all_issues_resolved.add(followup.payload.issue_id)
                    # Mark as resolved in section issues
                    for section_id, (raised, resolved) in section_issues.items():
                        if followup.payload.issue_id in raised:
                            resolved.add(followup.payload.issue_id)

        # Determine final challenge cycle
        final_cycle = 1
        if job_data["challenges"]:
            final_cycle = max(c.cycle_number for c in job_data["challenges"])

        # Build section traces
        section_traces: list[SectionTrace] = []
        for section in doc_model.sections:
            contributions: list[ChunkContribution] = []

            for chunk_id in section.source_refs:
                chunk_result = chunk_result_map.get(chunk_id)
                if chunk_result:
                    # Extract open questions as strings
                    open_questions = [
                        q.question for q in chunk_result.open_questions
                    ]
                    # Get line range from locator
                    line_range = None
                    locator = chunk_result.chunk_locator
                    if locator.get("start_line") and locator.get("end_line"):
                        line_range = (locator["start_line"], locator["end_line"])

                    contributions.append(
                        ChunkContribution(
                            chunk_id=chunk_id,
                            confidence=chunk_result.confidence,
                            open_questions=open_questions,
                            line_range=line_range,
                        )
                    )
                else:
                    # Chunk result not found, minimal contribution
                    contributions.append(
                        ChunkContribution(
                            chunk_id=chunk_id,
                            confidence=0.0,
                            open_questions=["Chunk result not found"],
                        )
                    )

            # Get issues for this section
            raised, resolved = section_issues.get(
                section.section_id, (set(), set())
            )

            section_traces.append(
                SectionTrace(
                    section_id=section.section_id,
                    section_title=section.title,
                    chunk_contributions=contributions,
                    challenger_iterations=final_cycle,
                    issues_raised=list(raised),
                    issues_resolved=list(resolved),
                )
            )

        return TraceReport(
            job_id=payload.job_id,
            artifact_id=payload.artifact_ref.artifact_id if payload.artifact_ref else "",
            artifact_version=payload.artifact_ref.artifact_version if payload.artifact_ref else "",
            section_traces=section_traces,
            total_chunks=len(job_data["chunks"]),
            total_sections=len(doc_model.sections),
            total_issues_raised=len(all_issues_raised),
            total_issues_resolved=len(all_issues_resolved),
            final_cycle=final_cycle,
            generated_at=datetime.utcnow().isoformat(),
        )

    def _generate_statistics(
        self,
        payload: DocFinalizePayload,
        job_data: dict[str, Any],
        trace_report: TraceReport,
    ) -> JobStatistics:
        """Generate summary statistics for the job.

        Args:
            payload: The finalize payload.
            job_data: Gathered job data.
            trace_report: Generated trace report.

        Returns:
            JobStatistics with high-level metrics.
        """
        # Calculate average confidence
        confidences: list[float] = []
        for result in job_data["chunk_results"]:
            confidences.append(result.confidence)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Count unresolved blockers
        blockers_remaining = 0
        for challenge in job_data["challenge_results"]:
            for issue in challenge.issues:
                if issue.severity == IssueSeverity.BLOCKER:
                    if issue.issue_id not in {
                        f.payload.issue_id
                        for f in job_data["followups"]
                        if f.status == WorkItemStatus.DONE
                        and hasattr(f.payload, "issue_id")
                    }:
                        blockers_remaining += 1

        # Calculate total lines analyzed
        total_lines = 0
        for result in job_data["chunk_results"]:
            locator = result.chunk_locator
            if locator.get("start_line") and locator.get("end_line"):
                total_lines += locator["end_line"] - locator["start_line"] + 1

        # Determine status
        status = "accepted"
        if blockers_remaining > 0:
            status = "completed_with_blockers"
        elif trace_report.total_issues_raised > trace_report.total_issues_resolved:
            status = "completed_with_warnings"

        return JobStatistics(
            job_id=payload.job_id,
            artifact_id=payload.artifact_ref.artifact_id if payload.artifact_ref else "",
            status=status,
            total_chunks=trace_report.total_chunks,
            total_merges=len(job_data["merges"]),
            challenger_cycles=trace_report.final_cycle,
            issues_raised=trace_report.total_issues_raised,
            issues_resolved=trace_report.total_issues_resolved,
            blockers_remaining=blockers_remaining,
            average_confidence=round(avg_confidence, 3),
            total_lines_analyzed=total_lines,
            final_doc_uri=payload.output_doc_uri,
            final_trace_uri=payload.output_trace_uri,
        )

    def _generate_final_doc(
        self,
        doc_content: str,
        doc_model: DocumentationModel,
        statistics: JobStatistics,
    ) -> str:
        """Generate the final Markdown documentation.

        Adds metadata header and footer with provenance information.

        Args:
            doc_content: Original documentation content.
            doc_model: Documentation model.
            statistics: Job statistics.

        Returns:
            Final Markdown document with metadata.
        """
        # Build header with metadata
        header_lines = [
            "---",
            f"artifact: {statistics.artifact_id}",
            f"generated: {datetime.utcnow().isoformat()}",
            f"status: {statistics.status}",
            f"confidence: {statistics.average_confidence:.1%}",
            f"challenger_cycles: {statistics.challenger_cycles}",
            "---",
            "",
        ]

        # Build footer with summary
        footer_lines = [
            "",
            "---",
            "",
            "## Document Summary",
            "",
            f"- **Total Sections**: {len(doc_model.sections)}",
            f"- **Source Lines Analyzed**: {statistics.total_lines_analyzed}",
            f"- **Issues Raised**: {statistics.issues_raised}",
            f"- **Issues Resolved**: {statistics.issues_resolved}",
            f"- **Average Confidence**: {statistics.average_confidence:.1%}",
            "",
            f"*Generated by Atlas Documentation System*",
            "",
        ]

        return "\n".join(header_lines) + doc_content + "\n".join(footer_lines)

    async def _generate_pdf(self, doc_content: str, output_uri: str) -> str | None:
        """Generate PDF from Markdown content.

        Note: This is a placeholder. PDF generation would require
        additional dependencies (e.g., weasyprint, pandoc).

        Args:
            doc_content: Markdown content.
            output_uri: URI for PDF output.

        Returns:
            Output URI if successful, None otherwise.
        """
        # PDF generation is optional and requires external tools
        # For now, log a warning and skip
        logger.warning(
            "PDF generation requested but not implemented. "
            "Would require additional dependencies (weasyprint, pandoc, etc.)."
        )
        return None

    async def _cleanup_intermediates(self, job_id: str) -> int:
        """Clean up intermediate artifacts.

        Removes chunk results, merge results, and other intermediate
        files while preserving final deliverables.

        Args:
            job_id: The job identifier.

        Returns:
            Number of artifacts deleted.
        """
        deleted = 0

        # List of URI patterns to clean up
        cleanup_patterns = [
            f"chunks/{job_id}",
            f"merges/{job_id}",
            f"followups/{job_id}",
            f"challenges/{job_id}",
        ]

        for pattern in cleanup_patterns:
            try:
                artifacts = await self.artifact_store.list_artifacts(pattern)
                for uri in artifacts:
                    if await self.artifact_store.delete(uri):
                        deleted += 1
                        logger.debug(f"Deleted intermediate artifact: {uri}")
            except Exception as e:
                logger.warning(f"Error cleaning up {pattern}: {e}")

        logger.info(f"Cleaned up {deleted} intermediate artifacts for job {job_id}")
        return deleted

    async def _output_exists(self, work_item: WorkItem) -> bool:
        """Check if output already exists (idempotency check).

        Args:
            work_item: The work item to check.

        Returns:
            True if all output artifacts exist.
        """
        if not isinstance(work_item.payload, DocFinalizePayload):
            return False

        payload = work_item.payload

        # Check all required outputs exist
        doc_exists = await self.artifact_store.exists(payload.output_doc_uri)
        trace_exists = await self.artifact_store.exists(payload.output_trace_uri)
        summary_exists = await self.artifact_store.exists(payload.output_summary_uri)

        return doc_exists and trace_exists and summary_exists

    async def _load_existing_result(
        self, payload: DocFinalizePayload
    ) -> FinalizeResult:
        """Load existing result for idempotent reprocessing.

        Args:
            payload: The finalize payload.

        Returns:
            FinalizeResult loaded from existing artifacts.
        """
        # Load statistics to get status
        try:
            stats_data = await self.artifact_store.read_json(payload.output_summary_uri)
            statistics = JobStatistics(**stats_data)
            status = statistics.status
        except Exception:
            status = "unknown"

        return FinalizeResult(
            job_id=payload.job_id,
            artifact_id=payload.artifact_ref.artifact_id if payload.artifact_ref else "",
            artifact_version=payload.artifact_ref.artifact_version if payload.artifact_ref else "",
            status=status,
            doc_uri=payload.output_doc_uri,
            pdf_uri=payload.output_pdf_uri,
            trace_uri=payload.output_trace_uri,
            summary_uri=payload.output_summary_uri,
            warnings=["Loaded from existing output (idempotent)"],
        )
