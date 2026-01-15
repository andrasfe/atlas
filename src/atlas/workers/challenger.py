"""Challenger worker for documentation review.

The Challenger reviews merged documentation and raises issues about
unclear, incomplete, inconsistent, or missing information.

Key Responsibilities:
- Review documentation against quality criteria
- Identify issues with severity levels
- Propose resolution plans with bounded scopes
- Provide routing hints for follow-up dispatch
"""

from abc import abstractmethod
from typing import Any

from atlas.workers.base import Worker
from atlas.models.work_item import WorkItem, DocChallengePayload
from atlas.models.results import (
    ChallengeResult,
    Issue,
    ResolutionPlan,
    DocumentationModel,
)
from atlas.models.enums import WorkItemType, IssueSeverity


class Challenger(Worker):
    """Worker for reviewing documentation quality.

    Challengers claim DOC_CHALLENGE work items and produce ChallengeResult
    artifacts containing issues and resolution plans.

    Design Principles:
        - Review for completeness, clarity, consistency
        - Identify specific issues with severity levels
        - Provide routing hints for follow-up dispatch
        - Propose bounded scopes for resolution

    Review Focus Areas:
        - Error handling coverage
        - I/O operations documentation
        - Restart/recovery logic
        - Data flow traceability
        - Cross-references and consistency

    Output Schema (ChallengeResult):
        - issues: List of identified issues with severity
        - resolution_plan: Recommended follow-up tasks

    Example Implementation:
        >>> class QualityChallenger(Challenger):
        ...     async def review_documentation(self, doc, doc_model, profile):
        ...         # Build review prompt
        ...         messages = self._build_prompt(doc, doc_model, profile)
        ...         # Get LLM review
        ...         response = await self.llm.complete_json(messages, REVIEW_SCHEMA)
        ...         # Create result
        ...         return ChallengeResult(**response)

    TODO: Implement concrete challengers for different review profiles.
    """

    @property
    def supported_work_types(self) -> list[WorkItemType]:
        """Challenger supports documentation review."""
        return [WorkItemType.DOC_CHALLENGE]

    async def process(self, work_item: WorkItem) -> ChallengeResult:
        """Process a challenge work item.

        Args:
            work_item: The challenge work item.

        Returns:
            ChallengeResult with issues and resolution plan.
        """
        payload = work_item.payload
        if not isinstance(payload, DocChallengePayload):
            raise ValueError("Expected DocChallengePayload")

        # Load documentation
        doc = await self.artifact_store.read_text(payload.doc_uri)
        doc_model = await self.artifact_store.read_json(payload.doc_model_uri)

        # Perform review
        result = await self.review_documentation(
            doc,
            DocumentationModel(**doc_model),
            payload.challenge_profile,
        )

        # Write result artifact
        await self.artifact_store.write_json(
            payload.output_uri,
            result.model_dump(),
        )

        return result

    @abstractmethod
    async def review_documentation(
        self,
        doc: str,
        doc_model: DocumentationModel,
        challenge_profile: str,
    ) -> ChallengeResult:
        """Review documentation and identify issues.

        Args:
            doc: The documentation text.
            doc_model: The documentation model with traceability.
            challenge_profile: What to look for.

        Returns:
            ChallengeResult with issues and resolution plan.

        TODO: Implement review logic for different profiles.
        """
        pass

    def create_issue(
        self,
        question: str,
        severity: IssueSeverity,
        doc_section_refs: list[str] | None = None,
        suspected_scopes: list[str] | None = None,
        routing_hints: dict[str, list[str]] | None = None,
    ) -> Issue:
        """Create an issue with appropriate routing information.

        Helper method for creating well-formed issues.

        Args:
            question: The problem statement or question.
            severity: Issue severity level.
            doc_section_refs: Which doc sections are unclear.
            suspected_scopes: Chunk IDs or paragraph names.
            routing_hints: Symbols/paragraphs/files mentioned.

        Returns:
            Issue with generated ID.
        """
        import hashlib

        # Generate deterministic issue ID
        id_content = f"{question}:{severity.value}"
        issue_id = hashlib.sha256(id_content.encode()).hexdigest()[:12]

        return Issue(
            issue_id=issue_id,
            severity=severity,
            question=question,
            doc_section_refs=doc_section_refs or [],
            suspected_scopes=suspected_scopes or [],
            routing_hints=routing_hints or {},
        )

    def create_resolution_plan(
        self,
        issues: list[Issue],
        doc_model: DocumentationModel,
    ) -> ResolutionPlan:
        """Create a resolution plan for identified issues.

        Uses the documentation model to map issues to bounded scopes.

        Args:
            issues: List of identified issues.
            doc_model: Documentation model for routing.

        Returns:
            ResolutionPlan with follow-up tasks.

        TODO: Implement intelligent routing based on doc_model.
        """
        from atlas.models.results import FollowupTask

        tasks = []
        for issue in issues:
            if issue.severity in [IssueSeverity.BLOCKER, IssueSeverity.MAJOR]:
                # Create follow-up task
                scope = self._compute_scope(issue, doc_model)
                tasks.append(
                    FollowupTask(
                        issue_id=issue.issue_id,
                        scope=scope,
                        description=issue.question,
                    )
                )

        requires_patch = len(tasks) > 0

        return ResolutionPlan(
            followup_tasks=tasks,
            requires_patch_merge=requires_patch,
        )

    def _compute_scope(
        self,
        issue: Issue,
        doc_model: DocumentationModel,
    ) -> dict[str, Any]:
        """Compute bounded scope for an issue.

        Uses routing information from the issue and doc model.

        Args:
            issue: The issue to scope.
            doc_model: Documentation model for lookup.

        Returns:
            Scope dictionary for follow-up.
        """
        scope: dict[str, Any] = {"issue_id": issue.issue_id}

        # Priority 1: Use suspected_scopes if they contain chunk IDs
        if issue.suspected_scopes:
            scope["chunk_ids"] = issue.suspected_scopes
            return scope

        # Priority 2: Use doc section source refs
        if issue.doc_section_refs:
            chunk_ids = set()
            for section_id in issue.doc_section_refs:
                for section in doc_model.sections:
                    if section.section_id == section_id:
                        chunk_ids.update(section.source_refs)
            if chunk_ids:
                scope["chunk_ids"] = list(chunk_ids)
                return scope

        # Priority 3: Use routing hints to look up in index
        if issue.routing_hints:
            chunk_ids = set()
            symbols = issue.routing_hints.get("symbols", [])
            for symbol in symbols:
                if symbol in doc_model.index.symbol_to_chunks:
                    chunk_ids.update(doc_model.index.symbol_to_chunks[symbol])
            paragraphs = issue.routing_hints.get("paragraphs", [])
            for para in paragraphs:
                if para in doc_model.index.paragraph_to_chunk:
                    chunk_ids.add(doc_model.index.paragraph_to_chunk[para])
            if chunk_ids:
                scope["chunk_ids"] = list(chunk_ids)
                return scope

        # Fallback: Mark as cross-cutting
        scope["type"] = "cross_cutting"
        return scope

    async def _output_exists(self, work_item: WorkItem) -> bool:
        """Check if output already exists."""
        output_uri = self._get_output_uri(work_item)
        if output_uri:
            return await self.artifact_store.exists(output_uri)
        return False
