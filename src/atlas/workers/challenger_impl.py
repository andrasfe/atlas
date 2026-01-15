"""Challenger worker implementation for documentation review.

The Challenger reviews merged documentation and raises issues about
unclear, incomplete, inconsistent, or missing information.

This implementation:
- Reviews documentation against quality criteria
- Identifies issues with severity levels (BLOCKER, MAJOR, MINOR, QUESTION)
- Proposes resolution plans with bounded scopes
- Provides routing hints for follow-up dispatch
"""

import logging
from typing import Any

from atlas.adapters.artifact_store import ArtifactStoreAdapter
from atlas.adapters.llm import LLMAdapter, LLMMessage, LLMRole
from atlas.adapters.ticket_system import TicketSystemAdapter
from atlas.models.enums import IssueSeverity
from atlas.models.results import (
    ChallengeResult,
    DocumentationModel,
    FollowupTask,
    Issue,
    ResolutionPlan,
)
from atlas.workers.challenger import Challenger

logger = logging.getLogger(__name__)


CHALLENGE_SYSTEM_PROMPT = """You are an expert documentation reviewer analyzing generated documentation for quality issues.

Your task is to review the documentation and identify:
1. BLOCKER issues - Critical gaps that make the documentation misleading or dangerous
2. MAJOR issues - Significant omissions that affect understanding
3. MINOR issues - Small improvements or clarifications needed
4. QUESTION - Clarification questions that need answers

Focus areas based on challenge profile:
- error_handling: Error handling coverage, FILE STATUS checks, ON ERROR clauses
- io_operations: I/O operations documentation, file access patterns
- restartability: Restart/recovery logic, checkpoint handling
- data_flow: Data flow traceability, variable transformations
- comprehensive: All of the above

For each issue, provide:
- A clear question or problem statement
- Which documentation sections are affected
- Suspected scopes (chunk IDs or paragraph names if known)
- Routing hints (symbols, paragraphs, files that might help answer)

Produce valid JSON:
{
  "issues": [
    {
      "issue_id": "unique-id",
      "severity": "blocker|major|minor|question",
      "question": "Clear problem statement or question",
      "doc_section_refs": ["section1", "section2"],
      "suspected_scopes": ["chunk_id_1", "paragraph_name"],
      "routing_hints": {
        "symbols": ["WS-FILE-STATUS", "INPUT-FILE"],
        "paragraphs": ["ERROR-HANDLING", "PROCESS-FILE"],
        "files": ["file1", "file2"]
      }
    }
  ],
  "summary": "Overall quality assessment",
  "recommendations": ["High-level recommendations"]
}

Be thorough but fair - only flag genuine issues, not stylistic preferences.
"""

CHALLENGE_USER_PROMPT = """Review the following documentation for {artifact_id}.

Challenge Profile: {challenge_profile}
Cycle: {cycle}

---DOCUMENTATION START---
{doc_content}
---DOCUMENTATION END---

---DOC MODEL (for routing context)---
Sections: {sections_summary}
Index available: {has_index}
---END DOC MODEL---

Identify all quality issues according to the challenge profile.
For each issue, provide clear routing information to help dispatch follow-up work.
"""

# Profile-specific focus areas
PROFILE_FOCUS = {
    "error_handling": """
Focus on:
- Are FILE STATUS variables checked after I/O operations?
- Are all error conditions documented (88-level status values)?
- Is error recovery logic clearly explained?
- Are ABEND conditions identified?
- Is ON ERROR handling documented?
""",
    "io_operations": """
Focus on:
- Are all files identified with their DD names?
- Are READ/WRITE/REWRITE/DELETE operations documented?
- Is the file access mode (sequential, random, dynamic) clear?
- Are record layouts referenced?
- Is file opening/closing logic clear?
""",
    "restartability": """
Focus on:
- Is checkpoint/restart logic documented?
- Are commit points identified?
- Is recovery from partial completion explained?
- Are counters and accumulators that need preserving identified?
""",
    "data_flow": """
Focus on:
- Are input-to-output transformations clear?
- Are intermediate variables and their purposes explained?
- Is the flow of data through paragraphs traceable?
- Are copybook structures and their usage documented?
""",
    "comprehensive": """
Review ALL aspects:
- Error handling completeness
- I/O operations coverage
- Restartability considerations
- Data flow clarity
- Overall documentation quality
""",
    "standard": """
General quality review:
- Is the documentation complete and accurate?
- Are there gaps in coverage?
- Is the information clear and understandable?
- Are there any inconsistencies?
""",
}


class ChallengerWorker(Challenger):
    """Concrete implementation of the Challenger worker.

    Reviews documentation and identifies issues with routing information
    for follow-up dispatch.

    Design Principles:
        - Review for completeness, clarity, consistency
        - Identify specific issues with severity levels
        - Provide routing hints for follow-up dispatch
        - Propose bounded scopes for resolution
    """

    def __init__(
        self,
        worker_id: str,
        ticket_system: TicketSystemAdapter,
        artifact_store: ArtifactStoreAdapter,
        llm: LLMAdapter,
    ):
        """Initialize the challenger worker.

        Args:
            worker_id: Unique identifier for this worker instance.
            ticket_system: Ticket system adapter.
            artifact_store: Artifact store adapter.
            llm: LLM adapter for review.
        """
        super().__init__(worker_id, ticket_system, artifact_store, llm)

    async def review_documentation(
        self,
        doc: str,
        doc_model: DocumentationModel,
        challenge_profile: str,
    ) -> ChallengeResult:
        """Review documentation and identify issues.

        Analyzes the documentation against quality criteria based on
        the challenge profile and produces issues with routing information.

        Args:
            doc: The documentation text.
            doc_model: The documentation model with traceability.
            challenge_profile: What to look for (error_handling, io_operations, etc.).

        Returns:
            ChallengeResult with issues and resolution plan.
        """
        logger.debug(f"Reviewing documentation with profile: {challenge_profile}")

        # Get profile-specific focus
        profile_focus = PROFILE_FOCUS.get(challenge_profile, PROFILE_FOCUS["standard"])

        # Build sections summary
        sections_summary = []
        for section in doc_model.sections[:20]:  # Limit for context
            sections_summary.append(
                f"- {section.section_id}: {section.title} (refs: {section.source_refs[:3]})"
            )

        # Build system prompt with profile focus
        system_prompt = CHALLENGE_SYSTEM_PROMPT + f"\n\nProfile-specific focus:\n{profile_focus}"

        user_prompt = CHALLENGE_USER_PROMPT.format(
            artifact_id=doc_model.metadata.get("artifact_id", "unknown"),
            challenge_profile=challenge_profile,
            cycle=doc_model.metadata.get("cycle", 1),
            doc_content=doc[:12000],  # Truncate for context
            sections_summary="\n".join(sections_summary) or "No sections",
            has_index="Yes" if doc_model.index.symbol_to_chunks else "No",
        )

        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content=system_prompt),
            LLMMessage(role=LLMRole.USER, content=user_prompt),
        ]

        try:
            response = await self.llm.complete_json(messages)
            issues = self._parse_issues(response.get("issues", []))
        except Exception as e:
            logger.error(f"LLM review failed: {e}")
            issues = []

        # Create resolution plan
        resolution_plan = self.create_resolution_plan(issues, doc_model)

        # Get artifact info from doc_model metadata
        artifact_id = doc_model.metadata.get("artifact_id", "")
        artifact_version = doc_model.metadata.get("artifact_version", "")
        job_id = doc_model.metadata.get("job_id", "")

        return ChallengeResult(
            job_id=job_id,
            artifact_id=artifact_id,
            artifact_version=artifact_version,
            issues=issues,
            resolution_plan=resolution_plan,
            metadata={
                "challenge_profile": challenge_profile,
                "doc_uri": doc_model.doc_uri,
                "issues_count": len(issues),
                "blockers_count": sum(
                    1 for i in issues if i.severity == IssueSeverity.BLOCKER
                ),
            },
        )

    def _parse_issues(self, issues_data: list[dict[str, Any]]) -> list[Issue]:
        """Parse issue data from LLM response.

        Args:
            issues_data: Raw issue data from LLM.

        Returns:
            List of Issue objects.
        """
        issues: list[Issue] = []

        for i, issue_data in enumerate(issues_data):
            # Parse severity
            severity_str = issue_data.get("severity", "question").lower()
            try:
                severity = IssueSeverity(severity_str)
            except ValueError:
                severity = IssueSeverity.QUESTION

            # Generate issue ID if not provided
            issue_id = issue_data.get("issue_id")
            if not issue_id:
                import hashlib
                content = f"{issue_data.get('question', '')}:{i}"
                issue_id = hashlib.sha256(content.encode()).hexdigest()[:12]

            # Parse routing hints
            routing_hints = issue_data.get("routing_hints", {})
            if not isinstance(routing_hints, dict):
                routing_hints = {}

            issues.append(Issue(
                issue_id=issue_id,
                severity=severity,
                question=issue_data.get("question", "Unknown issue"),
                doc_section_refs=issue_data.get("doc_section_refs", []),
                suspected_scopes=issue_data.get("suspected_scopes", []),
                routing_hints=routing_hints,
            ))

        return issues

    def create_resolution_plan(
        self,
        issues: list[Issue],
        doc_model: DocumentationModel,
    ) -> ResolutionPlan:
        """Create a resolution plan for identified issues.

        Maps issues to bounded follow-up scopes using the documentation
        model for routing.

        Args:
            issues: List of identified issues.
            doc_model: Documentation model for routing.

        Returns:
            ResolutionPlan with follow-up tasks.
        """
        tasks: list[FollowupTask] = []

        for issue in issues:
            # Only create tasks for blocker/major issues
            if issue.severity not in [IssueSeverity.BLOCKER, IssueSeverity.MAJOR]:
                continue

            # Compute scope using parent class method
            scope = self._compute_scope(issue, doc_model)

            # Add question to scope for follow-up worker
            scope["question"] = issue.question

            tasks.append(FollowupTask(
                issue_id=issue.issue_id,
                scope=scope,
                description=issue.question,
            ))

        requires_patch = len(tasks) > 0

        return ResolutionPlan(
            followup_tasks=tasks,
            requires_patch_merge=requires_patch,
        )

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
