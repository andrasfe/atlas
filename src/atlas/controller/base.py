"""Abstract base class for the workflow controller.

The Controller orchestrates the entire analysis workflow, managing
work items, gating dependencies, and routing challenger issues.

Key Responsibilities:
- Create chunk and merge tickets based on manifest
- Gate merge tickets until prerequisites complete
- Dispatch follow-up work for challenger issues
- Trigger patch merges and re-challenges
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from atlas.adapters.ticket_system import TicketSystemAdapter
from atlas.adapters.artifact_store import ArtifactStoreAdapter
from atlas.models.manifest import Manifest
from atlas.models.work_item import WorkItem
from atlas.models.results import ChallengeResult, Issue
from atlas.models.enums import WorkItemStatus


@dataclass
class ControllerConfig:
    """Configuration for the controller.

    Attributes:
        max_concurrent_chunks: Max concurrent chunk analysis.
        max_concurrent_merges: Max concurrent merges.
        max_challenge_iterations: Max challenger loop iterations.
        follow_up_batch_size: How many follow-ups to dispatch at once.
        poll_interval_seconds: How often to poll for state changes.
    """

    max_concurrent_chunks: int = 50
    max_concurrent_merges: int = 10
    max_challenge_iterations: int = 3
    follow_up_batch_size: int = 10
    poll_interval_seconds: float = 5.0


@dataclass
class ReconcileResult:
    """Result of a controller reconcile operation.

    Attributes:
        work_items_created: Number of new work items created.
        work_items_unblocked: Number of work items transitioned to READY.
        work_items_completed: Number of work items now DONE.
        errors: Any errors during reconciliation.
        phase: Current workflow phase.
    """

    work_items_created: int = 0
    work_items_unblocked: int = 0
    work_items_completed: int = 0
    errors: list[str] = field(default_factory=list)
    phase: str = "unknown"


class Controller(ABC):
    """Abstract interface for workflow orchestration.

    The Controller manages the end-to-end workflow for analyzing
    artifacts. It operates as a reconciliation loop, continuously
    observing state and advancing work.

    Design Principles:
        - Reconcile loop: observe state -> compute missing work -> create it -> gate/advance
        - Idempotent: safe to re-run after crashes/timeouts without duplicate work
        - Bounded: follow-ups target bounded scopes, not whole-program analysis

    Workflow Phases:
        A. Request & Plan - Create manifest with chunks and merge DAG
        B. Chunk Analysis - Scribes analyze chunks
        C. Hierarchical Merge - Aggregators merge results bottom-up
        D. Challenger Review - Challenger reviews and raises issues
        E. Follow-up Dispatch - Controller routes issues to follow-ups
        F. Re-challenge Loop - Optional re-challenge until acceptance

    Example Implementation:
        >>> class SimpleController(Controller):
        ...     async def reconcile(self, job_id: str) -> ReconcileResult:
        ...         # Get current state
        ...         manifest = await self._get_manifest(job_id)
        ...         work_items = await self.ticket_system.query_by_job(job_id)
        ...         # Compute missing work
        ...         missing = self._compute_missing_work(manifest, work_items)
        ...         # Create and gate work items
        ...         await self._create_work_items(missing)
        ...         await self._unblock_ready_items(work_items)
        ...         return ReconcileResult(...)

    TODO: Implement concrete controllers for:
        - Simple single-threaded controller
        - Distributed controller with worker coordination
    """

    def __init__(
        self,
        ticket_system: TicketSystemAdapter,
        artifact_store: ArtifactStoreAdapter,
        config: ControllerConfig | None = None,
    ):
        """Initialize the controller.

        Args:
            ticket_system: Ticket system adapter.
            artifact_store: Artifact store adapter.
            config: Controller configuration.
        """
        self.ticket_system = ticket_system
        self.artifact_store = artifact_store
        self.config = config or ControllerConfig()

    @abstractmethod
    async def reconcile(self, job_id: str) -> ReconcileResult:
        """Run one reconciliation cycle for a job.

        Observes current state, computes missing work, creates it,
        and gates/advances work items as appropriate.

        Args:
            job_id: The job identifier.

        Returns:
            ReconcileResult with actions taken.

        TODO: Implement reconciliation logic.
        """
        pass

    @abstractmethod
    async def initialize_job(
        self,
        manifest: Manifest,
    ) -> str:
        """Initialize a new analysis job from a manifest.

        Creates all chunk and merge work items in appropriate states.

        Args:
            manifest: The workflow manifest.

        Returns:
            The job ID.

        TODO: Implement job initialization.
        """
        pass

    @abstractmethod
    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get current status of a job.

        Returns progress metrics and current phase.

        Args:
            job_id: The job identifier.

        Returns:
            Status dictionary with metrics.

        TODO: Implement status reporting.
        """
        pass

    @abstractmethod
    async def advance_blocked_items(self, job_id: str) -> int:
        """Check and advance BLOCKED work items.

        For each BLOCKED item, check if dependencies are DONE
        and transition to READY if so.

        Args:
            job_id: The job identifier.

        Returns:
            Number of items advanced to READY.

        TODO: Implement dependency checking and status transitions.
        """
        pass

    @abstractmethod
    async def route_challenger_issues(
        self,
        job_id: str,
        challenge_result: ChallengeResult,
    ) -> list[WorkItem]:
        """Route challenger issues to follow-up work items.

        Uses the routing algorithm from the spec:
        1. If issue has suspected_scopes with chunk IDs, use those
        2. Else if issue references doc sections with source refs, use those
        3. Else if issue references symbols/paragraphs, consult indexes
        4. Else create bounded cross-cutting follow-up plan

        Args:
            job_id: The job identifier.
            challenge_result: The challenger's output.

        Returns:
            List of created follow-up work items.

        TODO: Implement issue routing logic.
        """
        pass

    @abstractmethod
    async def create_patch_merge(
        self,
        job_id: str,
        followup_work_ids: list[str],
    ) -> WorkItem:
        """Create a patch merge work item.

        Called when follow-ups are complete and documentation
        needs to be updated.

        Args:
            job_id: The job identifier.
            followup_work_ids: IDs of completed follow-up work items.

        Returns:
            The created patch merge work item.

        TODO: Implement patch merge creation.
        """
        pass

    def compute_idempotency_key(
        self,
        job_id: str,
        work_type: str,
        artifact_version: str,
        identifier: str,
    ) -> str:
        """Compute a stable idempotency key for work item creation.

        Args:
            job_id: Job identifier.
            work_type: Work type name.
            artifact_version: Artifact version hash.
            identifier: Type-specific identifier (chunk_id, merge_node_id, etc.).

        Returns:
            Stable idempotency key string.
        """
        return f"{job_id}:{work_type}:{artifact_version}:{identifier}"

    async def is_job_complete(self, job_id: str) -> bool:
        """Check if a job is complete.

        A job is complete when:
        - Root merge is DONE, AND
        - Either no challenger issues OR all follow-ups and patch merge DONE

        Args:
            job_id: The job identifier.

        Returns:
            True if job is complete.
        """
        status = await self.get_job_status(job_id)
        return status.get("phase") == "complete"

    async def should_rechallenge(
        self,
        job_id: str,
        cycle_number: int,
    ) -> bool:
        """Determine if another challenger cycle should run.

        Based on:
        - Max iterations not exceeded
        - Previous challenge had actionable issues
        - Policy allows continuation

        Args:
            job_id: The job identifier.
            cycle_number: Current cycle number.

        Returns:
            True if another challenge should run.
        """
        if cycle_number >= self.config.max_challenge_iterations:
            return False

        # TODO: Check previous challenge results and policy
        return True

    def scope_fits_context(
        self,
        scope: dict[str, Any],
        context_budget: int,
    ) -> bool:
        """Check if a follow-up scope fits within context budget.

        Args:
            scope: The proposed scope.
            context_budget: Maximum tokens.

        Returns:
            True if scope is bounded appropriately.
        """
        # TODO: Implement scope size estimation
        chunk_ids = scope.get("chunk_ids", [])
        return len(chunk_ids) <= 5  # Conservative default

    def split_cross_cutting_scope(
        self,
        issue: Issue,
        manifest: Manifest,
    ) -> list[dict[str, Any]]:
        """Split a cross-cutting issue into bounded scopes.

        When an issue requires whole-program analysis, split it
        into bounded follow-ups per division or merge level.

        Args:
            issue: The cross-cutting issue.
            manifest: The workflow manifest.

        Returns:
            List of bounded scope dictionaries.

        TODO: Implement intelligent scope splitting.
        """
        # TODO: Implement cross-cutting scope splitting
        # Default: one scope per procedure division chunk kind
        return [{"issue_id": issue.issue_id, "chunk_ids": [], "type": "cross_cutting"}]
