"""Controller reconcile loop implementation.

The reconciler implements the core reconciliation logic for the Atlas workflow,
managing work items through phases A-F as defined in the specification.

Key Responsibilities:
- Query for ready work items
- Route to appropriate workers
- Handle status transitions
- Support idempotent re-processing
"""

import logging
from datetime import datetime
from typing import Any

from atlas.adapters.artifact_store import ArtifactStoreAdapter
from atlas.adapters.ticket_system import TicketSystemAdapter
from atlas.controller.base import Controller, ControllerConfig, ReconcileResult
from atlas.models.enums import IssueSeverity, WorkItemStatus, WorkItemType
from atlas.models.manifest import Manifest, ChunkSpec, MergeNode
from atlas.models.results import ChallengeResult, DocumentationModel, Issue
from atlas.models.work_item import (
    WorkItem,
    WorkItemPayload,
    DocChunkPayload,
    DocMergePayload,
    DocChallengePayload,
    DocFollowupPayload,
    DocPatchMergePayload,
    ChunkLocator,
)

logger = logging.getLogger(__name__)


class ReconcileController(Controller):
    """Concrete implementation of the workflow controller.

    Implements the reconciliation loop that observes state, computes
    missing work, creates it, and gates/advances work items.

    The controller operates through six phases:
        A. Request & Plan - Create manifest with chunks and merge DAG
        B. Chunk Analysis - Scribes analyze chunks
        C. Hierarchical Merge - Aggregators merge results bottom-up
        D. Challenger Review - Challenger reviews and raises issues
        E. Follow-up Dispatch - Controller routes issues to follow-ups
        F. Re-challenge Loop - Optional re-challenge until acceptance

    Design Principles:
        - Idempotent: Safe to re-run after crashes without duplicate work
        - Bounded: Follow-ups target bounded scopes, not whole-program
        - Deterministic: Chunk boundaries and IDs are stable per artifact version
    """

    def __init__(
        self,
        ticket_system: TicketSystemAdapter,
        artifact_store: ArtifactStoreAdapter,
        config: ControllerConfig | None = None,
    ):
        """Initialize the reconcile controller.

        Args:
            ticket_system: Ticket system adapter for work item management.
            artifact_store: Artifact store adapter for manifest/result storage.
            config: Optional controller configuration.
        """
        super().__init__(ticket_system, artifact_store, config)
        self._manifest_cache: dict[str, Manifest] = {}

    async def reconcile(self, job_id: str) -> ReconcileResult:
        """Run one reconciliation cycle for a job.

        Observes current state, computes missing work, creates it,
        and gates/advances work items as appropriate.

        The reconciliation follows these steps:
        1. Load the job manifest
        2. Query all work items for the job
        3. Determine current phase based on work item states
        4. Execute phase-specific reconciliation logic
        5. Advance blocked items whose dependencies are complete

        Args:
            job_id: The job identifier.

        Returns:
            ReconcileResult with actions taken during this cycle.
        """
        result = ReconcileResult()
        errors: list[str] = []

        try:
            # Load manifest
            manifest = await self._get_manifest(job_id)
            if manifest is None:
                result.errors.append(f"Manifest not found for job {job_id}")
                result.phase = "error"
                return result

            # Query all work items for job
            work_items = await self.ticket_system.query_by_job(job_id)
            work_items_by_type = self._group_by_type(work_items)
            work_items_by_id = {item.work_id: item for item in work_items}

            # Determine current phase
            phase = self._determine_phase(manifest, work_items_by_type)
            result.phase = phase
            logger.info(f"Job {job_id} is in phase: {phase}")

            # Execute phase-specific logic
            if phase == "plan":
                # Phase A: Create chunk and merge work items
                created = await self._execute_phase_a(job_id, manifest, work_items_by_type)
                result.work_items_created = created

            elif phase == "chunk":
                # Phase B: Monitor chunk completion, advance merges when ready
                phase_b_result = await self._execute_phase_b(
                    job_id, manifest, work_items_by_type, work_items_by_id
                )
                result.work_items_unblocked = phase_b_result.get("unblocked", 0)
                if phase_b_result.get("errors"):
                    result.errors.extend(phase_b_result["errors"])

            elif phase == "merge":
                # Phase C: Advance merge items when dependencies complete
                unblocked = await self._advance_merge_items(
                    job_id, manifest, work_items_by_type, work_items_by_id
                )
                result.work_items_unblocked = unblocked

            elif phase == "challenge":
                # Phase D: Create challenge work item if needed
                created = await self._execute_phase_d(job_id, manifest, work_items_by_type)
                result.work_items_created = created

            elif phase == "followup":
                # Phase E: Process challenge result and dispatch follow-ups
                created = await self._execute_phase_e(job_id, manifest, work_items_by_type)
                result.work_items_created = created

            elif phase == "patch":
                # Phase E continued: Create patch merge when follow-ups complete
                created = await self._execute_patch_merge(job_id, manifest, work_items_by_type)
                result.work_items_created = created

            elif phase == "rechallenge":
                # Phase F: Re-run challenger on updated documentation
                created = await self._execute_phase_f(job_id, manifest, work_items_by_type)
                result.work_items_created = created

            elif phase == "complete":
                # Job is done
                logger.info(f"Job {job_id} is complete")

            # Always try to advance blocked items
            advanced = await self.advance_blocked_items(job_id)
            result.work_items_unblocked += advanced

            # Count completed items
            result.work_items_completed = sum(
                1 for item in work_items if item.status == WorkItemStatus.DONE
            )

        except Exception as e:
            logger.exception(f"Error during reconciliation for job {job_id}")
            errors.append(str(e))
            result.phase = "error"

        result.errors = errors
        return result

    async def initialize_job(self, manifest: Manifest) -> str:
        """Initialize a new analysis job from a manifest.

        Creates all chunk and merge work items in appropriate states.
        Chunk items are created as READY, merge items as BLOCKED.

        Args:
            manifest: The workflow manifest.

        Returns:
            The job ID.
        """
        job_id = manifest.job_id
        logger.info(f"Initializing job {job_id}")

        # Store manifest in artifact store
        manifest_uri = self._get_manifest_uri(job_id)
        await self.artifact_store.write_json(manifest_uri, manifest.model_dump())

        # Cache manifest
        self._manifest_cache[job_id] = manifest

        # Create chunk work items (READY)
        for chunk in manifest.chunks:
            await self._create_chunk_work_item(job_id, manifest, chunk)

        # Create merge work items (BLOCKED)
        for merge_node in manifest.merge_dag:
            await self._create_merge_work_item(job_id, manifest, merge_node)

        logger.info(
            f"Job {job_id} initialized with {len(manifest.chunks)} chunks "
            f"and {len(manifest.merge_dag)} merge nodes"
        )

        return job_id

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get current status of a job.

        Returns progress metrics and current phase.

        Args:
            job_id: The job identifier.

        Returns:
            Status dictionary with metrics including:
            - phase: Current workflow phase
            - chunks_total: Total chunk count
            - chunks_done: Completed chunk count
            - merges_total: Total merge count
            - merges_done: Completed merge count
            - is_complete: Whether job is finished
        """
        manifest = await self._get_manifest(job_id)
        work_items = await self.ticket_system.query_by_job(job_id)
        work_items_by_type = self._group_by_type(work_items)

        # Count by status
        chunks = work_items_by_type.get(WorkItemType.DOC_CHUNK, [])
        merges = work_items_by_type.get(WorkItemType.DOC_MERGE, [])
        challenges = work_items_by_type.get(WorkItemType.DOC_CHALLENGE, [])
        followups = work_items_by_type.get(WorkItemType.DOC_FOLLOWUP, [])

        chunks_done = sum(1 for c in chunks if c.status == WorkItemStatus.DONE)
        merges_done = sum(1 for m in merges if m.status == WorkItemStatus.DONE)
        challenges_done = sum(1 for c in challenges if c.status == WorkItemStatus.DONE)
        followups_done = sum(1 for f in followups if f.status == WorkItemStatus.DONE)

        # Determine phase
        phase = self._determine_phase(manifest, work_items_by_type) if manifest else "unknown"

        return {
            "job_id": job_id,
            "phase": phase,
            "chunks_total": len(chunks),
            "chunks_done": chunks_done,
            "chunks_in_progress": sum(
                1 for c in chunks if c.status == WorkItemStatus.IN_PROGRESS
            ),
            "merges_total": len(merges),
            "merges_done": merges_done,
            "merges_blocked": sum(1 for m in merges if m.status == WorkItemStatus.BLOCKED),
            "challenges_total": len(challenges),
            "challenges_done": challenges_done,
            "followups_total": len(followups),
            "followups_done": followups_done,
            "is_complete": phase == "complete",
        }

    async def advance_blocked_items(self, job_id: str) -> int:
        """Check and advance BLOCKED work items.

        For each BLOCKED item, check if dependencies are DONE
        and transition to READY if so.

        Args:
            job_id: The job identifier.

        Returns:
            Number of items advanced to READY.
        """
        advanced = 0

        # Query blocked items
        blocked_items = await self.ticket_system.query_by_status(
            WorkItemStatus.BLOCKED, job_id=job_id
        )

        for item in blocked_items:
            # Check if all dependencies are done
            deps_done = await self.ticket_system.check_dependencies_done(item.work_id)
            if deps_done:
                success = await self.ticket_system.update_status(
                    item.work_id,
                    WorkItemStatus.READY,
                    expected_status=WorkItemStatus.BLOCKED,
                )
                if success:
                    advanced += 1
                    logger.debug(
                        f"Advanced {item.work_id} from BLOCKED to READY"
                    )

        return advanced

    async def route_challenger_issues(
        self,
        job_id: str,
        challenge_result: ChallengeResult,
    ) -> list[WorkItem]:
        """Route challenger issues to follow-up work items.

        Uses the routing algorithm from the spec (section 9.2):
        1. If issue has suspected_scopes with chunk IDs, use those
        2. Else if issue references doc sections with source refs, use those
        3. Else if issue references symbols/paragraphs, consult indexes
        4. Else create bounded cross-cutting follow-up plan

        Args:
            job_id: The job identifier.
            challenge_result: The challenger's output.

        Returns:
            List of created follow-up work items.
        """
        manifest = await self._get_manifest(job_id)
        if manifest is None:
            return []

        # Get the doc model for routing
        work_items = await self.ticket_system.query_by_job(
            job_id, WorkItemType.DOC_MERGE
        )
        root_merge = None
        for item in work_items:
            if isinstance(item.payload, DocMergePayload):
                # Find the root merge
                merge_node = manifest.get_merge_node(item.payload.merge_node_id)
                if merge_node and merge_node.is_root and item.status == WorkItemStatus.DONE:
                    root_merge = item
                    break

        # Try to load doc model
        doc_model = None
        if root_merge and manifest.artifacts:
            doc_model_uri = self.artifact_store.generate_uri(
                manifest.artifacts.base_uri,
                manifest.artifacts.doc_model_path,
            )
            try:
                doc_model_data = await self.artifact_store.read_json(doc_model_uri)
                doc_model = DocumentationModel(**doc_model_data)
            except Exception:
                logger.warning(f"Could not load doc model from {doc_model_uri}")

        # Route each issue that requires follow-up
        followup_items: list[WorkItem] = []
        for issue in challenge_result.issues:
            if issue.severity not in [IssueSeverity.BLOCKER, IssueSeverity.MAJOR]:
                continue

            scopes = self._route_issue_to_scopes(issue, manifest, doc_model)

            for scope in scopes:
                # Check scope size constraint
                if not self.scope_fits_context(scope, manifest.context_budget):
                    # Split into smaller scopes
                    split_scopes = self.split_cross_cutting_scope(issue, manifest)
                    for split_scope in split_scopes:
                        work_item = await self._create_followup_work_item(
                            job_id, manifest, issue, split_scope
                        )
                        if work_item:
                            followup_items.append(work_item)
                else:
                    work_item = await self._create_followup_work_item(
                        job_id, manifest, issue, scope
                    )
                    if work_item:
                        followup_items.append(work_item)

        return followup_items

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
        """
        manifest = await self._get_manifest(job_id)
        if manifest is None:
            raise ValueError(f"Manifest not found for job {job_id}")

        # Gather follow-up answer URIs
        answer_uris: list[str] = []
        for work_id in followup_work_ids:
            item = await self.ticket_system.get_work_item(work_id)
            if item and isinstance(item.payload, DocFollowupPayload):
                answer_uris.append(item.payload.output_uri)

        # Get current doc URIs
        if manifest.artifacts is None:
            raise ValueError(f"Manifest has no artifacts config for job {job_id}")

        base_doc_uri = self.artifact_store.generate_uri(
            manifest.artifacts.base_uri,
            manifest.artifacts.doc_path,
        )
        base_doc_model_uri = self.artifact_store.generate_uri(
            manifest.artifacts.base_uri,
            manifest.artifacts.doc_model_path,
        )

        # Create output URIs for updated docs (include cycle number)
        cycle = manifest.cycle_number + 1
        output_doc_uri = self.artifact_store.generate_uri(
            manifest.artifacts.base_uri,
            f"cycle{cycle}/{manifest.artifacts.doc_path}",
        )
        output_doc_model_uri = self.artifact_store.generate_uri(
            manifest.artifacts.base_uri,
            f"cycle{cycle}/{manifest.artifacts.doc_model_path}",
        )

        # Create patch merge payload
        payload = DocPatchMergePayload(
            job_id=job_id,
            artifact_ref=manifest.artifact_ref,
            manifest_uri=self._get_manifest_uri(job_id),
            base_doc_uri=base_doc_uri,
            base_doc_model_uri=base_doc_model_uri,
            inputs=answer_uris,
            output_doc_uri=output_doc_uri,
            output_doc_model_uri=output_doc_model_uri,
        )

        # Compute idempotency key
        idem_key = self.compute_idempotency_key(
            job_id,
            WorkItemType.DOC_PATCH_MERGE.value,
            manifest.artifact_ref.artifact_version,
            f"cycle{cycle}",
        )

        # Check for existing item
        existing = await self.ticket_system.find_by_idempotency_key(idem_key)
        if existing:
            return existing

        # Create work item
        work_item = WorkItem(
            work_id=f"{job_id}-patch-merge-cycle{cycle}",
            work_type=WorkItemType.DOC_PATCH_MERGE,
            status=WorkItemStatus.READY,
            payload=payload,
            depends_on=followup_work_ids,
            cycle_number=cycle,
            idempotency_key=idem_key,
            created_at=datetime.utcnow().isoformat(),
        )

        await self.ticket_system.create_work_item(work_item)
        logger.info(f"Created patch merge work item for job {job_id} cycle {cycle}")

        return work_item

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------

    async def _get_manifest(self, job_id: str) -> Manifest | None:
        """Load manifest for a job, using cache if available.

        Args:
            job_id: The job identifier.

        Returns:
            Manifest if found, None otherwise.
        """
        if job_id in self._manifest_cache:
            return self._manifest_cache[job_id]

        manifest_uri = self._get_manifest_uri(job_id)
        try:
            data = await self.artifact_store.read_json(manifest_uri)
            manifest = Manifest(**data)
            self._manifest_cache[job_id] = manifest
            return manifest
        except Exception as e:
            logger.warning(f"Could not load manifest for job {job_id}: {e}")
            return None

    def _get_manifest_uri(self, job_id: str) -> str:
        """Get the manifest URI for a job."""
        return f"manifests/{job_id}/manifest.json"

    def _group_by_type(
        self, work_items: list[WorkItem]
    ) -> dict[WorkItemType, list[WorkItem]]:
        """Group work items by type."""
        result: dict[WorkItemType, list[WorkItem]] = {}
        for item in work_items:
            if item.work_type not in result:
                result[item.work_type] = []
            result[item.work_type].append(item)
        return result

    def _determine_phase(
        self,
        manifest: Manifest | None,
        work_items_by_type: dict[WorkItemType, list[WorkItem]],
    ) -> str:
        """Determine current workflow phase based on work item states.

        The workflow progresses through these phases:
        - plan: Need to create chunk and merge work items (Phase A)
        - chunk: Chunk analysis in progress (Phase B)
        - merge: Hierarchical merging in progress (Phase C)
        - challenge: Challenger review needed or in progress (Phase D)
        - followup: Follow-up work dispatched and in progress (Phase E)
        - patch: Patch merge needed or in progress (Phase E continued)
        - rechallenge: Re-challenge needed after patch merge (Phase F)
        - complete: Job finished successfully

        Args:
            manifest: The job manifest.
            work_items_by_type: Work items grouped by type.

        Returns:
            Phase name string.
        """
        if manifest is None:
            return "plan"

        chunks = work_items_by_type.get(WorkItemType.DOC_CHUNK, [])
        merges = work_items_by_type.get(WorkItemType.DOC_MERGE, [])
        challenges = work_items_by_type.get(WorkItemType.DOC_CHALLENGE, [])
        followups = work_items_by_type.get(WorkItemType.DOC_FOLLOWUP, [])
        patch_merges = work_items_by_type.get(WorkItemType.DOC_PATCH_MERGE, [])

        # Phase A (plan): If no work items exist, need to create them
        if not chunks and not merges:
            return "plan"

        # Phase B (chunk): Check if chunks are still being analyzed
        if chunks:
            chunks_done = all(c.status == WorkItemStatus.DONE for c in chunks)
            if not chunks_done:
                return "chunk"

        # Phase C (merge): Check if merges are complete
        if merges:
            merges_done = all(m.status == WorkItemStatus.DONE for m in merges)
            if not merges_done:
                return "merge"

        # Phase D (challenge): Check if challenger review is needed or in progress
        if not challenges:
            return "challenge"

        latest_challenge = max(challenges, key=lambda c: c.cycle_number)

        # If latest challenge is not done, still in challenge phase
        if latest_challenge.status != WorkItemStatus.DONE:
            return "challenge"

        # Phase E (followup/patch): Check if follow-ups exist and are in progress
        current_cycle_followups = [
            f for f in followups if f.cycle_number == latest_challenge.cycle_number
        ]

        if current_cycle_followups:
            # Check if all follow-ups are done
            followups_done = all(
                f.status == WorkItemStatus.DONE for f in current_cycle_followups
            )
            if not followups_done:
                return "followup"

            # Follow-ups done, check if patch merge is needed
            # Patch merge cycle_number = challenge cycle + 1
            next_patch_cycle = latest_challenge.cycle_number + 1
            current_cycle_patches = [
                p for p in patch_merges if p.cycle_number == next_patch_cycle
            ]

            if not current_cycle_patches:
                # Need to create patch merge
                return "patch"

            # Check if patch merge is complete
            latest_patch = current_cycle_patches[0]
            if latest_patch.status != WorkItemStatus.DONE:
                return "patch"

            # Phase F (rechallenge): Patch merge done, check if re-challenge needed
            # Re-challenge cycle uses the patch's cycle number
            next_challenge_cycle = next_patch_cycle
            if next_challenge_cycle <= self.config.max_challenge_iterations:
                # Check if re-challenge already exists
                rechallenge_exists = any(
                    c.cycle_number == next_challenge_cycle for c in challenges
                )
                if not rechallenge_exists:
                    return "rechallenge"

                # If rechallenge exists but not done, return challenge
                rechallenge = next(
                    (c for c in challenges if c.cycle_number == next_challenge_cycle),
                    None
                )
                if rechallenge and rechallenge.status != WorkItemStatus.DONE:
                    return "challenge"

        # If no follow-ups were created for this challenge, check why
        # Either no actionable issues or job should complete
        elif latest_challenge.status == WorkItemStatus.DONE:
            # Challenge done but no follow-ups - either complete or need to check issues
            # The reconcile loop will create follow-ups if needed
            # For now, assume complete if no follow-ups and challenge is done
            pass

        # Job is complete if we've reached here
        return "complete"

    async def _execute_phase_a(
        self,
        job_id: str,
        manifest: Manifest,
        work_items_by_type: dict[WorkItemType, list[WorkItem]],
    ) -> int:
        """Execute Phase A: Create chunk and merge work items.

        Args:
            job_id: The job identifier.
            manifest: The workflow manifest.
            work_items_by_type: Existing work items by type.

        Returns:
            Number of work items created.
        """
        created = 0
        existing_chunks = {
            item.payload.chunk_id
            for item in work_items_by_type.get(WorkItemType.DOC_CHUNK, [])
            if isinstance(item.payload, DocChunkPayload)
        }
        existing_merges = {
            item.payload.merge_node_id
            for item in work_items_by_type.get(WorkItemType.DOC_MERGE, [])
            if isinstance(item.payload, DocMergePayload)
        }

        # Create missing chunk items
        for chunk in manifest.chunks:
            if chunk.chunk_id not in existing_chunks:
                await self._create_chunk_work_item(job_id, manifest, chunk)
                created += 1

        # Create missing merge items
        for merge_node in manifest.merge_dag:
            if merge_node.merge_node_id not in existing_merges:
                await self._create_merge_work_item(job_id, manifest, merge_node)
                created += 1

        logger.info(f"Phase A: Created {created} work items for job {job_id}")
        return created

    async def _execute_phase_b(
        self,
        job_id: str,
        manifest: Manifest,
        work_items_by_type: dict[WorkItemType, list[WorkItem]],
        work_items_by_id: dict[str, WorkItem],
    ) -> dict[str, Any]:
        """Execute Phase B: Track chunk analysis completion and advance merges.

        Monitors chunk completion status, handles partial completion gracefully,
        and advances BLOCKED merge items whose dependencies are DONE.

        Phase B responsibilities:
        - Track overall chunk completion progress
        - Identify failed or stuck chunks
        - Advance merge items when their chunk dependencies complete
        - Report partial completion status for monitoring

        Args:
            job_id: The job identifier.
            manifest: The workflow manifest.
            work_items_by_type: Work items grouped by type.
            work_items_by_id: Work items by ID.

        Returns:
            Dictionary with:
            - unblocked: Number of merge items advanced to READY
            - chunks_done: Number of completed chunks
            - chunks_failed: Number of failed chunks
            - chunks_in_progress: Number of in-progress chunks
            - errors: List of error messages
        """
        result: dict[str, Any] = {
            "unblocked": 0,
            "chunks_done": 0,
            "chunks_failed": 0,
            "chunks_in_progress": 0,
            "chunks_ready": 0,
            "errors": [],
        }

        chunks = work_items_by_type.get(WorkItemType.DOC_CHUNK, [])
        if not chunks:
            return result

        # Count chunk statuses
        for chunk in chunks:
            if chunk.status == WorkItemStatus.DONE:
                result["chunks_done"] += 1
            elif chunk.status == WorkItemStatus.FAILED:
                result["chunks_failed"] += 1
            elif chunk.status == WorkItemStatus.IN_PROGRESS:
                result["chunks_in_progress"] += 1
            elif chunk.status == WorkItemStatus.READY:
                result["chunks_ready"] += 1

        total_chunks = len(chunks)
        logger.info(
            f"Phase B progress for job {job_id}: "
            f"{result['chunks_done']}/{total_chunks} done, "
            f"{result['chunks_in_progress']} in progress, "
            f"{result['chunks_failed']} failed"
        )

        # Check for failed chunks and decide how to proceed
        if result["chunks_failed"] > 0:
            failed_chunk_ids = [
                chunk.payload.chunk_id
                for chunk in chunks
                if chunk.status == WorkItemStatus.FAILED
                and isinstance(chunk.payload, DocChunkPayload)
            ]
            result["errors"].append(
                f"Job {job_id} has {result['chunks_failed']} failed chunks: "
                f"{failed_chunk_ids[:5]}{'...' if len(failed_chunk_ids) > 5 else ''}"
            )
            logger.warning(f"Job {job_id}: {result['chunks_failed']} chunks failed")

        # Advance merge items whose chunk dependencies are complete
        # This handles partial completion - merges can proceed when their
        # specific dependencies are done, even if other chunks are still running
        unblocked = await self._advance_merge_items(
            job_id, manifest, work_items_by_type, work_items_by_id
        )
        result["unblocked"] = unblocked

        if unblocked > 0:
            logger.info(f"Phase B: Advanced {unblocked} merge items for job {job_id}")

        return result

    async def _advance_merge_items(
        self,
        job_id: str,
        manifest: Manifest,
        work_items_by_type: dict[WorkItemType, list[WorkItem]],
        work_items_by_id: dict[str, WorkItem],
    ) -> int:
        """Advance merge items whose dependencies are complete (Phase C).

        Implements hierarchical merge coordination by tracking the merge tree
        bottom-up. When all inputs for a merge node are DONE, the controller
        transitions that merge ticket to READY. This continues until the root
        merge produces doc_uri and doc_model_uri.

        The algorithm processes merges level-by-level to ensure proper
        bottom-up ordering:
        1. Group merge nodes by level
        2. Process lowest level first (leaf merges)
        3. Advance to higher levels as dependencies complete

        Args:
            job_id: The job identifier.
            manifest: The workflow manifest.
            work_items_by_type: Work items grouped by type.
            work_items_by_id: Work items by ID.

        Returns:
            Number of items advanced to READY.
        """
        advanced = 0
        merge_items = work_items_by_type.get(WorkItemType.DOC_MERGE, [])

        if not merge_items:
            return 0

        # Build a map from merge_node_id to work item
        merge_by_node_id: dict[str, WorkItem] = {}
        for item in merge_items:
            if isinstance(item.payload, DocMergePayload):
                merge_by_node_id[item.payload.merge_node_id] = item

        # Group merge nodes by level for bottom-up processing
        merges_by_level: dict[int, list[MergeNode]] = {}
        for node in manifest.merge_dag:
            if node.level not in merges_by_level:
                merges_by_level[node.level] = []
            merges_by_level[node.level].append(node)

        # Process levels in ascending order (bottom-up)
        for level in sorted(merges_by_level.keys()):
            for merge_node in merges_by_level[level]:
                work_item = merge_by_node_id.get(merge_node.merge_node_id)
                if work_item is None:
                    continue

                if work_item.status != WorkItemStatus.BLOCKED:
                    continue

                # Check if all dependencies are done
                deps_done = await self.ticket_system.check_dependencies_done(
                    work_item.work_id
                )
                if deps_done:
                    success = await self.ticket_system.update_status(
                        work_item.work_id,
                        WorkItemStatus.READY,
                        expected_status=WorkItemStatus.BLOCKED,
                    )
                    if success:
                        advanced += 1
                        logger.info(
                            f"Phase C: Advanced merge {merge_node.merge_node_id} "
                            f"(level {level}) to READY"
                        )

        # Check if root merge completed
        root_merge = manifest.get_root_merge_node()
        if root_merge:
            root_item = merge_by_node_id.get(root_merge.merge_node_id)
            if root_item and root_item.status == WorkItemStatus.DONE:
                logger.info(
                    f"Phase C: Root merge {root_merge.merge_node_id} completed "
                    f"for job {job_id}"
                )

        return advanced

    async def _execute_phase_d(
        self,
        job_id: str,
        manifest: Manifest,
        work_items_by_type: dict[WorkItemType, list[WorkItem]],
    ) -> int:
        """Execute Phase D: Create challenge work item when root merge completes.

        After the root merge produces doc_uri and doc_model_uri, the controller
        creates a DOC_CHALLENGE work item. When the challenger completes:
        - If no issues above threshold: job can proceed to finalize/complete
        - If issues exist: controller creates follow-up work (Phase E)

        Args:
            job_id: The job identifier.
            manifest: The workflow manifest.
            work_items_by_type: Existing work items by type.

        Returns:
            Number of work items created (0 or 1).
        """
        challenges = work_items_by_type.get(WorkItemType.DOC_CHALLENGE, [])
        patch_merges = work_items_by_type.get(WorkItemType.DOC_PATCH_MERGE, [])

        # Determine cycle number based on completed challenges/patches
        cycle = 1
        if challenges:
            # If we have patch merges, the next challenge cycle follows the patch
            if patch_merges:
                latest_patch = max(patch_merges, key=lambda p: p.cycle_number)
                if latest_patch.status == WorkItemStatus.DONE:
                    # Patch completed, cycle is patch's cycle number
                    cycle = latest_patch.cycle_number
            else:
                # No patches yet, use challenge count
                cycle = max(c.cycle_number for c in challenges) + 1

        # Check if challenge already exists for this cycle
        existing_for_cycle = [
            c for c in challenges if c.cycle_number == cycle
        ]
        if existing_for_cycle:
            return 0

        # Check idempotency
        idem_key = self.compute_idempotency_key(
            job_id,
            WorkItemType.DOC_CHALLENGE.value,
            manifest.artifact_ref.artifact_version,
            f"cycle{cycle}",
        )

        existing = await self.ticket_system.find_by_idempotency_key(idem_key)
        if existing:
            return 0

        # Get doc URIs - for cycle > 1, use the patched doc from previous cycle
        if manifest.artifacts is None:
            logger.error(f"No artifacts config for job {job_id}")
            return 0

        if cycle > 1:
            # Use updated documentation from previous patch merge
            doc_uri = self.artifact_store.generate_uri(
                manifest.artifacts.base_uri,
                f"cycle{cycle}/{manifest.artifacts.doc_path}",
            )
            doc_model_uri = self.artifact_store.generate_uri(
                manifest.artifacts.base_uri,
                f"cycle{cycle}/{manifest.artifacts.doc_model_path}",
            )
        else:
            doc_uri = self.artifact_store.generate_uri(
                manifest.artifacts.base_uri,
                manifest.artifacts.doc_path,
            )
            doc_model_uri = self.artifact_store.generate_uri(
                manifest.artifacts.base_uri,
                manifest.artifacts.doc_model_path,
            )

        output_uri = self.artifact_store.generate_uri(
            manifest.artifacts.base_uri,
            f"challenges/challenge_cycle{cycle}.json",
        )

        # Create challenge payload
        payload = DocChallengePayload(
            job_id=job_id,
            artifact_ref=manifest.artifact_ref,
            manifest_uri=self._get_manifest_uri(job_id),
            doc_uri=doc_uri,
            doc_model_uri=doc_model_uri,
            challenge_profile=manifest.review_policy.challenge_profile,
            output_uri=output_uri,
        )

        # Create work item
        work_item = WorkItem(
            work_id=f"{job_id}-challenge-cycle{cycle}",
            work_type=WorkItemType.DOC_CHALLENGE,
            status=WorkItemStatus.READY,
            payload=payload,
            cycle_number=cycle,
            idempotency_key=idem_key,
            created_at=datetime.utcnow().isoformat(),
        )

        await self.ticket_system.create_work_item(work_item)
        logger.info(f"Phase D: Created challenge work item for job {job_id} cycle {cycle}")
        return 1

    async def _handle_challenge_result(
        self,
        job_id: str,
        challenge_item: WorkItem,
        manifest: Manifest,
    ) -> tuple[bool, int]:
        """Handle the result of a completed challenge.

        Determines next action based on challenge issues:
        - No BLOCKER/MAJOR issues: return (needs_followup=False, 0)
        - Has actionable issues: return (needs_followup=True, issue_count)

        Args:
            job_id: The job identifier.
            challenge_item: The completed challenge work item.
            manifest: The workflow manifest.

        Returns:
            Tuple of (needs_followup, actionable_issue_count).
        """
        if not isinstance(challenge_item.payload, DocChallengePayload):
            return False, 0

        try:
            result_data = await self.artifact_store.read_json(
                challenge_item.payload.output_uri
            )
            challenge_result = ChallengeResult(**result_data)
        except Exception as e:
            logger.error(f"Could not load challenge result: {e}")
            return False, 0

        # Count actionable issues (BLOCKER or MAJOR)
        actionable_issues = [
            issue for issue in challenge_result.issues
            if issue.severity in [IssueSeverity.BLOCKER, IssueSeverity.MAJOR]
        ]

        if not actionable_issues:
            logger.info(
                f"Phase D: Challenge cycle {challenge_item.cycle_number} has no "
                f"actionable issues for job {job_id}"
            )
            return False, 0

        logger.info(
            f"Phase D: Challenge cycle {challenge_item.cycle_number} has "
            f"{len(actionable_issues)} actionable issues for job {job_id}"
        )
        return True, len(actionable_issues)

    async def _execute_phase_e(
        self,
        job_id: str,
        manifest: Manifest,
        work_items_by_type: dict[WorkItemType, list[WorkItem]],
    ) -> int:
        """Execute Phase E: Dispatch follow-up work items.

        Args:
            job_id: The job identifier.
            manifest: The workflow manifest.
            work_items_by_type: Existing work items by type.

        Returns:
            Number of follow-up work items created.
        """
        challenges = work_items_by_type.get(WorkItemType.DOC_CHALLENGE, [])
        if not challenges:
            return 0

        # Get latest completed challenge
        completed_challenges = [
            c for c in challenges if c.status == WorkItemStatus.DONE
        ]
        if not completed_challenges:
            return 0

        latest_challenge = max(completed_challenges, key=lambda c: c.cycle_number)

        # Load challenge result
        if not isinstance(latest_challenge.payload, DocChallengePayload):
            return 0

        try:
            result_data = await self.artifact_store.read_json(
                latest_challenge.payload.output_uri
            )
            challenge_result = ChallengeResult(**result_data)
        except Exception as e:
            logger.error(f"Could not load challenge result: {e}")
            return 0

        # Check if follow-ups already exist for this cycle
        existing_followups = work_items_by_type.get(WorkItemType.DOC_FOLLOWUP, [])
        existing_issues = {
            f.payload.issue_id
            for f in existing_followups
            if isinstance(f.payload, DocFollowupPayload)
            and f.cycle_number == latest_challenge.cycle_number
        }

        # Route issues to follow-ups
        created = 0
        for issue in challenge_result.issues:
            if issue.issue_id in existing_issues:
                continue

            if issue.severity not in [IssueSeverity.BLOCKER, IssueSeverity.MAJOR]:
                continue

            # Route this issue
            work_items = await self._route_single_issue(
                job_id, manifest, issue, latest_challenge.cycle_number
            )
            created += len(work_items)

        return created

    async def _execute_patch_merge(
        self,
        job_id: str,
        manifest: Manifest,
        work_items_by_type: dict[WorkItemType, list[WorkItem]],
    ) -> int:
        """Create patch merge when follow-ups are complete.

        Args:
            job_id: The job identifier.
            manifest: The workflow manifest.
            work_items_by_type: Existing work items by type.

        Returns:
            Number of patch merge items created.
        """
        followups = work_items_by_type.get(WorkItemType.DOC_FOLLOWUP, [])
        if not followups:
            return 0

        # Get latest cycle's followups
        latest_cycle = max(f.cycle_number for f in followups)
        cycle_followups = [f for f in followups if f.cycle_number == latest_cycle]

        # Check all are done
        if not all(f.status == WorkItemStatus.DONE for f in cycle_followups):
            return 0

        # Check if patch merge already exists
        patch_merges = work_items_by_type.get(WorkItemType.DOC_PATCH_MERGE, [])
        existing_patches = [p for p in patch_merges if p.cycle_number == latest_cycle + 1]
        if existing_patches:
            return 0

        # Create patch merge
        followup_ids = [f.work_id for f in cycle_followups]
        await self.create_patch_merge(job_id, followup_ids)
        return 1

    async def _execute_phase_f(
        self,
        job_id: str,
        manifest: Manifest,
        work_items_by_type: dict[WorkItemType, list[WorkItem]],
    ) -> int:
        """Execute Phase F: Re-challenge loop after patch merge.

        After a patch merge completes, the controller MAY re-run DOC_CHALLENGE
        on the updated documentation. The loop continues until:
        - Issues are resolved (no BLOCKER/MAJOR issues)
        - Iteration limit is reached (config.max_challenge_iterations)
        - Only minor issues remain (if policy allows)

        This implements the optional re-challenge loop per spec section 8 Phase F.

        Args:
            job_id: The job identifier.
            manifest: The workflow manifest.
            work_items_by_type: Existing work items by type.

        Returns:
            Number of work items created (0 or 1).
        """
        patch_merges = work_items_by_type.get(WorkItemType.DOC_PATCH_MERGE, [])
        challenges = work_items_by_type.get(WorkItemType.DOC_CHALLENGE, [])

        if not patch_merges:
            return 0

        # Get the latest completed patch merge
        completed_patches = [
            p for p in patch_merges if p.status == WorkItemStatus.DONE
        ]
        if not completed_patches:
            return 0

        latest_patch = max(completed_patches, key=lambda p: p.cycle_number)

        # Determine the next challenge cycle number
        next_cycle = latest_patch.cycle_number

        # Check iteration limit
        if next_cycle > self.config.max_challenge_iterations:
            logger.info(
                f"Phase F: Max iterations ({self.config.max_challenge_iterations}) "
                f"reached for job {job_id}. Completing without re-challenge."
            )
            return 0

        # Check if challenge already exists for this cycle
        existing_for_cycle = [
            c for c in challenges if c.cycle_number == next_cycle
        ]
        if existing_for_cycle:
            return 0

        # Check the previous challenge result to decide if re-challenge is needed
        if challenges:
            latest_challenge = max(challenges, key=lambda c: c.cycle_number)
            if latest_challenge.status == WorkItemStatus.DONE:
                needs_followup, issue_count = await self._handle_challenge_result(
                    job_id, latest_challenge, manifest
                )
                if not needs_followup:
                    # No actionable issues in previous challenge, no need to re-challenge
                    logger.info(
                        f"Phase F: Previous challenge had no actionable issues. "
                        f"Skipping re-challenge for job {job_id}"
                    )
                    return 0

        # Create the re-challenge work item
        logger.info(
            f"Phase F: Creating re-challenge cycle {next_cycle} for job {job_id} "
            f"(iteration {next_cycle}/{self.config.max_challenge_iterations})"
        )

        return await self._execute_phase_d(job_id, manifest, work_items_by_type)

    async def _create_chunk_work_item(
        self,
        job_id: str,
        manifest: Manifest,
        chunk: ChunkSpec,
    ) -> WorkItem:
        """Create a chunk work item.

        Args:
            job_id: The job identifier.
            manifest: The workflow manifest.
            chunk: The chunk specification.

        Returns:
            The created work item.
        """
        # Compute idempotency key
        idem_key = self.compute_idempotency_key(
            job_id,
            WorkItemType.DOC_CHUNK.value,
            manifest.artifact_ref.artifact_version,
            chunk.chunk_id,
        )

        # Check for existing
        existing = await self.ticket_system.find_by_idempotency_key(idem_key)
        if existing:
            return existing

        # Determine result URI
        result_uri = chunk.result_uri
        if not result_uri and manifest.artifacts:
            result_uri = self.artifact_store.generate_uri(
                manifest.artifacts.base_uri,
                manifest.artifacts.chunk_results_path,
                chunk_id=chunk.chunk_id,
            )

        # Create payload
        payload = DocChunkPayload(
            job_id=job_id,
            artifact_ref=manifest.artifact_ref,
            manifest_uri=self._get_manifest_uri(job_id),
            chunk_id=chunk.chunk_id,
            chunk_locator=ChunkLocator(
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                division=chunk.division,
                section=chunk.section,
                paragraphs=chunk.paragraphs,
            ),
            result_uri=result_uri or f"chunks/{chunk.chunk_id}.json",
        )

        # Create work item (READY - no dependencies)
        work_item = WorkItem(
            work_id=f"{job_id}-chunk-{chunk.chunk_id}",
            work_type=WorkItemType.DOC_CHUNK,
            status=WorkItemStatus.READY,
            payload=payload,
            idempotency_key=idem_key,
            created_at=datetime.utcnow().isoformat(),
        )

        await self.ticket_system.create_work_item(work_item)
        return work_item

    async def _create_merge_work_item(
        self,
        job_id: str,
        manifest: Manifest,
        merge_node: MergeNode,
    ) -> WorkItem:
        """Create a merge work item.

        Args:
            job_id: The job identifier.
            manifest: The workflow manifest.
            merge_node: The merge node specification.

        Returns:
            The created work item.
        """
        # Compute idempotency key
        idem_key = self.compute_idempotency_key(
            job_id,
            WorkItemType.DOC_MERGE.value,
            manifest.artifact_ref.artifact_version,
            merge_node.merge_node_id,
        )

        # Check for existing
        existing = await self.ticket_system.find_by_idempotency_key(idem_key)
        if existing:
            return existing

        # Determine input URIs and dependencies
        input_uris: list[str] = []
        depends_on: list[str] = []

        for input_id in merge_node.input_ids:
            # Check if input is a chunk or another merge node
            chunk = manifest.get_chunk(input_id)
            if chunk:
                # It's a chunk - get its result URI
                result_uri = chunk.result_uri
                if not result_uri and manifest.artifacts:
                    result_uri = self.artifact_store.generate_uri(
                        manifest.artifacts.base_uri,
                        manifest.artifacts.chunk_results_path,
                        chunk_id=input_id,
                    )
                input_uris.append(result_uri or f"chunks/{input_id}.json")
                depends_on.append(f"{job_id}-chunk-{input_id}")
            else:
                # It's a merge node - get its output URI
                child_merge = manifest.get_merge_node(input_id)
                if child_merge:
                    output_uri = child_merge.output_uri
                    if not output_uri and manifest.artifacts:
                        output_uri = self.artifact_store.generate_uri(
                            manifest.artifacts.base_uri,
                            manifest.artifacts.merge_results_path,
                            merge_node_id=input_id,
                        )
                    input_uris.append(output_uri or f"merges/{input_id}.json")
                    depends_on.append(f"{job_id}-merge-{input_id}")

        # Determine output URI
        output_uri = merge_node.output_uri
        if not output_uri and manifest.artifacts:
            output_uri = self.artifact_store.generate_uri(
                manifest.artifacts.base_uri,
                manifest.artifacts.merge_results_path,
                merge_node_id=merge_node.merge_node_id,
            )

        # Create payload
        payload = DocMergePayload(
            job_id=job_id,
            artifact_ref=manifest.artifact_ref,
            manifest_uri=self._get_manifest_uri(job_id),
            merge_node_id=merge_node.merge_node_id,
            input_uris=input_uris,
            output_uri=output_uri or f"merges/{merge_node.merge_node_id}.json",
        )

        # Create work item (BLOCKED - has dependencies)
        work_item = WorkItem(
            work_id=f"{job_id}-merge-{merge_node.merge_node_id}",
            work_type=WorkItemType.DOC_MERGE,
            status=WorkItemStatus.BLOCKED,
            payload=payload,
            depends_on=depends_on,
            idempotency_key=idem_key,
            created_at=datetime.utcnow().isoformat(),
        )

        await self.ticket_system.create_work_item(work_item)
        return work_item

    async def _create_followup_work_item(
        self,
        job_id: str,
        manifest: Manifest,
        issue: Issue,
        scope: dict[str, Any],
    ) -> WorkItem | None:
        """Create a follow-up work item for an issue.

        Args:
            job_id: The job identifier.
            manifest: The workflow manifest.
            issue: The issue to address.
            scope: The bounded scope for analysis.

        Returns:
            The created work item, or None if already exists.
        """
        # Compute idempotency key
        scope_id = "-".join(scope.get("chunk_ids", ["cross"]))[:50]
        idem_key = self.compute_idempotency_key(
            job_id,
            WorkItemType.DOC_FOLLOWUP.value,
            manifest.artifact_ref.artifact_version,
            f"{issue.issue_id}-{scope_id}",
        )

        # Check for existing
        existing = await self.ticket_system.find_by_idempotency_key(idem_key)
        if existing:
            return None

        # Gather input URIs
        input_uris: list[str] = []
        chunk_ids = scope.get("chunk_ids", [])
        for chunk_id in chunk_ids:
            chunk = manifest.get_chunk(chunk_id)
            if chunk and manifest.artifacts:
                result_uri = self.artifact_store.generate_uri(
                    manifest.artifacts.base_uri,
                    manifest.artifacts.chunk_results_path,
                    chunk_id=chunk_id,
                )
                input_uris.append(result_uri)

        # Determine output URI
        output_uri = ""
        if manifest.artifacts:
            output_uri = self.artifact_store.generate_uri(
                manifest.artifacts.base_uri,
                f"followups/{issue.issue_id}_{scope_id}.json",
            )

        # Create payload
        payload = DocFollowupPayload(
            job_id=job_id,
            artifact_ref=manifest.artifact_ref,
            manifest_uri=self._get_manifest_uri(job_id),
            issue_id=issue.issue_id,
            scope=scope,
            inputs=input_uris,
            output_uri=output_uri or f"followups/{issue.issue_id}.json",
        )

        # Get current challenge cycle
        challenges = await self.ticket_system.query_by_job(
            job_id, WorkItemType.DOC_CHALLENGE
        )
        cycle = 1
        if challenges:
            cycle = max(c.cycle_number for c in challenges)

        # Create work item
        work_item = WorkItem(
            work_id=f"{job_id}-followup-{issue.issue_id[:8]}-{scope_id[:20]}",
            work_type=WorkItemType.DOC_FOLLOWUP,
            status=WorkItemStatus.READY,
            payload=payload,
            cycle_number=cycle,
            idempotency_key=idem_key,
            created_at=datetime.utcnow().isoformat(),
        )

        await self.ticket_system.create_work_item(work_item)
        return work_item

    def _route_issue_to_scopes(
        self,
        issue: Issue,
        manifest: Manifest,
        doc_model: DocumentationModel | None,
    ) -> list[dict[str, Any]]:
        """Route an issue to bounded scopes for follow-up work.

        Implements the routing algorithm from spec section 9.2:

        Priority 1: If suspected_scopes contains chunk IDs -> create follow-ups per chunk
        Priority 2: If issue references doc sections with source refs -> use those chunk IDs
        Priority 3: If issue references symbols/paragraphs -> consult indexes
        Priority 4: Create bounded cross-cutting follow-up plan

        Scope size constraints (per spec 9.3):
        - A follow-up MUST target 1 chunk, a small list (max 3-5), a merge node output,
          or a single concern within one division
        - If issue requires whole-program re-analysis, controller MUST split into
          bounded follow-ups

        Args:
            issue: The issue to route.
            manifest: The workflow manifest.
            doc_model: The documentation model for routing.

        Returns:
            List of scope dictionaries, each bounded appropriately.
        """
        scopes: list[dict[str, Any]] = []
        max_chunks_per_scope = 5  # Per spec 9.3: max configurable, e.g., 3-5

        # Priority 1: Use suspected_scopes if they contain valid chunk IDs
        if issue.suspected_scopes:
            valid_chunk_ids = [
                s for s in issue.suspected_scopes
                if manifest.get_chunk(s) is not None
            ]
            if valid_chunk_ids:
                # Split into bounded scopes if too many chunks
                for i in range(0, len(valid_chunk_ids), max_chunks_per_scope):
                    batch = valid_chunk_ids[i:i + max_chunks_per_scope]
                    scopes.append({
                        "issue_id": issue.issue_id,
                        "chunk_ids": batch,
                        "routing_method": "suspected_scopes",
                    })
                logger.debug(
                    f"Routed issue {issue.issue_id} via suspected_scopes to "
                    f"{len(scopes)} scope(s)"
                )
                return scopes

        # Priority 2: Use doc section source refs if available
        if doc_model and issue.doc_section_refs:
            chunk_ids: set[str] = set()
            for section_id in issue.doc_section_refs:
                for section in doc_model.sections:
                    if section.section_id == section_id:
                        # Validate chunk IDs against manifest
                        for ref in section.source_refs:
                            if manifest.get_chunk(ref) is not None:
                                chunk_ids.add(ref)
            if chunk_ids:
                chunk_id_list = list(chunk_ids)
                for i in range(0, len(chunk_id_list), max_chunks_per_scope):
                    batch = chunk_id_list[i:i + max_chunks_per_scope]
                    scopes.append({
                        "issue_id": issue.issue_id,
                        "chunk_ids": batch,
                        "routing_method": "doc_section_refs",
                        "doc_sections": issue.doc_section_refs,
                    })
                logger.debug(
                    f"Routed issue {issue.issue_id} via doc_section_refs to "
                    f"{len(scopes)} scope(s)"
                )
                return scopes

        # Priority 3: Use routing hints via index (symbols/paragraphs/files)
        if issue.routing_hints:
            chunk_ids_from_hints: set[str] = set()

            # Check symbols
            symbols = issue.routing_hints.get("symbols", [])
            for symbol in symbols:
                if doc_model and symbol in doc_model.index.symbol_to_chunks:
                    for chunk_id in doc_model.index.symbol_to_chunks[symbol]:
                        if manifest.get_chunk(chunk_id) is not None:
                            chunk_ids_from_hints.add(chunk_id)

            # Check paragraphs
            paragraphs = issue.routing_hints.get("paragraphs", [])
            for para in paragraphs:
                if doc_model and para in doc_model.index.paragraph_to_chunk:
                    chunk_id = doc_model.index.paragraph_to_chunk[para]
                    if manifest.get_chunk(chunk_id) is not None:
                        chunk_ids_from_hints.add(chunk_id)
                else:
                    # Fall back to searching manifest chunks by paragraph name
                    for chunk in manifest.chunks:
                        if para in chunk.paragraphs:
                            chunk_ids_from_hints.add(chunk.chunk_id)

            # Check files
            files = issue.routing_hints.get("files", [])
            for file_name in files:
                if doc_model and file_name in doc_model.index.file_to_chunks:
                    for chunk_id in doc_model.index.file_to_chunks[file_name]:
                        if manifest.get_chunk(chunk_id) is not None:
                            chunk_ids_from_hints.add(chunk_id)

            if chunk_ids_from_hints:
                chunk_id_list = list(chunk_ids_from_hints)
                for i in range(0, len(chunk_id_list), max_chunks_per_scope):
                    batch = chunk_id_list[i:i + max_chunks_per_scope]
                    scopes.append({
                        "issue_id": issue.issue_id,
                        "chunk_ids": batch,
                        "routing_method": "routing_hints",
                        "hints": issue.routing_hints,
                    })
                logger.debug(
                    f"Routed issue {issue.issue_id} via routing_hints to "
                    f"{len(scopes)} scope(s)"
                )
                return scopes

        # Priority 4: Cross-cutting follow-up plan
        # Create bounded follow-ups per division or per merge level
        logger.debug(
            f"Issue {issue.issue_id} has no specific scope, creating "
            f"cross-cutting follow-up plan"
        )
        scopes = self.split_cross_cutting_scope(issue, manifest)
        return scopes

    async def _route_single_issue(
        self,
        job_id: str,
        manifest: Manifest,
        issue: Issue,
        cycle: int,
    ) -> list[WorkItem]:
        """Route a single issue to follow-up work items.

        Args:
            job_id: The job identifier.
            manifest: The workflow manifest.
            issue: The issue to route.
            cycle: Current challenge cycle.

        Returns:
            List of created work items.
        """
        # Try to load doc model
        doc_model = None
        if manifest.artifacts:
            doc_model_uri = self.artifact_store.generate_uri(
                manifest.artifacts.base_uri,
                manifest.artifacts.doc_model_path,
            )
            try:
                data = await self.artifact_store.read_json(doc_model_uri)
                doc_model = DocumentationModel(**data)
            except Exception:
                pass

        # Route to scopes
        scopes = self._route_issue_to_scopes(issue, manifest, doc_model)

        # Create work items
        work_items: list[WorkItem] = []
        for scope in scopes:
            work_item = await self._create_followup_work_item(
                job_id, manifest, issue, scope
            )
            if work_item:
                work_items.append(work_item)

        return work_items

    def split_cross_cutting_scope(
        self,
        issue: Issue,
        manifest: Manifest,
    ) -> list[dict[str, Any]]:
        """Split a cross-cutting issue into bounded scopes.

        When an issue appears to require whole-program re-analysis, the controller
        MUST split it into bounded follow-ups rather than creating one giant task.
        Per spec section 9.3, this creates:
        - One follow-up per merge-level or per division (e.g., PROCEDURE parts)
        - Plus an "issue-specific merge" that consolidates answers

        Strategy:
        1. Group chunks by division (DATA, PROCEDURE, etc.)
        2. For large divisions, group by chunk_kind (PROCEDURE_PART1, PART2, etc.)
        3. Ensure each scope has max 5 chunks for bounded context
        4. Use merge nodes as natural grouping boundaries when available

        Args:
            issue: The cross-cutting issue.
            manifest: The workflow manifest.

        Returns:
            List of bounded scope dictionaries.
        """
        scopes: list[dict[str, Any]] = []
        max_chunks_per_scope = 5  # Per spec 9.3: configurable max

        # Strategy 1: Try to use merge nodes as natural boundaries
        # Each leaf merge node represents a bounded set of chunks
        leaf_merges = [
            node for node in manifest.merge_dag
            if node.level == 0 or all(
                manifest.get_chunk(input_id) is not None
                for input_id in node.input_ids
            )
        ]

        if leaf_merges:
            for merge_node in leaf_merges:
                # Get chunk IDs from this merge node's inputs
                chunk_ids = [
                    input_id for input_id in merge_node.input_ids
                    if manifest.get_chunk(input_id) is not None
                ]
                if chunk_ids:
                    # Split if needed
                    for i in range(0, len(chunk_ids), max_chunks_per_scope):
                        batch = chunk_ids[i:i + max_chunks_per_scope]
                        scopes.append({
                            "issue_id": issue.issue_id,
                            "chunk_ids": batch,
                            "type": "cross_cutting",
                            "routing_method": "merge_node_boundary",
                            "merge_node_id": merge_node.merge_node_id,
                        })

            if scopes:
                logger.debug(
                    f"Split cross-cutting issue {issue.issue_id} into "
                    f"{len(scopes)} scope(s) using merge node boundaries"
                )
                return scopes

        # Strategy 2: Group chunks by division/kind
        chunks_by_division: dict[str, list[str]] = {}
        for chunk in manifest.chunks:
            # Use division if available, otherwise chunk_kind
            key = chunk.division or chunk.chunk_kind.value
            if key not in chunks_by_division:
                chunks_by_division[key] = []
            chunks_by_division[key].append(chunk.chunk_id)

        # Create scopes per division, respecting max chunks per scope
        for division, chunk_ids in chunks_by_division.items():
            # Split large divisions into manageable batches
            for i in range(0, len(chunk_ids), max_chunks_per_scope):
                batch = chunk_ids[i:i + max_chunks_per_scope]
                part_num = (i // max_chunks_per_scope) + 1
                scopes.append({
                    "issue_id": issue.issue_id,
                    "chunk_ids": batch,
                    "type": "cross_cutting",
                    "routing_method": "division_split",
                    "division": division,
                    "part": part_num if len(chunk_ids) > max_chunks_per_scope else None,
                })

        # Fallback: If no chunks found, create a single scope that will
        # need manual review or LLM-based analysis without chunk context
        if not scopes:
            logger.warning(
                f"Cross-cutting issue {issue.issue_id} could not be routed "
                f"to specific chunks. Creating empty cross-cutting scope."
            )
            scopes.append({
                "issue_id": issue.issue_id,
                "chunk_ids": [],
                "type": "cross_cutting",
                "routing_method": "fallback",
                "question": issue.question,
            })

        logger.debug(
            f"Split cross-cutting issue {issue.issue_id} into "
            f"{len(scopes)} scope(s) using division boundaries"
        )
        return scopes
