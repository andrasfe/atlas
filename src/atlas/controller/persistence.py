"""Job state persistence and recovery.

Provides the ability to save and restore job state for checkpoint/resume
of long-running jobs.

Design Principles:
- Serialize complete job state to artifact store
- Support checkpoint creation at any phase
- Enable recovery after crashes/restarts
- Preserve idempotency guarantees

Usage:
    >>> persistence = JobStatePersistence(artifact_store, ticket_system)
    >>> state = await persistence.save_checkpoint("job-123")
    >>> # Later, after restart...
    >>> await persistence.restore_checkpoint("job-123", state.checkpoint_id)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from atlas.adapters.artifact_store import ArtifactStoreAdapter
from atlas.adapters.ticket_system import TicketSystemAdapter
from atlas.models.enums import WorkItemStatus, WorkItemType
from atlas.models.manifest import Manifest
from atlas.models.work_item import WorkItem
from atlas.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WorkItemSnapshot:
    """Snapshot of a work item's state.

    Attributes:
        work_id: Work item identifier.
        work_type: Type of work item.
        status: Current status.
        cycle_number: Challenge cycle number.
        depends_on: List of dependency work IDs.
        payload_summary: Summary of payload (not full payload).
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
    """

    work_id: str
    work_type: str
    status: str
    cycle_number: int = 1
    depends_on: list[str] = field(default_factory=list)
    payload_summary: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_work_item(cls, item: WorkItem) -> "WorkItemSnapshot":
        """Create snapshot from work item.

        Args:
            item: Work item to snapshot.

        Returns:
            WorkItemSnapshot instance.
        """
        # Extract key payload info without full serialization
        payload_summary = {
            "job_id": item.payload.job_id,
        }
        if hasattr(item.payload, "chunk_id"):
            payload_summary["chunk_id"] = item.payload.chunk_id
        if hasattr(item.payload, "merge_node_id"):
            payload_summary["merge_node_id"] = item.payload.merge_node_id
        if hasattr(item.payload, "issue_id"):
            payload_summary["issue_id"] = item.payload.issue_id

        return cls(
            work_id=item.work_id,
            work_type=item.work_type.value,
            status=item.status.value,
            cycle_number=item.cycle_number,
            depends_on=item.depends_on.copy(),
            payload_summary=payload_summary,
            created_at=item.created_at,
            updated_at=item.updated_at,
        )


@dataclass
class PhaseProgress:
    """Progress within a workflow phase.

    Attributes:
        phase: Current phase name.
        total: Total items in phase.
        completed: Completed items.
        in_progress: Items currently being processed.
        blocked: Items waiting on dependencies.
        failed: Items that failed.
    """

    phase: str
    total: int = 0
    completed: int = 0
    in_progress: int = 0
    blocked: int = 0
    failed: int = 0

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage.

        Returns:
            Progress as percentage (0-100).
        """
        if self.total == 0:
            return 100.0
        return (self.completed / self.total) * 100.0


@dataclass
class JobCheckpoint:
    """Complete checkpoint of job state.

    Attributes:
        checkpoint_id: Unique checkpoint identifier.
        job_id: Job identifier.
        created_at: When checkpoint was created.
        phase: Current workflow phase.
        cycle_number: Current challenge cycle.
        manifest_uri: URI to the manifest.
        work_items: Snapshots of all work items.
        phase_progress: Progress by phase.
        metadata: Additional checkpoint metadata.
    """

    checkpoint_id: str
    job_id: str
    created_at: str
    phase: str
    cycle_number: int = 1
    manifest_uri: str = ""
    work_items: list[WorkItemSnapshot] = field(default_factory=list)
    phase_progress: dict[str, PhaseProgress] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert checkpoint to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "checkpoint_id": self.checkpoint_id,
            "job_id": self.job_id,
            "created_at": self.created_at,
            "phase": self.phase,
            "cycle_number": self.cycle_number,
            "manifest_uri": self.manifest_uri,
            "work_items": [
                {
                    "work_id": w.work_id,
                    "work_type": w.work_type,
                    "status": w.status,
                    "cycle_number": w.cycle_number,
                    "depends_on": w.depends_on,
                    "payload_summary": w.payload_summary,
                    "created_at": w.created_at,
                    "updated_at": w.updated_at,
                }
                for w in self.work_items
            ],
            "phase_progress": {
                phase: {
                    "phase": p.phase,
                    "total": p.total,
                    "completed": p.completed,
                    "in_progress": p.in_progress,
                    "blocked": p.blocked,
                    "failed": p.failed,
                }
                for phase, p in self.phase_progress.items()
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobCheckpoint":
        """Create checkpoint from dictionary.

        Args:
            data: Dictionary data.

        Returns:
            JobCheckpoint instance.
        """
        work_items = [
            WorkItemSnapshot(
                work_id=w["work_id"],
                work_type=w["work_type"],
                status=w["status"],
                cycle_number=w.get("cycle_number", 1),
                depends_on=w.get("depends_on", []),
                payload_summary=w.get("payload_summary", {}),
                created_at=w.get("created_at"),
                updated_at=w.get("updated_at"),
            )
            for w in data.get("work_items", [])
        ]

        phase_progress = {
            phase: PhaseProgress(
                phase=p["phase"],
                total=p.get("total", 0),
                completed=p.get("completed", 0),
                in_progress=p.get("in_progress", 0),
                blocked=p.get("blocked", 0),
                failed=p.get("failed", 0),
            )
            for phase, p in data.get("phase_progress", {}).items()
        }

        return cls(
            checkpoint_id=data["checkpoint_id"],
            job_id=data["job_id"],
            created_at=data["created_at"],
            phase=data["phase"],
            cycle_number=data.get("cycle_number", 1),
            manifest_uri=data.get("manifest_uri", ""),
            work_items=work_items,
            phase_progress=phase_progress,
            metadata=data.get("metadata", {}),
        )


class JobStatePersistence:
    """Manages job state persistence and recovery.

    Provides checkpoint/resume functionality for long-running jobs,
    enabling recovery after crashes and restarts.

    Example:
        >>> persistence = JobStatePersistence(artifact_store, ticket_system)
        >>> checkpoint = await persistence.save_checkpoint("job-123")
        >>> print(f"Saved checkpoint: {checkpoint.checkpoint_id}")
        >>> # After restart
        >>> await persistence.restore_checkpoint("job-123", checkpoint.checkpoint_id)
    """

    def __init__(
        self,
        artifact_store: ArtifactStoreAdapter,
        ticket_system: TicketSystemAdapter,
        checkpoint_base_uri: str = "checkpoints",
    ) -> None:
        """Initialize persistence manager.

        Args:
            artifact_store: Artifact store for checkpoint storage.
            ticket_system: Ticket system for work item access.
            checkpoint_base_uri: Base URI for checkpoint storage.
        """
        self.artifact_store = artifact_store
        self.ticket_system = ticket_system
        self.checkpoint_base_uri = checkpoint_base_uri

    def _generate_checkpoint_id(self, job_id: str) -> str:
        """Generate a unique checkpoint ID.

        Args:
            job_id: Job identifier.

        Returns:
            Unique checkpoint ID.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{job_id}_checkpoint_{timestamp}"

    def _get_checkpoint_uri(self, job_id: str, checkpoint_id: str) -> str:
        """Get URI for a checkpoint.

        Args:
            job_id: Job identifier.
            checkpoint_id: Checkpoint identifier.

        Returns:
            Checkpoint URI.
        """
        return f"{self.checkpoint_base_uri}/{job_id}/{checkpoint_id}.json"

    def _get_latest_checkpoint_uri(self, job_id: str) -> str:
        """Get URI for latest checkpoint pointer.

        Args:
            job_id: Job identifier.

        Returns:
            Latest checkpoint pointer URI.
        """
        return f"{self.checkpoint_base_uri}/{job_id}/latest.json"

    async def save_checkpoint(
        self,
        job_id: str,
        manifest: Manifest | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> JobCheckpoint:
        """Save a checkpoint of current job state.

        Args:
            job_id: Job identifier.
            manifest: Optional manifest (will load if not provided).
            metadata: Additional metadata to include.

        Returns:
            The created checkpoint.
        """
        logger.info(f"Saving checkpoint for job {job_id}")

        # Generate checkpoint ID
        checkpoint_id = self._generate_checkpoint_id(job_id)
        created_at = datetime.now(timezone.utc).isoformat()

        # Query all work items for job
        work_items = await self.ticket_system.query_by_job(job_id)

        # Create snapshots
        snapshots = [WorkItemSnapshot.from_work_item(item) for item in work_items]

        # Determine phase and progress
        phase_progress = self._compute_phase_progress(work_items)
        current_phase = self._determine_current_phase(phase_progress)

        # Determine cycle number
        cycle_number = 1
        for item in work_items:
            if item.work_type == WorkItemType.DOC_CHALLENGE:
                cycle_number = max(cycle_number, item.cycle_number)

        # Get manifest URI
        manifest_uri = f"manifests/{job_id}/manifest.json"

        # Create checkpoint
        checkpoint = JobCheckpoint(
            checkpoint_id=checkpoint_id,
            job_id=job_id,
            created_at=created_at,
            phase=current_phase,
            cycle_number=cycle_number,
            manifest_uri=manifest_uri,
            work_items=snapshots,
            phase_progress=phase_progress,
            metadata=metadata or {},
        )

        # Save checkpoint
        checkpoint_uri = self._get_checkpoint_uri(job_id, checkpoint_id)
        await self.artifact_store.write_json(checkpoint_uri, checkpoint.to_dict())

        # Update latest pointer
        latest_uri = self._get_latest_checkpoint_uri(job_id)
        await self.artifact_store.write_json(
            latest_uri,
            {
                "checkpoint_id": checkpoint_id,
                "checkpoint_uri": checkpoint_uri,
                "created_at": created_at,
            },
        )

        logger.info(f"Saved checkpoint {checkpoint_id} for job {job_id}")
        return checkpoint

    async def load_checkpoint(
        self,
        job_id: str,
        checkpoint_id: str | None = None,
    ) -> JobCheckpoint | None:
        """Load a checkpoint.

        Args:
            job_id: Job identifier.
            checkpoint_id: Specific checkpoint ID, or None for latest.

        Returns:
            JobCheckpoint if found, None otherwise.
        """
        if checkpoint_id is None:
            # Load latest
            latest_uri = self._get_latest_checkpoint_uri(job_id)
            try:
                latest_data = await self.artifact_store.read_json(latest_uri)
                checkpoint_id = latest_data["checkpoint_id"]
            except Exception:
                logger.warning(f"No latest checkpoint found for job {job_id}")
                return None

        checkpoint_uri = self._get_checkpoint_uri(job_id, checkpoint_id)
        try:
            data = await self.artifact_store.read_json(checkpoint_uri)
            return JobCheckpoint.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None

    async def restore_checkpoint(
        self,
        job_id: str,
        checkpoint_id: str | None = None,
    ) -> bool:
        """Restore job state from a checkpoint.

        This restores work item statuses to match the checkpoint.
        Note: This does NOT recreate work items, only updates statuses.

        Args:
            job_id: Job identifier.
            checkpoint_id: Specific checkpoint ID, or None for latest.

        Returns:
            True if restoration succeeded.
        """
        checkpoint = await self.load_checkpoint(job_id, checkpoint_id)
        if checkpoint is None:
            logger.error(f"Cannot restore: checkpoint not found for job {job_id}")
            return False

        logger.info(f"Restoring job {job_id} from checkpoint {checkpoint.checkpoint_id}")

        # Restore work item statuses
        restored_count = 0
        for snapshot in checkpoint.work_items:
            try:
                item = await self.ticket_system.get_work_item(snapshot.work_id)
                if item is None:
                    logger.warning(f"Work item {snapshot.work_id} not found during restore")
                    continue

                # Restore status if changed
                target_status = WorkItemStatus(snapshot.status)
                if item.status != target_status:
                    # Use update_status without expected_status to force update
                    await self.ticket_system.update_status(
                        snapshot.work_id,
                        target_status,
                    )
                    restored_count += 1

            except Exception as e:
                logger.error(f"Failed to restore work item {snapshot.work_id}: {e}")

        logger.info(
            f"Restored {restored_count} work items from checkpoint {checkpoint.checkpoint_id}"
        )
        return True

    async def list_checkpoints(
        self,
        job_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """List available checkpoints for a job.

        Args:
            job_id: Job identifier.
            limit: Maximum number of checkpoints to return.

        Returns:
            List of checkpoint metadata dictionaries.
        """
        prefix = f"{self.checkpoint_base_uri}/{job_id}/"
        uris = await self.artifact_store.list_artifacts(prefix, limit=limit * 2)

        checkpoints = []
        for uri in uris:
            if uri.endswith(".json") and not uri.endswith("latest.json"):
                try:
                    data = await self.artifact_store.read_json(uri)
                    checkpoints.append({
                        "checkpoint_id": data.get("checkpoint_id"),
                        "created_at": data.get("created_at"),
                        "phase": data.get("phase"),
                        "uri": uri,
                    })
                except Exception:
                    continue

        # Sort by creation time (newest first) and limit
        checkpoints.sort(key=lambda c: c.get("created_at", ""), reverse=True)
        return checkpoints[:limit]

    async def delete_checkpoint(
        self,
        job_id: str,
        checkpoint_id: str,
    ) -> bool:
        """Delete a checkpoint.

        Args:
            job_id: Job identifier.
            checkpoint_id: Checkpoint to delete.

        Returns:
            True if deleted.
        """
        checkpoint_uri = self._get_checkpoint_uri(job_id, checkpoint_id)
        return await self.artifact_store.delete(checkpoint_uri)

    async def cleanup_old_checkpoints(
        self,
        job_id: str,
        keep_count: int = 5,
    ) -> int:
        """Clean up old checkpoints, keeping only the most recent.

        Args:
            job_id: Job identifier.
            keep_count: Number of checkpoints to keep.

        Returns:
            Number of checkpoints deleted.
        """
        checkpoints = await self.list_checkpoints(job_id, limit=100)

        deleted = 0
        for checkpoint in checkpoints[keep_count:]:
            checkpoint_id = checkpoint.get("checkpoint_id")
            if checkpoint_id:
                if await self.delete_checkpoint(job_id, checkpoint_id):
                    deleted += 1

        return deleted

    def _compute_phase_progress(
        self,
        work_items: list[WorkItem],
    ) -> dict[str, PhaseProgress]:
        """Compute progress for each phase.

        Args:
            work_items: All work items.

        Returns:
            Dictionary of phase name to progress.
        """
        # Group by work type
        by_type: dict[WorkItemType, list[WorkItem]] = {}
        for item in work_items:
            if item.work_type not in by_type:
                by_type[item.work_type] = []
            by_type[item.work_type].append(item)

        progress = {}

        # Chunk phase
        chunks = by_type.get(WorkItemType.DOC_CHUNK, [])
        if chunks:
            progress["chunk"] = PhaseProgress(
                phase="chunk",
                total=len(chunks),
                completed=sum(1 for c in chunks if c.status == WorkItemStatus.DONE),
                in_progress=sum(1 for c in chunks if c.status == WorkItemStatus.IN_PROGRESS),
                blocked=sum(1 for c in chunks if c.status == WorkItemStatus.BLOCKED),
                failed=sum(1 for c in chunks if c.status == WorkItemStatus.FAILED),
            )

        # Merge phase
        merges = by_type.get(WorkItemType.DOC_MERGE, [])
        if merges:
            progress["merge"] = PhaseProgress(
                phase="merge",
                total=len(merges),
                completed=sum(1 for m in merges if m.status == WorkItemStatus.DONE),
                in_progress=sum(1 for m in merges if m.status == WorkItemStatus.IN_PROGRESS),
                blocked=sum(1 for m in merges if m.status == WorkItemStatus.BLOCKED),
                failed=sum(1 for m in merges if m.status == WorkItemStatus.FAILED),
            )

        # Challenge phase
        challenges = by_type.get(WorkItemType.DOC_CHALLENGE, [])
        if challenges:
            progress["challenge"] = PhaseProgress(
                phase="challenge",
                total=len(challenges),
                completed=sum(1 for c in challenges if c.status == WorkItemStatus.DONE),
                in_progress=sum(1 for c in challenges if c.status == WorkItemStatus.IN_PROGRESS),
                blocked=sum(1 for c in challenges if c.status == WorkItemStatus.BLOCKED),
                failed=sum(1 for c in challenges if c.status == WorkItemStatus.FAILED),
            )

        # Followup phase
        followups = by_type.get(WorkItemType.DOC_FOLLOWUP, [])
        if followups:
            progress["followup"] = PhaseProgress(
                phase="followup",
                total=len(followups),
                completed=sum(1 for f in followups if f.status == WorkItemStatus.DONE),
                in_progress=sum(1 for f in followups if f.status == WorkItemStatus.IN_PROGRESS),
                blocked=sum(1 for f in followups if f.status == WorkItemStatus.BLOCKED),
                failed=sum(1 for f in followups if f.status == WorkItemStatus.FAILED),
            )

        return progress

    def _determine_current_phase(
        self,
        phase_progress: dict[str, PhaseProgress],
    ) -> str:
        """Determine current phase from progress.

        Args:
            phase_progress: Progress by phase.

        Returns:
            Current phase name.
        """
        # Check phases in order
        chunk_progress = phase_progress.get("chunk")
        if chunk_progress and chunk_progress.completed < chunk_progress.total:
            return "chunk"

        merge_progress = phase_progress.get("merge")
        if merge_progress and merge_progress.completed < merge_progress.total:
            return "merge"

        challenge_progress = phase_progress.get("challenge")
        if challenge_progress and challenge_progress.completed < challenge_progress.total:
            return "challenge"

        followup_progress = phase_progress.get("followup")
        if followup_progress and followup_progress.completed < followup_progress.total:
            return "followup"

        # All phases complete or not started
        if not phase_progress:
            return "plan"

        return "complete"


# Convenience function
async def create_job_checkpoint(
    artifact_store: ArtifactStoreAdapter,
    ticket_system: TicketSystemAdapter,
    job_id: str,
    metadata: dict[str, Any] | None = None,
) -> JobCheckpoint:
    """Convenience function to create a job checkpoint.

    Args:
        artifact_store: Artifact store.
        ticket_system: Ticket system.
        job_id: Job identifier.
        metadata: Optional metadata.

    Returns:
        The created checkpoint.
    """
    persistence = JobStatePersistence(artifact_store, ticket_system)
    return await persistence.save_checkpoint(job_id, metadata=metadata)


async def restore_job_from_checkpoint(
    artifact_store: ArtifactStoreAdapter,
    ticket_system: TicketSystemAdapter,
    job_id: str,
    checkpoint_id: str | None = None,
) -> bool:
    """Convenience function to restore a job from checkpoint.

    Args:
        artifact_store: Artifact store.
        ticket_system: Ticket system.
        job_id: Job identifier.
        checkpoint_id: Optional specific checkpoint ID.

    Returns:
        True if restoration succeeded.
    """
    persistence = JobStatePersistence(artifact_store, ticket_system)
    return await persistence.restore_checkpoint(job_id, checkpoint_id)
