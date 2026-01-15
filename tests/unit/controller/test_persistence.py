"""Tests for job state persistence and recovery."""

import pytest
from datetime import datetime, timezone

from atlas.controller.persistence import (
    JobStatePersistence,
    JobCheckpoint,
    WorkItemSnapshot,
    PhaseProgress,
    create_job_checkpoint,
    restore_job_from_checkpoint,
)
from atlas.models.enums import WorkItemStatus, WorkItemType
from atlas.models.work_item import WorkItem, DocChunkPayload, ChunkLocator
from atlas.models.artifact import ArtifactRef


class TestWorkItemSnapshot:
    """Tests for WorkItemSnapshot."""

    def test_from_work_item(self):
        """Test creating snapshot from work item."""
        payload = DocChunkPayload(
            job_id="job-123",
            artifact_ref=ArtifactRef(
                artifact_id="test.cbl",
                artifact_type="cobol",
                artifact_version="abc123",
                artifact_uri="s3://bucket/test.cbl",
            ),
            manifest_uri="s3://bucket/manifest.json",
            chunk_id="chunk-001",
            chunk_locator=ChunkLocator(start_line=1, end_line=100),
            result_uri="s3://bucket/results/chunk-001.json",
        )

        work_item = WorkItem(
            work_id="job-123-chunk-001",
            work_type=WorkItemType.DOC_CHUNK,
            status=WorkItemStatus.IN_PROGRESS,
            payload=payload,
            cycle_number=1,
            depends_on=["dep-1", "dep-2"],
        )

        snapshot = WorkItemSnapshot.from_work_item(work_item)

        assert snapshot.work_id == "job-123-chunk-001"
        assert snapshot.work_type == "doc_chunk"
        assert snapshot.status == "in_progress"
        assert snapshot.cycle_number == 1
        assert snapshot.depends_on == ["dep-1", "dep-2"]
        assert snapshot.payload_summary["job_id"] == "job-123"
        assert snapshot.payload_summary["chunk_id"] == "chunk-001"


class TestPhaseProgress:
    """Tests for PhaseProgress."""

    def test_progress_percent_empty(self):
        """Test progress with no items."""
        progress = PhaseProgress(phase="chunk", total=0)
        assert progress.progress_percent == 100.0

    def test_progress_percent(self):
        """Test progress calculation."""
        progress = PhaseProgress(phase="chunk", total=10, completed=5)
        assert progress.progress_percent == 50.0


class TestJobCheckpoint:
    """Tests for JobCheckpoint serialization."""

    def test_to_dict(self):
        """Test checkpoint serialization."""
        checkpoint = JobCheckpoint(
            checkpoint_id="job-123_checkpoint_20240115",
            job_id="job-123",
            created_at="2024-01-15T10:00:00Z",
            phase="chunk",
            cycle_number=1,
            manifest_uri="s3://bucket/manifest.json",
            work_items=[
                WorkItemSnapshot(
                    work_id="work-1",
                    work_type="DOC_CHUNK",
                    status="DONE",
                )
            ],
            phase_progress={
                "chunk": PhaseProgress(phase="chunk", total=10, completed=5),
            },
        )

        data = checkpoint.to_dict()

        assert data["checkpoint_id"] == "job-123_checkpoint_20240115"
        assert data["job_id"] == "job-123"
        assert data["phase"] == "chunk"
        assert len(data["work_items"]) == 1
        assert data["work_items"][0]["work_id"] == "work-1"
        assert "chunk" in data["phase_progress"]

    def test_from_dict(self):
        """Test checkpoint deserialization."""
        data = {
            "checkpoint_id": "job-123_checkpoint_20240115",
            "job_id": "job-123",
            "created_at": "2024-01-15T10:00:00Z",
            "phase": "merge",
            "cycle_number": 2,
            "manifest_uri": "s3://bucket/manifest.json",
            "work_items": [
                {
                    "work_id": "work-1",
                    "work_type": "DOC_CHUNK",
                    "status": "DONE",
                    "cycle_number": 1,
                    "depends_on": [],
                    "payload_summary": {},
                }
            ],
            "phase_progress": {
                "chunk": {
                    "phase": "chunk",
                    "total": 10,
                    "completed": 10,
                    "in_progress": 0,
                    "blocked": 0,
                    "failed": 0,
                }
            },
            "metadata": {"test": "value"},
        }

        checkpoint = JobCheckpoint.from_dict(data)

        assert checkpoint.checkpoint_id == "job-123_checkpoint_20240115"
        assert checkpoint.phase == "merge"
        assert checkpoint.cycle_number == 2
        assert len(checkpoint.work_items) == 1
        assert checkpoint.work_items[0].status == "DONE"
        assert "chunk" in checkpoint.phase_progress
        assert checkpoint.metadata["test"] == "value"


class TestJobStatePersistence:
    """Tests for JobStatePersistence."""

    @pytest.fixture
    def mock_artifact_store(self, mock_artifact_store):
        """Provide mock artifact store."""
        return mock_artifact_store

    @pytest.fixture
    def mock_ticket_system(self, mock_ticket_system):
        """Provide mock ticket system."""
        return mock_ticket_system

    @pytest.fixture
    def persistence(self, mock_artifact_store, mock_ticket_system):
        """Create persistence instance."""
        return JobStatePersistence(
            artifact_store=mock_artifact_store,
            ticket_system=mock_ticket_system,
        )

    @pytest.mark.asyncio
    async def test_save_checkpoint(self, persistence, mock_ticket_system):
        """Test saving a checkpoint."""
        # Create test work items
        payload = DocChunkPayload(
            job_id="job-123",
            artifact_ref=ArtifactRef(
                artifact_id="test.cbl",
                artifact_type="cobol",
                artifact_version="abc123",
                artifact_uri="s3://bucket/test.cbl",
            ),
            manifest_uri="s3://bucket/manifest.json",
            chunk_id="chunk-001",
            chunk_locator=ChunkLocator(start_line=1, end_line=100),
            result_uri="s3://bucket/results/chunk-001.json",
        )

        work_item = WorkItem(
            work_id="job-123-chunk-001",
            work_type=WorkItemType.DOC_CHUNK,
            status=WorkItemStatus.DONE,
            payload=payload,
        )

        await mock_ticket_system.create_work_item(work_item)

        # Save checkpoint
        checkpoint = await persistence.save_checkpoint("job-123")

        assert checkpoint.job_id == "job-123"
        assert checkpoint.checkpoint_id.startswith("job-123_checkpoint_")
        assert len(checkpoint.work_items) == 1
        assert checkpoint.work_items[0].status == "done"

    @pytest.mark.asyncio
    async def test_load_checkpoint(self, persistence, mock_artifact_store, mock_ticket_system):
        """Test loading a checkpoint."""
        # Create test work items
        payload = DocChunkPayload(
            job_id="job-123",
            artifact_ref=ArtifactRef(
                artifact_id="test.cbl",
                artifact_type="cobol",
                artifact_version="abc123",
                artifact_uri="s3://bucket/test.cbl",
            ),
            manifest_uri="s3://bucket/manifest.json",
            chunk_id="chunk-001",
            chunk_locator=ChunkLocator(start_line=1, end_line=100),
            result_uri="s3://bucket/results/chunk-001.json",
        )

        work_item = WorkItem(
            work_id="job-123-chunk-001",
            work_type=WorkItemType.DOC_CHUNK,
            status=WorkItemStatus.DONE,
            payload=payload,
        )

        await mock_ticket_system.create_work_item(work_item)

        # Save and load
        saved = await persistence.save_checkpoint("job-123")
        loaded = await persistence.load_checkpoint("job-123", saved.checkpoint_id)

        assert loaded is not None
        assert loaded.checkpoint_id == saved.checkpoint_id
        assert len(loaded.work_items) == 1

    @pytest.mark.asyncio
    async def test_load_latest_checkpoint(self, persistence, mock_ticket_system):
        """Test loading latest checkpoint."""
        # Create test work item
        payload = DocChunkPayload(
            job_id="job-123",
            artifact_ref=ArtifactRef(
                artifact_id="test.cbl",
                artifact_type="cobol",
                artifact_version="abc123",
                artifact_uri="s3://bucket/test.cbl",
            ),
            manifest_uri="s3://bucket/manifest.json",
            chunk_id="chunk-001",
            chunk_locator=ChunkLocator(start_line=1, end_line=100),
            result_uri="s3://bucket/results/chunk-001.json",
        )

        work_item = WorkItem(
            work_id="job-123-chunk-001",
            work_type=WorkItemType.DOC_CHUNK,
            status=WorkItemStatus.DONE,
            payload=payload,
        )

        await mock_ticket_system.create_work_item(work_item)

        # Save checkpoint
        saved = await persistence.save_checkpoint("job-123")

        # Load latest (without specifying ID)
        loaded = await persistence.load_checkpoint("job-123")

        assert loaded is not None
        assert loaded.checkpoint_id == saved.checkpoint_id

    @pytest.mark.asyncio
    async def test_restore_checkpoint(self, persistence, mock_ticket_system):
        """Test restoring from checkpoint."""
        # Create test work item in IN_PROGRESS state
        payload = DocChunkPayload(
            job_id="job-123",
            artifact_ref=ArtifactRef(
                artifact_id="test.cbl",
                artifact_type="cobol",
                artifact_version="abc123",
                artifact_uri="s3://bucket/test.cbl",
            ),
            manifest_uri="s3://bucket/manifest.json",
            chunk_id="chunk-001",
            chunk_locator=ChunkLocator(start_line=1, end_line=100),
            result_uri="s3://bucket/results/chunk-001.json",
        )

        work_item = WorkItem(
            work_id="job-123-chunk-001",
            work_type=WorkItemType.DOC_CHUNK,
            status=WorkItemStatus.DONE,
            payload=payload,
        )

        await mock_ticket_system.create_work_item(work_item)

        # Save checkpoint in DONE state
        checkpoint = await persistence.save_checkpoint("job-123")

        # Change status to something else
        await mock_ticket_system.update_status(
            "job-123-chunk-001",
            WorkItemStatus.FAILED,
        )

        # Verify it changed
        item = await mock_ticket_system.get_work_item("job-123-chunk-001")
        assert item.status == WorkItemStatus.FAILED

        # Restore from checkpoint
        success = await persistence.restore_checkpoint("job-123", checkpoint.checkpoint_id)
        assert success

        # Verify status restored
        item = await mock_ticket_system.get_work_item("job-123-chunk-001")
        assert item.status == WorkItemStatus.DONE

    @pytest.mark.asyncio
    async def test_load_nonexistent_checkpoint(self, persistence):
        """Test loading nonexistent checkpoint returns None."""
        loaded = await persistence.load_checkpoint("nonexistent-job")
        assert loaded is None

    def test_generate_checkpoint_id(self, persistence):
        """Test checkpoint ID generation."""
        checkpoint_id = persistence._generate_checkpoint_id("job-123")
        assert checkpoint_id.startswith("job-123_checkpoint_")

    def test_compute_phase_progress(self, persistence):
        """Test phase progress computation."""
        work_items = [
            WorkItem(
                work_id="chunk-1",
                work_type=WorkItemType.DOC_CHUNK,
                status=WorkItemStatus.DONE,
                payload=DocChunkPayload(
                    job_id="job-123",
                    artifact_ref=ArtifactRef(
                        artifact_id="test.cbl",
                        artifact_type="cobol",
                        artifact_version="abc123",
                        artifact_uri="s3://bucket/test.cbl",
                    ),
                    manifest_uri="uri",
                    chunk_id="c1",
                    chunk_locator=ChunkLocator(start_line=1, end_line=100),
                    result_uri="uri",
                ),
            ),
            WorkItem(
                work_id="chunk-2",
                work_type=WorkItemType.DOC_CHUNK,
                status=WorkItemStatus.IN_PROGRESS,
                payload=DocChunkPayload(
                    job_id="job-123",
                    artifact_ref=ArtifactRef(
                        artifact_id="test.cbl",
                        artifact_type="cobol",
                        artifact_version="abc123",
                        artifact_uri="s3://bucket/test.cbl",
                    ),
                    manifest_uri="uri",
                    chunk_id="c2",
                    chunk_locator=ChunkLocator(start_line=101, end_line=200),
                    result_uri="uri",
                ),
            ),
        ]

        progress = persistence._compute_phase_progress(work_items)

        assert "chunk" in progress
        assert progress["chunk"].total == 2
        assert progress["chunk"].completed == 1
        assert progress["chunk"].in_progress == 1

    def test_determine_current_phase(self, persistence):
        """Test current phase determination."""
        # Chunk phase incomplete
        progress = {
            "chunk": PhaseProgress(phase="chunk", total=10, completed=5),
        }
        assert persistence._determine_current_phase(progress) == "chunk"

        # Chunk complete, merge incomplete
        progress = {
            "chunk": PhaseProgress(phase="chunk", total=10, completed=10),
            "merge": PhaseProgress(phase="merge", total=3, completed=1),
        }
        assert persistence._determine_current_phase(progress) == "merge"

        # All complete
        progress = {
            "chunk": PhaseProgress(phase="chunk", total=10, completed=10),
            "merge": PhaseProgress(phase="merge", total=3, completed=3),
        }
        assert persistence._determine_current_phase(progress) == "complete"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_create_job_checkpoint(self, mock_artifact_store, mock_ticket_system):
        """Test convenience function for creating checkpoint."""
        # Create a work item
        payload = DocChunkPayload(
            job_id="job-123",
            artifact_ref=ArtifactRef(
                artifact_id="test.cbl",
                artifact_type="cobol",
                artifact_version="abc123",
                artifact_uri="s3://bucket/test.cbl",
            ),
            manifest_uri="s3://bucket/manifest.json",
            chunk_id="chunk-001",
            chunk_locator=ChunkLocator(start_line=1, end_line=100),
            result_uri="s3://bucket/results/chunk-001.json",
        )

        work_item = WorkItem(
            work_id="job-123-chunk-001",
            work_type=WorkItemType.DOC_CHUNK,
            status=WorkItemStatus.DONE,
            payload=payload,
        )

        await mock_ticket_system.create_work_item(work_item)

        checkpoint = await create_job_checkpoint(
            mock_artifact_store,
            mock_ticket_system,
            "job-123",
        )

        assert checkpoint.job_id == "job-123"
