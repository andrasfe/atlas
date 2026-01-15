"""Unit tests for the in-memory ticket system adapter.

These tests verify the MemoryTicketSystem implementation including:
- Work item creation and retrieval
- Status transitions with optimistic locking
- Claim/release with worker tracking
- Query operations
- Idempotency key handling
"""

import pytest
from datetime import datetime, timezone

from atlas.adapters.memory_ticket_system import (
    MemoryTicketSystem,
    WorkItemNotFoundError,
    InvalidStatusTransitionError,
    DuplicateWorkItemError,
)
from atlas.models.enums import WorkItemStatus, WorkItemType
from atlas.models.work_item import WorkItem, WorkItemPayload, DocChunkPayload, ChunkLocator
from atlas.models.artifact import ArtifactRef


@pytest.fixture
def ticket_system() -> MemoryTicketSystem:
    """Provide a fresh ticket system for each test."""
    return MemoryTicketSystem()


@pytest.fixture
def sample_artifact_ref() -> ArtifactRef:
    """Provide a sample artifact reference."""
    return ArtifactRef(
        artifact_id="TEST001.cbl",
        artifact_type="cobol",
        artifact_version="abc123",
        artifact_uri="file:///test/TEST001.cbl",
    )


@pytest.fixture
def sample_work_item(sample_artifact_ref: ArtifactRef) -> WorkItem:
    """Provide a sample work item."""
    return WorkItem(
        work_id="chunk-001",
        work_type=WorkItemType.DOC_CHUNK,
        status=WorkItemStatus.READY,
        payload=DocChunkPayload(
            job_id="job-001",
            artifact_ref=sample_artifact_ref,
            manifest_uri="file:///test/manifest.json",
            chunk_id="procedure_part_001",
            chunk_locator=ChunkLocator(start_line=100, end_line=200),
            result_uri="file:///test/results/chunk-001.json",
        ),
    )


class TestCreateWorkItem:
    """Tests for work item creation."""

    @pytest.mark.asyncio
    async def test_create_work_item_success(
        self,
        ticket_system: MemoryTicketSystem,
        sample_work_item: WorkItem,
    ) -> None:
        """Test successful work item creation."""
        work_id = await ticket_system.create_work_item(sample_work_item)

        assert work_id == "chunk-001"
        assert ticket_system.item_count == 1

    @pytest.mark.asyncio
    async def test_create_work_item_sets_timestamps(
        self,
        ticket_system: MemoryTicketSystem,
        sample_work_item: WorkItem,
    ) -> None:
        """Test that creation sets timestamps."""
        await ticket_system.create_work_item(sample_work_item)

        item = await ticket_system.get_work_item("chunk-001")
        assert item is not None
        assert item.created_at is not None
        assert item.updated_at is not None

    @pytest.mark.asyncio
    async def test_create_work_item_generates_id(
        self,
        ticket_system: MemoryTicketSystem,
        sample_artifact_ref: ArtifactRef,
    ) -> None:
        """Test that work ID is generated if not provided."""
        item = WorkItem(
            work_id="",  # Empty ID
            work_type=WorkItemType.DOC_CHUNK,
            status=WorkItemStatus.NEW,
            payload=WorkItemPayload(job_id="job-001"),
        )

        work_id = await ticket_system.create_work_item(item)
        assert work_id.startswith("work-")

    @pytest.mark.asyncio
    async def test_create_duplicate_raises_error(
        self,
        ticket_system: MemoryTicketSystem,
        sample_work_item: WorkItem,
    ) -> None:
        """Test that creating duplicate work ID raises error."""
        await ticket_system.create_work_item(sample_work_item)

        with pytest.raises(DuplicateWorkItemError):
            await ticket_system.create_work_item(sample_work_item)

    @pytest.mark.asyncio
    async def test_create_with_idempotency_key_returns_existing(
        self,
        ticket_system: MemoryTicketSystem,
        sample_work_item: WorkItem,
    ) -> None:
        """Test idempotency key returns existing item ID."""
        sample_work_item.idempotency_key = "unique-key-001"
        work_id1 = await ticket_system.create_work_item(sample_work_item)

        # Create another with same idempotency key but different work_id
        another_item = WorkItem(
            work_id="chunk-002",
            work_type=WorkItemType.DOC_CHUNK,
            status=WorkItemStatus.NEW,
            payload=WorkItemPayload(job_id="job-001"),
            idempotency_key="unique-key-001",
        )

        work_id2 = await ticket_system.create_work_item(another_item)

        assert work_id1 == work_id2
        assert ticket_system.item_count == 1


class TestGetWorkItem:
    """Tests for work item retrieval."""

    @pytest.mark.asyncio
    async def test_get_existing_item(
        self,
        ticket_system: MemoryTicketSystem,
        sample_work_item: WorkItem,
    ) -> None:
        """Test retrieving an existing work item."""
        await ticket_system.create_work_item(sample_work_item)

        item = await ticket_system.get_work_item("chunk-001")
        assert item is not None
        assert item.work_id == "chunk-001"
        assert item.work_type == WorkItemType.DOC_CHUNK

    @pytest.mark.asyncio
    async def test_get_nonexistent_item_returns_none(
        self,
        ticket_system: MemoryTicketSystem,
    ) -> None:
        """Test that getting nonexistent item returns None."""
        item = await ticket_system.get_work_item("nonexistent")
        assert item is None


class TestUpdateStatus:
    """Tests for status updates."""

    @pytest.mark.asyncio
    async def test_update_status_success(
        self,
        ticket_system: MemoryTicketSystem,
        sample_work_item: WorkItem,
    ) -> None:
        """Test successful status update."""
        await ticket_system.create_work_item(sample_work_item)

        result = await ticket_system.update_status(
            "chunk-001",
            WorkItemStatus.IN_PROGRESS,
        )

        assert result is True
        item = await ticket_system.get_work_item("chunk-001")
        assert item is not None
        assert item.status == WorkItemStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_update_status_with_expected_status(
        self,
        ticket_system: MemoryTicketSystem,
        sample_work_item: WorkItem,
    ) -> None:
        """Test optimistic locking with expected status."""
        await ticket_system.create_work_item(sample_work_item)

        # Should succeed with correct expected status
        result = await ticket_system.update_status(
            "chunk-001",
            WorkItemStatus.IN_PROGRESS,
            expected_status=WorkItemStatus.READY,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_update_status_fails_wrong_expected(
        self,
        ticket_system: MemoryTicketSystem,
        sample_work_item: WorkItem,
    ) -> None:
        """Test that update fails with wrong expected status."""
        await ticket_system.create_work_item(sample_work_item)

        result = await ticket_system.update_status(
            "chunk-001",
            WorkItemStatus.IN_PROGRESS,
            expected_status=WorkItemStatus.NEW,  # Wrong - item is READY
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_update_status_nonexistent_raises(
        self,
        ticket_system: MemoryTicketSystem,
    ) -> None:
        """Test that updating nonexistent item raises error."""
        with pytest.raises(WorkItemNotFoundError):
            await ticket_system.update_status(
                "nonexistent",
                WorkItemStatus.IN_PROGRESS,
            )

    @pytest.mark.asyncio
    async def test_update_status_invalid_transition_raises(
        self,
        ticket_system: MemoryTicketSystem,
        sample_work_item: WorkItem,
    ) -> None:
        """Test that invalid transition raises error."""
        sample_work_item.status = WorkItemStatus.NEW
        await ticket_system.create_work_item(sample_work_item)

        with pytest.raises(InvalidStatusTransitionError):
            await ticket_system.update_status(
                "chunk-001",
                WorkItemStatus.DONE,  # Can't go directly NEW -> DONE
            )


class TestClaimRelease:
    """Tests for claim and release operations."""

    @pytest.mark.asyncio
    async def test_claim_success(
        self,
        ticket_system: MemoryTicketSystem,
        sample_work_item: WorkItem,
    ) -> None:
        """Test successful claim."""
        await ticket_system.create_work_item(sample_work_item)

        result = await ticket_system.claim_work_item(
            "chunk-001",
            "worker-1",
            lease_duration_seconds=300,
        )

        assert result is True
        item = await ticket_system.get_work_item("chunk-001")
        assert item is not None
        assert item.status == WorkItemStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_claim_records_worker(
        self,
        ticket_system: MemoryTicketSystem,
        sample_work_item: WorkItem,
    ) -> None:
        """Test that claim records worker ID."""
        await ticket_system.create_work_item(sample_work_item)
        await ticket_system.claim_work_item("chunk-001", "worker-1")

        claim_info = ticket_system.get_claim_info("chunk-001")
        assert claim_info is not None
        assert claim_info["worker_id"] == "worker-1"

    @pytest.mark.asyncio
    async def test_claim_fails_if_already_claimed(
        self,
        ticket_system: MemoryTicketSystem,
        sample_work_item: WorkItem,
    ) -> None:
        """Test that claim fails if already claimed."""
        await ticket_system.create_work_item(sample_work_item)
        await ticket_system.claim_work_item("chunk-001", "worker-1")

        result = await ticket_system.claim_work_item("chunk-001", "worker-2")
        assert result is False

    @pytest.mark.asyncio
    async def test_claim_fails_if_not_ready(
        self,
        ticket_system: MemoryTicketSystem,
        sample_work_item: WorkItem,
    ) -> None:
        """Test that claim fails if item is not READY."""
        sample_work_item.status = WorkItemStatus.NEW
        await ticket_system.create_work_item(sample_work_item)

        result = await ticket_system.claim_work_item("chunk-001", "worker-1")
        assert result is False

    @pytest.mark.asyncio
    async def test_release_success(
        self,
        ticket_system: MemoryTicketSystem,
        sample_work_item: WorkItem,
    ) -> None:
        """Test successful release."""
        await ticket_system.create_work_item(sample_work_item)
        await ticket_system.claim_work_item("chunk-001", "worker-1")

        result = await ticket_system.release_work_item(
            "chunk-001",
            "worker-1",
            new_status=WorkItemStatus.READY,
        )

        assert result is True
        item = await ticket_system.get_work_item("chunk-001")
        assert item is not None
        assert item.status == WorkItemStatus.READY
        assert ticket_system.get_claim_info("chunk-001") is None

    @pytest.mark.asyncio
    async def test_release_fails_wrong_worker(
        self,
        ticket_system: MemoryTicketSystem,
        sample_work_item: WorkItem,
    ) -> None:
        """Test that release fails if wrong worker."""
        await ticket_system.create_work_item(sample_work_item)
        await ticket_system.claim_work_item("chunk-001", "worker-1")

        result = await ticket_system.release_work_item(
            "chunk-001",
            "worker-2",  # Wrong worker
        )
        assert result is False


class TestQueries:
    """Tests for query operations."""

    @pytest.fixture
    def populated_system(
        self,
        ticket_system: MemoryTicketSystem,
        sample_artifact_ref: ArtifactRef,
    ):
        """Populate ticket system with multiple items."""
        async def populate():
            items = [
                WorkItem(
                    work_id=f"chunk-{i:03d}",
                    work_type=WorkItemType.DOC_CHUNK,
                    status=WorkItemStatus.READY if i % 2 == 0 else WorkItemStatus.NEW,
                    payload=DocChunkPayload(
                        job_id="job-001" if i < 5 else "job-002",
                        artifact_ref=sample_artifact_ref,
                        manifest_uri="file:///test/manifest.json",
                        chunk_id=f"chunk_{i:03d}",
                        chunk_locator=ChunkLocator(start_line=i*100, end_line=(i+1)*100),
                        result_uri=f"file:///test/results/chunk-{i:03d}.json",
                    ),
                )
                for i in range(10)
            ]
            for item in items:
                await ticket_system.create_work_item(item)
            return ticket_system

        return populate

    @pytest.mark.asyncio
    async def test_query_by_status(
        self,
        populated_system,
    ) -> None:
        """Test query by status."""
        ts = await populated_system()

        ready_items = await ts.query_by_status(WorkItemStatus.READY)
        assert len(ready_items) == 5

        new_items = await ts.query_by_status(WorkItemStatus.NEW)
        assert len(new_items) == 5

    @pytest.mark.asyncio
    async def test_query_by_status_with_type_filter(
        self,
        populated_system,
    ) -> None:
        """Test query by status with type filter."""
        ts = await populated_system()

        items = await ts.query_by_status(
            WorkItemStatus.READY,
            work_type=WorkItemType.DOC_CHUNK,
        )
        assert len(items) == 5

        items = await ts.query_by_status(
            WorkItemStatus.READY,
            work_type=WorkItemType.DOC_MERGE,
        )
        assert len(items) == 0

    @pytest.mark.asyncio
    async def test_query_by_status_with_job_filter(
        self,
        populated_system,
    ) -> None:
        """Test query by status with job filter."""
        ts = await populated_system()

        items = await ts.query_by_status(
            WorkItemStatus.READY,
            job_id="job-001",
        )
        # job-001 has items 0-4, even ones (0,2,4) are READY
        assert len(items) == 3

    @pytest.mark.asyncio
    async def test_query_by_status_with_limit(
        self,
        populated_system,
    ) -> None:
        """Test query respects limit."""
        ts = await populated_system()

        items = await ts.query_by_status(WorkItemStatus.READY, limit=2)
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_query_by_job(
        self,
        populated_system,
    ) -> None:
        """Test query by job."""
        ts = await populated_system()

        job1_items = await ts.query_by_job("job-001")
        assert len(job1_items) == 5

        job2_items = await ts.query_by_job("job-002")
        assert len(job2_items) == 5

    @pytest.mark.asyncio
    async def test_query_by_job_with_type(
        self,
        populated_system,
    ) -> None:
        """Test query by job with type filter."""
        ts = await populated_system()

        items = await ts.query_by_job(
            "job-001",
            work_type=WorkItemType.DOC_CHUNK,
        )
        assert len(items) == 5

    @pytest.mark.asyncio
    async def test_get_ready_work_items(
        self,
        populated_system,
    ) -> None:
        """Test get ready work items."""
        ts = await populated_system()

        ready = await ts.get_ready_work_items()
        assert len(ready) == 5
        for item in ready:
            assert item.status == WorkItemStatus.READY


class TestDependencies:
    """Tests for dependency checking."""

    @pytest.mark.asyncio
    async def test_check_dependencies_no_deps(
        self,
        ticket_system: MemoryTicketSystem,
        sample_work_item: WorkItem,
    ) -> None:
        """Test that item with no dependencies is ready."""
        await ticket_system.create_work_item(sample_work_item)

        result = await ticket_system.check_dependencies_done("chunk-001")
        assert result is True

    @pytest.mark.asyncio
    async def test_check_dependencies_all_done(
        self,
        ticket_system: MemoryTicketSystem,
        sample_artifact_ref: ArtifactRef,
    ) -> None:
        """Test dependencies when all are done."""
        # Create dependency items
        for i in range(3):
            dep = WorkItem(
                work_id=f"dep-{i}",
                work_type=WorkItemType.DOC_CHUNK,
                status=WorkItemStatus.DONE,
                payload=WorkItemPayload(job_id="job-001"),
            )
            await ticket_system.create_work_item(dep)

        # Create item with dependencies
        item = WorkItem(
            work_id="merge-001",
            work_type=WorkItemType.DOC_MERGE,
            status=WorkItemStatus.BLOCKED,
            payload=WorkItemPayload(job_id="job-001"),
            depends_on=["dep-0", "dep-1", "dep-2"],
        )
        await ticket_system.create_work_item(item)

        result = await ticket_system.check_dependencies_done("merge-001")
        assert result is True

    @pytest.mark.asyncio
    async def test_check_dependencies_not_all_done(
        self,
        ticket_system: MemoryTicketSystem,
    ) -> None:
        """Test dependencies when some not done."""
        # Create dependency items (one not done)
        for i, status in enumerate([WorkItemStatus.DONE, WorkItemStatus.IN_PROGRESS, WorkItemStatus.DONE]):
            dep = WorkItem(
                work_id=f"dep-{i}",
                work_type=WorkItemType.DOC_CHUNK,
                status=status,
                payload=WorkItemPayload(job_id="job-001"),
            )
            await ticket_system.create_work_item(dep)

        # Create item with dependencies
        item = WorkItem(
            work_id="merge-001",
            work_type=WorkItemType.DOC_MERGE,
            status=WorkItemStatus.BLOCKED,
            payload=WorkItemPayload(job_id="job-001"),
            depends_on=["dep-0", "dep-1", "dep-2"],
        )
        await ticket_system.create_work_item(item)

        result = await ticket_system.check_dependencies_done("merge-001")
        assert result is False


class TestIdempotencyKey:
    """Tests for idempotency key lookup."""

    @pytest.mark.asyncio
    async def test_find_by_idempotency_key_found(
        self,
        ticket_system: MemoryTicketSystem,
        sample_work_item: WorkItem,
    ) -> None:
        """Test finding item by idempotency key."""
        sample_work_item.idempotency_key = "idem-key-001"
        await ticket_system.create_work_item(sample_work_item)

        item = await ticket_system.find_by_idempotency_key("idem-key-001")
        assert item is not None
        assert item.work_id == "chunk-001"

    @pytest.mark.asyncio
    async def test_find_by_idempotency_key_not_found(
        self,
        ticket_system: MemoryTicketSystem,
    ) -> None:
        """Test that missing idempotency key returns None."""
        item = await ticket_system.find_by_idempotency_key("nonexistent")
        assert item is None


class TestUtilities:
    """Tests for utility methods."""

    @pytest.mark.asyncio
    async def test_clear(
        self,
        ticket_system: MemoryTicketSystem,
        sample_work_item: WorkItem,
    ) -> None:
        """Test clearing the ticket system."""
        await ticket_system.create_work_item(sample_work_item)
        assert ticket_system.item_count == 1

        ticket_system.clear()

        assert ticket_system.item_count == 0
        item = await ticket_system.get_work_item("chunk-001")
        assert item is None

    @pytest.mark.asyncio
    async def test_item_count(
        self,
        ticket_system: MemoryTicketSystem,
    ) -> None:
        """Test item count property."""
        assert ticket_system.item_count == 0

        for i in range(5):
            item = WorkItem(
                work_id=f"item-{i}",
                work_type=WorkItemType.DOC_CHUNK,
                status=WorkItemStatus.NEW,
                payload=WorkItemPayload(job_id="job-001"),
            )
            await ticket_system.create_work_item(item)

        assert ticket_system.item_count == 5
