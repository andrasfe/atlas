"""Unit tests for the Controller reconcile loop."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from atlas.controller.reconciler import ReconcileController
from atlas.controller.base import ControllerConfig, ReconcileResult
from atlas.models.artifact import ArtifactRef
from atlas.models.enums import (
    WorkItemStatus,
    WorkItemType,
    ChunkKind,
    IssueSeverity,
)
from atlas.models.manifest import (
    Manifest,
    ChunkSpec,
    MergeNode,
    AnalysisProfile,
    SplitterProfile,
    ReviewPolicy,
    ArtifactOutputConfig,
)
from atlas.models.work_item import (
    WorkItem,
    DocChunkPayload,
    DocMergePayload,
    DocChallengePayload,
    DocFollowupPayload,
    ChunkLocator,
)
from atlas.models.results import (
    ChallengeResult,
    Issue,
    ResolutionPlan,
    FollowupTask,
    DocumentationModel,
    DocIndex,
    Section,
)


@pytest.fixture
def artifact_ref() -> ArtifactRef:
    """Create sample artifact reference."""
    return ArtifactRef(
        artifact_id="TEST001.cbl",
        artifact_type="cobol",
        artifact_version="abc123",
        artifact_uri="s3://bucket/sources/TEST001.cbl",
    )


@pytest.fixture
def sample_manifest(artifact_ref: ArtifactRef) -> Manifest:
    """Create sample manifest with chunks and merge DAG."""
    chunks = [
        ChunkSpec(
            chunk_id="chunk_001",
            chunk_kind=ChunkKind.PROCEDURE_PART,
            start_line=1,
            end_line=100,
            division="PROCEDURE",
            paragraphs=["MAIN-LOGIC"],
        ),
        ChunkSpec(
            chunk_id="chunk_002",
            chunk_kind=ChunkKind.PROCEDURE_PART,
            start_line=101,
            end_line=200,
            division="PROCEDURE",
            paragraphs=["PROCESS-RECORD"],
        ),
        ChunkSpec(
            chunk_id="chunk_003",
            chunk_kind=ChunkKind.DATA_DIVISION,
            start_line=201,
            end_line=300,
            division="DATA",
        ),
    ]

    merge_dag = [
        MergeNode(
            merge_node_id="merge_procedure",
            input_ids=["chunk_001", "chunk_002"],
            is_root=False,
            level=1,
        ),
        MergeNode(
            merge_node_id="merge_root",
            input_ids=["merge_procedure", "chunk_003"],
            is_root=True,
            level=2,
        ),
    ]

    return Manifest(
        job_id="test-job-001",
        artifact_ref=artifact_ref,
        analysis_profile=AnalysisProfile(name="test"),
        splitter_profile=SplitterProfile(name="test"),
        context_budget=4000,
        chunks=chunks,
        merge_dag=merge_dag,
        review_policy=ReviewPolicy(challenge_profile="standard"),
        artifacts=ArtifactOutputConfig(base_uri="s3://bucket/results"),
    )


@pytest.fixture
def controller(mock_ticket_system, mock_artifact_store) -> ReconcileController:
    """Create controller with mock adapters."""
    return ReconcileController(
        ticket_system=mock_ticket_system,
        artifact_store=mock_artifact_store,
        config=ControllerConfig(max_challenge_iterations=3),
    )


class TestReconcileController:
    """Tests for ReconcileController."""

    @pytest.mark.asyncio
    async def test_initialize_job_creates_chunks_and_merges(
        self,
        controller: ReconcileController,
        sample_manifest: Manifest,
        mock_artifact_store,
    ) -> None:
        """Test that initialize_job creates all work items."""
        job_id = await controller.initialize_job(sample_manifest)

        assert job_id == "test-job-001"

        # Check manifest was written
        assert mock_artifact_store._artifacts  # Some artifact written

        # Check work items were created
        items = controller.ticket_system._items
        assert len(items) > 0

        # Should have 3 chunk items
        chunk_items = [
            i for i in items.values() if i.work_type == WorkItemType.DOC_CHUNK
        ]
        assert len(chunk_items) == 3

        # Should have 2 merge items
        merge_items = [
            i for i in items.values() if i.work_type == WorkItemType.DOC_MERGE
        ]
        assert len(merge_items) == 2

        # Chunk items should be READY
        for item in chunk_items:
            assert item.status == WorkItemStatus.READY

        # Merge items should be BLOCKED
        for item in merge_items:
            assert item.status == WorkItemStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_reconcile_phase_a_creates_missing_items(
        self,
        controller: ReconcileController,
        sample_manifest: Manifest,
        mock_artifact_store,
    ) -> None:
        """Test reconcile creates missing work items in phase A."""
        # Store manifest
        manifest_uri = controller._get_manifest_uri(sample_manifest.job_id)
        await mock_artifact_store.write_json(manifest_uri, sample_manifest.model_dump())

        # Reconcile should create all items
        result = await controller.reconcile(sample_manifest.job_id)

        assert result.phase == "plan"
        assert result.work_items_created == 5  # 3 chunks + 2 merges

    @pytest.mark.asyncio
    async def test_reconcile_phase_b_monitors_chunks(
        self,
        controller: ReconcileController,
        sample_manifest: Manifest,
        mock_artifact_store,
    ) -> None:
        """Test reconcile detects chunk phase when chunks in progress."""
        # Initialize job
        await controller.initialize_job(sample_manifest)

        # Mark one chunk as done, others still ready
        items = controller.ticket_system._items
        for work_id, item in items.items():
            if item.work_type == WorkItemType.DOC_CHUNK:
                if "chunk_001" in work_id:
                    item.status = WorkItemStatus.DONE
                break

        result = await controller.reconcile(sample_manifest.job_id)
        assert result.phase == "chunk"

    @pytest.mark.asyncio
    async def test_reconcile_advances_blocked_merges(
        self,
        controller: ReconcileController,
        sample_manifest: Manifest,
    ) -> None:
        """Test reconcile advances merges when dependencies complete."""
        await controller.initialize_job(sample_manifest)

        # Mark all chunks as done
        items = controller.ticket_system._items
        for item in items.values():
            if item.work_type == WorkItemType.DOC_CHUNK:
                item.status = WorkItemStatus.DONE

        result = await controller.reconcile(sample_manifest.job_id)

        # Should be in merge phase and have unblocked items
        assert result.phase == "merge"
        assert result.work_items_unblocked > 0

    @pytest.mark.asyncio
    async def test_get_job_status(
        self,
        controller: ReconcileController,
        sample_manifest: Manifest,
    ) -> None:
        """Test get_job_status returns correct metrics."""
        await controller.initialize_job(sample_manifest)

        status = await controller.get_job_status(sample_manifest.job_id)

        assert status["job_id"] == "test-job-001"
        assert status["chunks_total"] == 3
        assert status["chunks_done"] == 0
        assert status["merges_total"] == 2
        assert status["merges_blocked"] == 2
        assert status["is_complete"] is False

    @pytest.mark.asyncio
    async def test_advance_blocked_items(
        self,
        controller: ReconcileController,
        sample_manifest: Manifest,
    ) -> None:
        """Test advance_blocked_items transitions items correctly."""
        await controller.initialize_job(sample_manifest)

        # Mark chunk_001 and chunk_002 as done
        items = controller.ticket_system._items
        for work_id, item in items.items():
            if "chunk_001" in work_id or "chunk_002" in work_id:
                item.status = WorkItemStatus.DONE

        advanced = await controller.advance_blocked_items(sample_manifest.job_id)

        # merge_procedure should be advanced (depends on chunk_001, chunk_002)
        assert advanced == 1

        # Check the merge item is now READY
        for item in items.values():
            if item.work_type == WorkItemType.DOC_MERGE:
                if isinstance(item.payload, DocMergePayload):
                    if item.payload.merge_node_id == "merge_procedure":
                        assert item.status == WorkItemStatus.READY

    @pytest.mark.asyncio
    async def test_route_challenger_issues(
        self,
        controller: ReconcileController,
        sample_manifest: Manifest,
        mock_artifact_store,
    ) -> None:
        """Test routing challenger issues to follow-up work items."""
        await controller.initialize_job(sample_manifest)

        # Create challenge result with issues
        challenge_result = ChallengeResult(
            job_id="test-job-001",
            artifact_id="TEST001.cbl",
            artifact_version="abc123",
            issues=[
                Issue(
                    issue_id="issue-001",
                    severity=IssueSeverity.MAJOR,
                    question="What happens when file read fails?",
                    suspected_scopes=["chunk_001"],
                    routing_hints={"paragraphs": ["ERROR-HANDLING"]},
                ),
                Issue(
                    issue_id="issue-002",
                    severity=IssueSeverity.MINOR,  # Should be skipped
                    question="Minor style issue",
                ),
            ],
            resolution_plan=ResolutionPlan(
                followup_tasks=[],
                requires_patch_merge=True,
            ),
        )

        followups = await controller.route_challenger_issues(
            sample_manifest.job_id, challenge_result
        )

        # Should create 1 follow-up (minor issue skipped)
        assert len(followups) == 1
        assert followups[0].work_type == WorkItemType.DOC_FOLLOWUP

    @pytest.mark.asyncio
    async def test_create_patch_merge(
        self,
        controller: ReconcileController,
        sample_manifest: Manifest,
        mock_artifact_store,
    ) -> None:
        """Test creating patch merge work item."""
        await controller.initialize_job(sample_manifest)

        # Create a follow-up item first
        followup_id = "test-job-001-followup-test"
        followup_payload = DocFollowupPayload(
            job_id="test-job-001",
            issue_id="issue-001",
            scope={"chunk_ids": ["chunk_001"], "question": "Test question"},
            inputs=["s3://bucket/input.json"],
            output_uri="s3://bucket/followup.json",
        )
        controller.ticket_system._items[followup_id] = WorkItem(
            work_id=followup_id,
            work_type=WorkItemType.DOC_FOLLOWUP,
            status=WorkItemStatus.DONE,
            payload=followup_payload,
        )

        patch_merge = await controller.create_patch_merge(
            sample_manifest.job_id, [followup_id]
        )

        assert patch_merge.work_type == WorkItemType.DOC_PATCH_MERGE
        assert patch_merge.status == WorkItemStatus.READY

    def test_compute_idempotency_key(
        self,
        controller: ReconcileController,
    ) -> None:
        """Test idempotency key computation."""
        key = controller.compute_idempotency_key(
            "job-001", "doc_chunk", "abc123", "chunk_001"
        )

        assert "job-001" in key
        assert "doc_chunk" in key
        assert "abc123" in key
        assert "chunk_001" in key

    def test_determine_phase_plan(
        self,
        controller: ReconcileController,
        sample_manifest: Manifest,
    ) -> None:
        """Test phase determination for plan phase."""
        phase = controller._determine_phase(sample_manifest, {})
        assert phase == "plan"

    def test_determine_phase_chunk(
        self,
        controller: ReconcileController,
        sample_manifest: Manifest,
    ) -> None:
        """Test phase determination for chunk phase."""
        # Create work items with some chunks not done
        chunks = [
            WorkItem(
                work_id="chunk-1",
                work_type=WorkItemType.DOC_CHUNK,
                status=WorkItemStatus.READY,
                payload=DocChunkPayload(
                    job_id="test",
                    chunk_id="chunk_001",
                    chunk_locator=ChunkLocator(start_line=1, end_line=100),
                    result_uri="s3://bucket/chunk_001.json",
                ),
            ),
        ]
        merges = [
            WorkItem(
                work_id="merge-1",
                work_type=WorkItemType.DOC_MERGE,
                status=WorkItemStatus.BLOCKED,
                payload=DocMergePayload(
                    job_id="test",
                    merge_node_id="merge_001",
                    input_uris=["s3://bucket/chunk_001.json"],
                    output_uri="s3://bucket/merge_001.json",
                ),
            ),
        ]

        work_items_by_type = {
            WorkItemType.DOC_CHUNK: chunks,
            WorkItemType.DOC_MERGE: merges,
        }

        phase = controller._determine_phase(sample_manifest, work_items_by_type)
        assert phase == "chunk"

    def test_split_cross_cutting_scope(
        self,
        controller: ReconcileController,
        sample_manifest: Manifest,
    ) -> None:
        """Test splitting cross-cutting issues into bounded scopes."""
        issue = Issue(
            issue_id="cross-cutting-001",
            severity=IssueSeverity.BLOCKER,
            question="How is error handling done?",
        )

        scopes = controller.split_cross_cutting_scope(issue, sample_manifest)

        # Should create scopes grouped by division
        assert len(scopes) >= 1
        for scope in scopes:
            assert "issue_id" in scope
            assert scope["type"] == "cross_cutting"


class TestReconcileResult:
    """Tests for ReconcileResult dataclass."""

    def test_default_values(self) -> None:
        """Test ReconcileResult has correct defaults."""
        result = ReconcileResult()

        assert result.work_items_created == 0
        assert result.work_items_unblocked == 0
        assert result.work_items_completed == 0
        assert result.errors == []
        assert result.phase == "unknown"

    def test_with_values(self) -> None:
        """Test ReconcileResult with values."""
        result = ReconcileResult(
            work_items_created=5,
            work_items_unblocked=2,
            work_items_completed=3,
            errors=["Error 1"],
            phase="chunk",
        )

        assert result.work_items_created == 5
        assert result.phase == "chunk"
        assert len(result.errors) == 1
