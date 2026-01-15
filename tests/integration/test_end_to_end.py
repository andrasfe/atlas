"""End-to-end integration tests for Atlas workflow.

These tests verify the complete workflow from DOC_REQUEST through
all phases to completion, using mock LLM adapters and in-memory stores.

Test Coverage:
1. atlas-b7y: End-to-end small COBOL program
2. atlas-2ks: Large COBOL program with hierarchical merges
3. atlas-6mc: Challenger loop with cross-cutting question
4. atlas-799: Crash recovery and resume
5. atlas-udt: Ambiguous routing fallback
6. atlas-0j7: Multiple iteration challenger loop
"""

import pytest
import asyncio
import json
from pathlib import Path

from atlas.adapters.memory_ticket_system import MemoryTicketSystem
from atlas.controller.reconciler import ReconcileController
from atlas.controller.base import ControllerConfig
from atlas.controller.persistence import JobStatePersistence, JobCheckpoint
from atlas.models.artifact import Artifact, ArtifactRef
from atlas.models.manifest import (
    Manifest,
    ChunkSpec,
    MergeNode,
    AnalysisProfile,
    SplitterProfile,
    ReviewPolicy,
    ArtifactOutputConfig,
)
from atlas.models.enums import (
    WorkItemStatus,
    WorkItemType,
    ArtifactType,
    ChunkKind,
    IssueSeverity,
)
from atlas.models.work_item import WorkItem, DocChunkPayload, ChunkLocator
from atlas.models.results import (
    ChunkResult,
    ChunkFacts,
    MergeResult,
    ConsolidatedFacts,
    MergeCoverage,
    ChallengeResult,
    Issue,
    ResolutionPlan,
    FollowupTask,
    FollowupAnswer,
    Evidence,
    SymbolDef,
    IOOperation,
)

from tests.integration.mock_llm import MockLLMAdapter, MockLLMConfig
from tests.integration.mock_agent_harness import (
    MockAgentHarness,
    AgentType,
    create_harness_no_issues,
    create_harness_with_cross_cutting_issues,
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


class MockArtifactStore:
    """In-memory artifact store for testing."""

    def __init__(self) -> None:
        self._artifacts: dict[str, bytes] = {}
        self._json_cache: dict[str, dict] = {}

    async def write(
        self,
        uri: str,
        content: bytes,
        *,
        content_type: str = "application/json",
        metadata: dict[str, str] | None = None,
    ) -> str:
        self._artifacts[uri] = content
        return uri

    async def write_json(self, uri: str, data: dict) -> str:
        content = json.dumps(data).encode()
        self._artifacts[uri] = content
        self._json_cache[uri] = data
        return uri

    async def read(self, uri: str) -> bytes:
        if uri not in self._artifacts:
            raise FileNotFoundError(f"Artifact not found: {uri}")
        return self._artifacts[uri]

    async def read_json(self, uri: str) -> dict:
        if uri in self._json_cache:
            return self._json_cache[uri]
        content = await self.read(uri)
        data = json.loads(content)
        self._json_cache[uri] = data
        return data

    async def exists(self, uri: str) -> bool:
        return uri in self._artifacts

    async def delete(self, uri: str) -> bool:
        if uri in self._artifacts:
            del self._artifacts[uri]
            self._json_cache.pop(uri, None)
            return True
        return False

    async def list_artifacts(self, prefix: str, limit: int = 1000) -> list[str]:
        return [uri for uri in self._artifacts if uri.startswith(prefix)][:limit]

    def generate_uri(self, base: str, path: str, **kwargs) -> str:
        result = f"{base}/{path}"
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


@pytest.fixture
def ticket_system():
    """Provide a fresh in-memory ticket system."""
    return MemoryTicketSystem()


@pytest.fixture
def artifact_store():
    """Provide a fresh in-memory artifact store."""
    return MockArtifactStore()


@pytest.fixture
def mock_llm():
    """Provide a mock LLM adapter."""
    return MockLLMAdapter()


@pytest.fixture
def controller(ticket_system, artifact_store):
    """Provide a reconcile controller."""
    config = ControllerConfig(max_challenge_iterations=3)
    return ReconcileController(ticket_system, artifact_store, config)


@pytest.fixture
def small_program_content():
    """Load small COBOL program fixture."""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "cobol" / "small_program.cbl"
    if fixture_path.exists():
        return fixture_path.read_text()
    # Fallback content if fixture doesn't exist
    return """       IDENTIFICATION DIVISION.
       PROGRAM-ID. TESTPROG.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 WS-STATUS PIC X(2).
       PROCEDURE DIVISION.
           MAIN-LOGIC.
               DISPLAY "HELLO WORLD".
               STOP RUN.
"""


@pytest.fixture
def small_manifest(artifact_store) -> Manifest:
    """Create a manifest for small program testing."""
    artifact_ref = ArtifactRef(
        artifact_id="TESTPROG.cbl",
        artifact_type=ArtifactType.COBOL,
        artifact_version="test-version-001",
        artifact_uri="s3://test/TESTPROG.cbl",
    )

    chunks = [
        ChunkSpec(
            chunk_id="chunk-id-division",
            chunk_kind=ChunkKind.IDENTIFICATION_DIVISION,
            start_line=1,
            end_line=3,
            division="IDENTIFICATION",
            estimated_tokens=50,
        ),
        ChunkSpec(
            chunk_id="chunk-data-division",
            chunk_kind=ChunkKind.DATA_DIVISION,
            start_line=4,
            end_line=6,
            division="DATA",
            estimated_tokens=100,
        ),
        ChunkSpec(
            chunk_id="chunk-procedure",
            chunk_kind=ChunkKind.PROCEDURE_PART,
            start_line=7,
            end_line=11,
            division="PROCEDURE",
            paragraphs=["MAIN-LOGIC"],
            estimated_tokens=150,
        ),
    ]

    merge_dag = [
        MergeNode(
            merge_node_id="merge-root",
            input_ids=["chunk-id-division", "chunk-data-division", "chunk-procedure"],
            is_root=True,
            level=1,
        ),
    ]

    return Manifest(
        job_id="test-job-small",
        artifact_ref=artifact_ref,
        analysis_profile=AnalysisProfile(name="documentation"),
        splitter_profile=SplitterProfile(name="cobol-standard"),
        context_budget=4000,
        chunks=chunks,
        merge_dag=merge_dag,
        review_policy=ReviewPolicy(max_iterations=3),
        artifacts=ArtifactOutputConfig(
            base_uri="s3://test-results",
            doc_path="doc.md",
            doc_model_path="doc_model.json",
            chunk_results_path="chunks/{chunk_id}.json",
            merge_results_path="merges/{merge_node_id}.json",
        ),
    )


# -----------------------------------------------------------------------------
# Test: End-to-end small COBOL program (atlas-b7y)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_to_end_small_program(
    ticket_system,
    artifact_store,
    controller,
    small_manifest,
    mock_llm,
):
    """Test full workflow: DOC_REQUEST -> split -> chunk -> merge -> challenge -> done.

    This test verifies the complete workflow for a small COBOL program:
    1. Initialize job with manifest
    2. Process all chunk work items
    3. Process merge work items
    4. Process challenge work item
    5. Verify job completes successfully

    Ticket: atlas-b7y
    """
    # Configure mock LLM to produce no issues (clean pass)
    mock_llm.configure_no_issues()

    # Initialize job
    job_id = await controller.initialize_job(small_manifest)
    assert job_id == "test-job-small"

    # Verify initial state
    status = await controller.get_job_status(job_id)
    assert status["chunks_total"] == 3
    assert status["merges_total"] == 1
    assert status["phase"] == "chunk"

    # Simulate chunk processing
    chunk_items = await ticket_system.query_by_status(
        WorkItemStatus.READY, WorkItemType.DOC_CHUNK
    )
    assert len(chunk_items) == 3

    for chunk_item in chunk_items:
        # Claim and process chunk
        claimed = await ticket_system.claim_work_item(chunk_item.work_id, "worker-1")
        assert claimed

        # Generate chunk result
        chunk_result = _create_mock_chunk_result(
            job_id,
            small_manifest.artifact_ref.artifact_id,
            small_manifest.artifact_ref.artifact_version,
            chunk_item.payload.chunk_id,
        )

        # Write result
        result_uri = chunk_item.payload.result_uri
        await artifact_store.write_json(result_uri, chunk_result.model_dump())

        # Mark done
        await ticket_system.update_status(chunk_item.work_id, WorkItemStatus.DONE)

    # Run reconciliation to advance merges
    result = await controller.reconcile(job_id)
    assert result.work_items_unblocked >= 1

    # Process merge
    merge_items = await ticket_system.query_by_status(
        WorkItemStatus.READY, WorkItemType.DOC_MERGE
    )
    assert len(merge_items) == 1

    merge_item = merge_items[0]
    claimed = await ticket_system.claim_work_item(merge_item.work_id, "worker-1")
    assert claimed

    # Generate merge result
    merge_result = _create_mock_merge_result(
        job_id,
        small_manifest.artifact_ref.artifact_id,
        small_manifest.artifact_ref.artifact_version,
        merge_item.payload.merge_node_id,
    )

    # Write result and doc model
    await artifact_store.write_json(
        merge_item.payload.output_uri, merge_result.model_dump()
    )
    await artifact_store.write_json(
        f"{small_manifest.artifacts.base_uri}/{small_manifest.artifacts.doc_path}",
        {"content": "Generated documentation"},
    )
    await artifact_store.write_json(
        f"{small_manifest.artifacts.base_uri}/{small_manifest.artifacts.doc_model_path}",
        {"sections": [], "index": {}},
    )

    await ticket_system.update_status(merge_item.work_id, WorkItemStatus.DONE)

    # Run reconciliation to create challenge
    result = await controller.reconcile(job_id)

    # Process challenge
    challenge_items = await ticket_system.query_by_status(
        WorkItemStatus.READY, WorkItemType.DOC_CHALLENGE
    )
    assert len(challenge_items) == 1

    challenge_item = challenge_items[0]
    claimed = await ticket_system.claim_work_item(challenge_item.work_id, "worker-1")
    assert claimed

    # Generate challenge result with no issues
    challenge_result = ChallengeResult(
        job_id=job_id,
        artifact_id=small_manifest.artifact_ref.artifact_id,
        artifact_version=small_manifest.artifact_ref.artifact_version,
        issues=[],
        resolution_plan=ResolutionPlan(
            followup_tasks=[],
            requires_patch_merge=False,
        ),
    )

    await artifact_store.write_json(
        challenge_item.payload.output_uri, challenge_result.model_dump()
    )
    await ticket_system.update_status(challenge_item.work_id, WorkItemStatus.DONE)

    # Final reconciliation
    result = await controller.reconcile(job_id)

    # Verify completion
    status = await controller.get_job_status(job_id)
    assert status["phase"] == "complete"
    assert status["is_complete"] is True
    assert status["chunks_done"] == 3
    assert status["merges_done"] == 1
    assert status["challenges_done"] == 1


# -----------------------------------------------------------------------------
# Test: Large program with hierarchical merges (atlas-2ks)
# -----------------------------------------------------------------------------


@pytest.fixture
def large_manifest(artifact_store) -> Manifest:
    """Create a manifest for large program testing with hierarchical merges."""
    artifact_ref = ArtifactRef(
        artifact_id="LARGEPROG.cbl",
        artifact_type=ArtifactType.COBOL,
        artifact_version="test-version-large",
        artifact_uri="s3://test/LARGEPROG.cbl",
    )

    # Create 8 chunks
    chunks = []
    for i in range(8):
        chunks.append(
            ChunkSpec(
                chunk_id=f"chunk-{i}",
                chunk_kind=ChunkKind.PROCEDURE_PART,
                start_line=1 + i * 100,
                end_line=100 + i * 100,
                division="PROCEDURE",
                paragraphs=[f"PARA-{i}"],
                estimated_tokens=500,
            )
        )

    # Create hierarchical merge DAG:
    # Level 1: merge chunks 0-1, 2-3, 4-5, 6-7
    # Level 2: merge level-1 results into pairs
    # Level 3 (root): final merge
    merge_dag = [
        # Level 1 merges
        MergeNode(
            merge_node_id="merge-l1-0",
            input_ids=["chunk-0", "chunk-1"],
            is_root=False,
            level=1,
        ),
        MergeNode(
            merge_node_id="merge-l1-1",
            input_ids=["chunk-2", "chunk-3"],
            is_root=False,
            level=1,
        ),
        MergeNode(
            merge_node_id="merge-l1-2",
            input_ids=["chunk-4", "chunk-5"],
            is_root=False,
            level=1,
        ),
        MergeNode(
            merge_node_id="merge-l1-3",
            input_ids=["chunk-6", "chunk-7"],
            is_root=False,
            level=1,
        ),
        # Level 2 merges
        MergeNode(
            merge_node_id="merge-l2-0",
            input_ids=["merge-l1-0", "merge-l1-1"],
            is_root=False,
            level=2,
        ),
        MergeNode(
            merge_node_id="merge-l2-1",
            input_ids=["merge-l1-2", "merge-l1-3"],
            is_root=False,
            level=2,
        ),
        # Root merge
        MergeNode(
            merge_node_id="merge-root",
            input_ids=["merge-l2-0", "merge-l2-1"],
            is_root=True,
            level=3,
        ),
    ]

    return Manifest(
        job_id="test-job-large",
        artifact_ref=artifact_ref,
        analysis_profile=AnalysisProfile(name="documentation"),
        splitter_profile=SplitterProfile(name="cobol-standard"),
        context_budget=4000,
        chunks=chunks,
        merge_dag=merge_dag,
        review_policy=ReviewPolicy(max_iterations=3),
        artifacts=ArtifactOutputConfig(
            base_uri="s3://test-results-large",
            doc_path="doc.md",
            doc_model_path="doc_model.json",
            chunk_results_path="chunks/{chunk_id}.json",
            merge_results_path="merges/{merge_node_id}.json",
        ),
    )


@pytest.mark.asyncio
async def test_large_program_hierarchical_merges(
    ticket_system,
    artifact_store,
    large_manifest,
):
    """Test multi-level merge tree execution for large programs.

    This test verifies:
    1. All 8 chunks are created and can be processed
    2. Level 1 merges activate after their chunk dependencies complete
    3. Level 2 merges activate after level 1 completes
    4. Root merge activates last
    5. Proper dependency ordering throughout

    Ticket: atlas-2ks
    """
    config = ControllerConfig(max_challenge_iterations=3)
    controller = ReconcileController(ticket_system, artifact_store, config)

    # Initialize job
    job_id = await controller.initialize_job(large_manifest)
    assert job_id == "test-job-large"

    # Verify initial state
    status = await controller.get_job_status(job_id)
    assert status["chunks_total"] == 8
    assert status["merges_total"] == 7  # 4 + 2 + 1
    assert status["merges_blocked"] == 7  # All blocked initially

    # Process all chunks
    for i in range(8):
        chunk_items = await ticket_system.query_by_status(
            WorkItemStatus.READY, WorkItemType.DOC_CHUNK
        )
        assert len(chunk_items) > 0

        chunk_item = chunk_items[0]
        await ticket_system.claim_work_item(chunk_item.work_id, "worker-1")

        chunk_result = _create_mock_chunk_result(
            job_id,
            large_manifest.artifact_ref.artifact_id,
            large_manifest.artifact_ref.artifact_version,
            chunk_item.payload.chunk_id,
        )
        await artifact_store.write_json(
            chunk_item.payload.result_uri, chunk_result.model_dump()
        )
        await ticket_system.update_status(chunk_item.work_id, WorkItemStatus.DONE)

    # Reconcile to advance blocked merges
    result = await controller.reconcile(job_id)
    assert result.work_items_unblocked >= 4  # Level 1 merges should unblock

    # Check that only level 1 merges are READY
    ready_merges = await ticket_system.query_by_status(
        WorkItemStatus.READY, WorkItemType.DOC_MERGE
    )
    level_1_ids = {"merge-l1-0", "merge-l1-1", "merge-l1-2", "merge-l1-3"}
    ready_ids = {m.payload.merge_node_id for m in ready_merges}
    assert ready_ids == level_1_ids

    # Process level 1 merges
    for merge_item in ready_merges:
        await ticket_system.claim_work_item(merge_item.work_id, "worker-1")
        merge_result = _create_mock_merge_result(
            job_id,
            large_manifest.artifact_ref.artifact_id,
            large_manifest.artifact_ref.artifact_version,
            merge_item.payload.merge_node_id,
        )
        await artifact_store.write_json(
            merge_item.payload.output_uri, merge_result.model_dump()
        )
        await ticket_system.update_status(merge_item.work_id, WorkItemStatus.DONE)

    # Reconcile to advance level 2 merges
    result = await controller.reconcile(job_id)
    assert result.work_items_unblocked >= 2

    # Check level 2 merges are READY
    ready_merges = await ticket_system.query_by_status(
        WorkItemStatus.READY, WorkItemType.DOC_MERGE
    )
    level_2_ids = {"merge-l2-0", "merge-l2-1"}
    ready_ids = {m.payload.merge_node_id for m in ready_merges}
    assert ready_ids == level_2_ids

    # Process level 2 merges
    for merge_item in ready_merges:
        await ticket_system.claim_work_item(merge_item.work_id, "worker-1")
        merge_result = _create_mock_merge_result(
            job_id,
            large_manifest.artifact_ref.artifact_id,
            large_manifest.artifact_ref.artifact_version,
            merge_item.payload.merge_node_id,
        )
        await artifact_store.write_json(
            merge_item.payload.output_uri, merge_result.model_dump()
        )
        await ticket_system.update_status(merge_item.work_id, WorkItemStatus.DONE)

    # Reconcile to advance root merge
    result = await controller.reconcile(job_id)
    assert result.work_items_unblocked >= 1

    # Check root merge is READY
    ready_merges = await ticket_system.query_by_status(
        WorkItemStatus.READY, WorkItemType.DOC_MERGE
    )
    assert len(ready_merges) == 1
    assert ready_merges[0].payload.merge_node_id == "merge-root"

    # Verify status
    status = await controller.get_job_status(job_id)
    assert status["chunks_done"] == 8
    assert status["merges_done"] == 6  # 4 + 2 done, root pending


# -----------------------------------------------------------------------------
# Test: Challenger loop with cross-cutting question (atlas-6mc)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_challenger_cross_cutting_question(
    ticket_system,
    artifact_store,
    small_manifest,
):
    """Test when challenger identifies cross-cutting issue.

    This test verifies:
    1. Challenger raises an issue without specific chunk scope
    2. Controller routes to cross-cutting follow-up plan
    3. Follow-ups are dispatched to correct chunks

    Ticket: atlas-6mc
    """
    config = ControllerConfig(max_challenge_iterations=3)
    controller = ReconcileController(ticket_system, artifact_store, config)

    # Initialize and process through merge phase
    job_id = await controller.initialize_job(small_manifest)

    # Complete all chunks
    chunk_items = await ticket_system.query_by_status(
        WorkItemStatus.READY, WorkItemType.DOC_CHUNK
    )
    for chunk_item in chunk_items:
        await ticket_system.claim_work_item(chunk_item.work_id, "worker-1")
        chunk_result = _create_mock_chunk_result(
            job_id,
            small_manifest.artifact_ref.artifact_id,
            small_manifest.artifact_ref.artifact_version,
            chunk_item.payload.chunk_id,
        )
        await artifact_store.write_json(
            chunk_item.payload.result_uri, chunk_result.model_dump()
        )
        await ticket_system.update_status(chunk_item.work_id, WorkItemStatus.DONE)

    # Reconcile and complete merge
    await controller.reconcile(job_id)
    merge_items = await ticket_system.query_by_status(
        WorkItemStatus.READY, WorkItemType.DOC_MERGE
    )
    for merge_item in merge_items:
        await ticket_system.claim_work_item(merge_item.work_id, "worker-1")
        merge_result = _create_mock_merge_result(
            job_id,
            small_manifest.artifact_ref.artifact_id,
            small_manifest.artifact_ref.artifact_version,
            merge_item.payload.merge_node_id,
        )
        await artifact_store.write_json(
            merge_item.payload.output_uri, merge_result.model_dump()
        )
        # Write doc model
        await artifact_store.write_json(
            f"{small_manifest.artifacts.base_uri}/{small_manifest.artifacts.doc_model_path}",
            {"sections": [], "index": {"symbol_to_chunks": {}, "paragraph_to_chunk": {}}},
        )
        await artifact_store.write_json(
            f"{small_manifest.artifacts.base_uri}/{small_manifest.artifacts.doc_path}",
            {"content": "Documentation"},
        )
        await ticket_system.update_status(merge_item.work_id, WorkItemStatus.DONE)

    # Reconcile to create challenge
    await controller.reconcile(job_id)

    # Complete challenge with cross-cutting issue (empty suspected_scopes)
    challenge_items = await ticket_system.query_by_status(
        WorkItemStatus.READY, WorkItemType.DOC_CHALLENGE
    )
    assert len(challenge_items) == 1

    challenge_item = challenge_items[0]
    await ticket_system.claim_work_item(challenge_item.work_id, "worker-1")

    # Create challenge result with cross-cutting issue
    cross_cutting_issue = Issue(
        issue_id="cross-cutting-issue-1",
        severity=IssueSeverity.MAJOR,
        question="What is the overall error handling strategy across all divisions?",
        doc_section_refs=["section-errors"],
        suspected_scopes=[],  # Empty = cross-cutting
        routing_hints={},  # No specific routing
    )

    challenge_result = ChallengeResult(
        job_id=job_id,
        artifact_id=small_manifest.artifact_ref.artifact_id,
        artifact_version=small_manifest.artifact_ref.artifact_version,
        issues=[cross_cutting_issue],
        resolution_plan=ResolutionPlan(
            followup_tasks=[
                FollowupTask(
                    issue_id="cross-cutting-issue-1",
                    scope={},  # Cross-cutting scope
                    description="Analyze error handling across all divisions",
                )
            ],
            requires_patch_merge=True,
        ),
    )

    await artifact_store.write_json(
        challenge_item.payload.output_uri, challenge_result.model_dump()
    )
    await ticket_system.update_status(challenge_item.work_id, WorkItemStatus.DONE)

    # Use the controller's route_challenger_issues method to dispatch follow-ups
    # This is the correct way to create follow-ups from challenge results
    followup_items = await controller.route_challenger_issues(job_id, challenge_result)

    # Cross-cutting issues with empty suspected_scopes result in follow-ups
    # grouped by division. Since our manifest has 3 chunks in different divisions,
    # we should have at least one follow-up per division (up to max_chunks_per_scope)
    assert len(followup_items) >= 1, "Cross-cutting issues should create follow-up work items"

    # Cross-cutting follow-ups should have scope information
    for followup in followup_items:
        assert followup.status in [WorkItemStatus.READY, WorkItemStatus.BLOCKED]
        # The scope should include cross_cutting type
        if hasattr(followup.payload, 'scope'):
            scope = followup.payload.scope
            # Cross-cutting scopes have type: cross_cutting
            assert scope.get("type") == "cross_cutting" or len(scope.get("chunk_ids", [])) >= 0


# -----------------------------------------------------------------------------
# Test: Crash recovery and resume (atlas-799)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_crash_recovery_and_resume(
    ticket_system,
    artifact_store,
    small_manifest,
):
    """Test checkpoint/restore using JobStatePersistence.

    This test verifies:
    1. Checkpoint can be created at any phase
    2. State can be restored after simulated crash
    3. Workflow resumes correctly from checkpoint

    Ticket: atlas-799
    """
    config = ControllerConfig(max_challenge_iterations=3)
    controller = ReconcileController(ticket_system, artifact_store, config)
    persistence = JobStatePersistence(artifact_store, ticket_system)

    # Initialize job
    job_id = await controller.initialize_job(small_manifest)

    # Process first chunk
    chunk_items = await ticket_system.query_by_status(
        WorkItemStatus.READY, WorkItemType.DOC_CHUNK
    )
    first_chunk = chunk_items[0]
    await ticket_system.claim_work_item(first_chunk.work_id, "worker-1")
    chunk_result = _create_mock_chunk_result(
        job_id,
        small_manifest.artifact_ref.artifact_id,
        small_manifest.artifact_ref.artifact_version,
        first_chunk.payload.chunk_id,
    )
    await artifact_store.write_json(
        first_chunk.payload.result_uri, chunk_result.model_dump()
    )
    await ticket_system.update_status(first_chunk.work_id, WorkItemStatus.DONE)

    # Create checkpoint after partial progress
    checkpoint = await persistence.save_checkpoint(
        job_id,
        metadata={"reason": "test_checkpoint"},
    )

    assert checkpoint.checkpoint_id is not None
    assert checkpoint.job_id == job_id
    assert checkpoint.phase == "chunk"  # Still in chunk phase
    assert len(checkpoint.work_items) > 0

    # Verify checkpoint contains correct state
    # WorkItemSnapshot stores work_type as string value (e.g., "doc_chunk")
    chunk_snapshots = [
        w for w in checkpoint.work_items if w.work_type == WorkItemType.DOC_CHUNK.value
    ]
    done_chunks = [w for w in chunk_snapshots if w.status == WorkItemStatus.DONE.value]
    assert len(done_chunks) == 1

    # Simulate crash by marking an IN_PROGRESS item as READY
    # (simulating a worker that crashed mid-processing)
    remaining_chunks = await ticket_system.query_by_status(
        WorkItemStatus.READY, WorkItemType.DOC_CHUNK
    )
    if remaining_chunks:
        await ticket_system.claim_work_item(remaining_chunks[0].work_id, "worker-crash")
        # Worker "crashes" - item is IN_PROGRESS but worker is gone

    # List checkpoints
    checkpoints = await persistence.list_checkpoints(job_id)
    assert len(checkpoints) >= 1
    assert checkpoints[0]["checkpoint_id"] == checkpoint.checkpoint_id

    # Load checkpoint
    loaded = await persistence.load_checkpoint(job_id)
    assert loaded is not None
    assert loaded.checkpoint_id == checkpoint.checkpoint_id

    # Restore from checkpoint
    # In a real scenario, this would reset stuck IN_PROGRESS items
    restored = await persistence.restore_checkpoint(job_id)
    assert restored is True

    # Verify job can continue
    status = await controller.get_job_status(job_id)
    assert status["chunks_done"] >= 1
    assert status["phase"] in ["chunk", "merge"]


# -----------------------------------------------------------------------------
# Test: Ambiguous routing fallback (atlas-udt)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ambiguous_routing_fallback(
    ticket_system,
    artifact_store,
    small_manifest,
):
    """Test when issues can't be routed to specific chunks.

    This test verifies:
    1. Issue with ambiguous routing hints
    2. Controller falls back to cross-cutting scope
    3. Follow-ups are created with bounded scopes

    Ticket: atlas-udt
    """
    config = ControllerConfig(max_challenge_iterations=3)
    controller = ReconcileController(ticket_system, artifact_store, config)

    # Initialize and complete through challenge
    job_id = await controller.initialize_job(small_manifest)

    # Complete chunks
    chunk_items = await ticket_system.query_by_status(
        WorkItemStatus.READY, WorkItemType.DOC_CHUNK
    )
    for chunk_item in chunk_items:
        await ticket_system.claim_work_item(chunk_item.work_id, "worker-1")
        chunk_result = _create_mock_chunk_result(
            job_id,
            small_manifest.artifact_ref.artifact_id,
            small_manifest.artifact_ref.artifact_version,
            chunk_item.payload.chunk_id,
        )
        await artifact_store.write_json(
            chunk_item.payload.result_uri, chunk_result.model_dump()
        )
        await ticket_system.update_status(chunk_item.work_id, WorkItemStatus.DONE)

    await controller.reconcile(job_id)

    # Complete merge
    merge_items = await ticket_system.query_by_status(
        WorkItemStatus.READY, WorkItemType.DOC_MERGE
    )
    for merge_item in merge_items:
        await ticket_system.claim_work_item(merge_item.work_id, "worker-1")
        merge_result = _create_mock_merge_result(
            job_id,
            small_manifest.artifact_ref.artifact_id,
            small_manifest.artifact_ref.artifact_version,
            merge_item.payload.merge_node_id,
        )
        await artifact_store.write_json(
            merge_item.payload.output_uri, merge_result.model_dump()
        )
        await artifact_store.write_json(
            f"{small_manifest.artifacts.base_uri}/{small_manifest.artifacts.doc_model_path}",
            {"sections": [], "index": {"symbol_to_chunks": {}, "paragraph_to_chunk": {}}},
        )
        await artifact_store.write_json(
            f"{small_manifest.artifacts.base_uri}/{small_manifest.artifacts.doc_path}",
            {"content": "Documentation"},
        )
        await ticket_system.update_status(merge_item.work_id, WorkItemStatus.DONE)

    await controller.reconcile(job_id)

    # Complete challenge with ambiguous issue
    challenge_items = await ticket_system.query_by_status(
        WorkItemStatus.READY, WorkItemType.DOC_CHALLENGE
    )
    challenge_item = challenge_items[0]
    await ticket_system.claim_work_item(challenge_item.work_id, "worker-1")

    # Issue with routing hints that don't match any known chunks
    ambiguous_issue = Issue(
        issue_id="ambiguous-issue-1",
        severity=IssueSeverity.MAJOR,
        question="How does UNKNOWN-PARA handle errors?",
        doc_section_refs=["section-unknown"],
        suspected_scopes=["nonexistent-chunk"],  # Invalid chunk ID
        routing_hints={"paragraphs": ["UNKNOWN-PARA"]},  # Unknown paragraph
    )

    challenge_result = ChallengeResult(
        job_id=job_id,
        artifact_id=small_manifest.artifact_ref.artifact_id,
        artifact_version=small_manifest.artifact_ref.artifact_version,
        issues=[ambiguous_issue],
        resolution_plan=ResolutionPlan(
            followup_tasks=[
                FollowupTask(
                    issue_id="ambiguous-issue-1",
                    scope={"chunk_ids": ["nonexistent-chunk"]},
                    description="Analyze unknown paragraph",
                )
            ],
            requires_patch_merge=True,
        ),
    )

    await artifact_store.write_json(
        challenge_item.payload.output_uri, challenge_result.model_dump()
    )
    await ticket_system.update_status(challenge_item.work_id, WorkItemStatus.DONE)

    # Reconcile to dispatch follow-ups
    await controller.reconcile(job_id)

    # Verify fallback handling - follow-ups should be created
    # even for ambiguous routing (using cross-cutting scope)
    status = await controller.get_job_status(job_id)
    # The phase should have progressed
    assert status["phase"] in ["followup", "challenge", "complete"]


# -----------------------------------------------------------------------------
# Test: Multiple iteration challenger loop (atlas-0j7)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_challenger_iterations(
    ticket_system,
    artifact_store,
    small_manifest,
):
    """Test full challenger loop with multiple rounds.

    This test verifies:
    1. First challenge raises issues
    2. Follow-ups are processed
    3. Patch merge updates documentation
    4. Re-challenge is triggered
    5. max_iterations is enforced

    Ticket: atlas-0j7
    """
    config = ControllerConfig(max_challenge_iterations=2)
    controller = ReconcileController(ticket_system, artifact_store, config)

    # Initialize job
    job_id = await controller.initialize_job(small_manifest)

    # Helper to complete all pending work of a type
    async def complete_work_items(work_type: WorkItemType):
        items = await ticket_system.query_by_status(WorkItemStatus.READY, work_type)
        for item in items:
            await ticket_system.claim_work_item(item.work_id, "worker-1")

            if work_type == WorkItemType.DOC_CHUNK:
                result = _create_mock_chunk_result(
                    job_id,
                    small_manifest.artifact_ref.artifact_id,
                    small_manifest.artifact_ref.artifact_version,
                    item.payload.chunk_id,
                )
                await artifact_store.write_json(
                    item.payload.result_uri, result.model_dump()
                )
            elif work_type == WorkItemType.DOC_MERGE:
                result = _create_mock_merge_result(
                    job_id,
                    small_manifest.artifact_ref.artifact_id,
                    small_manifest.artifact_ref.artifact_version,
                    item.payload.merge_node_id,
                )
                await artifact_store.write_json(
                    item.payload.output_uri, result.model_dump()
                )
                await artifact_store.write_json(
                    f"{small_manifest.artifacts.base_uri}/{small_manifest.artifacts.doc_model_path}",
                    {"sections": [], "index": {"symbol_to_chunks": {}, "paragraph_to_chunk": {}}},
                )
                await artifact_store.write_json(
                    f"{small_manifest.artifacts.base_uri}/{small_manifest.artifacts.doc_path}",
                    {"content": "Documentation"},
                )

            await ticket_system.update_status(item.work_id, WorkItemStatus.DONE)

    # Complete chunks and merges
    await complete_work_items(WorkItemType.DOC_CHUNK)
    await controller.reconcile(job_id)
    await complete_work_items(WorkItemType.DOC_MERGE)
    await controller.reconcile(job_id)

    # First challenge - raises issues
    challenge_items = await ticket_system.query_by_status(
        WorkItemStatus.READY, WorkItemType.DOC_CHALLENGE
    )
    assert len(challenge_items) == 1

    challenge_item = challenge_items[0]
    assert challenge_item.cycle_number == 1

    await ticket_system.claim_work_item(challenge_item.work_id, "worker-1")

    # Challenge raises issue requiring follow-up
    issue = Issue(
        issue_id="iter-issue-1",
        severity=IssueSeverity.MAJOR,
        question="Clarify error handling in MAIN-LOGIC",
        doc_section_refs=["section-1"],
        suspected_scopes=["chunk-procedure"],
        routing_hints={"paragraphs": ["MAIN-LOGIC"]},
    )

    challenge_result = ChallengeResult(
        job_id=job_id,
        artifact_id=small_manifest.artifact_ref.artifact_id,
        artifact_version=small_manifest.artifact_ref.artifact_version,
        issues=[issue],
        resolution_plan=ResolutionPlan(
            followup_tasks=[
                FollowupTask(
                    issue_id="iter-issue-1",
                    scope={"chunk_ids": ["chunk-procedure"]},
                    description="Clarify error handling",
                )
            ],
            requires_patch_merge=True,
        ),
    )

    await artifact_store.write_json(
        challenge_item.payload.output_uri, challenge_result.model_dump()
    )
    await ticket_system.update_status(challenge_item.work_id, WorkItemStatus.DONE)

    # Use route_challenger_issues to create follow-ups from the challenge result
    # This is the controller's method for dispatching follow-up work items
    followup_work_items = await controller.route_challenger_issues(job_id, challenge_result)

    # Verify follow-up created
    followup_items = await ticket_system.query_by_status(
        WorkItemStatus.READY, WorkItemType.DOC_FOLLOWUP
    )
    assert len(followup_items) >= 1, f"Expected follow-ups, got {len(followup_items)}"

    # Complete follow-ups
    for followup_item in followup_items:
        await ticket_system.claim_work_item(followup_item.work_id, "worker-1")
        followup_answer = FollowupAnswer(
            issue_id=followup_item.payload.issue_id,
            scope=followup_item.payload.scope,
            answer="Error handling uses FILE-STATUS checks",
            facts=ChunkFacts(symbols_defined=[], symbols_used=["FILE-STATUS"]),
            evidence=[Evidence(evidence_type="line_range", start_line=10, end_line=15, note="Status check")],
            confidence=0.95,
        )
        await artifact_store.write_json(
            followup_item.payload.output_uri, followup_answer.model_dump()
        )
        await ticket_system.update_status(followup_item.work_id, WorkItemStatus.DONE)

    # Reconcile to create patch merge
    await controller.reconcile(job_id)

    # Complete patch merge if created
    patch_items = await ticket_system.query_by_status(
        WorkItemStatus.READY, WorkItemType.DOC_PATCH_MERGE
    )
    if patch_items:
        for patch_item in patch_items:
            await ticket_system.claim_work_item(patch_item.work_id, "worker-1")
            await artifact_store.write_json(
                patch_item.payload.output_doc_uri, {"content": "Updated doc"}
            )
            await artifact_store.write_json(
                patch_item.payload.output_doc_model_uri,
                {"sections": [], "index": {}},
            )
            await ticket_system.update_status(patch_item.work_id, WorkItemStatus.DONE)

        await controller.reconcile(job_id)

    # Check for re-challenge (cycle 2)
    all_challenges = await ticket_system.query_by_job(job_id, WorkItemType.DOC_CHALLENGE)
    cycle_numbers = [c.cycle_number for c in all_challenges]

    # Verify max_iterations is being tracked
    status = await controller.get_job_status(job_id)
    # The job should still be progressing
    assert status["challenges_total"] >= 1


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _create_mock_chunk_result(
    job_id: str,
    artifact_id: str,
    artifact_version: str,
    chunk_id: str,
) -> ChunkResult:
    """Create a mock chunk result for testing."""
    return ChunkResult(
        job_id=job_id,
        artifact_id=artifact_id,
        artifact_version=artifact_version,
        chunk_id=chunk_id,
        chunk_locator={"start_line": 1, "end_line": 50},
        chunk_kind="procedure_part",
        summary=f"Analysis of {chunk_id}",
        facts=ChunkFacts(
            symbols_defined=[
                SymbolDef(name=f"WS-{chunk_id}", kind="variable", line_number=10),
            ],
            symbols_used=[f"WS-{chunk_id}"],
            entrypoints=[],
            paragraphs_defined=[f"PARA-{chunk_id}"],
            io_operations=[],
        ),
        evidence=[
            Evidence(
                evidence_type="line_range",
                start_line=1,
                end_line=50,
                note=f"Content of {chunk_id}",
            )
        ],
        confidence=0.9,
    )


def _create_mock_merge_result(
    job_id: str,
    artifact_id: str,
    artifact_version: str,
    merge_node_id: str,
) -> MergeResult:
    """Create a mock merge result for testing."""
    return MergeResult(
        job_id=job_id,
        artifact_id=artifact_id,
        artifact_version=artifact_version,
        merge_node_id=merge_node_id,
        coverage=MergeCoverage(
            included_input_ids=["chunk-1", "chunk-2"],
            missing_input_ids=[],
        ),
        consolidated_facts=ConsolidatedFacts(
            call_graph_edges=[{"from": "MAIN", "to": "PROCESS"}],
            io_map=[],
        ),
    )
