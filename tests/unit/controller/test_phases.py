"""Unit tests for Controller workflow phases C, D, E, and F.

Tests the phase-specific logic in the ReconcileController:
- Phase C: Hierarchical merge coordination
- Phase D: Challenger review handling
- Phase E: Follow-up dispatch and patch merge
- Phase F: Re-challenge loop
"""

import pytest

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit
from unittest.mock import AsyncMock, patch
import json

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
    DocPatchMergePayload,
    DocFinalizePayload,
    ChunkLocator,
)
from atlas.models.results import (
    ChallengeResult,
    Issue,
    ResolutionPlan,
)


@pytest.fixture
def artifact_ref() -> ArtifactRef:
    """Create sample artifact reference."""
    return ArtifactRef(
        artifact_id="TESTPROG.cbl",
        artifact_type="cobol",
        artifact_version="version123",
        artifact_uri="s3://bucket/sources/TESTPROG.cbl",
    )


@pytest.fixture
def hierarchical_manifest(artifact_ref: ArtifactRef) -> Manifest:
    """Create manifest with hierarchical merge structure."""
    chunks = [
        ChunkSpec(
            chunk_id="chunk_1",
            chunk_kind=ChunkKind.PROCEDURE_PART,
            start_line=1,
            end_line=100,
            division="PROCEDURE",
        ),
        ChunkSpec(
            chunk_id="chunk_2",
            chunk_kind=ChunkKind.PROCEDURE_PART,
            start_line=101,
            end_line=200,
            division="PROCEDURE",
        ),
        ChunkSpec(
            chunk_id="chunk_3",
            chunk_kind=ChunkKind.DATA_DIVISION,
            start_line=201,
            end_line=300,
            division="DATA",
        ),
        ChunkSpec(
            chunk_id="chunk_4",
            chunk_kind=ChunkKind.PROCEDURE_PART,
            start_line=301,
            end_line=400,
            division="PROCEDURE",
        ),
    ]

    merge_dag = [
        # Level 0: Leaf merges
        MergeNode(
            merge_node_id="merge_level0_proc",
            input_ids=["chunk_1", "chunk_2"],
            level=0,
        ),
        MergeNode(
            merge_node_id="merge_level0_data",
            input_ids=["chunk_3"],
            level=0,
        ),
        # Level 1: Intermediate merge
        MergeNode(
            merge_node_id="merge_level1",
            input_ids=["merge_level0_proc", "chunk_4"],
            level=1,
        ),
        # Level 2: Root merge
        MergeNode(
            merge_node_id="merge_root",
            input_ids=["merge_level1", "merge_level0_data"],
            is_root=True,
            level=2,
        ),
    ]

    return Manifest(
        job_id="test-phases-job",
        artifact_ref=artifact_ref,
        analysis_profile=AnalysisProfile(name="test"),
        splitter_profile=SplitterProfile(name="test"),
        context_budget=4000,
        chunks=chunks,
        merge_dag=merge_dag,
        review_policy=ReviewPolicy(
            challenge_profile="standard",
            max_iterations=3,
        ),
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


class TestPhaseCHierarchicalMerge:
    """Tests for Phase C: Hierarchical merge coordination."""

    @pytest.mark.asyncio
    async def test_advance_merges_bottom_up(
        self,
        controller: ReconcileController,
        hierarchical_manifest: Manifest,
    ) -> None:
        """Merges are advanced bottom-up by level."""
        await controller.initialize_job(hierarchical_manifest)

        # Mark all chunks as DONE
        items = controller.ticket_system._items
        for item in items.values():
            if item.work_type == WorkItemType.DOC_CHUNK:
                item.status = WorkItemStatus.DONE

        # Get work items grouped
        work_items = list(items.values())
        work_items_by_type = controller._group_by_type(work_items)
        work_items_by_id = {item.work_id: item for item in work_items}

        # Advance merges
        advanced = await controller._advance_merge_items(
            hierarchical_manifest.job_id,
            hierarchical_manifest,
            work_items_by_type,
            work_items_by_id,
        )

        # Level 0 merges should be advanced (chunk deps done)
        assert advanced >= 2  # merge_level0_proc and merge_level0_data

        # Verify level 0 merges are READY
        for item in items.values():
            if isinstance(item.payload, DocMergePayload):
                if item.payload.merge_node_id in ["merge_level0_proc", "merge_level0_data"]:
                    assert item.status == WorkItemStatus.READY

    @pytest.mark.asyncio
    async def test_parent_merge_waits_for_children(
        self,
        controller: ReconcileController,
        hierarchical_manifest: Manifest,
    ) -> None:
        """Parent merges remain BLOCKED until child merges complete."""
        await controller.initialize_job(hierarchical_manifest)

        # Mark only chunks as DONE (not merges)
        items = controller.ticket_system._items
        for item in items.values():
            if item.work_type == WorkItemType.DOC_CHUNK:
                item.status = WorkItemStatus.DONE

        # Get work items
        work_items = list(items.values())
        work_items_by_type = controller._group_by_type(work_items)
        work_items_by_id = {item.work_id: item for item in work_items}

        # Advance first round
        await controller._advance_merge_items(
            hierarchical_manifest.job_id,
            hierarchical_manifest,
            work_items_by_type,
            work_items_by_id,
        )

        # Level 1 merge should still be BLOCKED (depends on merge_level0_proc)
        for item in items.values():
            if isinstance(item.payload, DocMergePayload):
                if item.payload.merge_node_id == "merge_level1":
                    assert item.status == WorkItemStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_root_merge_advances_last(
        self,
        controller: ReconcileController,
        hierarchical_manifest: Manifest,
    ) -> None:
        """Root merge advances only when all intermediate merges complete."""
        await controller.initialize_job(hierarchical_manifest)

        items = controller.ticket_system._items

        # Complete all chunks
        for item in items.values():
            if item.work_type == WorkItemType.DOC_CHUNK:
                item.status = WorkItemStatus.DONE

        # Complete level 0 merges
        for item in items.values():
            if isinstance(item.payload, DocMergePayload):
                if "level0" in item.payload.merge_node_id:
                    item.status = WorkItemStatus.DONE

        # Complete level 1 merge
        for item in items.values():
            if isinstance(item.payload, DocMergePayload):
                if item.payload.merge_node_id == "merge_level1":
                    item.status = WorkItemStatus.DONE

        # Advance merges
        work_items = list(items.values())
        work_items_by_type = controller._group_by_type(work_items)
        work_items_by_id = {item.work_id: item for item in work_items}

        advanced = await controller._advance_merge_items(
            hierarchical_manifest.job_id,
            hierarchical_manifest,
            work_items_by_type,
            work_items_by_id,
        )

        # Root merge should now be READY
        root_item = next(
            (
                item for item in items.values()
                if isinstance(item.payload, DocMergePayload)
                and item.payload.merge_node_id == "merge_root"
            ),
            None,
        )
        assert root_item is not None
        assert root_item.status == WorkItemStatus.READY


class TestPhaseDChallengerReview:
    """Tests for Phase D: Challenger review handling."""

    @pytest.mark.asyncio
    async def test_create_challenge_after_root_merge(
        self,
        controller: ReconcileController,
        hierarchical_manifest: Manifest,
    ) -> None:
        """Challenge work item created after root merge completes."""
        await controller.initialize_job(hierarchical_manifest)

        items = controller.ticket_system._items

        # Complete all chunks and merges
        for item in items.values():
            item.status = WorkItemStatus.DONE

        work_items = list(items.values())
        work_items_by_type = controller._group_by_type(work_items)

        # Execute phase D
        created = await controller._execute_phase_d(
            hierarchical_manifest.job_id,
            hierarchical_manifest,
            work_items_by_type,
        )

        assert created == 1

        # Verify challenge was created
        challenges = [
            item for item in items.values()
            if item.work_type == WorkItemType.DOC_CHALLENGE
        ]
        assert len(challenges) == 1
        assert challenges[0].cycle_number == 1
        assert challenges[0].status == WorkItemStatus.READY

    @pytest.mark.asyncio
    async def test_challenge_uses_updated_doc_for_cycle_2(
        self,
        controller: ReconcileController,
        hierarchical_manifest: Manifest,
        mock_artifact_store,
    ) -> None:
        """Cycle 2+ challenges use updated doc from patch merge."""
        await controller.initialize_job(hierarchical_manifest)

        items = controller.ticket_system._items

        # Complete initial workflow
        for item in items.values():
            item.status = WorkItemStatus.DONE

        # Create cycle 1 challenge (completed)
        challenge1 = WorkItem(
            work_id="test-phases-job-challenge-cycle1",
            work_type=WorkItemType.DOC_CHALLENGE,
            status=WorkItemStatus.DONE,
            payload=DocChallengePayload(
                job_id="test-phases-job",
                doc_uri="s3://bucket/results/doc/documentation.md",
                doc_model_uri="s3://bucket/results/doc/doc_model.json",
                challenge_profile="standard",
                output_uri="s3://bucket/results/challenges/challenge_cycle1.json",
            ),
            cycle_number=1,
        )
        items[challenge1.work_id] = challenge1

        # Create patch merge (completed)
        patch1 = WorkItem(
            work_id="test-phases-job-patch-merge-cycle2",
            work_type=WorkItemType.DOC_PATCH_MERGE,
            status=WorkItemStatus.DONE,
            payload=DocPatchMergePayload(
                job_id="test-phases-job",
                base_doc_uri="s3://bucket/results/doc/documentation.md",
                base_doc_model_uri="s3://bucket/results/doc/doc_model.json",
                inputs=["s3://bucket/results/followups/answer.json"],
                output_doc_uri="s3://bucket/results/cycle2/doc/documentation.md",
                output_doc_model_uri="s3://bucket/results/cycle2/doc/doc_model.json",
            ),
            cycle_number=2,
        )
        items[patch1.work_id] = patch1

        work_items = list(items.values())
        work_items_by_type = controller._group_by_type(work_items)

        # Execute phase D for cycle 2
        created = await controller._execute_phase_d(
            hierarchical_manifest.job_id,
            hierarchical_manifest,
            work_items_by_type,
        )

        assert created == 1

        # Find the new challenge
        cycle2_challenges = [
            item for item in items.values()
            if item.work_type == WorkItemType.DOC_CHALLENGE
            and item.cycle_number == 2
        ]
        assert len(cycle2_challenges) == 1

        # Verify it uses cycle2 doc paths
        payload = cycle2_challenges[0].payload
        assert isinstance(payload, DocChallengePayload)
        assert "cycle2" in payload.doc_uri

    @pytest.mark.asyncio
    async def test_handle_challenge_result_with_issues(
        self,
        controller: ReconcileController,
        hierarchical_manifest: Manifest,
        mock_artifact_store,
    ) -> None:
        """Challenge result with BLOCKER/MAJOR issues triggers follow-ups."""
        # Store challenge result
        challenge_result = ChallengeResult(
            job_id="test-phases-job",
            artifact_id="TESTPROG.cbl",
            artifact_version="version123",
            issues=[
                Issue(
                    issue_id="issue-1",
                    severity=IssueSeverity.BLOCKER,
                    question="Critical issue",
                ),
                Issue(
                    issue_id="issue-2",
                    severity=IssueSeverity.MINOR,
                    question="Minor issue",
                ),
            ],
            resolution_plan=ResolutionPlan(requires_patch_merge=True),
        )

        output_uri = "s3://bucket/results/challenges/challenge_cycle1.json"
        await mock_artifact_store.write_json(
            output_uri, challenge_result.model_dump()
        )

        # Create challenge work item
        challenge_item = WorkItem(
            work_id="test-challenge",
            work_type=WorkItemType.DOC_CHALLENGE,
            status=WorkItemStatus.DONE,
            payload=DocChallengePayload(
                job_id="test-phases-job",
                doc_uri="s3://bucket/results/doc/documentation.md",
                doc_model_uri="s3://bucket/results/doc/doc_model.json",
                challenge_profile="standard",
                output_uri=output_uri,
            ),
            cycle_number=1,
        )

        needs_followup, count = await controller._handle_challenge_result(
            "test-phases-job", challenge_item, hierarchical_manifest
        )

        assert needs_followup is True
        assert count == 1  # Only BLOCKER counts


class TestPhaseEFollowupDispatch:
    """Tests for Phase E: Follow-up dispatch and patch merge."""

    @pytest.mark.asyncio
    async def test_dispatch_followups_for_blocker_issues(
        self,
        controller: ReconcileController,
        hierarchical_manifest: Manifest,
        mock_artifact_store,
    ) -> None:
        """Follow-ups created for BLOCKER and MAJOR issues only."""
        await controller.initialize_job(hierarchical_manifest)

        items = controller.ticket_system._items

        # Complete chunks and merges
        for item in items.values():
            item.status = WorkItemStatus.DONE

        # Create completed challenge
        challenge_result = ChallengeResult(
            job_id="test-phases-job",
            artifact_id="TESTPROG.cbl",
            artifact_version="version123",
            issues=[
                Issue(
                    issue_id="blocker-1",
                    severity=IssueSeverity.BLOCKER,
                    question="Blocker issue",
                    suspected_scopes=["chunk_1"],
                ),
                Issue(
                    issue_id="major-1",
                    severity=IssueSeverity.MAJOR,
                    question="Major issue",
                    suspected_scopes=["chunk_2"],
                ),
                Issue(
                    issue_id="minor-1",
                    severity=IssueSeverity.MINOR,
                    question="Minor issue - should skip",
                ),
            ],
            resolution_plan=ResolutionPlan(requires_patch_merge=True),
        )

        output_uri = "s3://bucket/results/challenges/challenge_cycle1.json"
        await mock_artifact_store.write_json(
            output_uri, challenge_result.model_dump()
        )

        challenge = WorkItem(
            work_id="test-phases-job-challenge-cycle1",
            work_type=WorkItemType.DOC_CHALLENGE,
            status=WorkItemStatus.DONE,
            payload=DocChallengePayload(
                job_id="test-phases-job",
                doc_uri="s3://bucket/results/doc/documentation.md",
                doc_model_uri="s3://bucket/results/doc/doc_model.json",
                challenge_profile="standard",
                output_uri=output_uri,
            ),
            cycle_number=1,
        )
        items[challenge.work_id] = challenge

        work_items = list(items.values())
        work_items_by_type = controller._group_by_type(work_items)

        # Execute phase E
        created = await controller._execute_phase_e(
            hierarchical_manifest.job_id,
            hierarchical_manifest,
            work_items_by_type,
        )

        # Should create 2 follow-ups (BLOCKER + MAJOR)
        assert created == 2

    @pytest.mark.asyncio
    async def test_create_patch_merge_when_followups_done(
        self,
        controller: ReconcileController,
        hierarchical_manifest: Manifest,
    ) -> None:
        """Patch merge created when all follow-ups complete."""
        await controller.initialize_job(hierarchical_manifest)

        items = controller.ticket_system._items

        # Complete everything
        for item in items.values():
            item.status = WorkItemStatus.DONE

        # Create completed follow-ups
        followup1 = WorkItem(
            work_id="test-phases-job-followup-1",
            work_type=WorkItemType.DOC_FOLLOWUP,
            status=WorkItemStatus.DONE,
            payload=DocFollowupPayload(
                job_id="test-phases-job",
                issue_id="issue-1",
                scope={"chunk_ids": ["chunk_1"]},
                inputs=["s3://bucket/inputs/1.json"],
                output_uri="s3://bucket/followups/1.json",
            ),
            cycle_number=1,
        )
        items[followup1.work_id] = followup1

        followup2 = WorkItem(
            work_id="test-phases-job-followup-2",
            work_type=WorkItemType.DOC_FOLLOWUP,
            status=WorkItemStatus.DONE,
            payload=DocFollowupPayload(
                job_id="test-phases-job",
                issue_id="issue-2",
                scope={"chunk_ids": ["chunk_2"]},
                inputs=["s3://bucket/inputs/2.json"],
                output_uri="s3://bucket/followups/2.json",
            ),
            cycle_number=1,
        )
        items[followup2.work_id] = followup2

        work_items = list(items.values())
        work_items_by_type = controller._group_by_type(work_items)

        # Execute patch merge creation
        created = await controller._execute_patch_merge(
            hierarchical_manifest.job_id,
            hierarchical_manifest,
            work_items_by_type,
        )

        assert created == 1

        # Verify patch merge was created
        patch_merges = [
            item for item in items.values()
            if item.work_type == WorkItemType.DOC_PATCH_MERGE
        ]
        assert len(patch_merges) == 1
        assert patch_merges[0].cycle_number == 2  # Next cycle


class TestPhaseFRechallengeLoop:
    """Tests for Phase F: Re-challenge loop."""

    @pytest.mark.asyncio
    async def test_rechallenge_after_patch_merge(
        self,
        controller: ReconcileController,
        hierarchical_manifest: Manifest,
        mock_artifact_store,
    ) -> None:
        """Re-challenge created after patch merge completes."""
        await controller.initialize_job(hierarchical_manifest)

        items = controller.ticket_system._items

        # Complete initial workflow
        for item in items.values():
            item.status = WorkItemStatus.DONE

        # Create cycle 1 challenge with issues
        challenge_result = ChallengeResult(
            job_id="test-phases-job",
            artifact_id="TESTPROG.cbl",
            artifact_version="version123",
            issues=[
                Issue(
                    issue_id="issue-1",
                    severity=IssueSeverity.BLOCKER,
                    question="Issue",
                ),
            ],
            resolution_plan=ResolutionPlan(requires_patch_merge=True),
        )

        output_uri = "s3://bucket/results/challenges/challenge_cycle1.json"
        await mock_artifact_store.write_json(
            output_uri, challenge_result.model_dump()
        )

        challenge1 = WorkItem(
            work_id="test-phases-job-challenge-cycle1",
            work_type=WorkItemType.DOC_CHALLENGE,
            status=WorkItemStatus.DONE,
            payload=DocChallengePayload(
                job_id="test-phases-job",
                doc_uri="s3://bucket/results/doc/documentation.md",
                doc_model_uri="s3://bucket/results/doc/doc_model.json",
                challenge_profile="standard",
                output_uri=output_uri,
            ),
            cycle_number=1,
        )
        items[challenge1.work_id] = challenge1

        # Create completed patch merge
        patch1 = WorkItem(
            work_id="test-phases-job-patch-merge-cycle2",
            work_type=WorkItemType.DOC_PATCH_MERGE,
            status=WorkItemStatus.DONE,
            payload=DocPatchMergePayload(
                job_id="test-phases-job",
                base_doc_uri="s3://bucket/results/doc/documentation.md",
                base_doc_model_uri="s3://bucket/results/doc/doc_model.json",
                inputs=["s3://bucket/followups/1.json"],
                output_doc_uri="s3://bucket/results/cycle2/doc/documentation.md",
                output_doc_model_uri="s3://bucket/results/cycle2/doc/doc_model.json",
            ),
            cycle_number=2,
        )
        items[patch1.work_id] = patch1

        work_items = list(items.values())
        work_items_by_type = controller._group_by_type(work_items)

        # Execute phase F
        created = await controller._execute_phase_f(
            hierarchical_manifest.job_id,
            hierarchical_manifest,
            work_items_by_type,
        )

        assert created == 1

        # Verify re-challenge was created
        challenges = [
            item for item in items.values()
            if item.work_type == WorkItemType.DOC_CHALLENGE
        ]
        assert len(challenges) == 2
        cycle2_challenge = [c for c in challenges if c.cycle_number == 2]
        assert len(cycle2_challenge) == 1

    @pytest.mark.asyncio
    async def test_max_iterations_stops_rechallenge(
        self,
        controller: ReconcileController,
        hierarchical_manifest: Manifest,
        mock_artifact_store,
    ) -> None:
        """Re-challenge stops when max iterations reached."""
        # Set max iterations to 2
        controller.config.max_challenge_iterations = 2

        await controller.initialize_job(hierarchical_manifest)

        items = controller.ticket_system._items

        # Complete initial workflow
        for item in items.values():
            item.status = WorkItemStatus.DONE

        # Create challenges and patches up to limit
        for cycle in range(1, 3):
            challenge_result = ChallengeResult(
                job_id="test-phases-job",
                artifact_id="TESTPROG.cbl",
                artifact_version="version123",
                issues=[
                    Issue(
                        issue_id=f"issue-{cycle}",
                        severity=IssueSeverity.BLOCKER,
                        question=f"Issue cycle {cycle}",
                    ),
                ],
            )

            output_uri = f"s3://bucket/results/challenges/challenge_cycle{cycle}.json"
            await mock_artifact_store.write_json(
                output_uri, challenge_result.model_dump()
            )

            challenge = WorkItem(
                work_id=f"test-phases-job-challenge-cycle{cycle}",
                work_type=WorkItemType.DOC_CHALLENGE,
                status=WorkItemStatus.DONE,
                payload=DocChallengePayload(
                    job_id="test-phases-job",
                    doc_uri=f"s3://bucket/results/cycle{cycle}/doc/documentation.md",
                    doc_model_uri=f"s3://bucket/results/cycle{cycle}/doc/doc_model.json",
                    challenge_profile="standard",
                    output_uri=output_uri,
                ),
                cycle_number=cycle,
            )
            items[challenge.work_id] = challenge

            if cycle < 2:
                patch = WorkItem(
                    work_id=f"test-phases-job-patch-merge-cycle{cycle + 1}",
                    work_type=WorkItemType.DOC_PATCH_MERGE,
                    status=WorkItemStatus.DONE,
                    payload=DocPatchMergePayload(
                        job_id="test-phases-job",
                        base_doc_uri=f"s3://bucket/results/cycle{cycle}/doc/documentation.md",
                        base_doc_model_uri=f"s3://bucket/results/cycle{cycle}/doc/doc_model.json",
                        inputs=[f"s3://bucket/followups/{cycle}.json"],
                        output_doc_uri=f"s3://bucket/results/cycle{cycle + 1}/doc/documentation.md",
                        output_doc_model_uri=f"s3://bucket/results/cycle{cycle + 1}/doc/doc_model.json",
                    ),
                    cycle_number=cycle + 1,
                )
                items[patch.work_id] = patch

        # Create final patch that would trigger cycle 3 rechallenge
        final_patch = WorkItem(
            work_id="test-phases-job-patch-merge-cycle3",
            work_type=WorkItemType.DOC_PATCH_MERGE,
            status=WorkItemStatus.DONE,
            payload=DocPatchMergePayload(
                job_id="test-phases-job",
                base_doc_uri="s3://bucket/results/cycle2/doc/documentation.md",
                base_doc_model_uri="s3://bucket/results/cycle2/doc/doc_model.json",
                inputs=["s3://bucket/followups/2.json"],
                output_doc_uri="s3://bucket/results/cycle3/doc/documentation.md",
                output_doc_model_uri="s3://bucket/results/cycle3/doc/doc_model.json",
            ),
            cycle_number=3,
        )
        items[final_patch.work_id] = final_patch

        work_items = list(items.values())
        work_items_by_type = controller._group_by_type(work_items)

        # Execute phase F - should NOT create re-challenge (max reached)
        created = await controller._execute_phase_f(
            hierarchical_manifest.job_id,
            hierarchical_manifest,
            work_items_by_type,
        )

        # Should not create new challenge (max iterations = 2)
        assert created == 0


class TestPhaseDetermination:
    """Tests for phase determination logic."""

    def test_determine_phase_plan(
        self,
        controller: ReconcileController,
        hierarchical_manifest: Manifest,
    ) -> None:
        """Phase is 'plan' when no work items exist."""
        phase = controller._determine_phase(hierarchical_manifest, {})
        assert phase == "plan"

    def test_determine_phase_chunk(
        self,
        controller: ReconcileController,
        hierarchical_manifest: Manifest,
    ) -> None:
        """Phase is 'chunk' when chunks are not all done."""
        chunks = [
            WorkItem(
                work_id="c1",
                work_type=WorkItemType.DOC_CHUNK,
                status=WorkItemStatus.READY,
                payload=DocChunkPayload(
                    job_id="test",
                    chunk_id="chunk_1",
                    chunk_locator=ChunkLocator(start_line=1, end_line=100),
                    result_uri="s3://test/c1.json",
                ),
            ),
        ]
        work_items_by_type = {WorkItemType.DOC_CHUNK: chunks}

        phase = controller._determine_phase(hierarchical_manifest, work_items_by_type)
        assert phase == "chunk"

    def test_determine_phase_merge(
        self,
        controller: ReconcileController,
        hierarchical_manifest: Manifest,
    ) -> None:
        """Phase is 'merge' when chunks done but merges not."""
        chunks = [
            WorkItem(
                work_id="c1",
                work_type=WorkItemType.DOC_CHUNK,
                status=WorkItemStatus.DONE,
                payload=DocChunkPayload(
                    job_id="test",
                    chunk_id="chunk_1",
                    chunk_locator=ChunkLocator(start_line=1, end_line=100),
                    result_uri="s3://test/c1.json",
                ),
            ),
        ]
        merges = [
            WorkItem(
                work_id="m1",
                work_type=WorkItemType.DOC_MERGE,
                status=WorkItemStatus.BLOCKED,
                payload=DocMergePayload(
                    job_id="test",
                    merge_node_id="merge_1",
                    input_uris=["s3://test/c1.json"],
                    output_uri="s3://test/m1.json",
                ),
            ),
        ]
        work_items_by_type = {
            WorkItemType.DOC_CHUNK: chunks,
            WorkItemType.DOC_MERGE: merges,
        }

        phase = controller._determine_phase(hierarchical_manifest, work_items_by_type)
        assert phase == "merge"

    def test_determine_phase_challenge(
        self,
        controller: ReconcileController,
        hierarchical_manifest: Manifest,
    ) -> None:
        """Phase is 'challenge' when merges done but no challenge exists."""
        chunks = [
            WorkItem(
                work_id="c1",
                work_type=WorkItemType.DOC_CHUNK,
                status=WorkItemStatus.DONE,
                payload=DocChunkPayload(
                    job_id="test",
                    chunk_id="chunk_1",
                    chunk_locator=ChunkLocator(start_line=1, end_line=100),
                    result_uri="s3://test/c1.json",
                ),
            ),
        ]
        merges = [
            WorkItem(
                work_id="m1",
                work_type=WorkItemType.DOC_MERGE,
                status=WorkItemStatus.DONE,
                payload=DocMergePayload(
                    job_id="test",
                    merge_node_id="merge_1",
                    input_uris=["s3://test/c1.json"],
                    output_uri="s3://test/m1.json",
                ),
            ),
        ]
        work_items_by_type = {
            WorkItemType.DOC_CHUNK: chunks,
            WorkItemType.DOC_MERGE: merges,
        }

        phase = controller._determine_phase(hierarchical_manifest, work_items_by_type)
        assert phase == "challenge"

    def test_determine_phase_finalize(
        self,
        controller: ReconcileController,
        hierarchical_manifest: Manifest,
    ) -> None:
        """Phase is 'finalize' when challenge done with no follow-ups (before finalize exists)."""
        chunks = [
            WorkItem(
                work_id="c1",
                work_type=WorkItemType.DOC_CHUNK,
                status=WorkItemStatus.DONE,
                payload=DocChunkPayload(
                    job_id="test",
                    chunk_id="chunk_1",
                    chunk_locator=ChunkLocator(start_line=1, end_line=100),
                    result_uri="s3://test/c1.json",
                ),
            ),
        ]
        merges = [
            WorkItem(
                work_id="m1",
                work_type=WorkItemType.DOC_MERGE,
                status=WorkItemStatus.DONE,
                payload=DocMergePayload(
                    job_id="test",
                    merge_node_id="merge_1",
                    input_uris=["s3://test/c1.json"],
                    output_uri="s3://test/m1.json",
                ),
            ),
        ]
        challenges = [
            WorkItem(
                work_id="ch1",
                work_type=WorkItemType.DOC_CHALLENGE,
                status=WorkItemStatus.DONE,
                payload=DocChallengePayload(
                    job_id="test",
                    doc_uri="s3://test/doc.md",
                    doc_model_uri="s3://test/model.json",
                    challenge_profile="standard",
                    output_uri="s3://test/challenge.json",
                ),
                cycle_number=1,
            ),
        ]
        work_items_by_type = {
            WorkItemType.DOC_CHUNK: chunks,
            WorkItemType.DOC_MERGE: merges,
            WorkItemType.DOC_CHALLENGE: challenges,
        }

        phase = controller._determine_phase(hierarchical_manifest, work_items_by_type)
        assert phase == "finalize"

    def test_determine_phase_complete(
        self,
        controller: ReconcileController,
        hierarchical_manifest: Manifest,
    ) -> None:
        """Phase is 'complete' when finalize is done."""
        chunks = [
            WorkItem(
                work_id="c1",
                work_type=WorkItemType.DOC_CHUNK,
                status=WorkItemStatus.DONE,
                payload=DocChunkPayload(
                    job_id="test",
                    chunk_id="chunk_1",
                    chunk_locator=ChunkLocator(start_line=1, end_line=100),
                    result_uri="s3://test/c1.json",
                ),
            ),
        ]
        merges = [
            WorkItem(
                work_id="m1",
                work_type=WorkItemType.DOC_MERGE,
                status=WorkItemStatus.DONE,
                payload=DocMergePayload(
                    job_id="test",
                    merge_node_id="merge_1",
                    input_uris=["s3://test/c1.json"],
                    output_uri="s3://test/m1.json",
                ),
            ),
        ]
        challenges = [
            WorkItem(
                work_id="ch1",
                work_type=WorkItemType.DOC_CHALLENGE,
                status=WorkItemStatus.DONE,
                payload=DocChallengePayload(
                    job_id="test",
                    doc_uri="s3://test/doc.md",
                    doc_model_uri="s3://test/model.json",
                    challenge_profile="standard",
                    output_uri="s3://test/challenge.json",
                ),
                cycle_number=1,
            ),
        ]
        finalizes = [
            WorkItem(
                work_id="fin1",
                work_type=WorkItemType.DOC_FINALIZE,
                status=WorkItemStatus.DONE,
                payload=DocFinalizePayload(
                    job_id="test",
                    doc_uri="s3://test/doc.md",
                    doc_model_uri="s3://test/model.json",
                    output_doc_uri="s3://test/final/doc.md",
                    output_trace_uri="s3://test/final/trace.json",
                    output_summary_uri="s3://test/final/summary.json",
                ),
            ),
        ]
        work_items_by_type = {
            WorkItemType.DOC_CHUNK: chunks,
            WorkItemType.DOC_MERGE: merges,
            WorkItemType.DOC_CHALLENGE: challenges,
            WorkItemType.DOC_FINALIZE: finalizes,
        }

        phase = controller._determine_phase(hierarchical_manifest, work_items_by_type)
        assert phase == "complete"
