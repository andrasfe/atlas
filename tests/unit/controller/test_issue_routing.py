"""Unit tests for the challenger issue routing algorithm.

Tests the routing algorithm defined in spec section 9.2:
1. If suspected_scopes contains chunk IDs -> use those
2. Else if issue references doc sections with source refs -> use those
3. Else if issue references symbols/paragraphs -> consult indexes
4. Else create bounded cross-cutting follow-up plan

Also tests scope size constraints from spec section 9.3:
- Follow-ups MUST target bounded scopes (1 chunk, small list max 3-5, or merge node)
- Cross-cutting issues MUST be split into bounded follow-ups
"""

import pytest

from atlas.controller.reconciler import ReconcileController
from atlas.controller.base import ControllerConfig
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
from atlas.models.results import (
    Issue,
    DocumentationModel,
    DocIndex,
    Section,
)


@pytest.fixture
def artifact_ref() -> ArtifactRef:
    """Create sample artifact reference."""
    return ArtifactRef(
        artifact_id="TESTPROG.cbl",
        artifact_type="cobol",
        artifact_version="hash123",
        artifact_uri="s3://bucket/sources/TESTPROG.cbl",
    )


@pytest.fixture
def manifest_with_chunks(artifact_ref: ArtifactRef) -> Manifest:
    """Create manifest with multiple chunks across divisions."""
    chunks = [
        # DATA DIVISION chunks
        ChunkSpec(
            chunk_id="data_working_storage",
            chunk_kind=ChunkKind.WORKING_STORAGE,
            start_line=50,
            end_line=150,
            division="DATA",
            section="WORKING-STORAGE",
            paragraphs=[],
        ),
        ChunkSpec(
            chunk_id="data_file_section",
            chunk_kind=ChunkKind.FILE_SECTION,
            start_line=10,
            end_line=49,
            division="DATA",
            section="FILE",
            paragraphs=[],
        ),
        # PROCEDURE DIVISION chunks
        ChunkSpec(
            chunk_id="proc_part_1",
            chunk_kind=ChunkKind.PROCEDURE_PART,
            start_line=200,
            end_line=350,
            division="PROCEDURE",
            paragraphs=["MAIN-LOGIC", "INIT-ROUTINE"],
        ),
        ChunkSpec(
            chunk_id="proc_part_2",
            chunk_kind=ChunkKind.PROCEDURE_PART,
            start_line=351,
            end_line=500,
            division="PROCEDURE",
            paragraphs=["PROCESS-RECORD", "VALIDATE-INPUT"],
        ),
        ChunkSpec(
            chunk_id="proc_part_3",
            chunk_kind=ChunkKind.PROCEDURE_PART,
            start_line=501,
            end_line=650,
            division="PROCEDURE",
            paragraphs=["WRITE-OUTPUT", "CLEANUP"],
        ),
        ChunkSpec(
            chunk_id="proc_part_4",
            chunk_kind=ChunkKind.PROCEDURE_PART,
            start_line=651,
            end_line=800,
            division="PROCEDURE",
            paragraphs=["ERROR-HANDLER", "ABEND-ROUTINE"],
        ),
    ]

    merge_dag = [
        MergeNode(
            merge_node_id="merge_data",
            input_ids=["data_working_storage", "data_file_section"],
            level=0,
        ),
        MergeNode(
            merge_node_id="merge_proc_12",
            input_ids=["proc_part_1", "proc_part_2"],
            level=0,
        ),
        MergeNode(
            merge_node_id="merge_proc_34",
            input_ids=["proc_part_3", "proc_part_4"],
            level=0,
        ),
        MergeNode(
            merge_node_id="merge_procedure",
            input_ids=["merge_proc_12", "merge_proc_34"],
            level=1,
        ),
        MergeNode(
            merge_node_id="merge_root",
            input_ids=["merge_data", "merge_procedure"],
            is_root=True,
            level=2,
        ),
    ]

    return Manifest(
        job_id="test-routing-job",
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
def doc_model_with_index() -> DocumentationModel:
    """Create documentation model with routing indexes."""
    return DocumentationModel(
        doc_uri="s3://bucket/results/doc/documentation.md",
        sections=[
            Section(
                section_id="sec_overview",
                title="Program Overview",
                source_refs=["proc_part_1"],
            ),
            Section(
                section_id="sec_data_structures",
                title="Data Structures",
                source_refs=["data_working_storage", "data_file_section"],
            ),
            Section(
                section_id="sec_error_handling",
                title="Error Handling",
                source_refs=["proc_part_4"],
            ),
            Section(
                section_id="sec_io_operations",
                title="I/O Operations",
                source_refs=["proc_part_2", "proc_part_3"],
            ),
        ],
        index=DocIndex(
            symbol_to_chunks={
                "WS-STATUS-CODE": ["data_working_storage"],
                "FILE-RECORD": ["data_file_section"],
                "ERROR-FLAG": ["data_working_storage", "proc_part_4"],
                "INPUT-BUFFER": ["data_working_storage", "proc_part_2"],
            },
            paragraph_to_chunk={
                "MAIN-LOGIC": "proc_part_1",
                "INIT-ROUTINE": "proc_part_1",
                "PROCESS-RECORD": "proc_part_2",
                "VALIDATE-INPUT": "proc_part_2",
                "WRITE-OUTPUT": "proc_part_3",
                "CLEANUP": "proc_part_3",
                "ERROR-HANDLER": "proc_part_4",
                "ABEND-ROUTINE": "proc_part_4",
            },
            file_to_chunks={
                "TESTPROG.cbl": [
                    "data_working_storage",
                    "data_file_section",
                    "proc_part_1",
                    "proc_part_2",
                    "proc_part_3",
                    "proc_part_4",
                ],
            },
        ),
    )


@pytest.fixture
def controller(mock_ticket_system, mock_artifact_store) -> ReconcileController:
    """Create controller with mock adapters."""
    return ReconcileController(
        ticket_system=mock_ticket_system,
        artifact_store=mock_artifact_store,
        config=ControllerConfig(max_challenge_iterations=3),
    )


class TestIssueRoutingPriority1SuspectedScopes:
    """Tests for Priority 1: suspected_scopes with chunk IDs."""

    def test_route_with_valid_chunk_ids(
        self,
        controller: ReconcileController,
        manifest_with_chunks: Manifest,
    ) -> None:
        """Issue with valid chunk IDs in suspected_scopes routes directly."""
        issue = Issue(
            issue_id="issue-001",
            severity=IssueSeverity.MAJOR,
            question="What happens in the error handler?",
            suspected_scopes=["proc_part_4"],
        )

        scopes = controller._route_issue_to_scopes(
            issue, manifest_with_chunks, None
        )

        assert len(scopes) == 1
        assert scopes[0]["chunk_ids"] == ["proc_part_4"]
        assert scopes[0]["routing_method"] == "suspected_scopes"

    def test_route_with_multiple_valid_chunk_ids(
        self,
        controller: ReconcileController,
        manifest_with_chunks: Manifest,
    ) -> None:
        """Issue with multiple valid chunk IDs creates single scope."""
        issue = Issue(
            issue_id="issue-002",
            severity=IssueSeverity.BLOCKER,
            question="How do these chunks interact?",
            suspected_scopes=["proc_part_1", "proc_part_2"],
        )

        scopes = controller._route_issue_to_scopes(
            issue, manifest_with_chunks, None
        )

        assert len(scopes) == 1
        assert set(scopes[0]["chunk_ids"]) == {"proc_part_1", "proc_part_2"}

    def test_route_with_invalid_chunk_ids_falls_through(
        self,
        controller: ReconcileController,
        manifest_with_chunks: Manifest,
    ) -> None:
        """Issue with invalid chunk IDs falls through to other priorities."""
        issue = Issue(
            issue_id="issue-003",
            severity=IssueSeverity.MAJOR,
            question="Unknown scope question",
            suspected_scopes=["nonexistent_chunk"],
        )

        scopes = controller._route_issue_to_scopes(
            issue, manifest_with_chunks, None
        )

        # Should fall through to cross-cutting (Priority 4)
        assert len(scopes) >= 1
        assert scopes[0]["type"] == "cross_cutting"

    def test_route_splits_large_chunk_list(
        self,
        controller: ReconcileController,
        manifest_with_chunks: Manifest,
    ) -> None:
        """Issue with >5 chunk IDs is split into multiple scopes."""
        issue = Issue(
            issue_id="issue-004",
            severity=IssueSeverity.MAJOR,
            question="Question spanning many chunks",
            suspected_scopes=[
                "data_working_storage",
                "data_file_section",
                "proc_part_1",
                "proc_part_2",
                "proc_part_3",
                "proc_part_4",
            ],
        )

        scopes = controller._route_issue_to_scopes(
            issue, manifest_with_chunks, None
        )

        # Should be split into at least 2 scopes (6 chunks, max 5 per scope)
        assert len(scopes) >= 2
        total_chunks = sum(len(s["chunk_ids"]) for s in scopes)
        assert total_chunks == 6


class TestIssueRoutingPriority2DocSectionRefs:
    """Tests for Priority 2: doc section source refs."""

    def test_route_via_doc_section_refs(
        self,
        controller: ReconcileController,
        manifest_with_chunks: Manifest,
        doc_model_with_index: DocumentationModel,
    ) -> None:
        """Issue referencing doc section uses section's source_refs."""
        issue = Issue(
            issue_id="issue-005",
            severity=IssueSeverity.MAJOR,
            question="Clarify the error handling section",
            doc_section_refs=["sec_error_handling"],
        )

        scopes = controller._route_issue_to_scopes(
            issue, manifest_with_chunks, doc_model_with_index
        )

        assert len(scopes) == 1
        assert scopes[0]["chunk_ids"] == ["proc_part_4"]
        assert scopes[0]["routing_method"] == "doc_section_refs"

    def test_route_via_multiple_doc_sections(
        self,
        controller: ReconcileController,
        manifest_with_chunks: Manifest,
        doc_model_with_index: DocumentationModel,
    ) -> None:
        """Issue referencing multiple doc sections aggregates source_refs."""
        issue = Issue(
            issue_id="issue-006",
            severity=IssueSeverity.MAJOR,
            question="How do data structures relate to I/O?",
            doc_section_refs=["sec_data_structures", "sec_io_operations"],
        )

        scopes = controller._route_issue_to_scopes(
            issue, manifest_with_chunks, doc_model_with_index
        )

        # Should combine source_refs from both sections
        all_chunks = set()
        for scope in scopes:
            all_chunks.update(scope["chunk_ids"])

        expected = {
            "data_working_storage",
            "data_file_section",
            "proc_part_2",
            "proc_part_3",
        }
        assert all_chunks == expected


class TestIssueRoutingPriority3RoutingHints:
    """Tests for Priority 3: routing hints via indexes."""

    def test_route_via_symbol_hint(
        self,
        controller: ReconcileController,
        manifest_with_chunks: Manifest,
        doc_model_with_index: DocumentationModel,
    ) -> None:
        """Issue with symbol routing hint uses symbol_to_chunks index."""
        issue = Issue(
            issue_id="issue-007",
            severity=IssueSeverity.MAJOR,
            question="What is WS-STATUS-CODE used for?",
            routing_hints={"symbols": ["WS-STATUS-CODE"]},
        )

        scopes = controller._route_issue_to_scopes(
            issue, manifest_with_chunks, doc_model_with_index
        )

        assert len(scopes) == 1
        assert scopes[0]["chunk_ids"] == ["data_working_storage"]
        assert scopes[0]["routing_method"] == "routing_hints"

    def test_route_via_paragraph_hint(
        self,
        controller: ReconcileController,
        manifest_with_chunks: Manifest,
        doc_model_with_index: DocumentationModel,
    ) -> None:
        """Issue with paragraph routing hint uses paragraph_to_chunk index."""
        issue = Issue(
            issue_id="issue-008",
            severity=IssueSeverity.MAJOR,
            question="Explain the ERROR-HANDLER paragraph",
            routing_hints={"paragraphs": ["ERROR-HANDLER"]},
        )

        scopes = controller._route_issue_to_scopes(
            issue, manifest_with_chunks, doc_model_with_index
        )

        assert len(scopes) == 1
        assert scopes[0]["chunk_ids"] == ["proc_part_4"]

    def test_route_via_multiple_hints(
        self,
        controller: ReconcileController,
        manifest_with_chunks: Manifest,
        doc_model_with_index: DocumentationModel,
    ) -> None:
        """Issue with multiple routing hints aggregates all matches."""
        issue = Issue(
            issue_id="issue-009",
            severity=IssueSeverity.MAJOR,
            question="How does ERROR-FLAG relate to ERROR-HANDLER?",
            routing_hints={
                "symbols": ["ERROR-FLAG"],
                "paragraphs": ["ERROR-HANDLER"],
            },
        )

        scopes = controller._route_issue_to_scopes(
            issue, manifest_with_chunks, doc_model_with_index
        )

        # ERROR-FLAG is in data_working_storage and proc_part_4
        # ERROR-HANDLER is in proc_part_4
        all_chunks = set()
        for scope in scopes:
            all_chunks.update(scope["chunk_ids"])

        assert "proc_part_4" in all_chunks
        assert "data_working_storage" in all_chunks

    def test_route_via_paragraph_fallback_to_manifest(
        self,
        controller: ReconcileController,
        manifest_with_chunks: Manifest,
    ) -> None:
        """Paragraph hint without doc model falls back to manifest search."""
        issue = Issue(
            issue_id="issue-010",
            severity=IssueSeverity.MAJOR,
            question="What does MAIN-LOGIC do?",
            routing_hints={"paragraphs": ["MAIN-LOGIC"]},
        )

        # No doc model provided - should fall back to manifest
        scopes = controller._route_issue_to_scopes(
            issue, manifest_with_chunks, None
        )

        # Should find MAIN-LOGIC in proc_part_1 by searching manifest chunks
        all_chunks = set()
        for scope in scopes:
            all_chunks.update(scope["chunk_ids"])

        assert "proc_part_1" in all_chunks


class TestIssueRoutingPriority4CrossCutting:
    """Tests for Priority 4: cross-cutting follow-up plan."""

    def test_cross_cutting_creates_scopes(
        self,
        controller: ReconcileController,
        manifest_with_chunks: Manifest,
    ) -> None:
        """Cross-cutting issue creates bounded scopes."""
        issue = Issue(
            issue_id="issue-011",
            severity=IssueSeverity.BLOCKER,
            question="How is error handling done across the program?",
        )

        scopes = controller._route_issue_to_scopes(
            issue, manifest_with_chunks, None
        )

        # Should create multiple scopes (either by merge node or division)
        assert len(scopes) >= 1
        # All scopes should be cross-cutting type
        for scope in scopes:
            assert scope["type"] == "cross_cutting"
            # Each scope should have a routing method
            assert "routing_method" in scope

    def test_cross_cutting_respects_max_chunks(
        self,
        controller: ReconcileController,
        manifest_with_chunks: Manifest,
    ) -> None:
        """Cross-cutting scopes respect max 5 chunks per scope."""
        issue = Issue(
            issue_id="issue-012",
            severity=IssueSeverity.BLOCKER,
            question="Explain the entire program flow",
        )

        scopes = controller._route_issue_to_scopes(
            issue, manifest_with_chunks, None
        )

        for scope in scopes:
            assert len(scope.get("chunk_ids", [])) <= 5

    def test_cross_cutting_uses_merge_node_boundaries_when_available(
        self,
        controller: ReconcileController,
        manifest_with_chunks: Manifest,
    ) -> None:
        """Cross-cutting prefers merge node boundaries for natural grouping."""
        issue = Issue(
            issue_id="issue-013",
            severity=IssueSeverity.BLOCKER,
            question="Explain the overall data flow",
        )

        scopes = controller.split_cross_cutting_scope(
            issue, manifest_with_chunks
        )

        # Should use merge node boundaries when available
        assert len(scopes) >= 1
        for scope in scopes:
            assert scope["type"] == "cross_cutting"


class TestScopeSizeConstraints:
    """Tests for scope size constraints per spec 9.3."""

    def test_scope_fits_context_single_chunk(
        self,
        controller: ReconcileController,
    ) -> None:
        """Single chunk scope fits context budget."""
        scope = {"chunk_ids": ["chunk_001"]}
        assert controller.scope_fits_context(scope, 4000) is True

    def test_scope_fits_context_small_list(
        self,
        controller: ReconcileController,
    ) -> None:
        """Small chunk list (<=5) fits context budget."""
        scope = {"chunk_ids": ["c1", "c2", "c3", "c4", "c5"]}
        assert controller.scope_fits_context(scope, 4000) is True

    def test_scope_exceeds_context_large_list(
        self,
        controller: ReconcileController,
    ) -> None:
        """Large chunk list (>5) exceeds context budget."""
        scope = {"chunk_ids": ["c1", "c2", "c3", "c4", "c5", "c6"]}
        assert controller.scope_fits_context(scope, 4000) is False


class TestSplitCrossCuttingScope:
    """Tests for split_cross_cutting_scope method."""

    def test_split_creates_bounded_scopes(
        self,
        controller: ReconcileController,
        manifest_with_chunks: Manifest,
    ) -> None:
        """Split creates bounded scopes with max 5 chunks each."""
        issue = Issue(
            issue_id="cross-001",
            severity=IssueSeverity.BLOCKER,
            question="Cross-cutting question",
        )

        scopes = controller.split_cross_cutting_scope(issue, manifest_with_chunks)

        for scope in scopes:
            assert len(scope.get("chunk_ids", [])) <= 5
            assert scope["type"] == "cross_cutting"
            assert scope["issue_id"] == "cross-001"

    def test_split_groups_by_division(
        self,
        controller: ReconcileController,
        manifest_with_chunks: Manifest,
    ) -> None:
        """Split groups chunks by division when using division strategy."""
        issue = Issue(
            issue_id="cross-002",
            severity=IssueSeverity.BLOCKER,
            question="What happens during processing?",
        )

        scopes = controller.split_cross_cutting_scope(issue, manifest_with_chunks)

        # Verify grouping
        for scope in scopes:
            if scope.get("routing_method") == "division_split":
                chunk_ids = scope["chunk_ids"]
                # All chunks in a scope should be from same division
                divisions = set()
                for chunk_id in chunk_ids:
                    chunk = manifest_with_chunks.get_chunk(chunk_id)
                    if chunk:
                        divisions.add(chunk.division)
                assert len(divisions) <= 1

    def test_split_handles_empty_manifest(
        self,
        controller: ReconcileController,
        artifact_ref: ArtifactRef,
    ) -> None:
        """Split handles manifest with no chunks gracefully."""
        empty_manifest = Manifest(
            job_id="empty-job",
            artifact_ref=artifact_ref,
            chunks=[],
            merge_dag=[],
        )

        issue = Issue(
            issue_id="cross-003",
            severity=IssueSeverity.BLOCKER,
            question="Question with no chunks",
        )

        scopes = controller.split_cross_cutting_scope(issue, empty_manifest)

        # Should create a fallback scope
        assert len(scopes) >= 1
        assert scopes[0]["routing_method"] == "fallback"


class TestIssueRoutingIntegration:
    """Integration tests for the full routing flow."""

    @pytest.mark.asyncio
    async def test_route_challenger_issues_creates_followups(
        self,
        controller: ReconcileController,
        manifest_with_chunks: Manifest,
        mock_artifact_store,
    ) -> None:
        """route_challenger_issues creates follow-up work items."""
        # Initialize job
        await controller.initialize_job(manifest_with_chunks)

        # Import here to avoid circular imports
        from atlas.models.results import ChallengeResult, ResolutionPlan

        # Create challenge result with issues
        challenge_result = ChallengeResult(
            job_id="test-routing-job",
            artifact_id="TESTPROG.cbl",
            artifact_version="hash123",
            issues=[
                Issue(
                    issue_id="issue-route-001",
                    severity=IssueSeverity.BLOCKER,
                    question="Critical issue in error handler",
                    suspected_scopes=["proc_part_4"],
                ),
                Issue(
                    issue_id="issue-route-002",
                    severity=IssueSeverity.MAJOR,
                    question="Major issue in processing",
                    routing_hints={"paragraphs": ["PROCESS-RECORD"]},
                ),
                Issue(
                    issue_id="issue-route-003",
                    severity=IssueSeverity.MINOR,  # Should be skipped
                    question="Minor style issue",
                ),
            ],
            resolution_plan=ResolutionPlan(requires_patch_merge=True),
        )

        followups = await controller.route_challenger_issues(
            "test-routing-job", challenge_result
        )

        # Should create 2 follow-ups (MINOR skipped)
        assert len(followups) >= 2

        # Verify follow-up types
        for followup in followups:
            assert followup.work_type == WorkItemType.DOC_FOLLOWUP
            assert followup.status == WorkItemStatus.READY
