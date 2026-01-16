"""Unit tests for data models.

Comprehensive tests for all core data models including:
- Enums and status transitions
- Artifact and ArtifactRef
- WorkItem and payloads
- Manifest and related structures
- Results (ChunkResult, MergeResult, etc.)
"""

import json
import pytest

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit
from pydantic import ValidationError

from atlas.models.enums import (
    WorkItemStatus,
    WorkItemType,
    IssueSeverity,
    ChunkKind,
)
from atlas.models.artifact import Artifact, ArtifactRef
from atlas.models.work_item import (
    WorkItem,
    DocChunkPayload,
    DocMergePayload,
    DocChallengePayload,
    DocFollowupPayload,
    DocPatchMergePayload,
    ChunkLocator,
)
from atlas.models.manifest import Manifest, ChunkSpec, MergeNode, SplitterProfile
from atlas.models.results import (
    ChunkResult,
    ChunkFacts,
    MergeResult,
    MergeCoverage,
    MergeConflict,
    ConsolidatedFacts,
    ChallengeResult,
    ResolutionPlan,
    FollowupTask,
    FollowupAnswer,
    Issue,
    DocumentationModel,
    DocIndex,
    Section,
    Evidence,
    SymbolDef,
    IOOperation,
    ErrorHandlingPattern,
    CallTarget,
    OpenQuestion,
)


# =============================================================================
# ENUM TESTS
# =============================================================================

class TestWorkItemStatus:
    """Tests for WorkItemStatus enum and transitions."""

    def test_valid_transitions_from_new(self) -> None:
        """NEW can transition to READY or CANCELED."""
        assert WorkItemStatus.NEW.can_transition_to(WorkItemStatus.READY)
        assert WorkItemStatus.NEW.can_transition_to(WorkItemStatus.CANCELED)
        assert not WorkItemStatus.NEW.can_transition_to(WorkItemStatus.DONE)
        assert not WorkItemStatus.NEW.can_transition_to(WorkItemStatus.IN_PROGRESS)

    def test_valid_transitions_from_ready(self) -> None:
        """READY can transition to IN_PROGRESS, BLOCKED, or CANCELED."""
        assert WorkItemStatus.READY.can_transition_to(WorkItemStatus.IN_PROGRESS)
        assert WorkItemStatus.READY.can_transition_to(WorkItemStatus.BLOCKED)
        assert WorkItemStatus.READY.can_transition_to(WorkItemStatus.CANCELED)
        assert not WorkItemStatus.READY.can_transition_to(WorkItemStatus.DONE)
        assert not WorkItemStatus.READY.can_transition_to(WorkItemStatus.NEW)

    def test_valid_transitions_from_in_progress(self) -> None:
        """IN_PROGRESS can transition to DONE, FAILED, or CANCELED."""
        assert WorkItemStatus.IN_PROGRESS.can_transition_to(WorkItemStatus.DONE)
        assert WorkItemStatus.IN_PROGRESS.can_transition_to(WorkItemStatus.FAILED)
        assert WorkItemStatus.IN_PROGRESS.can_transition_to(WorkItemStatus.CANCELED)
        assert not WorkItemStatus.IN_PROGRESS.can_transition_to(WorkItemStatus.READY)
        assert not WorkItemStatus.IN_PROGRESS.can_transition_to(WorkItemStatus.NEW)

    def test_valid_transitions_from_blocked(self) -> None:
        """BLOCKED can transition to READY or CANCELED."""
        assert WorkItemStatus.BLOCKED.can_transition_to(WorkItemStatus.READY)
        assert WorkItemStatus.BLOCKED.can_transition_to(WorkItemStatus.CANCELED)
        assert not WorkItemStatus.BLOCKED.can_transition_to(WorkItemStatus.DONE)

    def test_canceled_is_terminal(self) -> None:
        """CANCELED cannot transition to any state."""
        for status in WorkItemStatus:
            assert not WorkItemStatus.CANCELED.can_transition_to(status)

    def test_done_limited_transitions(self) -> None:
        """DONE can only transition to CANCELED (rare but allowed)."""
        # DONE can transition to CANCELED (rare)
        assert WorkItemStatus.DONE.can_transition_to(WorkItemStatus.CANCELED)
        # But not to other states
        assert not WorkItemStatus.DONE.can_transition_to(WorkItemStatus.READY)
        assert not WorkItemStatus.DONE.can_transition_to(WorkItemStatus.IN_PROGRESS)
        assert not WorkItemStatus.DONE.can_transition_to(WorkItemStatus.BLOCKED)
        assert not WorkItemStatus.DONE.can_transition_to(WorkItemStatus.NEW)

    def test_failed_can_retry(self) -> None:
        """FAILED can transition to READY for retry."""
        assert WorkItemStatus.FAILED.can_transition_to(WorkItemStatus.READY)
        assert not WorkItemStatus.FAILED.can_transition_to(WorkItemStatus.IN_PROGRESS)

    def test_all_statuses_have_transitions(self) -> None:
        """All statuses should have defined transitions (even if empty)."""
        for status in WorkItemStatus:
            # Just ensure can_transition_to doesn't raise
            status.can_transition_to(WorkItemStatus.NEW)


class TestWorkItemType:
    """Tests for WorkItemType enum."""

    def test_all_work_types_exist(self) -> None:
        """All expected work types should exist."""
        expected_types = [
            "DOC_REQUEST",
            "DOC_CHUNK",
            "DOC_MERGE",
            "DOC_CHALLENGE",
            "DOC_FOLLOWUP",
            "DOC_PATCH_MERGE",
        ]
        for type_name in expected_types:
            assert hasattr(WorkItemType, type_name)

    def test_work_type_values(self) -> None:
        """Work type values should be lowercase."""
        assert WorkItemType.DOC_REQUEST.value == "doc_request"
        assert WorkItemType.DOC_CHUNK.value == "doc_chunk"
        assert WorkItemType.DOC_MERGE.value == "doc_merge"


class TestIssueSeverity:
    """Tests for IssueSeverity enum."""

    def test_severity_values(self) -> None:
        """Test severity values exist."""
        assert IssueSeverity.BLOCKER.value == "blocker"
        assert IssueSeverity.MAJOR.value == "major"
        assert IssueSeverity.MINOR.value == "minor"
        assert IssueSeverity.QUESTION.value == "question"

    def test_severity_all_levels(self) -> None:
        """Test all severity levels exist."""
        expected = ["BLOCKER", "MAJOR", "MINOR", "QUESTION"]
        for level in expected:
            assert hasattr(IssueSeverity, level)


class TestChunkKind:
    """Tests for ChunkKind enum."""

    def test_cobol_chunk_kinds(self) -> None:
        """Test COBOL-specific chunk kinds exist."""
        expected_kinds = [
            "IDENTIFICATION_DIVISION",
            "ENVIRONMENT_DIVISION",
            "DATA_DIVISION",
            "PROCEDURE_DIVISION",
            "WORKING_STORAGE",
            "LINKAGE_SECTION",
            "FILE_SECTION",
            "PROCEDURE_PART",
            "GENERIC",
        ]
        for kind_name in expected_kinds:
            assert hasattr(ChunkKind, kind_name)


# =============================================================================
# ARTIFACT TESTS
# =============================================================================

class TestArtifact:
    """Tests for Artifact model."""

    def test_artifact_creation(self, sample_artifact: Artifact) -> None:
        """Test artifact can be created with required fields."""
        assert sample_artifact.artifact_id == "TEST001.cbl"
        assert sample_artifact.artifact_type == "cobol"
        assert sample_artifact.artifact_version == "abc123def456"

    def test_artifact_to_ref(self, sample_artifact: Artifact) -> None:
        """Test artifact can be converted to reference."""
        ref = sample_artifact.to_ref()
        assert isinstance(ref, ArtifactRef)
        assert ref.artifact_id == sample_artifact.artifact_id
        assert ref.artifact_version == sample_artifact.artifact_version
        assert ref.artifact_uri == sample_artifact.artifact_uri

    def test_artifact_with_metadata(self) -> None:
        """Test artifact with metadata."""
        artifact = Artifact(
            artifact_id="TEST.cbl",
            artifact_type="cobol",
            artifact_version="v1.0",
            artifact_uri="file:///test.cbl",
            metadata={"lines": 500, "modified": "2024-01-15"},
        )
        assert artifact.metadata["lines"] == 500

    def test_artifact_serialization(self, sample_artifact: Artifact) -> None:
        """Test artifact can be serialized and deserialized."""
        data = sample_artifact.model_dump()
        restored = Artifact(**data)
        assert restored.artifact_id == sample_artifact.artifact_id
        assert restored.artifact_version == sample_artifact.artifact_version

    def test_artifact_json_serialization(self, sample_artifact: Artifact) -> None:
        """Test artifact JSON round-trip."""
        json_str = sample_artifact.model_dump_json()
        data = json.loads(json_str)
        restored = Artifact(**data)
        assert restored == sample_artifact


class TestArtifactRef:
    """Tests for ArtifactRef model."""

    def test_artifact_ref_creation(self) -> None:
        """Test ArtifactRef can be created with all required fields."""
        ref = ArtifactRef(
            artifact_id="TEST.cbl",
            artifact_type="cobol",
            artifact_version="abc123",
            artifact_uri="file:///test.cbl",
        )
        assert ref.artifact_id == "TEST.cbl"
        assert ref.artifact_type == "cobol"
        assert ref.artifact_version == "abc123"

    def test_artifact_ref_requires_all_fields(self) -> None:
        """Test ArtifactRef requires all fields."""
        with pytest.raises(ValidationError):
            ArtifactRef(
                artifact_id="TEST.cbl",
                artifact_version="abc123",
            )

    def test_artifact_ref_equality(self) -> None:
        """Test ArtifactRef equality comparison."""
        ref1 = ArtifactRef(
            artifact_id="TEST.cbl",
            artifact_type="cobol",
            artifact_version="abc123",
            artifact_uri="file:///test.cbl",
        )
        ref2 = ArtifactRef(
            artifact_id="TEST.cbl",
            artifact_type="cobol",
            artifact_version="abc123",
            artifact_uri="file:///test.cbl",
        )
        ref3 = ArtifactRef(
            artifact_id="TEST.cbl",
            artifact_type="cobol",
            artifact_version="def456",
            artifact_uri="file:///test.cbl",
        )
        assert ref1 == ref2
        assert ref1 != ref3


# =============================================================================
# WORK ITEM TESTS
# =============================================================================

class TestWorkItem:
    """Tests for WorkItem model."""

    def test_work_item_creation(self, sample_work_item: WorkItem) -> None:
        """Test work item can be created."""
        assert sample_work_item.work_id == "chunk-001"
        assert sample_work_item.work_type == WorkItemType.DOC_CHUNK
        assert sample_work_item.status == WorkItemStatus.READY

    def test_can_start_when_ready(self, sample_work_item: WorkItem) -> None:
        """Test can_start returns True when READY."""
        assert sample_work_item.can_start()

    def test_cannot_start_when_blocked(self, sample_work_item: WorkItem) -> None:
        """Test can_start returns False when BLOCKED."""
        sample_work_item.status = WorkItemStatus.BLOCKED
        assert not sample_work_item.can_start()

    def test_cannot_start_when_done(self, sample_work_item: WorkItem) -> None:
        """Test can_start returns False when DONE."""
        sample_work_item.status = WorkItemStatus.DONE
        assert not sample_work_item.can_start()

    def test_compute_idempotency_key(self, sample_work_item: WorkItem) -> None:
        """Test idempotency key computation."""
        key = sample_work_item.compute_idempotency_key()
        assert "test-job-001" in key
        assert "doc_chunk" in key

    def test_work_item_with_depends_on(self) -> None:
        """Test work item with dependencies."""
        work_item = WorkItem(
            work_id="merge-001",
            work_type=WorkItemType.DOC_MERGE,
            status=WorkItemStatus.BLOCKED,
            depends_on=["chunk-001", "chunk-002"],
            payload=DocMergePayload(
                job_id="job-001",
                merge_node_id="merge_1",
                input_uris=["uri1", "uri2"],
                output_uri="output/merge.json",
            ),
        )
        assert len(work_item.depends_on) == 2
        assert "chunk-001" in work_item.depends_on

    def test_work_item_serialization(self, sample_work_item: WorkItem) -> None:
        """Test work item serialization."""
        data = sample_work_item.model_dump()
        assert data["work_id"] == "chunk-001"
        assert data["status"] == "ready"


class TestDocChunkPayload:
    """Tests for DocChunkPayload."""

    def test_chunk_payload_creation(self) -> None:
        """Test DocChunkPayload creation."""
        payload = DocChunkPayload(
            job_id="job-001",
            chunk_id="chunk-001",
            chunk_locator=ChunkLocator(start_line=1, end_line=100),
            result_uri="output/chunk-001.json",
        )
        assert payload.chunk_id == "chunk-001"
        assert payload.chunk_locator.start_line == 1
        assert payload.result_uri == "output/chunk-001.json"

    def test_chunk_payload_with_artifact_ref(self) -> None:
        """Test DocChunkPayload with artifact reference."""
        ref = ArtifactRef(
            artifact_id="TEST.cbl",
            artifact_type="cobol",
            artifact_version="abc123",
            artifact_uri="file:///test.cbl",
        )
        payload = DocChunkPayload(
            job_id="job-001",
            artifact_ref=ref,
            chunk_id="chunk-001",
            chunk_locator=ChunkLocator(start_line=1, end_line=100),
            result_uri="output/chunk-001.json",
        )
        assert payload.artifact_ref is not None
        assert payload.artifact_ref.artifact_id == "TEST.cbl"


class TestDocMergePayload:
    """Tests for DocMergePayload."""

    def test_merge_payload_creation(self) -> None:
        """Test DocMergePayload creation."""
        payload = DocMergePayload(
            job_id="job-001",
            merge_node_id="merge_1",
            input_uris=["chunk1.json", "chunk2.json"],
            output_uri="output/merge.json",
        )
        assert payload.merge_node_id == "merge_1"
        assert len(payload.input_uris) == 2

    def test_merge_payload_empty_inputs(self) -> None:
        """Test DocMergePayload with empty input_uris."""
        payload = DocMergePayload(
            job_id="job-001",
            merge_node_id="merge_1",
            input_uris=[],
            output_uri="output/merge.json",
        )
        assert len(payload.input_uris) == 0


class TestDocChallengePayload:
    """Tests for DocChallengePayload."""

    def test_challenge_payload_creation(self) -> None:
        """Test DocChallengePayload creation."""
        payload = DocChallengePayload(
            job_id="job-001",
            doc_uri="output/final.md",
            doc_model_uri="output/doc_model.json",
            challenge_profile="standard",
            output_uri="output/challenge.json",
        )
        assert payload.doc_uri == "output/final.md"
        assert payload.challenge_profile == "standard"


class TestDocFollowupPayload:
    """Tests for DocFollowupPayload."""

    def test_followup_payload_creation(self) -> None:
        """Test DocFollowupPayload creation."""
        payload = DocFollowupPayload(
            job_id="job-001",
            issue_id="issue-001",
            scope={"chunk_ids": ["chunk-001"], "question": "What does this do?"},
            inputs=["chunk-001.json"],
            output_uri="output/followup.json",
        )
        assert payload.issue_id == "issue-001"
        assert "chunk_ids" in payload.scope


class TestDocPatchMergePayload:
    """Tests for DocPatchMergePayload."""

    def test_patch_merge_payload_creation(self) -> None:
        """Test DocPatchMergePayload creation."""
        payload = DocPatchMergePayload(
            job_id="job-001",
            base_doc_uri="output/doc.md",
            base_doc_model_uri="output/doc_model.json",
            inputs=["followup1.json", "followup2.json"],
            output_doc_uri="output/patched_doc.md",
            output_doc_model_uri="output/patched_model.json",
        )
        assert len(payload.inputs) == 2
        assert payload.output_doc_uri == "output/patched_doc.md"


# =============================================================================
# MANIFEST TESTS
# =============================================================================

class TestManifest:
    """Tests for Manifest model."""

    def test_manifest_creation(self, sample_manifest: Manifest) -> None:
        """Test manifest can be created."""
        assert sample_manifest.job_id == "test-job-001"
        assert len(sample_manifest.chunks) == 1
        assert len(sample_manifest.merge_dag) == 2

    def test_get_chunk(self, sample_manifest: Manifest) -> None:
        """Test getting chunk by ID."""
        chunk = sample_manifest.get_chunk("test001_cbl_procedure_part_001")
        assert chunk is not None
        assert chunk.chunk_id == "test001_cbl_procedure_part_001"

    def test_get_chunk_not_found(self, sample_manifest: Manifest) -> None:
        """Test getting non-existent chunk."""
        chunk = sample_manifest.get_chunk("nonexistent")
        assert chunk is None

    def test_get_root_merge_node(self, sample_manifest: Manifest) -> None:
        """Test getting root merge node."""
        root = sample_manifest.get_root_merge_node()
        assert root is not None
        assert root.is_root

    def test_get_root_merge_node_none(self) -> None:
        """Test getting root when none exists."""
        manifest = Manifest(
            job_id="job-001",
            artifact_ref=ArtifactRef(
                artifact_id="test.cbl",
                artifact_type="cobol",
                artifact_version="v1",
                artifact_uri="file:///test.cbl",
            ),
            chunks=[],
            merge_dag=[
                MergeNode(merge_node_id="merge_1", input_ids=[], is_root=False),
            ],
        )
        root = manifest.get_root_merge_node()
        assert root is None

    def test_manifest_serialization(self, sample_manifest: Manifest) -> None:
        """Test manifest serialization."""
        data = sample_manifest.model_dump()
        restored = Manifest(**data)
        assert restored.job_id == sample_manifest.job_id
        assert len(restored.chunks) == len(sample_manifest.chunks)


class TestChunkSpec:
    """Tests for ChunkSpec model."""

    def test_chunk_spec_creation(self) -> None:
        """Test ChunkSpec creation."""
        spec = ChunkSpec(
            chunk_id="chunk-001",
            chunk_kind=ChunkKind.PROCEDURE_PART,
            start_line=100,
            end_line=200,
            estimated_tokens=500,
        )
        assert spec.chunk_id == "chunk-001"
        assert spec.start_line == 100

    def test_chunk_spec_with_paragraphs(self) -> None:
        """Test ChunkSpec with paragraph list."""
        spec = ChunkSpec(
            chunk_id="chunk-001",
            chunk_kind=ChunkKind.PROCEDURE_PART,
            start_line=100,
            end_line=200,
            paragraphs=["MAIN-LOGIC", "PROCESS-DATA"],
        )
        assert len(spec.paragraphs) == 2

    def test_chunk_spec_with_division(self) -> None:
        """Test ChunkSpec with division info."""
        spec = ChunkSpec(
            chunk_id="chunk-001",
            chunk_kind=ChunkKind.WORKING_STORAGE,
            start_line=50,
            end_line=150,
            division="DATA",
            section="WORKING-STORAGE",
        )
        assert spec.division == "DATA"
        assert spec.section == "WORKING-STORAGE"


class TestMergeNode:
    """Tests for MergeNode model."""

    def test_merge_node_creation(self) -> None:
        """Test MergeNode creation."""
        node = MergeNode(
            merge_node_id="merge_1",
            input_ids=["chunk-001", "chunk-002"],
            is_root=False,
        )
        assert node.merge_node_id == "merge_1"
        assert len(node.input_ids) == 2

    def test_root_merge_node(self) -> None:
        """Test root MergeNode."""
        node = MergeNode(
            merge_node_id="root_merge",
            input_ids=["merge_1", "merge_2"],
            is_root=True,
        )
        assert node.is_root

    def test_merge_node_empty_inputs(self) -> None:
        """Test MergeNode with empty inputs."""
        node = MergeNode(
            merge_node_id="leaf_merge",
            input_ids=[],
            is_root=False,
        )
        assert len(node.input_ids) == 0


class TestSplitterProfile:
    """Tests for SplitterProfile model."""

    def test_splitter_profile_defaults(self) -> None:
        """Test SplitterProfile default values."""
        profile = SplitterProfile()
        assert profile.max_chunk_tokens > 0
        assert profile.overlap_lines >= 0
        assert profile.prefer_semantic is True

    def test_splitter_profile_custom(self) -> None:
        """Test SplitterProfile with custom values."""
        profile = SplitterProfile(
            max_chunk_tokens=2000,
            overlap_lines=10,
            prefer_semantic=False,
        )
        assert profile.max_chunk_tokens == 2000
        assert profile.overlap_lines == 10
        assert profile.prefer_semantic is False


# =============================================================================
# RESULT MODEL TESTS
# =============================================================================

class TestChunkResult:
    """Tests for ChunkResult model."""

    def test_chunk_result_creation(self) -> None:
        """Test chunk result can be created."""
        result = ChunkResult(
            job_id="job-001",
            artifact_id="TEST.cbl",
            artifact_version="abc123",
            chunk_id="chunk-001",
            chunk_locator={"start_line": 1, "end_line": 100},
            chunk_kind="procedure_part",
            summary="Test summary",
            confidence=0.85,
        )
        assert result.job_id == "job-001"
        assert result.confidence == 0.85

    def test_chunk_result_with_facts(self) -> None:
        """Test chunk result with facts."""
        facts = ChunkFacts(
            paragraphs_defined=["MAIN-LOGIC", "PROCESS"],
            symbols_used=["WS-COUNTER", "WS-FLAG"],
        )
        result = ChunkResult(
            job_id="job-001",
            artifact_id="TEST.cbl",
            artifact_version="abc123",
            chunk_id="chunk-001",
            chunk_locator={"start_line": 1, "end_line": 100},
            chunk_kind="procedure_part",
            facts=facts,
        )
        assert len(result.facts.paragraphs_defined) == 2
        assert "MAIN-LOGIC" in result.facts.paragraphs_defined

    def test_chunk_result_confidence_bounds(self) -> None:
        """Test confidence must be between 0 and 1."""
        # Valid confidence
        result = ChunkResult(
            job_id="job-001",
            artifact_id="TEST.cbl",
            artifact_version="abc123",
            chunk_id="chunk-001",
            chunk_locator={},
            chunk_kind="generic",
            confidence=0.0,
        )
        assert result.confidence == 0.0

        result = ChunkResult(
            job_id="job-001",
            artifact_id="TEST.cbl",
            artifact_version="abc123",
            chunk_id="chunk-001",
            chunk_locator={},
            chunk_kind="generic",
            confidence=1.0,
        )
        assert result.confidence == 1.0

    def test_chunk_result_with_evidence(self) -> None:
        """Test chunk result with evidence."""
        result = ChunkResult(
            job_id="job-001",
            artifact_id="TEST.cbl",
            artifact_version="abc123",
            chunk_id="chunk-001",
            chunk_locator={},
            chunk_kind="generic",
            evidence=[
                Evidence(
                    evidence_type="line_range",
                    start_line=10,
                    end_line=20,
                    note="Key logic here",
                ),
            ],
        )
        assert len(result.evidence) == 1
        assert result.evidence[0].start_line == 10

    def test_chunk_result_with_open_questions(self) -> None:
        """Test chunk result with open questions."""
        result = ChunkResult(
            job_id="job-001",
            artifact_id="TEST.cbl",
            artifact_version="abc123",
            chunk_id="chunk-001",
            chunk_locator={},
            chunk_kind="generic",
            open_questions=[
                OpenQuestion(
                    question="What is the purpose of WS-FLAG?",
                    context_needed="Variable definition",
                ),
            ],
        )
        assert len(result.open_questions) == 1


class TestChunkFacts:
    """Tests for ChunkFacts model."""

    def test_chunk_facts_empty(self) -> None:
        """Test empty ChunkFacts."""
        facts = ChunkFacts()
        assert len(facts.symbols_defined) == 0
        assert len(facts.calls) == 0

    def test_chunk_facts_with_symbols(self) -> None:
        """Test ChunkFacts with symbols."""
        facts = ChunkFacts(
            symbols_defined=[
                SymbolDef(name="WS-COUNTER", kind="variable", line_number=50),
                SymbolDef(
                    name="CUSTOMER-RECORD",
                    kind="record",
                    attributes={"level": "01"},
                ),
            ],
        )
        assert len(facts.symbols_defined) == 2

    def test_chunk_facts_with_calls(self) -> None:
        """Test ChunkFacts with calls."""
        facts = ChunkFacts(
            calls=[
                CallTarget(target="PROCESS-DATA", call_type="perform"),
                CallTarget(target="ERRLOG", call_type="call", is_external=True),
            ],
        )
        assert len(facts.calls) == 2
        assert facts.calls[1].is_external

    def test_chunk_facts_with_io(self) -> None:
        """Test ChunkFacts with I/O operations."""
        facts = ChunkFacts(
            io_operations=[
                IOOperation(
                    operation="READ",
                    file_name="INPUT-FILE",
                    status_check=True,
                    line_number=150,
                ),
                IOOperation(
                    operation="WRITE",
                    file_name="OUTPUT-FILE",
                    record_name="OUTPUT-RECORD",
                ),
            ],
        )
        assert len(facts.io_operations) == 2

    def test_chunk_facts_with_error_handling(self) -> None:
        """Test ChunkFacts with error handling patterns."""
        facts = ChunkFacts(
            error_handling=[
                ErrorHandlingPattern(
                    pattern_type="file_status",
                    description="Checks FILE STATUS after READ",
                    line_numbers=[155, 156],
                    related_symbols=["WS-INPUT-STATUS"],
                ),
            ],
        )
        assert len(facts.error_handling) == 1


class TestMergeResult:
    """Tests for MergeResult model."""

    def test_merge_result_creation(self) -> None:
        """Test MergeResult creation."""
        result = MergeResult(
            job_id="job-001",
            artifact_id="TEST.cbl",
            artifact_version="abc123",
            merge_node_id="merge_1",
        )
        assert result.merge_node_id == "merge_1"

    def test_merge_result_with_coverage(self) -> None:
        """Test MergeResult with coverage info."""
        result = MergeResult(
            job_id="job-001",
            artifact_id="TEST.cbl",
            artifact_version="abc123",
            merge_node_id="merge_1",
            coverage=MergeCoverage(
                included_input_ids=["chunk-001", "chunk-002"],
                missing_input_ids=["chunk-003"],
            ),
        )
        assert len(result.coverage.included_input_ids) == 2
        assert len(result.coverage.missing_input_ids) == 1

    def test_merge_result_with_conflicts(self) -> None:
        """Test MergeResult with conflicts."""
        result = MergeResult(
            job_id="job-001",
            artifact_id="TEST.cbl",
            artifact_version="abc123",
            merge_node_id="merge_1",
            conflicts=[
                MergeConflict(
                    description="Disagreement on WS-FLAG purpose",
                    input_ids=["chunk-001", "chunk-002"],
                    suggested_followup_scope="chunk-001",
                ),
            ],
        )
        assert len(result.conflicts) == 1

    def test_merge_result_with_consolidated_facts(self) -> None:
        """Test MergeResult with consolidated facts."""
        result = MergeResult(
            job_id="job-001",
            artifact_id="TEST.cbl",
            artifact_version="abc123",
            merge_node_id="merge_1",
            consolidated_facts=ConsolidatedFacts(
                call_graph_edges=[
                    {"from": "MAIN", "to": "PROCESS", "type": "perform"},
                ],
                symbols=[
                    SymbolDef(name="WS-COUNTER", kind="variable"),
                ],
            ),
        )
        assert len(result.consolidated_facts.call_graph_edges) == 1


class TestChallengeResult:
    """Tests for ChallengeResult model."""

    def test_challenge_result_creation(self) -> None:
        """Test ChallengeResult creation."""
        result = ChallengeResult(
            job_id="job-001",
            artifact_id="TEST.cbl",
            artifact_version="abc123",
        )
        assert len(result.issues) == 0

    def test_challenge_result_with_issues(self) -> None:
        """Test ChallengeResult with issues."""
        result = ChallengeResult(
            job_id="job-001",
            artifact_id="TEST.cbl",
            artifact_version="abc123",
            issues=[
                Issue(
                    issue_id="issue-001",
                    severity=IssueSeverity.MAJOR,
                    question="What happens when file read fails?",
                    suspected_scopes=["chunk-001"],
                ),
                Issue(
                    issue_id="issue-002",
                    severity=IssueSeverity.BLOCKER,
                    question="Error handling is missing",
                ),
            ],
        )
        assert len(result.issues) == 2

    def test_has_blockers(self) -> None:
        """Test has_blockers method."""
        result = ChallengeResult(
            job_id="job-001",
            artifact_id="TEST.cbl",
            artifact_version="abc123",
            issues=[
                Issue(
                    issue_id="issue-001",
                    severity=IssueSeverity.MAJOR,
                    question="Minor concern",
                ),
            ],
        )
        assert not result.has_blockers()

        result.issues.append(
            Issue(
                issue_id="issue-002",
                severity=IssueSeverity.BLOCKER,
                question="Critical issue",
            )
        )
        assert result.has_blockers()

    def test_issues_by_severity(self) -> None:
        """Test issues_by_severity method."""
        result = ChallengeResult(
            job_id="job-001",
            artifact_id="TEST.cbl",
            artifact_version="abc123",
            issues=[
                Issue(issue_id="i1", severity=IssueSeverity.MAJOR, question="Q1"),
                Issue(issue_id="i2", severity=IssueSeverity.MAJOR, question="Q2"),
                Issue(issue_id="i3", severity=IssueSeverity.BLOCKER, question="Q3"),
                Issue(issue_id="i4", severity=IssueSeverity.MINOR, question="Q4"),
            ],
        )
        by_severity = result.issues_by_severity()
        assert len(by_severity[IssueSeverity.MAJOR]) == 2
        assert len(by_severity[IssueSeverity.BLOCKER]) == 1
        assert len(by_severity[IssueSeverity.MINOR]) == 1

    def test_challenge_result_with_resolution_plan(self) -> None:
        """Test ChallengeResult with resolution plan."""
        result = ChallengeResult(
            job_id="job-001",
            artifact_id="TEST.cbl",
            artifact_version="abc123",
            issues=[
                Issue(
                    issue_id="issue-001",
                    severity=IssueSeverity.MAJOR,
                    question="Unclear logic",
                ),
            ],
            resolution_plan=ResolutionPlan(
                followup_tasks=[
                    FollowupTask(
                        issue_id="issue-001",
                        scope={"chunk_ids": ["chunk-001"]},
                        description="Investigate error handling",
                    ),
                ],
                requires_patch_merge=True,
            ),
        )
        assert result.resolution_plan.requires_patch_merge
        assert len(result.resolution_plan.followup_tasks) == 1


class TestFollowupAnswer:
    """Tests for FollowupAnswer model."""

    def test_followup_answer_creation(self) -> None:
        """Test FollowupAnswer creation."""
        answer = FollowupAnswer(
            issue_id="issue-001",
            scope={"chunk_ids": ["chunk-001"]},
            answer="The error handling works by...",
            confidence=0.9,
        )
        assert answer.issue_id == "issue-001"
        assert answer.confidence == 0.9

    def test_followup_answer_with_facts(self) -> None:
        """Test FollowupAnswer with facts."""
        answer = FollowupAnswer(
            issue_id="issue-001",
            scope={"chunk_ids": ["chunk-001"]},
            answer="Found the following...",
            facts=ChunkFacts(
                paragraphs_defined=["ERROR-HANDLER"],
            ),
        )
        assert len(answer.facts.paragraphs_defined) == 1

    def test_followup_answer_with_evidence(self) -> None:
        """Test FollowupAnswer with evidence."""
        answer = FollowupAnswer(
            issue_id="issue-001",
            scope={"chunk_ids": ["chunk-001"]},
            answer="Evidence shows...",
            evidence=[
                Evidence(
                    evidence_type="line_range",
                    start_line=100,
                    end_line=120,
                    note="Error handling code",
                ),
            ],
        )
        assert len(answer.evidence) == 1


class TestDocumentationModel:
    """Tests for DocumentationModel."""

    def test_documentation_model_creation(self) -> None:
        """Test DocumentationModel creation."""
        model = DocumentationModel(
            doc_uri="output/doc.md",
        )
        assert model.doc_uri == "output/doc.md"
        assert len(model.sections) == 0

    def test_documentation_model_with_sections(self) -> None:
        """Test DocumentationModel with sections."""
        model = DocumentationModel(
            doc_uri="output/doc.md",
            sections=[
                Section(
                    section_id="overview",
                    title="Overview",
                    content="This program processes...",
                    source_refs=["chunk-001"],
                ),
                Section(
                    section_id="data_structures",
                    title="Data Structures",
                    content="The following data structures...",
                    source_refs=["chunk-002"],
                ),
            ],
        )
        assert len(model.sections) == 2

    def test_documentation_model_with_index(self) -> None:
        """Test DocumentationModel with index."""
        model = DocumentationModel(
            doc_uri="output/doc.md",
            index=DocIndex(
                symbol_to_chunks={
                    "WS-COUNTER": ["chunk-001", "chunk-002"],
                    "CUSTOMER-RECORD": ["chunk-003"],
                },
                paragraph_to_chunk={
                    "MAIN-LOGIC": "chunk-001",
                    "PROCESS-DATA": "chunk-001",
                },
            ),
        )
        assert len(model.index.symbol_to_chunks) == 2
        assert "WS-COUNTER" in model.index.symbol_to_chunks


class TestEvidence:
    """Tests for Evidence model."""

    def test_evidence_creation(self) -> None:
        """Test Evidence creation."""
        evidence = Evidence(
            evidence_type="line_range",
            start_line=10,
            end_line=20,
        )
        assert evidence.start_line == 10

    def test_evidence_with_note(self) -> None:
        """Test Evidence with note."""
        evidence = Evidence(
            evidence_type="symbol_ref",
            note="References WS-COUNTER",
            source_ref="chunk-001",
        )
        assert evidence.note == "References WS-COUNTER"


class TestSymbolDef:
    """Tests for SymbolDef model."""

    def test_symbol_def_creation(self) -> None:
        """Test SymbolDef creation."""
        symbol = SymbolDef(
            name="WS-COUNTER",
            kind="variable",
        )
        assert symbol.name == "WS-COUNTER"

    def test_symbol_def_with_attributes(self) -> None:
        """Test SymbolDef with attributes."""
        symbol = SymbolDef(
            name="CUSTOMER-RECORD",
            kind="record",
            attributes={
                "level": "01",
                "picture": None,
                "usage": "DISPLAY",
            },
            line_number=50,
        )
        assert symbol.attributes["level"] == "01"


class TestIOOperation:
    """Tests for IOOperation model."""

    def test_io_operation_creation(self) -> None:
        """Test IOOperation creation."""
        io_op = IOOperation(
            operation="READ",
            file_name="INPUT-FILE",
        )
        assert io_op.operation == "READ"

    def test_io_operation_with_status_check(self) -> None:
        """Test IOOperation with status check."""
        io_op = IOOperation(
            operation="READ",
            file_name="INPUT-FILE",
            record_name="INPUT-RECORD",
            status_check=True,
            line_number=150,
        )
        assert io_op.status_check


class TestCallTarget:
    """Tests for CallTarget model."""

    def test_call_target_perform(self) -> None:
        """Test internal PERFORM call."""
        call = CallTarget(
            target="PROCESS-DATA",
            call_type="perform",
            is_external=False,
        )
        assert call.target == "PROCESS-DATA"
        assert not call.is_external

    def test_call_target_external(self) -> None:
        """Test external CALL."""
        call = CallTarget(
            target="ERRLOG",
            call_type="call",
            is_external=True,
            line_number=200,
        )
        assert call.is_external
        assert call.call_type == "call"
