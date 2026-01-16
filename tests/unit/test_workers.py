"""Unit tests for worker implementations."""

import json
import pytest

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit
from unittest.mock import AsyncMock, MagicMock, patch

from atlas.models.artifact import ArtifactRef
from atlas.models.enums import (
    WorkItemStatus,
    WorkItemType,
    ChunkKind,
    IssueSeverity,
)
from atlas.models.work_item import (
    WorkItem,
    DocChunkPayload,
    DocMergePayload,
    DocChallengePayload,
    DocFollowupPayload,
    DocFinalizePayload,
    ChunkLocator,
)
from atlas.models.results import (
    ChunkResult,
    ChunkFacts,
    MergeResult,
    ChallengeResult,
    FollowupAnswer,
    DocumentationModel,
    DocIndex,
    Section,
    Issue,
    ResolutionPlan,
    FinalizeResult,
    TraceReport,
    JobStatistics,
)
from atlas.workers.scribe_impl import ScribeWorker
from atlas.workers.aggregator_impl import AggregatorWorker
from atlas.workers.challenger_impl import ChallengerWorker
from atlas.workers.followup_impl import FollowupWorker
from atlas.workers.finalize_impl import FinalizeWorker


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
def scribe_worker(
    mock_ticket_system, mock_artifact_store, mock_llm
) -> ScribeWorker:
    """Create ScribeWorker with mocks."""
    return ScribeWorker(
        worker_id="scribe-001",
        ticket_system=mock_ticket_system,
        artifact_store=mock_artifact_store,
        llm=mock_llm,
    )


@pytest.fixture
def aggregator_worker(
    mock_ticket_system, mock_artifact_store, mock_llm
) -> AggregatorWorker:
    """Create AggregatorWorker with mocks."""
    return AggregatorWorker(
        worker_id="aggregator-001",
        ticket_system=mock_ticket_system,
        artifact_store=mock_artifact_store,
        llm=mock_llm,
    )


@pytest.fixture
def challenger_worker(
    mock_ticket_system, mock_artifact_store, mock_llm
) -> ChallengerWorker:
    """Create ChallengerWorker with mocks."""
    return ChallengerWorker(
        worker_id="challenger-001",
        ticket_system=mock_ticket_system,
        artifact_store=mock_artifact_store,
        llm=mock_llm,
    )


@pytest.fixture
def followup_worker(
    mock_ticket_system, mock_artifact_store, mock_llm
) -> FollowupWorker:
    """Create FollowupWorker with mocks."""
    return FollowupWorker(
        worker_id="followup-001",
        ticket_system=mock_ticket_system,
        artifact_store=mock_artifact_store,
        llm=mock_llm,
    )


@pytest.fixture
def finalize_worker(
    mock_ticket_system, mock_artifact_store, mock_llm
) -> FinalizeWorker:
    """Create FinalizeWorker with mocks."""
    return FinalizeWorker(
        worker_id="finalize-001",
        ticket_system=mock_ticket_system,
        artifact_store=mock_artifact_store,
        llm=mock_llm,
    )


class TestScribeWorker:
    """Tests for ScribeWorker."""

    def test_supported_work_types(self, scribe_worker: ScribeWorker) -> None:
        """Test scribe supports correct work types."""
        assert WorkItemType.DOC_CHUNK in scribe_worker.supported_work_types
        assert WorkItemType.DOC_FOLLOWUP in scribe_worker.supported_work_types

    @pytest.mark.asyncio
    async def test_analyze_chunk(
        self,
        scribe_worker: ScribeWorker,
        artifact_ref: ArtifactRef,
        mock_llm,
    ) -> None:
        """Test chunk analysis produces ChunkResult."""
        # Configure mock LLM response
        mock_llm._responses = [json.dumps({
            "summary": "Main processing logic",
            "facts": {
                "symbols_defined": [{"name": "WS-COUNT", "kind": "variable"}],
                "symbols_used": ["INPUT-FILE", "OUTPUT-FILE"],
                "paragraphs_defined": ["MAIN-LOGIC", "PROCESS-RECORD"],
                "calls": [{"target": "PROCESS-RECORD", "call_type": "perform"}],
                "io_operations": [{"operation": "READ", "file_name": "INPUT-FILE"}],
                "error_handling": [{"pattern_type": "file_status", "description": "Checks status"}],
            },
            "evidence": [{"evidence_type": "line_range", "start_line": 1, "end_line": 50}],
            "open_questions": [],
            "confidence": 0.9,
        })]

        payload = DocChunkPayload(
            job_id="job-001",
            artifact_ref=artifact_ref,
            manifest_uri="s3://bucket/manifest.json",
            chunk_id="chunk_001",
            chunk_locator=ChunkLocator(start_line=1, end_line=100),
            result_uri="s3://bucket/chunks/chunk_001.json",
        )

        content = "PROCEDURE DIVISION.\nMAIN-LOGIC.\n    PERFORM PROCESS-RECORD."
        manifest = {"chunks": [{"chunk_id": "chunk_001", "chunk_kind": "procedure_part"}]}

        result = await scribe_worker.analyze_chunk(content, payload, manifest)

        assert isinstance(result, ChunkResult)
        assert result.chunk_id == "chunk_001"
        assert result.summary == "Main processing logic"
        assert len(result.facts.paragraphs_defined) == 2
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_answer_followup(
        self,
        scribe_worker: ScribeWorker,
        mock_llm,
    ) -> None:
        """Test follow-up question answering."""
        mock_llm._responses = [json.dumps({
            "answer": "The error handling uses FILE STATUS checks.",
            "facts": {
                "error_handling": [{"pattern_type": "file_status", "description": "Checks FS-CODE"}]
            },
            "evidence": [{"evidence_type": "line_range", "start_line": 50, "end_line": 60}],
            "confidence": 0.85,
        })]

        inputs = [
            {"chunk_id": "chunk_001", "summary": "Main logic", "facts": {}},
        ]

        answer = await scribe_worker.answer_followup(
            issue_id="issue-001",
            scope={"question": "How is error handling done?", "chunk_ids": ["chunk_001"]},
            inputs=inputs,
        )

        assert isinstance(answer, FollowupAnswer)
        assert answer.issue_id == "issue-001"
        assert "FILE STATUS" in answer.answer
        assert answer.confidence == 0.85

    def test_parse_chunk_response(
        self, scribe_worker: ScribeWorker, artifact_ref: ArtifactRef
    ) -> None:
        """Test parsing LLM response into ChunkResult."""
        response = {
            "summary": "Test summary",
            "facts": {
                "symbols_defined": [{"name": "WS-VAR", "kind": "variable"}],
                "symbols_used": ["OTHER-VAR"],
                "paragraphs_defined": ["PARA-1"],
                "calls": [],
                "io_operations": [],
                "error_handling": [],
            },
            "evidence": [],
            "open_questions": [{"question": "What is WS-VAR?"}],
            "confidence": 0.75,
        }

        payload = DocChunkPayload(
            job_id="job-001",
            artifact_ref=artifact_ref,
            manifest_uri="s3://bucket/manifest.json",
            chunk_id="chunk_001",
            chunk_locator=ChunkLocator(start_line=1, end_line=100),
            result_uri="s3://bucket/chunks/chunk_001.json",
        )

        result = scribe_worker._parse_chunk_response(
            response, payload, ChunkKind.PROCEDURE_PART
        )

        assert result.summary == "Test summary"
        assert len(result.facts.symbols_defined) == 1
        assert result.facts.symbols_defined[0].name == "WS-VAR"
        assert len(result.open_questions) == 1
        assert result.confidence == 0.75


class TestAggregatorWorker:
    """Tests for AggregatorWorker."""

    def test_supported_work_types(self, aggregator_worker: AggregatorWorker) -> None:
        """Test aggregator supports correct work types."""
        assert WorkItemType.DOC_MERGE in aggregator_worker.supported_work_types
        assert WorkItemType.DOC_PATCH_MERGE in aggregator_worker.supported_work_types

    @pytest.mark.asyncio
    async def test_merge_results(
        self,
        aggregator_worker: AggregatorWorker,
        artifact_ref: ArtifactRef,
        mock_llm,
    ) -> None:
        """Test merging chunk results."""
        mock_llm._responses = [json.dumps({
            "consolidated_facts": {},
            "conflicts": [],
            "narrative_sections": [
                {"section_id": "sec1", "title": "Overview", "content": "Test content"}
            ],
        })]

        inputs = [
            {
                "chunk_id": "chunk_001",
                "summary": "Main logic",
                "facts": {
                    "calls": [{"target": "PARA-1", "call_type": "perform"}],
                    "io_operations": [{"operation": "READ", "file_name": "FILE-1"}],
                },
            },
            {
                "chunk_id": "chunk_002",
                "summary": "Secondary logic",
                "facts": {
                    "calls": [{"target": "PARA-2", "call_type": "perform"}],
                },
            },
        ]

        payload = DocMergePayload(
            job_id="job-001",
            artifact_ref=artifact_ref,
            manifest_uri="s3://bucket/manifest.json",
            merge_node_id="merge_001",
            input_uris=["s3://bucket/chunk_001.json", "s3://bucket/chunk_002.json"],
            output_uri="s3://bucket/merge_001.json",
        )

        manifest = {
            "artifact_ref": {"artifact_id": "TEST.cbl", "artifact_version": "abc"},
            "merge_dag": [{"merge_node_id": "merge_001", "input_ids": ["chunk_001", "chunk_002"]}],
        }

        result = await aggregator_worker.merge_results(inputs, payload, manifest)

        assert isinstance(result, MergeResult)
        assert result.merge_node_id == "merge_001"
        assert len(result.coverage.included_input_ids) == 2

    def test_merge_facts(self, aggregator_worker: AggregatorWorker) -> None:
        """Test programmatic fact merging."""
        inputs = [
            {
                "chunk_id": "chunk_001",
                "facts": {
                    "calls": [{"target": "PARA-1", "call_type": "perform"}],
                    "io_operations": [{"operation": "READ", "file_name": "FILE-1"}],
                    "symbols_defined": [{"name": "VAR-1", "kind": "variable"}],
                },
            },
            {
                "chunk_id": "chunk_002",
                "facts": {
                    "calls": [{"target": "PARA-2", "call_type": "perform"}],
                    "io_operations": [{"operation": "WRITE", "file_name": "FILE-2"}],
                    "symbols_defined": [{"name": "VAR-2", "kind": "variable"}],
                },
            },
        ]

        result = aggregator_worker._merge_facts(inputs)

        assert len(result.call_graph_edges) == 2
        assert len(result.io_map) == 2
        assert len(result.symbols) == 2


class TestChallengerWorker:
    """Tests for ChallengerWorker."""

    def test_supported_work_types(self, challenger_worker: ChallengerWorker) -> None:
        """Test challenger supports correct work types."""
        assert WorkItemType.DOC_CHALLENGE in challenger_worker.supported_work_types
        assert len(challenger_worker.supported_work_types) == 1

    @pytest.mark.asyncio
    async def test_review_documentation(
        self,
        challenger_worker: ChallengerWorker,
        mock_llm,
    ) -> None:
        """Test documentation review produces issues."""
        mock_llm._responses = [json.dumps({
            "issues": [
                {
                    "issue_id": "issue-001",
                    "severity": "major",
                    "question": "Error handling is not documented",
                    "doc_section_refs": ["section-1"],
                    "suspected_scopes": ["chunk_001"],
                    "routing_hints": {"paragraphs": ["ERROR-HANDLER"]},
                },
                {
                    "issue_id": "issue-002",
                    "severity": "minor",
                    "question": "Minor formatting issue",
                },
            ],
            "summary": "Documentation has gaps",
            "recommendations": ["Add error handling docs"],
        })]

        doc = "# Documentation\n\nSome content..."
        doc_model = DocumentationModel(
            doc_uri="s3://bucket/doc.md",
            sections=[
                Section(
                    section_id="section-1",
                    title="Overview",
                    source_refs=["chunk_001"],
                ),
            ],
            index=DocIndex(),
            metadata={"artifact_id": "TEST.cbl", "job_id": "job-001"},
        )

        result = await challenger_worker.review_documentation(
            doc, doc_model, "error_handling"
        )

        assert isinstance(result, ChallengeResult)
        assert len(result.issues) == 2
        assert result.issues[0].severity == IssueSeverity.MAJOR
        assert result.resolution_plan.requires_patch_merge

    def test_create_issue(self, challenger_worker: ChallengerWorker) -> None:
        """Test issue creation with routing info."""
        issue = challenger_worker.create_issue(
            question="What happens on error?",
            severity=IssueSeverity.BLOCKER,
            doc_section_refs=["sec-1"],
            suspected_scopes=["chunk_001"],
            routing_hints={"symbols": ["WS-STATUS"]},
        )

        assert issue.severity == IssueSeverity.BLOCKER
        assert issue.question == "What happens on error?"
        assert "sec-1" in issue.doc_section_refs
        assert len(issue.issue_id) > 0

    def test_create_resolution_plan(
        self, challenger_worker: ChallengerWorker
    ) -> None:
        """Test resolution plan creation."""
        issues = [
            Issue(
                issue_id="issue-001",
                severity=IssueSeverity.BLOCKER,
                question="Critical issue",
                suspected_scopes=["chunk_001"],
            ),
            Issue(
                issue_id="issue-002",
                severity=IssueSeverity.MINOR,  # Should be skipped
                question="Minor issue",
            ),
        ]

        doc_model = DocumentationModel(
            doc_uri="s3://bucket/doc.md",
            sections=[],
            index=DocIndex(),
        )

        plan = challenger_worker.create_resolution_plan(issues, doc_model)

        # Only blocker/major issues get tasks
        assert len(plan.followup_tasks) == 1
        assert plan.followup_tasks[0].issue_id == "issue-001"
        assert plan.requires_patch_merge

    def test_parse_issues(self, challenger_worker: ChallengerWorker) -> None:
        """Test parsing issues from LLM response."""
        issues_data = [
            {
                "issue_id": "test-001",
                "severity": "blocker",
                "question": "Critical problem",
                "doc_section_refs": ["sec-1"],
                "suspected_scopes": ["chunk_001"],
            },
            {
                "severity": "invalid_severity",  # Should default to QUESTION
                "question": "Unknown severity",
            },
        ]

        issues = challenger_worker._parse_issues(issues_data)

        assert len(issues) == 2
        assert issues[0].severity == IssueSeverity.BLOCKER
        assert issues[0].issue_id == "test-001"
        assert issues[1].severity == IssueSeverity.QUESTION


class TestFollowupWorker:
    """Tests for FollowupWorker."""

    def test_supported_work_types(self, followup_worker: FollowupWorker) -> None:
        """Test followup worker supports correct work types."""
        assert WorkItemType.DOC_FOLLOWUP in followup_worker.supported_work_types
        assert len(followup_worker.supported_work_types) == 1

    @pytest.mark.asyncio
    async def test_process_followup(
        self,
        followup_worker: FollowupWorker,
        artifact_ref: ArtifactRef,
        mock_llm,
        mock_artifact_store,
    ) -> None:
        """Test processing a follow-up work item."""
        mock_llm._responses = [json.dumps({
            "answer": "The error handling uses FILE STATUS WS-STATUS.",
            "facts": {
                "error_handling": [{"pattern_type": "file_status", "description": "Checks WS-STATUS"}],
            },
            "evidence": [{"evidence_type": "line_range", "start_line": 100, "end_line": 110}],
            "confidence": 0.9,
        })]

        # Store input artifact
        input_uri = "s3://bucket/chunk_001.json"
        await mock_artifact_store.write_json(input_uri, {
            "chunk_id": "chunk_001",
            "summary": "Main logic",
            "facts": {},
        })

        payload = DocFollowupPayload(
            job_id="job-001",
            artifact_ref=artifact_ref,
            manifest_uri="s3://bucket/manifest.json",
            issue_id="issue-001",
            scope={"question": "How is error handling done?", "chunk_ids": ["chunk_001"]},
            inputs=[input_uri],
            output_uri="s3://bucket/followup_001.json",
        )

        work_item = WorkItem(
            work_id="followup-001",
            work_type=WorkItemType.DOC_FOLLOWUP,
            status=WorkItemStatus.IN_PROGRESS,
            payload=payload,
        )

        result = await followup_worker.process(work_item)

        assert isinstance(result, FollowupAnswer)
        assert result.issue_id == "issue-001"
        assert "FILE STATUS" in result.answer
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_investigate_cross_cutting(
        self,
        followup_worker: FollowupWorker,
        mock_llm,
    ) -> None:
        """Test cross-cutting investigation."""
        mock_llm._responses = [json.dumps({
            "answer": "Error handling spans multiple paragraphs.",
            "facts": {},
            "evidence": [],
            "confidence": 0.8,
        })]

        answer = await followup_worker._investigate(
            issue_id="issue-001",
            scope={
                "question": "Explain error handling",
                "type": "cross_cutting",
                "chunk_ids": ["chunk_001", "chunk_002"],
            },
            prior_analysis=[],
            source_code="",
            is_cross_cutting=True,
        )

        assert answer.issue_id == "issue-001"
        assert answer.confidence == 0.8

    def test_parse_response(self, followup_worker: FollowupWorker) -> None:
        """Test parsing LLM response into FollowupAnswer."""
        response = {
            "answer": "Detailed answer here",
            "facts": {
                "symbols_defined": [{"name": "WS-STATUS", "kind": "variable"}],
                "error_handling": [{"pattern_type": "file_status"}],
            },
            "evidence": [{"start_line": 50, "end_line": 60}],
            "confidence": 0.85,
            "additional_context_needed": "Need copybook definition",
        }

        answer = followup_worker._parse_response(
            response,
            issue_id="issue-001",
            scope={"chunk_ids": ["chunk_001"]},
        )

        assert answer.answer == "Detailed answer here"
        assert len(answer.facts.symbols_defined) == 1
        assert answer.confidence == 0.85
        assert "additional_context_needed" in answer.metadata

    def test_extract_scope_lines(self, followup_worker: FollowupWorker) -> None:
        """Test extracting source lines for specific chunks."""
        source = "\n".join([f"Line {i}" for i in range(1, 201)])
        manifest = {
            "chunks": [
                {"chunk_id": "chunk_001", "start_line": 1, "end_line": 50},
                {"chunk_id": "chunk_002", "start_line": 51, "end_line": 100},
            ],
        }

        result = followup_worker._extract_scope_lines(
            source, manifest, ["chunk_001"]
        )

        assert "chunk_001" in result
        assert "Line 1" in result
        assert "Line 51" not in result


class TestWorkerIdempotency:
    """Tests for worker idempotency behavior."""

    @pytest.mark.asyncio
    async def test_scribe_checks_output_exists(
        self,
        scribe_worker: ScribeWorker,
        mock_artifact_store,
        artifact_ref: ArtifactRef,
    ) -> None:
        """Test scribe checks if output already exists."""
        output_uri = "s3://bucket/chunks/chunk_001.json"

        # Store existing output
        await mock_artifact_store.write_json(output_uri, {"chunk_id": "chunk_001"})

        work_item = WorkItem(
            work_id="chunk-001",
            work_type=WorkItemType.DOC_CHUNK,
            status=WorkItemStatus.READY,
            payload=DocChunkPayload(
                job_id="job-001",
                artifact_ref=artifact_ref,
                manifest_uri="s3://bucket/manifest.json",
                chunk_id="chunk_001",
                chunk_locator=ChunkLocator(start_line=1, end_line=100),
                result_uri=output_uri,
            ),
        )

        exists = await scribe_worker._output_exists(work_item)
        assert exists is True


# ============================================================================
# Extended Scribe Worker Tests
# ============================================================================


class TestScribeWorkerExtended:
    """Extended tests for ScribeWorker with more coverage."""

    @pytest.mark.asyncio
    async def test_analyze_chunk_with_all_fact_types(
        self,
        scribe_worker: ScribeWorker,
        artifact_ref: ArtifactRef,
        mock_llm,
    ) -> None:
        """Test chunk analysis with comprehensive facts."""
        mock_llm._responses = [json.dumps({
            "summary": "Comprehensive processing logic",
            "facts": {
                "symbols_defined": [
                    {"name": "WS-COUNT", "kind": "variable", "attributes": {"level": "01", "picture": "9(5)"}, "line_number": 10},
                    {"name": "WS-FLAG", "kind": "variable", "attributes": {"level": "01", "picture": "X"}, "line_number": 11},
                ],
                "symbols_used": ["INPUT-FILE", "OUTPUT-FILE", "WS-COUNT", "WS-FLAG"],
                "entrypoints": ["MAIN-ENTRY"],
                "paragraphs_defined": ["MAIN-LOGIC", "PROCESS-RECORD", "ERROR-HANDLER"],
                "calls": [
                    {"target": "PROCESS-RECORD", "call_type": "perform", "is_external": False, "line_number": 20},
                    {"target": "EXT-VALIDATION", "call_type": "call", "is_external": True, "line_number": 30},
                ],
                "io_operations": [
                    {"operation": "OPEN", "file_name": "INPUT-FILE", "line_number": 15, "status_check": True},
                    {"operation": "READ", "file_name": "INPUT-FILE", "record_name": "INPUT-REC", "line_number": 25, "status_check": True},
                    {"operation": "WRITE", "file_name": "OUTPUT-FILE", "record_name": "OUTPUT-REC", "line_number": 35, "status_check": False},
                ],
                "error_handling": [
                    {"pattern_type": "file_status", "description": "FILE STATUS check after OPEN", "line_numbers": [16, 17], "related_symbols": ["WS-FS"]},
                    {"pattern_type": "evaluate", "description": "EVALUATE block for record types", "line_numbers": [40, 41, 42], "related_symbols": ["WS-REC-TYPE"]},
                ],
            },
            "evidence": [
                {"evidence_type": "line_range", "start_line": 1, "end_line": 100, "note": "Main logic section"},
                {"evidence_type": "source_ref", "start_line": 50, "end_line": 60, "note": "Error handling block"},
            ],
            "open_questions": [
                {"question": "What is EXT-VALIDATION?", "context_needed": "External program source", "suspected_location": "EXTLIB"},
                {"question": "What error codes are expected?", "context_needed": "Error code documentation"},
            ],
            "confidence": 0.85,
        })]

        payload = DocChunkPayload(
            job_id="job-001",
            artifact_ref=artifact_ref,
            manifest_uri="s3://bucket/manifest.json",
            chunk_id="chunk_002",
            chunk_locator=ChunkLocator(start_line=1, end_line=100, division="PROCEDURE", section="MAIN-SECTION", paragraphs=["MAIN-LOGIC"]),
            result_uri="s3://bucket/chunks/chunk_002.json",
        )

        content = "PROCEDURE DIVISION.\nMAIN-LOGIC.\n    PERFORM PROCESS-RECORD."
        manifest = {"chunks": [{"chunk_id": "chunk_002", "chunk_kind": "procedure_part"}]}

        result = await scribe_worker.analyze_chunk(content, payload, manifest)

        # Verify comprehensive fact extraction
        assert len(result.facts.symbols_defined) == 2
        assert result.facts.symbols_defined[0].attributes.get("level") == "01"
        assert len(result.facts.symbols_used) == 4
        assert "MAIN-ENTRY" in result.facts.entrypoints
        assert len(result.facts.calls) == 2
        assert result.facts.calls[1].is_external is True
        assert len(result.facts.io_operations) == 3
        assert result.facts.io_operations[0].status_check is True
        assert len(result.facts.error_handling) == 2
        assert len(result.evidence) == 2
        assert len(result.open_questions) == 2

    @pytest.mark.asyncio
    async def test_analyze_chunk_with_llm_error(
        self,
        scribe_worker: ScribeWorker,
        artifact_ref: ArtifactRef,
        mock_llm,
    ) -> None:
        """Test chunk analysis handles LLM errors gracefully."""
        # Make LLM return invalid JSON - MockLLM catches this and returns {"content": ...}
        mock_llm._responses = ["not valid json at all"]

        payload = DocChunkPayload(
            job_id="job-001",
            artifact_ref=artifact_ref,
            manifest_uri="s3://bucket/manifest.json",
            chunk_id="chunk_error",
            chunk_locator=ChunkLocator(start_line=1, end_line=50),
            result_uri="s3://bucket/chunks/chunk_error.json",
        )

        content = "SOME COBOL CODE"
        manifest = {"chunks": []}

        result = await scribe_worker.analyze_chunk(content, payload, manifest)

        # When JSON parsing fails, MockLLM returns {"content": ...} which
        # results in a minimal result with low/no data extracted
        # The exact behavior depends on how scribe handles missing fields
        assert isinstance(result, ChunkResult)
        assert result.chunk_id == "chunk_error"

    @pytest.mark.asyncio
    async def test_analyze_chunk_generic_kind(
        self,
        scribe_worker: ScribeWorker,
        artifact_ref: ArtifactRef,
        mock_llm,
    ) -> None:
        """Test chunk analysis with generic chunk kind."""
        mock_llm._responses = [json.dumps({
            "summary": "Generic chunk",
            "facts": {"symbols_defined": [], "symbols_used": []},
            "evidence": [],
            "open_questions": [],
            "confidence": 0.5,
        })]

        payload = DocChunkPayload(
            job_id="job-001",
            artifact_ref=artifact_ref,
            manifest_uri="s3://bucket/manifest.json",
            chunk_id="chunk_generic",
            chunk_locator=ChunkLocator(start_line=1, end_line=10),
            result_uri="s3://bucket/chunks/chunk_generic.json",
        )

        # No chunk spec in manifest - should use GENERIC kind
        result = await scribe_worker.analyze_chunk("CODE", payload, {})
        assert result.chunk_kind == "generic"

    @pytest.mark.asyncio
    async def test_answer_followup_no_inputs(
        self,
        scribe_worker: ScribeWorker,
        mock_llm,
    ) -> None:
        """Test follow-up with empty inputs."""
        mock_llm._responses = [json.dumps({
            "answer": "Unable to answer without context.",
            "facts": {},
            "evidence": [],
            "confidence": 0.3,
        })]

        answer = await scribe_worker.answer_followup(
            issue_id="issue-empty",
            scope={"question": "What happens?"},
            inputs=[],
        )

        assert answer.confidence == 0.3
        assert answer.issue_id == "issue-empty"


# ============================================================================
# Extended Aggregator Worker Tests
# ============================================================================


class TestAggregatorWorkerExtended:
    """Extended tests for AggregatorWorker with more coverage."""

    def test_merge_facts_empty_inputs(self, aggregator_worker: AggregatorWorker) -> None:
        """Test merging with empty inputs."""
        result = aggregator_worker._merge_facts([])

        assert len(result.call_graph_edges) == 0
        assert len(result.io_map) == 0
        assert len(result.symbols) == 0

    def test_merge_facts_deduplicate(self, aggregator_worker: AggregatorWorker) -> None:
        """Test that merge collects calls from multiple chunks."""
        inputs = [
            {
                "chunk_id": "chunk_001",
                "facts": {
                    "calls": [
                        {"target": "PARA-1", "call_type": "perform"},
                        {"target": "PARA-2", "call_type": "perform"},
                    ],
                },
            },
            {
                "chunk_id": "chunk_002",
                "facts": {
                    "calls": [
                        {"target": "PARA-1", "call_type": "perform"},  # Duplicate target
                        {"target": "PARA-3", "call_type": "perform"},
                    ],
                },
            },
        ]

        result = aggregator_worker._merge_facts(inputs)

        # Result is ConsolidatedFacts with call_graph_edges
        # Each edge is a CallGraphEdge object or dict
        assert len(result.call_graph_edges) >= 3  # At least 3 edges from the inputs

    def test_merge_facts_preserves_io_details(self, aggregator_worker: AggregatorWorker) -> None:
        """Test I/O operations are preserved with all details."""
        inputs = [
            {
                "chunk_id": "chunk_001",
                "facts": {
                    "io_operations": [
                        {
                            "operation": "READ",
                            "file_name": "FILE-1",
                            "record_name": "REC-1",
                            "line_number": 100,
                            "status_check": True,
                        },
                    ],
                },
            },
        ]

        result = aggregator_worker._merge_facts(inputs)

        assert len(result.io_map) == 1
        io_op = result.io_map[0]
        assert io_op.operation == "READ"
        assert io_op.file_name == "FILE-1"
        assert io_op.status_check is True

    @pytest.mark.asyncio
    async def test_merge_results_with_conflicts(
        self,
        aggregator_worker: AggregatorWorker,
        artifact_ref: ArtifactRef,
        mock_llm,
    ) -> None:
        """Test merge that produces conflicts."""
        mock_llm._responses = [json.dumps({
            "consolidated_facts": {},
            "conflicts": [
                {
                    "description": "Conflicting information about FILE STATUS handling",
                    "input_ids": ["chunk_001", "chunk_002"],
                    "suggested_followup_scope": "Lines 100-120",
                },
            ],
            "narrative_sections": [],
        })]

        inputs = [
            {"chunk_id": "chunk_001", "summary": "Logic A", "facts": {}},
            {"chunk_id": "chunk_002", "summary": "Logic B", "facts": {}},
        ]

        payload = DocMergePayload(
            job_id="job-001",
            artifact_ref=artifact_ref,
            manifest_uri="s3://bucket/manifest.json",
            merge_node_id="merge_conflict",
            input_uris=["uri1", "uri2"],
            output_uri="s3://bucket/merge_conflict.json",
        )

        manifest = {
            "artifact_ref": {"artifact_id": "TEST.cbl", "artifact_version": "abc"},
            "merge_dag": [{"merge_node_id": "merge_conflict", "input_ids": ["chunk_001", "chunk_002"]}],
        }

        result = await aggregator_worker.merge_results(inputs, payload, manifest)

        assert len(result.conflicts) == 1
        assert "FILE STATUS" in result.conflicts[0].description


# ============================================================================
# Extended Challenger Worker Tests
# ============================================================================


class TestChallengerWorkerExtended:
    """Extended tests for ChallengerWorker with more coverage."""

    def test_create_issue_generates_id(
        self, challenger_worker: ChallengerWorker
    ) -> None:
        """Test issue ID is automatically generated."""
        issue = challenger_worker.create_issue(
            question="Test question?",
            severity=IssueSeverity.QUESTION,
        )

        assert issue.issue_id is not None
        assert len(issue.issue_id) > 0
        # Should be unique per call
        issue2 = challenger_worker.create_issue(
            question="Another question?",
            severity=IssueSeverity.MINOR,
        )
        assert issue.issue_id != issue2.issue_id

    def test_parse_issues_missing_fields(
        self, challenger_worker: ChallengerWorker
    ) -> None:
        """Test parsing issues with missing optional fields."""
        issues_data = [
            {
                "question": "Only has question",
            },
            {
                "severity": "major",
                "question": "Has severity",
                "doc_section_refs": ["sec-1"],
                "suspected_scopes": [],
            },
        ]

        issues = challenger_worker._parse_issues(issues_data)

        assert len(issues) == 2
        assert issues[0].severity == IssueSeverity.QUESTION  # Default
        assert issues[0].issue_id is not None  # Generated
        assert issues[1].severity == IssueSeverity.MAJOR

    def test_create_resolution_plan_no_issues(
        self, challenger_worker: ChallengerWorker
    ) -> None:
        """Test resolution plan with no issues."""
        doc_model = DocumentationModel(
            doc_uri="s3://bucket/doc.md",
            sections=[],
            index=DocIndex(),
        )

        plan = challenger_worker.create_resolution_plan([], doc_model)

        assert len(plan.followup_tasks) == 0
        assert not plan.requires_patch_merge

    def test_create_resolution_plan_all_minor(
        self, challenger_worker: ChallengerWorker
    ) -> None:
        """Test resolution plan with only minor issues (no tasks)."""
        issues = [
            Issue(
                issue_id="minor-001",
                severity=IssueSeverity.MINOR,
                question="Minor formatting",
            ),
            Issue(
                issue_id="question-001",
                severity=IssueSeverity.QUESTION,
                question="Clarification needed",
            ),
        ]

        doc_model = DocumentationModel(
            doc_uri="s3://bucket/doc.md",
            sections=[],
            index=DocIndex(),
        )

        plan = challenger_worker.create_resolution_plan(issues, doc_model)

        # Minor and question severity don't generate tasks
        assert len(plan.followup_tasks) == 0
        assert not plan.requires_patch_merge

    @pytest.mark.asyncio
    async def test_review_documentation_no_issues(
        self,
        challenger_worker: ChallengerWorker,
        mock_llm,
    ) -> None:
        """Test review that finds no issues."""
        mock_llm._responses = [json.dumps({
            "issues": [],
            "summary": "Documentation is complete and accurate.",
            "recommendations": [],
        })]

        doc_model = DocumentationModel(
            doc_uri="s3://bucket/doc.md",
            sections=[Section(section_id="s1", title="Overview")],
            index=DocIndex(),
        )

        result = await challenger_worker.review_documentation(
            "# Good Documentation", doc_model, "completeness"
        )

        assert len(result.issues) == 0
        assert not result.has_blockers()


# ============================================================================
# Extended Followup Worker Tests
# ============================================================================


class TestFollowupWorkerExtended:
    """Extended tests for FollowupWorker with more coverage."""

    def test_parse_response_minimal(self, followup_worker: FollowupWorker) -> None:
        """Test parsing minimal response."""
        response = {
            "answer": "Simple answer",
            "confidence": 0.7,
        }

        answer = followup_worker._parse_response(
            response,
            issue_id="issue-minimal",
            scope={"question": "What?"},
        )

        assert answer.answer == "Simple answer"
        assert answer.confidence == 0.7
        assert answer.facts is not None

    def test_parse_response_with_all_fields(self, followup_worker: FollowupWorker) -> None:
        """Test parsing response with all optional fields."""
        response = {
            "answer": "Comprehensive answer",
            "facts": {
                "symbols_defined": [{"name": "VAR", "kind": "variable"}],
                "symbols_used": ["OTHER"],
                "entrypoints": ["ENTRY"],
                "paragraphs_defined": ["PARA"],
                "calls": [{"target": "SUB", "call_type": "perform"}],
                "io_operations": [{"operation": "READ", "file_name": "FILE"}],
                "error_handling": [{"pattern_type": "evaluate", "description": "Error check"}],
            },
            "evidence": [
                {"evidence_type": "line_range", "start_line": 10, "end_line": 20},
            ],
            "confidence": 0.95,
            "additional_context_needed": "Need more info",
            "suggested_followup": "Check error codes",
        }

        answer = followup_worker._parse_response(
            response,
            issue_id="issue-full",
            scope={"chunk_ids": ["c1", "c2"]},
        )

        assert len(answer.facts.symbols_defined) == 1
        assert len(answer.facts.calls) == 1
        assert len(answer.evidence) == 1
        assert "additional_context_needed" in answer.metadata

    def test_extract_scope_lines_no_match(self, followup_worker: FollowupWorker) -> None:
        """Test extracting lines for non-existent chunk."""
        source = "\n".join([f"Line {i}" for i in range(1, 101)])
        manifest = {
            "chunks": [
                {"chunk_id": "chunk_001", "start_line": 1, "end_line": 50},
            ],
        }

        result = followup_worker._extract_scope_lines(
            source, manifest, ["nonexistent_chunk"]
        )

        # Should return empty or indicate no match
        assert "chunk_001" not in result

    def test_extract_scope_lines_multiple_chunks(self, followup_worker: FollowupWorker) -> None:
        """Test extracting lines for multiple chunks."""
        source = "\n".join([f"Line {i}" for i in range(1, 201)])
        manifest = {
            "chunks": [
                {"chunk_id": "chunk_001", "start_line": 1, "end_line": 50},
                {"chunk_id": "chunk_002", "start_line": 51, "end_line": 100},
                {"chunk_id": "chunk_003", "start_line": 101, "end_line": 150},
            ],
        }

        result = followup_worker._extract_scope_lines(
            source, manifest, ["chunk_001", "chunk_003"]
        )

        assert "chunk_001" in result
        assert "chunk_003" in result
        assert "chunk_002" not in result

    @pytest.mark.asyncio
    async def test_investigate_with_prior_analysis(
        self,
        followup_worker: FollowupWorker,
        mock_llm,
    ) -> None:
        """Test investigation with prior analysis context."""
        mock_llm._responses = [json.dumps({
            "answer": "Based on prior analysis, the error handling is...",
            "facts": {},
            "evidence": [],
            "confidence": 0.85,
        })]

        prior_analysis = [
            {"chunk_id": "c1", "summary": "Main logic", "facts": {"calls": []}},
            {"chunk_id": "c2", "summary": "Error handler", "facts": {"error_handling": []}},
        ]

        answer = await followup_worker._investigate(
            issue_id="issue-prior",
            scope={"question": "How is error handling done?", "chunk_ids": ["c1", "c2"]},
            prior_analysis=prior_analysis,
            source_code="COBOL CODE HERE",
            is_cross_cutting=False,
        )

        assert answer.issue_id == "issue-prior"
        # Verify LLM was called with context
        assert len(mock_llm._calls) == 1


# ============================================================================
# Worker Base Class Tests
# ============================================================================


class TestWorkerBase:
    """Tests for base worker functionality."""

    def test_worker_id(self, scribe_worker: ScribeWorker) -> None:
        """Test worker has ID."""
        assert scribe_worker.worker_id == "scribe-001"

    def test_worker_adapters(
        self,
        scribe_worker: ScribeWorker,
        mock_ticket_system,
        mock_artifact_store,
        mock_llm,
    ) -> None:
        """Test worker has adapters configured."""
        assert scribe_worker.ticket_system is mock_ticket_system
        assert scribe_worker.artifact_store is mock_artifact_store
        assert scribe_worker.llm is mock_llm


# ============================================================================
# LLM Fixture Integration Tests
# ============================================================================


class TestWithLLMFixtures:
    """Tests using mock LLM response fixtures."""

    @pytest.mark.asyncio
    async def test_scribe_with_fixture_response(
        self,
        scribe_worker: ScribeWorker,
        artifact_ref: ArtifactRef,
        mock_llm,
    ) -> None:
        """Test scribe with response from fixtures."""
        from tests.fixtures.llm_responses import get_mock_response

        # Load fixture response
        fixture = get_mock_response("chunk_analysis", "procedure_division")
        mock_llm._responses = [json.dumps(fixture)]

        payload = DocChunkPayload(
            job_id="job-fixture",
            artifact_ref=artifact_ref,
            manifest_uri="s3://bucket/manifest.json",
            chunk_id="chunk_fixture",
            chunk_locator=ChunkLocator(start_line=1, end_line=200),
            result_uri="s3://bucket/chunks/chunk_fixture.json",
        )

        result = await scribe_worker.analyze_chunk(
            "PROCEDURE DIVISION CONTENT", payload, {}
        )

        # Verify fixture data was used
        assert result.confidence == 0.85
        assert "business logic" in result.summary.lower() or "processing" in result.summary.lower()
        assert len(result.facts.symbols_defined) > 0
        assert len(result.facts.calls) > 0

    @pytest.mark.asyncio
    async def test_challenger_with_fixture_response(
        self,
        challenger_worker: ChallengerWorker,
        mock_llm,
    ) -> None:
        """Test challenger with response from fixtures."""
        from tests.fixtures.llm_responses import get_mock_response

        # Load fixture response
        fixture = get_mock_response("challenge_responses", "major_issues")
        mock_llm._responses = [json.dumps(fixture)]

        doc_model = DocumentationModel(
            doc_uri="s3://bucket/doc.md",
            sections=[Section(section_id="s1", title="Processing")],
            index=DocIndex(),
        )

        result = await challenger_worker.review_documentation(
            "# Documentation", doc_model, "accuracy"
        )

        # Verify fixture issues were parsed
        assert len(result.issues) == 2
        assert any(i.severity == IssueSeverity.MAJOR for i in result.issues)


# ============================================================================
# Finalize Worker Tests
# ============================================================================


class TestFinalizeWorker:
    """Tests for FinalizeWorker."""

    def test_supported_work_types(self, finalize_worker: FinalizeWorker) -> None:
        """Test finalize worker supports correct work types."""
        assert WorkItemType.DOC_FINALIZE in finalize_worker.supported_work_types
        assert len(finalize_worker.supported_work_types) == 1

    @pytest.mark.asyncio
    async def test_process_finalize(
        self,
        finalize_worker: FinalizeWorker,
        artifact_ref: ArtifactRef,
        mock_artifact_store,
        mock_ticket_system,
    ) -> None:
        """Test processing a finalize work item."""
        # Store documentation
        doc_uri = "s3://bucket/doc.md"
        await mock_artifact_store.write_text(doc_uri, "# Test Documentation\n\nContent here.")

        # Store documentation model
        doc_model_uri = "s3://bucket/doc_model.json"
        doc_model = DocumentationModel(
            doc_uri=doc_uri,
            sections=[
                Section(
                    section_id="section-1",
                    title="Overview",
                    source_refs=["chunk_001"],
                ),
            ],
            index=DocIndex(),
            metadata={"artifact_id": "TEST.cbl", "job_id": "job-001"},
        )
        await mock_artifact_store.write_json(doc_model_uri, doc_model.model_dump())

        # Store chunk result
        chunk_result = ChunkResult(
            job_id="job-001",
            artifact_id="TEST001.cbl",
            artifact_version="abc123",
            chunk_id="chunk_001",
            chunk_kind="procedure_part",
            chunk_locator={"start_line": 1, "end_line": 100},
            summary="Main logic",
            facts=ChunkFacts(),
            evidence=[],
            open_questions=[],
            confidence=0.9,
        )

        # Create chunk work item
        chunk_item = WorkItem(
            work_id="job-001-chunk-chunk_001",
            work_type=WorkItemType.DOC_CHUNK,
            status=WorkItemStatus.DONE,
            payload=DocChunkPayload(
                job_id="job-001",
                artifact_ref=artifact_ref,
                manifest_uri="s3://bucket/manifest.json",
                chunk_id="chunk_001",
                chunk_locator=ChunkLocator(start_line=1, end_line=100),
                result_uri="s3://bucket/chunks/chunk_001.json",
            ),
        )
        await mock_ticket_system.create_work_item(chunk_item)
        await mock_artifact_store.write_json(
            "s3://bucket/chunks/chunk_001.json",
            chunk_result.model_dump(),
        )

        # Create challenge work item (done, no issues)
        challenge_item = WorkItem(
            work_id="job-001-challenge-cycle1",
            work_type=WorkItemType.DOC_CHALLENGE,
            status=WorkItemStatus.DONE,
            payload=DocChallengePayload(
                job_id="job-001",
                artifact_ref=artifact_ref,
                manifest_uri="s3://bucket/manifest.json",
                doc_uri=doc_uri,
                doc_model_uri=doc_model_uri,
                challenge_profile="completeness",
                output_uri="s3://bucket/challenges/challenge_1.json",
            ),
            cycle_number=1,
        )
        await mock_ticket_system.create_work_item(challenge_item)

        # Store challenge result
        challenge_result = ChallengeResult(
            job_id="job-001",
            artifact_id="TEST001.cbl",
            artifact_version="abc123",
            doc_uri=doc_uri,
            challenge_profile="completeness",
            issues=[],
            summary="Documentation is complete",
            recommendations=[],
        )
        await mock_artifact_store.write_json(
            "s3://bucket/challenges/challenge_1.json",
            challenge_result.model_dump(),
        )

        # Create finalize payload
        payload = DocFinalizePayload(
            job_id="job-001",
            artifact_ref=artifact_ref,
            manifest_uri="s3://bucket/manifest.json",
            doc_uri=doc_uri,
            doc_model_uri=doc_model_uri,
            output_doc_uri="s3://bucket/final/documentation.md",
            output_trace_uri="s3://bucket/final/trace.json",
            output_summary_uri="s3://bucket/final/summary.json",
            generate_pdf=False,
            cleanup_intermediates=False,
        )

        work_item = WorkItem(
            work_id="job-001-finalize",
            work_type=WorkItemType.DOC_FINALIZE,
            status=WorkItemStatus.IN_PROGRESS,
            payload=payload,
        )

        result = await finalize_worker.process(work_item)

        assert isinstance(result, FinalizeResult)
        assert result.job_id == "job-001"
        assert result.status == "accepted"
        assert result.doc_uri == "s3://bucket/final/documentation.md"
        assert result.trace_uri == "s3://bucket/final/trace.json"
        assert result.summary_uri == "s3://bucket/final/summary.json"
        assert result.pdf_uri is None  # PDF not requested

    @pytest.mark.asyncio
    async def test_generate_trace_report(
        self,
        finalize_worker: FinalizeWorker,
        artifact_ref: ArtifactRef,
        mock_artifact_store,
        mock_ticket_system,
    ) -> None:
        """Test trace report generation."""
        # Set up documentation model
        doc_model = DocumentationModel(
            doc_uri="s3://bucket/doc.md",
            sections=[
                Section(
                    section_id="sec-1",
                    title="Overview",
                    source_refs=["chunk_001", "chunk_002"],
                ),
                Section(
                    section_id="sec-2",
                    title="Details",
                    source_refs=["chunk_003"],
                ),
            ],
            index=DocIndex(),
        )

        # Create chunk results
        chunk_results = [
            ChunkResult(
                job_id="job-001",
                artifact_id="TEST001.cbl",
                artifact_version="abc123",
                chunk_id="chunk_001",
                chunk_kind="procedure_part",
                chunk_locator={"start_line": 1, "end_line": 50},
                summary="Main logic",
                facts=ChunkFacts(),
                evidence=[],
                open_questions=[],
                confidence=0.9,
            ),
            ChunkResult(
                job_id="job-001",
                artifact_id="TEST001.cbl",
                artifact_version="abc123",
                chunk_id="chunk_002",
                chunk_kind="data_section",
                chunk_locator={"start_line": 51, "end_line": 100},
                summary="Data definitions",
                facts=ChunkFacts(),
                evidence=[],
                open_questions=[{"question": "What is WS-VAR?"}],
                confidence=0.8,
            ),
        ]

        # Set up job data
        job_data = {
            "chunks": [
                WorkItem(
                    work_id="chunk_001",
                    work_type=WorkItemType.DOC_CHUNK,
                    status=WorkItemStatus.DONE,
                    payload=DocChunkPayload(
                        job_id="job-001",
                        artifact_ref=artifact_ref,
                        manifest_uri="uri",
                        chunk_id="chunk_001",
                        chunk_locator=ChunkLocator(start_line=1, end_line=50),
                        result_uri="s3://bucket/chunk_001.json",
                    ),
                ),
            ],
            "chunk_results": chunk_results,
            "challenges": [],
            "challenge_results": [],
            "followups": [],
            "patch_merges": [],
            "merges": [],
        }

        payload = DocFinalizePayload(
            job_id="job-001",
            artifact_ref=artifact_ref,
            manifest_uri="s3://bucket/manifest.json",
            doc_uri="s3://bucket/doc.md",
            doc_model_uri="s3://bucket/doc_model.json",
            output_doc_uri="s3://bucket/final/doc.md",
            output_trace_uri="s3://bucket/final/trace.json",
            output_summary_uri="s3://bucket/final/summary.json",
        )

        trace = await finalize_worker._generate_trace_report(
            payload, doc_model, job_data
        )

        assert isinstance(trace, TraceReport)
        assert trace.job_id == "job-001"
        assert len(trace.section_traces) == 2
        assert trace.section_traces[0].section_id == "sec-1"
        assert len(trace.section_traces[0].chunk_contributions) == 2

    def test_generate_statistics(
        self,
        finalize_worker: FinalizeWorker,
        artifact_ref: ArtifactRef,
    ) -> None:
        """Test statistics generation."""
        payload = DocFinalizePayload(
            job_id="job-001",
            artifact_ref=artifact_ref,
            manifest_uri="s3://bucket/manifest.json",
            doc_uri="s3://bucket/doc.md",
            doc_model_uri="s3://bucket/doc_model.json",
            output_doc_uri="s3://bucket/final/doc.md",
            output_trace_uri="s3://bucket/final/trace.json",
            output_summary_uri="s3://bucket/final/summary.json",
        )

        job_data = {
            "chunks": [MagicMock()] * 5,
            "chunk_results": [
                MagicMock(confidence=0.9, chunk_locator={"start_line": 1, "end_line": 50}),
                MagicMock(confidence=0.8, chunk_locator={"start_line": 51, "end_line": 100}),
            ],
            "challenges": [MagicMock(cycle_number=1)],
            "challenge_results": [MagicMock(issues=[])],
            "followups": [],
            "patch_merges": [],
            "merges": [MagicMock()] * 3,
        }

        trace_report = TraceReport(
            job_id="job-001",
            artifact_id="TEST.cbl",
            artifact_version="abc123",
            total_chunks=5,
            total_sections=2,
            total_issues_raised=0,
            total_issues_resolved=0,
            final_cycle=1,
        )

        stats = finalize_worker._generate_statistics(payload, job_data, trace_report)

        assert isinstance(stats, JobStatistics)
        assert stats.job_id == "job-001"
        assert stats.total_chunks == 5
        assert stats.total_merges == 3
        assert stats.challenger_cycles == 1
        assert stats.average_confidence == 0.85
        assert stats.status == "accepted"

    def test_generate_final_doc(
        self,
        finalize_worker: FinalizeWorker,
    ) -> None:
        """Test final Markdown document generation."""
        doc_content = "# Documentation\n\nThis is the content."
        doc_model = DocumentationModel(
            doc_uri="s3://bucket/doc.md",
            sections=[
                Section(section_id="s1", title="Overview"),
                Section(section_id="s2", title="Details"),
            ],
            index=DocIndex(),
        )
        statistics = JobStatistics(
            job_id="job-001",
            artifact_id="TEST.cbl",
            status="accepted",
            total_chunks=10,
            total_merges=5,
            challenger_cycles=2,
            issues_raised=3,
            issues_resolved=3,
            average_confidence=0.85,
            total_lines_analyzed=500,
        )

        result = finalize_worker._generate_final_doc(
            doc_content, doc_model, statistics
        )

        # Check header metadata
        assert "artifact: TEST.cbl" in result
        assert "status: accepted" in result
        assert "confidence: 85.0%" in result

        # Check footer summary
        assert "Total Sections" in result
        assert "Source Lines Analyzed**: 500" in result
        assert "Issues Raised**: 3" in result
        assert "Issues Resolved**: 3" in result
        assert "Atlas Documentation System" in result

        # Check original content preserved
        assert "# Documentation" in result
        assert "This is the content." in result

    @pytest.mark.asyncio
    async def test_output_exists_check(
        self,
        finalize_worker: FinalizeWorker,
        artifact_ref: ArtifactRef,
        mock_artifact_store,
    ) -> None:
        """Test idempotency check for existing outputs."""
        # Store all output artifacts
        await mock_artifact_store.write_text(
            "s3://bucket/final/doc.md", "Final doc"
        )
        await mock_artifact_store.write_json(
            "s3://bucket/final/trace.json", {"job_id": "job-001"}
        )
        await mock_artifact_store.write_json(
            "s3://bucket/final/summary.json", {"job_id": "job-001"}
        )

        payload = DocFinalizePayload(
            job_id="job-001",
            artifact_ref=artifact_ref,
            manifest_uri="s3://bucket/manifest.json",
            doc_uri="s3://bucket/doc.md",
            doc_model_uri="s3://bucket/doc_model.json",
            output_doc_uri="s3://bucket/final/doc.md",
            output_trace_uri="s3://bucket/final/trace.json",
            output_summary_uri="s3://bucket/final/summary.json",
        )

        work_item = WorkItem(
            work_id="job-001-finalize",
            work_type=WorkItemType.DOC_FINALIZE,
            status=WorkItemStatus.IN_PROGRESS,
            payload=payload,
        )

        exists = await finalize_worker._output_exists(work_item)
        assert exists is True

    @pytest.mark.asyncio
    async def test_output_not_exists_partial(
        self,
        finalize_worker: FinalizeWorker,
        artifact_ref: ArtifactRef,
        mock_artifact_store,
    ) -> None:
        """Test idempotency check when outputs are partial."""
        # Only store some outputs
        await mock_artifact_store.write_text(
            "s3://bucket/final/doc.md", "Final doc"
        )
        # Missing trace and summary

        payload = DocFinalizePayload(
            job_id="job-001",
            artifact_ref=artifact_ref,
            manifest_uri="s3://bucket/manifest.json",
            doc_uri="s3://bucket/doc.md",
            doc_model_uri="s3://bucket/doc_model.json",
            output_doc_uri="s3://bucket/final/doc.md",
            output_trace_uri="s3://bucket/final/trace.json",
            output_summary_uri="s3://bucket/final/summary.json",
        )

        work_item = WorkItem(
            work_id="job-001-finalize",
            work_type=WorkItemType.DOC_FINALIZE,
            status=WorkItemStatus.IN_PROGRESS,
            payload=payload,
        )

        exists = await finalize_worker._output_exists(work_item)
        assert exists is False


class TestFinalizeWorkerStatuses:
    """Tests for finalize status determination."""

    def test_status_completed_with_blockers(
        self,
        finalize_worker: FinalizeWorker,
        artifact_ref: ArtifactRef,
    ) -> None:
        """Test status when blockers remain."""
        payload = DocFinalizePayload(
            job_id="job-001",
            artifact_ref=artifact_ref,
            manifest_uri="s3://bucket/manifest.json",
            doc_uri="s3://bucket/doc.md",
            doc_model_uri="s3://bucket/doc_model.json",
            output_doc_uri="s3://bucket/final/doc.md",
            output_trace_uri="s3://bucket/final/trace.json",
            output_summary_uri="s3://bucket/final/summary.json",
        )

        # Mock blocker issues
        blocker_issue = Issue(
            issue_id="blocker-001",
            severity=IssueSeverity.BLOCKER,
            question="Critical issue",
        )
        challenge_result = MagicMock()
        challenge_result.issues = [blocker_issue]

        job_data = {
            "chunks": [],
            "chunk_results": [],
            "challenges": [],
            "challenge_results": [challenge_result],
            "followups": [],  # No follow-ups to resolve the blocker
            "patch_merges": [],
            "merges": [],
        }

        trace_report = TraceReport(
            job_id="job-001",
            artifact_id="TEST.cbl",
            artifact_version="abc123",
            total_issues_raised=1,
            total_issues_resolved=0,
            final_cycle=1,
        )

        stats = finalize_worker._generate_statistics(payload, job_data, trace_report)

        assert stats.blockers_remaining == 1
        assert stats.status == "completed_with_blockers"
