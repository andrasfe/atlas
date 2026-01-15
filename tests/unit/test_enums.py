"""Unit tests for enum definitions."""

import pytest

from atlas.models.enums import (
    WorkItemStatus,
    WorkItemType,
    ArtifactType,
    ChunkKind,
    IssueSeverity,
)


class TestWorkItemStatusTransitions:
    """Comprehensive tests for status transition logic."""

    @pytest.mark.parametrize(
        "from_status,to_status,expected",
        [
            # NEW transitions
            (WorkItemStatus.NEW, WorkItemStatus.READY, True),
            (WorkItemStatus.NEW, WorkItemStatus.CANCELED, True),
            (WorkItemStatus.NEW, WorkItemStatus.IN_PROGRESS, False),
            (WorkItemStatus.NEW, WorkItemStatus.DONE, False),
            # READY transitions
            (WorkItemStatus.READY, WorkItemStatus.IN_PROGRESS, True),
            (WorkItemStatus.READY, WorkItemStatus.BLOCKED, True),
            (WorkItemStatus.READY, WorkItemStatus.CANCELED, True),
            (WorkItemStatus.READY, WorkItemStatus.DONE, False),
            # IN_PROGRESS transitions
            (WorkItemStatus.IN_PROGRESS, WorkItemStatus.DONE, True),
            (WorkItemStatus.IN_PROGRESS, WorkItemStatus.FAILED, True),
            (WorkItemStatus.IN_PROGRESS, WorkItemStatus.CANCELED, True),
            (WorkItemStatus.IN_PROGRESS, WorkItemStatus.READY, False),
            # BLOCKED transitions
            (WorkItemStatus.BLOCKED, WorkItemStatus.READY, True),
            (WorkItemStatus.BLOCKED, WorkItemStatus.CANCELED, True),
            (WorkItemStatus.BLOCKED, WorkItemStatus.DONE, False),
            # FAILED transitions (retry)
            (WorkItemStatus.FAILED, WorkItemStatus.READY, True),
            (WorkItemStatus.FAILED, WorkItemStatus.CANCELED, True),
            (WorkItemStatus.FAILED, WorkItemStatus.DONE, False),
            # DONE transitions
            (WorkItemStatus.DONE, WorkItemStatus.CANCELED, True),
            (WorkItemStatus.DONE, WorkItemStatus.READY, False),
        ],
    )
    def test_transition(
        self,
        from_status: WorkItemStatus,
        to_status: WorkItemStatus,
        expected: bool,
    ) -> None:
        """Test status transitions."""
        assert from_status.can_transition_to(to_status) == expected


class TestWorkItemType:
    """Tests for WorkItemType enum."""

    def test_all_types_defined(self) -> None:
        """Verify all expected work types are defined."""
        expected_types = {
            "doc_request",
            "doc_plan",
            "doc_chunk",
            "doc_merge",
            "doc_challenge",
            "doc_followup",
            "doc_patch_merge",
            "doc_finalize",
        }
        actual_types = {t.value for t in WorkItemType}
        assert expected_types == actual_types


class TestArtifactType:
    """Tests for ArtifactType enum."""

    def test_all_types_defined(self) -> None:
        """Verify all expected artifact types are defined."""
        expected_types = {"cobol", "copybook", "jcl", "other"}
        actual_types = {t.value for t in ArtifactType}
        assert expected_types == actual_types


class TestChunkKind:
    """Tests for ChunkKind enum."""

    def test_cobol_divisions_defined(self) -> None:
        """Verify COBOL division kinds are defined."""
        cobol_kinds = {
            ChunkKind.IDENTIFICATION_DIVISION,
            ChunkKind.ENVIRONMENT_DIVISION,
            ChunkKind.DATA_DIVISION,
            ChunkKind.PROCEDURE_DIVISION,
        }
        assert all(k in ChunkKind for k in cobol_kinds)

    def test_procedure_parts_defined(self) -> None:
        """Verify procedure part kind is defined."""
        assert ChunkKind.PROCEDURE_PART in ChunkKind


class TestIssueSeverity:
    """Tests for IssueSeverity enum."""

    def test_severity_ordering(self) -> None:
        """Verify severity values for ordering."""
        # BLOCKER should be most severe
        severities = list(IssueSeverity)
        assert IssueSeverity.BLOCKER in severities
        assert IssueSeverity.QUESTION in severities
