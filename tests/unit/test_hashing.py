"""Unit tests for hashing utilities and idempotency key generation."""

import pytest

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit

from atlas.utils.hashing import (
    compute_content_hash,
    compute_idempotency_key,
    compute_work_item_idempotency_key,
    compute_work_item_key_from_work_item,
    compute_artifact_uri,
    compute_result_uri,
    content_hash_short,
)
from atlas.models.enums import WorkItemType, WorkItemStatus
from atlas.models.work_item import (
    WorkItem,
    DocChunkPayload,
    DocMergePayload,
    DocFollowupPayload,
    DocChallengePayload,
    ChunkLocator,
)
from atlas.models.artifact import ArtifactRef


class TestComputeContentHash:
    """Tests for compute_content_hash function."""

    def test_hash_string(self) -> None:
        """Test hashing a string."""
        result = compute_content_hash("hello world")
        # Known SHA-256 hash of "hello world"
        expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        assert result == expected

    def test_hash_bytes(self) -> None:
        """Test hashing bytes."""
        result = compute_content_hash(b"hello world")
        expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        assert result == expected

    def test_hash_empty(self) -> None:
        """Test hashing empty content."""
        result = compute_content_hash("")
        # SHA-256 of empty string
        expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert result == expected

    def test_hash_deterministic(self) -> None:
        """Test that hashing is deterministic."""
        content = "test content for hashing"
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)
        assert hash1 == hash2

    def test_hash_different_content(self) -> None:
        """Test that different content produces different hashes."""
        hash1 = compute_content_hash("content1")
        hash2 = compute_content_hash("content2")
        assert hash1 != hash2

    def test_hash_length(self) -> None:
        """Test hash is 64 hex characters."""
        result = compute_content_hash("any content")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


class TestComputeIdempotencyKey:
    """Tests for compute_idempotency_key function."""

    def test_basic_key(self) -> None:
        """Test basic idempotency key computation."""
        key = compute_idempotency_key("job-123", "doc_chunk", "abc123", "chunk-001")
        assert len(key) == 64

    def test_key_deterministic(self) -> None:
        """Test that keys are deterministic."""
        key1 = compute_idempotency_key("job-123", "doc_chunk", "abc123")
        key2 = compute_idempotency_key("job-123", "doc_chunk", "abc123")
        assert key1 == key2

    def test_key_different_parts(self) -> None:
        """Test that different parts produce different keys."""
        key1 = compute_idempotency_key("job-123", "doc_chunk", "abc123")
        key2 = compute_idempotency_key("job-456", "doc_chunk", "abc123")
        assert key1 != key2

    def test_key_order_matters(self) -> None:
        """Test that part order affects the key."""
        key1 = compute_idempotency_key("a", "b", "c")
        key2 = compute_idempotency_key("c", "b", "a")
        assert key1 != key2

    def test_key_with_single_part(self) -> None:
        """Test key computation with single part."""
        key = compute_idempotency_key("single")
        assert len(key) == 64

    def test_key_filters_empty_strings(self) -> None:
        """Test that empty strings are filtered."""
        key1 = compute_idempotency_key("a", "", "b")
        key2 = compute_idempotency_key("a", "b")
        assert key1 == key2


class TestComputeWorkItemIdempotencyKey:
    """Tests for compute_work_item_idempotency_key function."""

    def test_chunk_key(self) -> None:
        """Test key for DOC_CHUNK work item."""
        key = compute_work_item_idempotency_key(
            job_id="job-123",
            work_type=WorkItemType.DOC_CHUNK,
            artifact_version="abc123",
            chunk_id="procedure_part_001",
        )
        assert len(key) == 64

    def test_merge_key(self) -> None:
        """Test key for DOC_MERGE work item."""
        key = compute_work_item_idempotency_key(
            job_id="job-123",
            work_type=WorkItemType.DOC_MERGE,
            artifact_version="abc123",
            merge_node_id="merge_procedure",
        )
        assert len(key) == 64

    def test_followup_key(self) -> None:
        """Test key for DOC_FOLLOWUP work item."""
        key = compute_work_item_idempotency_key(
            job_id="job-123",
            work_type=WorkItemType.DOC_FOLLOWUP,
            artifact_version="abc123",
            issue_id="issue-001",
        )
        assert len(key) == 64

    def test_with_string_work_type(self) -> None:
        """Test key with string work type."""
        key1 = compute_work_item_idempotency_key(
            job_id="job-123",
            work_type="doc_chunk",
            artifact_version="abc123",
            chunk_id="chunk-001",
        )
        key2 = compute_work_item_idempotency_key(
            job_id="job-123",
            work_type=WorkItemType.DOC_CHUNK,
            artifact_version="abc123",
            chunk_id="chunk-001",
        )
        assert key1 == key2

    def test_with_cycle_number(self) -> None:
        """Test key with cycle number."""
        key1 = compute_work_item_idempotency_key(
            job_id="job-123",
            work_type=WorkItemType.DOC_CHUNK,
            artifact_version="abc123",
            chunk_id="chunk-001",
            cycle_number=1,
        )
        key2 = compute_work_item_idempotency_key(
            job_id="job-123",
            work_type=WorkItemType.DOC_CHUNK,
            artifact_version="abc123",
            chunk_id="chunk-001",
            cycle_number=2,
        )
        assert key1 != key2

    def test_with_generic_identifier(self) -> None:
        """Test key with generic identifier."""
        key = compute_work_item_idempotency_key(
            job_id="job-123",
            work_type=WorkItemType.DOC_REQUEST,
            artifact_version="abc123",
            identifier="request-001",
        )
        assert len(key) == 64

    def test_deterministic(self) -> None:
        """Test key computation is deterministic."""
        kwargs = {
            "job_id": "job-123",
            "work_type": WorkItemType.DOC_CHUNK,
            "artifact_version": "abc123",
            "chunk_id": "chunk-001",
        }
        key1 = compute_work_item_idempotency_key(**kwargs)
        key2 = compute_work_item_idempotency_key(**kwargs)
        assert key1 == key2


class TestComputeWorkItemKeyFromWorkItem:
    """Tests for compute_work_item_key_from_work_item function."""

    @pytest.fixture
    def sample_artifact_ref(self) -> ArtifactRef:
        """Provide a sample artifact reference."""
        return ArtifactRef(
            artifact_id="TEST.cbl",
            artifact_type="cobol",
            artifact_version="abc123def456",
            artifact_uri="s3://bucket/TEST.cbl",
        )

    def test_chunk_work_item(self, sample_artifact_ref: ArtifactRef) -> None:
        """Test key from DOC_CHUNK work item."""
        work_item = WorkItem(
            work_id="chunk-001",
            work_type=WorkItemType.DOC_CHUNK,
            status=WorkItemStatus.READY,
            payload=DocChunkPayload(
                job_id="job-123",
                artifact_ref=sample_artifact_ref,
                chunk_id="procedure_part_001",
                chunk_locator=ChunkLocator(start_line=1, end_line=100),
                result_uri="s3://results/chunk.json",
            ),
        )
        key = compute_work_item_key_from_work_item(work_item)
        assert len(key) == 64

    def test_merge_work_item(self, sample_artifact_ref: ArtifactRef) -> None:
        """Test key from DOC_MERGE work item."""
        work_item = WorkItem(
            work_id="merge-001",
            work_type=WorkItemType.DOC_MERGE,
            status=WorkItemStatus.BLOCKED,
            payload=DocMergePayload(
                job_id="job-123",
                artifact_ref=sample_artifact_ref,
                merge_node_id="merge_procedure",
                input_uris=["s3://results/chunk1.json", "s3://results/chunk2.json"],
                output_uri="s3://results/merge.json",
            ),
        )
        key = compute_work_item_key_from_work_item(work_item)
        assert len(key) == 64

    def test_followup_work_item(self, sample_artifact_ref: ArtifactRef) -> None:
        """Test key from DOC_FOLLOWUP work item."""
        work_item = WorkItem(
            work_id="followup-001",
            work_type=WorkItemType.DOC_FOLLOWUP,
            status=WorkItemStatus.READY,
            payload=DocFollowupPayload(
                job_id="job-123",
                artifact_ref=sample_artifact_ref,
                issue_id="issue-001",
                scope={"chunk_ids": ["chunk-001"]},
                inputs=["s3://results/chunk1.json"],
                output_uri="s3://results/followup.json",
            ),
        )
        key = compute_work_item_key_from_work_item(work_item)
        assert len(key) == 64

    def test_challenge_work_item(self, sample_artifact_ref: ArtifactRef) -> None:
        """Test key from DOC_CHALLENGE work item."""
        work_item = WorkItem(
            work_id="challenge-001",
            work_type=WorkItemType.DOC_CHALLENGE,
            status=WorkItemStatus.READY,
            payload=DocChallengePayload(
                job_id="job-123",
                artifact_ref=sample_artifact_ref,
                doc_uri="s3://docs/doc.md",
                doc_model_uri="s3://docs/model.json",
                challenge_profile="comprehensive",
                output_uri="s3://results/challenge.json",
            ),
        )
        key = compute_work_item_key_from_work_item(work_item)
        assert len(key) == 64

    def test_key_changes_with_cycle(self, sample_artifact_ref: ArtifactRef) -> None:
        """Test that key changes with cycle number."""
        work_item1 = WorkItem(
            work_id="chunk-001",
            work_type=WorkItemType.DOC_CHUNK,
            status=WorkItemStatus.READY,
            cycle_number=1,
            payload=DocChunkPayload(
                job_id="job-123",
                artifact_ref=sample_artifact_ref,
                chunk_id="procedure_part_001",
                chunk_locator=ChunkLocator(start_line=1, end_line=100),
                result_uri="s3://results/chunk.json",
            ),
        )
        work_item2 = WorkItem(
            work_id="chunk-001",
            work_type=WorkItemType.DOC_CHUNK,
            status=WorkItemStatus.READY,
            cycle_number=2,
            payload=DocChunkPayload(
                job_id="job-123",
                artifact_ref=sample_artifact_ref,
                chunk_id="procedure_part_001",
                chunk_locator=ChunkLocator(start_line=1, end_line=100),
                result_uri="s3://results/chunk.json",
            ),
        )
        key1 = compute_work_item_key_from_work_item(work_item1)
        key2 = compute_work_item_key_from_work_item(work_item2)
        assert key1 != key2


class TestComputeArtifactUri:
    """Tests for compute_artifact_uri function."""

    def test_basic_uri(self) -> None:
        """Test basic URI generation."""
        uri = compute_artifact_uri(
            "s3://bucket/artifacts",
            "DRKBM100.cbl",
            "abc123",
        )
        assert uri == "s3://bucket/artifacts/DRKBM100.cbl@abc123"

    def test_with_suffix(self) -> None:
        """Test URI with suffix."""
        uri = compute_artifact_uri(
            "s3://bucket",
            "TEST.cbl",
            "def456",
            ".json",
        )
        assert uri == "s3://bucket/TEST.cbl@def456.json"

    def test_trailing_slash_handled(self) -> None:
        """Test trailing slash is handled."""
        uri = compute_artifact_uri(
            "s3://bucket/",
            "TEST.cbl",
            "abc",
        )
        assert uri == "s3://bucket/TEST.cbl@abc"

    def test_spaces_in_artifact_id(self) -> None:
        """Test spaces in artifact ID are sanitized."""
        uri = compute_artifact_uri(
            "s3://bucket",
            "my file.cbl",
            "abc",
        )
        assert uri == "s3://bucket/my_file.cbl@abc"


class TestComputeResultUri:
    """Tests for compute_result_uri function."""

    def test_basic_result_uri(self) -> None:
        """Test basic result URI generation."""
        uri = compute_result_uri(
            "s3://results",
            "job-123",
            "chunks",
            "procedure_part_001",
        )
        assert uri == "s3://results/job-123/cycle-1/chunks/procedure_part_001.json"

    def test_with_cycle_number(self) -> None:
        """Test result URI with cycle number."""
        uri = compute_result_uri(
            "s3://results",
            "job-123",
            "merges",
            "merge_root",
            cycle_number=2,
        )
        assert uri == "s3://results/job-123/cycle-2/merges/merge_root.json"

    def test_with_custom_suffix(self) -> None:
        """Test result URI with custom suffix."""
        uri = compute_result_uri(
            "s3://results",
            "job-123",
            "docs",
            "final",
            suffix=".md",
        )
        assert uri == "s3://results/job-123/cycle-1/docs/final.md"


class TestContentHashShort:
    """Tests for content_hash_short function."""

    def test_default_length(self) -> None:
        """Test default short hash length."""
        short_hash = content_hash_short("hello world")
        assert len(short_hash) == 8
        # Should be prefix of full hash
        full_hash = compute_content_hash("hello world")
        assert full_hash.startswith(short_hash)

    def test_custom_length(self) -> None:
        """Test custom short hash length."""
        short_hash = content_hash_short("hello world", length=12)
        assert len(short_hash) == 12

    def test_max_length(self) -> None:
        """Test max length is capped at 64."""
        short_hash = content_hash_short("hello world", length=100)
        assert len(short_hash) == 64

    def test_deterministic(self) -> None:
        """Test short hash is deterministic."""
        hash1 = content_hash_short("test content")
        hash2 = content_hash_short("test content")
        assert hash1 == hash2
