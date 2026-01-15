"""Unit tests for the filesystem artifact store adapter.

These tests verify the FilesystemArtifactStore implementation including:
- Write and read operations
- URI to path conversion
- JSON and text convenience methods
- Metadata handling
- Error handling
"""

import json
import pytest
import tempfile
from pathlib import Path

from atlas.adapters.filesystem_store import (
    FilesystemArtifactStore,
    ArtifactNotFoundError,
    ArtifactWriteError,
    ArtifactReadError,
)


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def artifact_store(temp_dir: Path) -> FilesystemArtifactStore:
    """Provide a filesystem artifact store for testing."""
    return FilesystemArtifactStore(temp_dir)


class TestWriteRead:
    """Tests for write and read operations."""

    @pytest.mark.asyncio
    async def test_write_and_read_bytes(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test writing and reading bytes."""
        content = b"Hello, World!"
        uri = "test/hello.txt"

        returned_uri = await artifact_store.write(uri, content)
        assert returned_uri == uri

        read_content = await artifact_store.read(uri)
        assert read_content == content

    @pytest.mark.asyncio
    async def test_write_creates_directories(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test that write creates parent directories."""
        content = b"nested content"
        uri = "deep/nested/path/file.txt"

        await artifact_store.write(uri, content)

        path = artifact_store._uri_to_path(uri)
        assert path.exists()

    @pytest.mark.asyncio
    async def test_write_idempotent(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test that writing same content is idempotent."""
        content = b"idempotent content"
        uri = "test/idempotent.txt"

        await artifact_store.write(uri, content)
        await artifact_store.write(uri, content)

        read_content = await artifact_store.read(uri)
        assert read_content == content

    @pytest.mark.asyncio
    async def test_read_nonexistent_raises(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test that reading nonexistent artifact raises error."""
        with pytest.raises(ArtifactNotFoundError):
            await artifact_store.read("nonexistent/file.txt")


class TestURIConversion:
    """Tests for URI to path conversion."""

    @pytest.mark.asyncio
    async def test_relative_uri(
        self,
        artifact_store: FilesystemArtifactStore,
        temp_dir: Path,
    ) -> None:
        """Test relative URI conversion."""
        uri = "relative/path/file.txt"
        path = artifact_store._uri_to_path(uri)

        assert path == (temp_dir / "relative/path/file.txt").resolve()

    @pytest.mark.asyncio
    async def test_file_scheme_relative(
        self,
        artifact_store: FilesystemArtifactStore,
        temp_dir: Path,
    ) -> None:
        """Test file:// scheme with relative path."""
        uri = "file://relative/path/file.txt"
        path = artifact_store._uri_to_path(uri)

        assert path == (temp_dir / "relative/path/file.txt").resolve()

    @pytest.mark.asyncio
    async def test_file_scheme_absolute(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test file:// scheme with absolute path."""
        uri = "file:///absolute/path/file.txt"
        path = artifact_store._uri_to_path(uri)

        assert path == Path("/absolute/path/file.txt").resolve()

    @pytest.mark.asyncio
    async def test_absolute_path_without_scheme(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test absolute path without file:// scheme."""
        uri = "/absolute/path/file.txt"
        path = artifact_store._uri_to_path(uri)

        assert path == Path("/absolute/path/file.txt").resolve()


class TestJSONOperations:
    """Tests for JSON convenience methods."""

    @pytest.mark.asyncio
    async def test_write_read_json(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test JSON write and read."""
        data = {
            "name": "test",
            "values": [1, 2, 3],
            "nested": {"key": "value"},
        }
        uri = "test/data.json"

        await artifact_store.write_json(uri, data)
        read_data = await artifact_store.read_json(uri)

        assert read_data == data

    @pytest.mark.asyncio
    async def test_write_json_creates_valid_json(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test that write_json creates valid JSON."""
        data = {"key": "value"}
        uri = "test/valid.json"

        await artifact_store.write_json(uri, data)
        raw_content = await artifact_store.read(uri)

        # Should be valid JSON
        parsed = json.loads(raw_content.decode("utf-8"))
        assert parsed == data

    @pytest.mark.asyncio
    async def test_read_json_invalid_raises(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test that read_json raises on invalid JSON."""
        await artifact_store.write("test/invalid.json", b"not json content")

        with pytest.raises(json.JSONDecodeError):
            await artifact_store.read_json("test/invalid.json")


class TestTextOperations:
    """Tests for text convenience methods."""

    @pytest.mark.asyncio
    async def test_write_read_text(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test text write and read."""
        text = "Hello, this is text content.\nWith multiple lines."
        uri = "test/text.txt"

        await artifact_store.write_text(uri, text)
        read_text = await artifact_store.read_text(uri)

        assert read_text == text

    @pytest.mark.asyncio
    async def test_write_text_unicode(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test text with unicode characters."""
        text = "Unicode: caf\u00e9, \u4e2d\u6587, \U0001f600"
        uri = "test/unicode.txt"

        await artifact_store.write_text(uri, text)
        read_text = await artifact_store.read_text(uri)

        assert read_text == text


class TestExists:
    """Tests for existence checking."""

    @pytest.mark.asyncio
    async def test_exists_true(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test exists returns True for existing artifact."""
        await artifact_store.write("test/exists.txt", b"content")

        assert await artifact_store.exists("test/exists.txt") is True

    @pytest.mark.asyncio
    async def test_exists_false(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test exists returns False for nonexistent artifact."""
        assert await artifact_store.exists("test/notexists.txt") is False


class TestDelete:
    """Tests for delete operations."""

    @pytest.mark.asyncio
    async def test_delete_existing(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test deleting existing artifact."""
        await artifact_store.write("test/delete.txt", b"content")

        result = await artifact_store.delete("test/delete.txt")

        assert result is True
        assert await artifact_store.exists("test/delete.txt") is False

    @pytest.mark.asyncio
    async def test_delete_nonexistent(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test deleting nonexistent artifact returns False."""
        result = await artifact_store.delete("test/notexists.txt")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_removes_metadata(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test that delete removes metadata file."""
        await artifact_store.write(
            "test/with_meta.txt",
            b"content",
            metadata={"key": "value"},
        )

        # Verify metadata file exists
        path = artifact_store._uri_to_path("test/with_meta.txt")
        meta_path = path.with_suffix(".txt.meta.json")
        assert meta_path.exists()

        # Delete
        await artifact_store.delete("test/with_meta.txt")

        # Metadata file should be gone
        assert not meta_path.exists()


class TestMetadata:
    """Tests for metadata operations."""

    @pytest.mark.asyncio
    async def test_get_metadata_existing(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test getting metadata for existing artifact."""
        await artifact_store.write(
            "test/meta.txt",
            b"content",
            metadata={"custom": "value"},
        )

        artifact = await artifact_store.get_metadata("test/meta.txt")

        assert artifact is not None
        assert artifact.artifact_id == "meta.txt"
        assert artifact.metadata.get("custom") == "value"

    @pytest.mark.asyncio
    async def test_get_metadata_nonexistent(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test getting metadata for nonexistent artifact."""
        artifact = await artifact_store.get_metadata("test/notexists.txt")
        assert artifact is None

    @pytest.mark.asyncio
    async def test_metadata_includes_content_hash(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test that metadata includes content hash."""
        content = b"hash me"
        await artifact_store.write("test/hash.txt", content)

        artifact = await artifact_store.get_metadata("test/hash.txt")

        assert artifact is not None
        expected_hash = await artifact_store.compute_hash(content)
        assert artifact.metadata.get("content_hash") == expected_hash


class TestListArtifacts:
    """Tests for listing artifacts."""

    @pytest.mark.asyncio
    async def test_list_by_prefix(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test listing artifacts by prefix."""
        # Create multiple artifacts
        await artifact_store.write("job1/chunk1.json", b"{}")
        await artifact_store.write("job1/chunk2.json", b"{}")
        await artifact_store.write("job2/chunk1.json", b"{}")

        job1_artifacts = await artifact_store.list_artifacts("job1")

        assert len(job1_artifacts) == 2
        assert all("job1" in uri for uri in job1_artifacts)

    @pytest.mark.asyncio
    async def test_list_empty_prefix(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test listing with nonexistent prefix."""
        artifacts = await artifact_store.list_artifacts("nonexistent")
        assert artifacts == []

    @pytest.mark.asyncio
    async def test_list_respects_limit(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test that list respects limit."""
        # Create multiple artifacts
        for i in range(10):
            await artifact_store.write(f"test/file{i}.txt", b"content")

        artifacts = await artifact_store.list_artifacts("test", limit=5)

        assert len(artifacts) == 5

    @pytest.mark.asyncio
    async def test_list_excludes_metadata_files(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test that list excludes .meta.json files."""
        await artifact_store.write(
            "test/data.json",
            b"{}",
            metadata={"key": "value"},
        )

        artifacts = await artifact_store.list_artifacts("test")

        # Should only have the main file, not the metadata file
        assert len(artifacts) == 1
        assert "meta.json" not in artifacts[0]


class TestComputeHash:
    """Tests for hash computation."""

    @pytest.mark.asyncio
    async def test_compute_hash_deterministic(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test that hash is deterministic."""
        content = b"hash me"

        hash1 = await artifact_store.compute_hash(content)
        hash2 = await artifact_store.compute_hash(content)

        assert hash1 == hash2

    @pytest.mark.asyncio
    async def test_compute_hash_different_content(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test that different content produces different hash."""
        hash1 = await artifact_store.compute_hash(b"content1")
        hash2 = await artifact_store.compute_hash(b"content2")

        assert hash1 != hash2

    @pytest.mark.asyncio
    async def test_compute_hash_format(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test hash format is SHA-256 hex."""
        content = b"test"
        hash_value = await artifact_store.compute_hash(content)

        # SHA-256 produces 64 character hex string
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)


class TestGenerateURI:
    """Tests for URI generation."""

    @pytest.mark.asyncio
    async def test_generate_uri_simple(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test simple URI generation."""
        uri = artifact_store.generate_uri(
            "s3://bucket/job-123",
            "chunks/{chunk_id}.json",
            chunk_id="chunk-001",
        )

        assert uri == "s3://bucket/job-123/chunks/chunk-001.json"

    @pytest.mark.asyncio
    async def test_generate_uri_strips_slashes(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test that URI generation handles trailing/leading slashes."""
        uri = artifact_store.generate_uri(
            "s3://bucket/job-123/",
            "/chunks/{chunk_id}.json",
            chunk_id="chunk-001",
        )

        assert uri == "s3://bucket/job-123/chunks/chunk-001.json"


class TestUtilities:
    """Tests for utility methods."""

    @pytest.mark.asyncio
    async def test_root_path(
        self,
        artifact_store: FilesystemArtifactStore,
        temp_dir: Path,
    ) -> None:
        """Test root_path property."""
        assert artifact_store.root_path == temp_dir

    @pytest.mark.asyncio
    async def test_clear_metadata_cache(
        self,
        artifact_store: FilesystemArtifactStore,
    ) -> None:
        """Test clearing metadata cache."""
        await artifact_store.write(
            "test/cached.txt",
            b"content",
            metadata={"key": "value"},
        )

        # Access metadata to populate cache
        await artifact_store.get_metadata("test/cached.txt")
        assert "test/cached.txt" in artifact_store._metadata_cache

        # Clear cache
        artifact_store.clear_metadata_cache()
        assert len(artifact_store._metadata_cache) == 0
