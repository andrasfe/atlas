"""Filesystem-based artifact store adapter.

This adapter provides a local filesystem implementation of the
ArtifactStoreAdapter interface. It stores artifacts as files on disk,
deriving file paths from URIs.

Features:
- URI-based path derivation (file:// scheme or direct paths)
- Content-addressable hashing using SHA-256
- JSON and text convenience methods
- Proper error handling for common filesystem operations
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from atlas.adapters.artifact_store import ArtifactStoreAdapter
from atlas.models.artifact import Artifact


class ArtifactNotFoundError(Exception):
    """Raised when an artifact is not found."""
    pass


class ArtifactWriteError(Exception):
    """Raised when writing an artifact fails."""
    pass


class ArtifactReadError(Exception):
    """Raised when reading an artifact fails."""
    pass


class InvalidURIError(Exception):
    """Raised when a URI cannot be parsed."""
    pass


def _sync_write_file(path: Path, content: bytes) -> None:
    """Synchronous file write helper."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)


def _sync_read_file(path: Path) -> bytes:
    """Synchronous file read helper."""
    with open(path, "rb") as f:
        return f.read()


def _sync_write_text(path: Path, text: str) -> None:
    """Synchronous text write helper."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _sync_read_text(path: Path) -> str:
    """Synchronous text read helper."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _sync_delete_file(path: Path) -> bool:
    """Synchronous file delete helper."""
    if path.exists():
        os.remove(path)
        return True
    return False


class FilesystemArtifactStore(ArtifactStoreAdapter):
    """Filesystem implementation of the artifact store adapter.

    This implementation stores artifacts as files on the local filesystem.
    URIs are mapped to file paths using a configurable base directory.

    URI Schemes Supported:
        - file://path/to/file - Standard file URI
        - /path/to/file - Direct absolute path
        - relative/path - Relative to base_dir

    Thread Safety:
        File operations are run in a thread pool using asyncio.to_thread
        for async compatibility. Concurrent writes to the same file may
        cause issues; ensure proper coordination.

    Example:
        >>> store = FilesystemArtifactStore("/var/atlas/artifacts")
        >>> uri = await store.write("job-123/chunk-001.json", data)
        >>> content = await store.read(uri)
    """

    def __init__(
        self,
        base_dir: str | Path,
        create_dirs: bool = True,
    ) -> None:
        """Initialize the filesystem artifact store.

        Args:
            base_dir: Base directory for storing artifacts.
            create_dirs: Whether to create directories automatically.
        """
        self.base_dir = Path(base_dir).resolve()
        self.create_dirs = create_dirs

        # Create base directory if needed
        if create_dirs:
            self.base_dir.mkdir(parents=True, exist_ok=True)

        # Metadata storage: maps URI to metadata dict
        self._metadata_cache: dict[str, dict[str, str]] = {}

    def _uri_to_path(self, uri: str) -> Path:
        """Convert a URI to a filesystem path.

        Handles various URI formats:
        - file:///absolute/path -> /absolute/path
        - file://relative/path -> base_dir/relative/path
        - /absolute/path -> /absolute/path
        - relative/path -> base_dir/relative/path

        Args:
            uri: The artifact URI.

        Returns:
            Resolved filesystem path.

        Raises:
            InvalidURIError: If URI format is invalid.
        """
        if uri.startswith("file://"):
            # Remove file:// prefix
            path_part = uri[7:]
            if path_part.startswith("/"):
                # Absolute path: file:///absolute/path
                return Path(path_part).resolve()
            else:
                # Relative path: file://relative/path
                return (self.base_dir / path_part).resolve()
        elif uri.startswith("/"):
            # Absolute path without scheme
            return Path(uri).resolve()
        else:
            # Relative path
            return (self.base_dir / uri).resolve()

    def _path_to_uri(self, path: Path) -> str:
        """Convert a filesystem path to a URI.

        Args:
            path: The filesystem path.

        Returns:
            File URI string.
        """
        return f"file://{path}"

    async def write(
        self,
        uri: str,
        content: bytes,
        *,
        content_type: str = "application/json",
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Write content to the filesystem.

        Creates parent directories if they don't exist.

        Args:
            uri: The target URI.
            content: The content to write.
            content_type: MIME type of the content.
            metadata: Optional metadata to store with the artifact.

        Returns:
            The URI where content was written.

        Raises:
            ArtifactWriteError: If write fails.
        """
        path = self._uri_to_path(uri)

        try:
            # Write content in thread pool
            await asyncio.to_thread(_sync_write_file, path, content)

            # Store metadata
            full_metadata = metadata.copy() if metadata else {}
            full_metadata["content_type"] = content_type
            full_metadata["size_bytes"] = str(len(content))
            full_metadata["content_hash"] = await self.compute_hash(content)

            self._metadata_cache[uri] = full_metadata

            # Also write metadata file
            metadata_path = path.with_suffix(path.suffix + ".meta.json")
            metadata_json = json.dumps(full_metadata, indent=2)
            await asyncio.to_thread(_sync_write_text, metadata_path, metadata_json)

            return uri

        except OSError as e:
            raise ArtifactWriteError(f"Failed to write artifact {uri}: {e}") from e

    async def read(self, uri: str) -> bytes:
        """Read content from the filesystem.

        Args:
            uri: The artifact URI.

        Returns:
            The content as bytes.

        Raises:
            ArtifactNotFoundError: If artifact doesn't exist.
            ArtifactReadError: If read fails.
        """
        path = self._uri_to_path(uri)

        if not path.exists():
            raise ArtifactNotFoundError(f"Artifact not found: {uri}")

        try:
            return await asyncio.to_thread(_sync_read_file, path)
        except OSError as e:
            raise ArtifactReadError(f"Failed to read artifact {uri}: {e}") from e

    async def exists(self, uri: str) -> bool:
        """Check if an artifact exists.

        Args:
            uri: The artifact URI.

        Returns:
            True if artifact exists, False otherwise.
        """
        path = self._uri_to_path(uri)
        return path.exists() and path.is_file()

    async def delete(self, uri: str) -> bool:
        """Delete an artifact.

        Also removes the associated metadata file if present.

        Args:
            uri: The artifact URI.

        Returns:
            True if deleted, False if didn't exist.
        """
        path = self._uri_to_path(uri)

        if not path.exists():
            return False

        try:
            # Delete main file
            await asyncio.to_thread(_sync_delete_file, path)

            # Delete metadata file if exists
            metadata_path = path.with_suffix(path.suffix + ".meta.json")
            if metadata_path.exists():
                await asyncio.to_thread(_sync_delete_file, metadata_path)

            # Remove from cache
            self._metadata_cache.pop(uri, None)

            return True

        except OSError:
            return False

    async def get_metadata(self, uri: str) -> Artifact | None:
        """Get artifact metadata.

        Attempts to load from metadata file or cache.

        Args:
            uri: The artifact URI.

        Returns:
            Artifact metadata if exists, None otherwise.
        """
        path = self._uri_to_path(uri)

        if not path.exists():
            return None

        # Try cache first
        if uri in self._metadata_cache:
            meta = self._metadata_cache[uri]
        else:
            # Try to load from metadata file
            metadata_path = path.with_suffix(path.suffix + ".meta.json")
            if metadata_path.exists():
                try:
                    content = await asyncio.to_thread(_sync_read_text, metadata_path)
                    meta = json.loads(content)
                    self._metadata_cache[uri] = meta
                except (OSError, json.JSONDecodeError):
                    meta = {}
            else:
                meta = {}

        # Construct artifact from file info
        artifact_id = path.name
        content_hash = meta.get("content_hash", "")
        if not content_hash:
            # Compute hash if not in metadata
            try:
                content = await self.read(uri)
                content_hash = await self.compute_hash(content)
            except (ArtifactNotFoundError, ArtifactReadError):
                content_hash = "unknown"

        return Artifact(
            artifact_id=artifact_id,
            artifact_type=meta.get("artifact_type", "other"),
            artifact_version=content_hash[:12] if content_hash else "unknown",
            artifact_uri=uri,
            metadata=meta,
        )

    async def list_artifacts(
        self,
        prefix: str,
        limit: int = 1000,
    ) -> list[str]:
        """List artifacts by URI prefix.

        Walks the directory tree under the prefix path.

        Args:
            prefix: URI prefix to filter by.
            limit: Maximum number of results.

        Returns:
            List of artifact URIs.
        """
        base_path = self._uri_to_path(prefix)

        if not base_path.exists():
            return []

        results: list[str] = []

        if base_path.is_file():
            # Prefix points to a single file
            return [prefix]

        # Walk directory tree
        for root, _, files in os.walk(base_path):
            root_path = Path(root)
            for filename in sorted(files):
                # Skip metadata files
                if filename.endswith(".meta.json"):
                    continue

                file_path = root_path / filename
                relative_path = file_path.relative_to(self.base_dir)
                uri = str(relative_path)

                results.append(uri)

                if len(results) >= limit:
                    return results

        return results

    async def compute_hash(self, content: bytes) -> str:
        """Compute SHA-256 content hash for versioning.

        Args:
            content: The content to hash.

        Returns:
            SHA-256 hex digest string.
        """
        return hashlib.sha256(content).hexdigest()

    # Convenience methods inherited from base class use the above methods

    async def write_json(
        self,
        uri: str,
        data: dict[str, Any],
        *,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Write JSON data to the artifact store.

        Convenience method for structured data.

        Args:
            uri: The target URI.
            data: The data to serialize as JSON.
            metadata: Optional metadata.

        Returns:
            The URI where content was written.
        """
        content = json.dumps(data, indent=2).encode("utf-8")
        return await self.write(
            uri,
            content,
            content_type="application/json",
            metadata=metadata,
        )

    async def read_json(self, uri: str) -> dict[str, Any]:
        """Read JSON data from the artifact store.

        Convenience method for structured data.

        Args:
            uri: The artifact URI.

        Returns:
            Parsed JSON data.

        Raises:
            ArtifactNotFoundError: If artifact doesn't exist.
            json.JSONDecodeError: If content is not valid JSON.
        """
        content = await self.read(uri)
        return json.loads(content.decode("utf-8"))

    async def write_text(
        self,
        uri: str,
        text: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Write text content to the artifact store.

        Args:
            uri: The target URI.
            text: The text content.
            metadata: Optional metadata.

        Returns:
            The URI where content was written.
        """
        return await self.write(
            uri,
            text.encode("utf-8"),
            content_type="text/plain",
            metadata=metadata,
        )

    async def read_text(self, uri: str) -> str:
        """Read text content from the artifact store.

        Args:
            uri: The artifact URI.

        Returns:
            Text content.
        """
        content = await self.read(uri)
        return content.decode("utf-8")

    # Additional utility methods

    def clear_metadata_cache(self) -> None:
        """Clear the in-memory metadata cache.

        Useful for testing or when metadata files may have been modified
        externally.
        """
        self._metadata_cache.clear()

    @property
    def root_path(self) -> Path:
        """Get the root storage path.

        Returns:
            The base directory path.
        """
        return self.base_dir
