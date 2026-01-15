"""Abstract base class for artifact store adapters.

Artifacts are the source of truth in Atlas. The artifact store
provides versioned storage for inputs (source code) and outputs
(chunk results, merge results, documentation).

Key Requirements:
- Versioned storage with content hashing
- Deterministic URIs for idempotent writes
- Efficient retrieval by URI
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from atlas.models.artifact import Artifact


class ArtifactStoreAdapter(ABC):
    """Abstract interface for artifact storage.

    Implementations should handle:
    - Versioned storage (content-addressable or explicit versioning)
    - URI generation and resolution
    - Content type handling (JSON, text, binary)

    Design Principle:
        Artifacts are the source of truth; tickets are pointers.
        Write outputs to deterministic URIs derived from payloads.
        If output already exists and passes validation, reuse it.

    Example Implementation:
        >>> class S3ArtifactStore(ArtifactStoreAdapter):
        ...     async def write(self, uri: str, content: bytes) -> str:
        ...         bucket, key = self._parse_uri(uri)
        ...         await self.s3_client.put_object(Bucket=bucket, Key=key, Body=content)
        ...         return uri

    TODO: Implement concrete adapters for:
        - S3/GCS/Azure Blob
        - Local filesystem
        - In-memory (for testing)
    """

    @abstractmethod
    async def write(
        self,
        uri: str,
        content: bytes,
        *,
        content_type: str = "application/json",
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Write content to the artifact store.

        Should be idempotent - writing the same content to the same
        URI should succeed without error.

        Args:
            uri: The target URI.
            content: The content to write.
            content_type: MIME type of the content.
            metadata: Optional metadata to store with the artifact.

        Returns:
            The URI where content was written (may include version).

        Raises:
            ArtifactWriteError: If write fails.

        TODO: Implement with idempotent write semantics.
        """
        pass

    @abstractmethod
    async def read(self, uri: str) -> bytes:
        """Read content from the artifact store.

        Args:
            uri: The artifact URI.

        Returns:
            The content as bytes.

        Raises:
            ArtifactNotFoundError: If artifact doesn't exist.
            ArtifactReadError: If read fails.

        TODO: Implement with appropriate error handling.
        """
        pass

    @abstractmethod
    async def exists(self, uri: str) -> bool:
        """Check if an artifact exists.

        Args:
            uri: The artifact URI.

        Returns:
            True if artifact exists, False otherwise.

        TODO: Implement with efficient existence check.
        """
        pass

    @abstractmethod
    async def delete(self, uri: str) -> bool:
        """Delete an artifact.

        Args:
            uri: The artifact URI.

        Returns:
            True if deleted, False if didn't exist.

        TODO: Implement with appropriate error handling.
        """
        pass

    @abstractmethod
    async def get_metadata(self, uri: str) -> Artifact | None:
        """Get artifact metadata.

        Args:
            uri: The artifact URI.

        Returns:
            Artifact metadata if exists, None otherwise.

        TODO: Implement metadata retrieval.
        """
        pass

    @abstractmethod
    async def list_artifacts(
        self,
        prefix: str,
        limit: int = 1000,
    ) -> list[str]:
        """List artifacts by URI prefix.

        Args:
            prefix: URI prefix to filter by.
            limit: Maximum number of results.

        Returns:
            List of artifact URIs.

        TODO: Implement with efficient listing.
        """
        pass

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
        import json
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
            JSONDecodeError: If content is not valid JSON.
        """
        import json
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

    @abstractmethod
    async def compute_hash(self, content: bytes) -> str:
        """Compute content hash for versioning.

        Args:
            content: The content to hash.

        Returns:
            Hash string (e.g., SHA-256 hex digest).

        TODO: Implement with consistent hashing algorithm.
        """
        pass

    def generate_uri(
        self,
        base_uri: str,
        path_template: str,
        **kwargs: str,
    ) -> str:
        """Generate a deterministic URI from a template.

        Args:
            base_uri: Base URI (e.g., "s3://bucket/job-123").
            path_template: Path template with placeholders.
            **kwargs: Values to substitute in template.

        Returns:
            Complete URI.

        Example:
            >>> store.generate_uri(
            ...     "s3://bucket/job-123",
            ...     "chunks/{chunk_id}.json",
            ...     chunk_id="chunk-001"
            ... )
            "s3://bucket/job-123/chunks/chunk-001.json"
        """
        path = path_template.format(**kwargs)
        return f"{base_uri.rstrip('/')}/{path.lstrip('/')}"
