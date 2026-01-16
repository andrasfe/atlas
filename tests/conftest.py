"""Pytest configuration and shared fixtures.

This module provides fixtures for testing Atlas components:
- Mock adapters (ticket system, artifact store)
- Sample data (artifacts, work items, manifests)
- Test utilities

Note: Atlas does NOT include LLM mocks. The integrating system provides
workers that wrap their own agents with LLM access.
"""

import pytest
from typing import Any, AsyncIterator

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
from atlas.models.work_item import WorkItem, DocChunkPayload, ChunkLocator
from atlas.models.enums import (
    WorkItemStatus,
    WorkItemType,
    ArtifactType,
    ChunkKind,
)
from atlas.adapters.ticket_system import TicketSystemAdapter
from atlas.adapters.artifact_store import ArtifactStoreAdapter


# -----------------------------------------------------------------------------
# Mock Adapters
# -----------------------------------------------------------------------------


class MockTicketSystem(TicketSystemAdapter):
    """In-memory ticket system for testing.

    TODO: Implement all abstract methods for testing.
    """

    def __init__(self) -> None:
        self._items: dict[str, WorkItem] = {}
        self._claims: dict[str, str] = {}  # work_id -> worker_id

    async def create_work_item(self, work_item: WorkItem) -> str:
        self._items[work_item.work_id] = work_item
        return work_item.work_id

    async def get_work_item(self, work_id: str) -> WorkItem | None:
        return self._items.get(work_id)

    async def update_status(
        self,
        work_id: str,
        new_status: WorkItemStatus,
        *,
        expected_status: WorkItemStatus | None = None,
    ) -> bool:
        item = self._items.get(work_id)
        if item is None:
            return False
        if expected_status and item.status != expected_status:
            return False
        item.status = new_status
        return True

    async def claim_work_item(
        self,
        work_id: str,
        worker_id: str,
        lease_duration_seconds: int = 300,
    ) -> bool:
        if work_id in self._claims:
            return False
        item = self._items.get(work_id)
        if item is None or item.status != WorkItemStatus.READY:
            return False
        self._claims[work_id] = worker_id
        item.status = WorkItemStatus.IN_PROGRESS
        return True

    async def release_work_item(
        self,
        work_id: str,
        worker_id: str,
        new_status: WorkItemStatus = WorkItemStatus.READY,
    ) -> bool:
        if self._claims.get(work_id) != worker_id:
            return False
        del self._claims[work_id]
        item = self._items.get(work_id)
        if item:
            item.status = new_status
        return True

    async def query_by_status(
        self,
        status: WorkItemStatus,
        work_type: WorkItemType | None = None,
        job_id: str | None = None,
        limit: int = 100,
    ) -> list[WorkItem]:
        results = []
        for item in self._items.values():
            if item.status != status:
                continue
            if work_type and item.work_type != work_type:
                continue
            if job_id and item.payload.job_id != job_id:
                continue
            results.append(item)
            if len(results) >= limit:
                break
        return results

    async def query_by_job(
        self,
        job_id: str,
        work_type: WorkItemType | None = None,
    ) -> list[WorkItem]:
        results = []
        for item in self._items.values():
            if item.payload.job_id != job_id:
                continue
            if work_type and item.work_type != work_type:
                continue
            results.append(item)
        return results

    async def get_ready_work_items(
        self,
        work_type: WorkItemType | None = None,
        limit: int = 10,
    ) -> list[WorkItem]:
        return await self.query_by_status(WorkItemStatus.READY, work_type, limit=limit)

    async def check_dependencies_done(self, work_id: str) -> bool:
        item = self._items.get(work_id)
        if item is None:
            return False
        for dep_id in item.depends_on:
            dep = self._items.get(dep_id)
            if dep is None or dep.status != WorkItemStatus.DONE:
                return False
        return True

    async def find_by_idempotency_key(self, idempotency_key: str) -> WorkItem | None:
        for item in self._items.values():
            if item.idempotency_key == idempotency_key:
                return item
        return None


class MockArtifactStore(ArtifactStoreAdapter):
    """In-memory artifact store for testing.

    TODO: Implement all abstract methods for testing.
    """

    def __init__(self) -> None:
        self._artifacts: dict[str, bytes] = {}
        self._metadata: dict[str, dict[str, str]] = {}

    async def write(
        self,
        uri: str,
        content: bytes,
        *,
        content_type: str = "application/json",
        metadata: dict[str, str] | None = None,
    ) -> str:
        self._artifacts[uri] = content
        self._metadata[uri] = metadata or {}
        return uri

    async def read(self, uri: str) -> bytes:
        if uri not in self._artifacts:
            raise FileNotFoundError(f"Artifact not found: {uri}")
        return self._artifacts[uri]

    async def exists(self, uri: str) -> bool:
        return uri in self._artifacts

    async def delete(self, uri: str) -> bool:
        if uri in self._artifacts:
            del self._artifacts[uri]
            self._metadata.pop(uri, None)
            return True
        return False

    async def get_metadata(self, uri: str) -> Artifact | None:
        if uri not in self._artifacts:
            return None
        return Artifact(
            artifact_id=uri.split("/")[-1],
            artifact_type="other",
            artifact_version="test",
            artifact_uri=uri,
            metadata=self._metadata.get(uri, {}),
        )

    async def list_artifacts(self, prefix: str, limit: int = 1000) -> list[str]:
        return [uri for uri in self._artifacts if uri.startswith(prefix)][:limit]

    async def compute_hash(self, content: bytes) -> str:
        import hashlib
        return hashlib.sha256(content).hexdigest()


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_ticket_system() -> MockTicketSystem:
    """Provide a mock ticket system."""
    return MockTicketSystem()


@pytest.fixture
def mock_artifact_store() -> MockArtifactStore:
    """Provide a mock artifact store."""
    return MockArtifactStore()


@pytest.fixture
def sample_artifact() -> Artifact:
    """Provide a sample artifact."""
    return Artifact(
        artifact_id="TEST001.cbl",
        artifact_type=ArtifactType.COBOL,
        artifact_version="abc123def456",
        artifact_uri="s3://test-bucket/sources/TEST001.cbl@abc123def456",
        metadata={"size_bytes": 5000, "lines": 150},
    )


@pytest.fixture
def sample_artifact_ref(sample_artifact: Artifact) -> ArtifactRef:
    """Provide a sample artifact reference."""
    return sample_artifact.to_ref()


@pytest.fixture
def sample_chunk_spec() -> ChunkSpec:
    """Provide a sample chunk specification."""
    return ChunkSpec(
        chunk_id="test001_cbl_procedure_part_001",
        chunk_kind=ChunkKind.PROCEDURE_PART,
        start_line=100,
        end_line=200,
        division="PROCEDURE",
        paragraphs=["MAIN-LOGIC", "PROCESS-RECORD"],
        estimated_tokens=2500,
        result_uri="s3://test-bucket/results/chunks/procedure_part_001.json",
    )


@pytest.fixture
def sample_merge_node() -> MergeNode:
    """Provide a sample merge node."""
    return MergeNode(
        merge_node_id="merge_procedure",
        input_ids=["procedure_part_001", "procedure_part_002"],
        is_root=False,
        level=1,
        output_uri="s3://test-bucket/results/merges/merge_procedure.json",
    )


@pytest.fixture
def sample_manifest(
    sample_artifact_ref: ArtifactRef,
    sample_chunk_spec: ChunkSpec,
    sample_merge_node: MergeNode,
) -> Manifest:
    """Provide a sample manifest."""
    root_merge = MergeNode(
        merge_node_id="merge_root",
        input_ids=["merge_procedure", "merge_data"],
        is_root=True,
        level=2,
    )
    return Manifest(
        job_id="test-job-001",
        artifact_ref=sample_artifact_ref,
        analysis_profile=AnalysisProfile(name="test"),
        splitter_profile=SplitterProfile(name="test"),
        context_budget=4000,
        chunks=[sample_chunk_spec],
        merge_dag=[sample_merge_node, root_merge],
        review_policy=ReviewPolicy(),
        artifacts=ArtifactOutputConfig(base_uri="s3://test-bucket/results"),
    )


@pytest.fixture
def sample_work_item(sample_artifact_ref: ArtifactRef) -> WorkItem:
    """Provide a sample work item."""
    return WorkItem(
        work_id="chunk-001",
        work_type=WorkItemType.DOC_CHUNK,
        status=WorkItemStatus.READY,
        payload=DocChunkPayload(
            job_id="test-job-001",
            artifact_ref=sample_artifact_ref,
            manifest_uri="s3://test-bucket/manifests/test-job-001.json",
            chunk_id="procedure_part_001",
            chunk_locator=ChunkLocator(start_line=100, end_line=200),
            result_uri="s3://test-bucket/results/chunks/procedure_part_001.json",
        ),
    )
