"""Abstract base class for ticket system adapters.

Ticket systems vary. This adapter interface allows the controller
to work with any ticket system (Jira, Linear, custom, etc.) by
implementing the required operations.

Key Requirements:
- Support for leasing/claiming work items
- Atomic status transitions
- Idempotent ticket creation (via idempotency keys)
- Query by status and type
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from atlas.models.enums import WorkItemStatus, WorkItemType
from atlas.models.work_item import WorkItem, WorkItemPayload


class TicketSystemAdapter(ABC):
    """Abstract interface for ticket system integration.

    Implementations should handle:
    - Mapping canonical statuses to system-specific statuses
    - Implementing lease/claim semantics or optimistic locking
    - Ensuring idempotent ticket creation

    Design Principle:
        Works with any ticket system via adapters; no hard-coded ticket schema.
        Tickets are pointers to artifacts - keep payloads small.

    Example Implementation:
        >>> class JiraAdapter(TicketSystemAdapter):
        ...     async def create_work_item(self, work_item: WorkItem) -> str:
        ...         # Map to Jira issue and create
        ...         jira_issue = self._to_jira_issue(work_item)
        ...         result = await self.jira_client.create_issue(jira_issue)
        ...         return result.key

    TODO: Implement concrete adapters for:
        - Jira
        - Linear
        - In-memory (for testing)
        - SQLite-based (for simple deployments)
    """

    @abstractmethod
    async def create_work_item(
        self,
        work_item: WorkItem,
    ) -> str:
        """Create a new work item in the ticket system.

        If idempotency_key is set on the work item and a matching item
        exists, this should return the existing item's ID without
        creating a duplicate.

        Args:
            work_item: The work item to create.

        Returns:
            The work ID assigned by the ticket system.

        Raises:
            DuplicateWorkItemError: If idempotency check fails unexpectedly.

        TODO: Implement with appropriate error handling and idempotency checks.
        """
        pass

    @abstractmethod
    async def get_work_item(self, work_id: str) -> WorkItem | None:
        """Retrieve a work item by ID.

        Args:
            work_id: The work item identifier.

        Returns:
            The WorkItem if found, None otherwise.

        TODO: Implement with proper mapping from system-specific format.
        """
        pass

    @abstractmethod
    async def update_status(
        self,
        work_id: str,
        new_status: WorkItemStatus,
        *,
        expected_status: WorkItemStatus | None = None,
    ) -> bool:
        """Update the status of a work item.

        Should implement optimistic locking if expected_status is provided.

        Args:
            work_id: The work item identifier.
            new_status: The target status.
            expected_status: If provided, only update if current status matches.

        Returns:
            True if update succeeded, False if status didn't match expected.

        Raises:
            WorkItemNotFoundError: If work item doesn't exist.
            InvalidStatusTransitionError: If transition is not allowed.

        TODO: Implement with optimistic locking and transition validation.
        """
        pass

    @abstractmethod
    async def claim_work_item(
        self,
        work_id: str,
        worker_id: str,
        lease_duration_seconds: int = 300,
    ) -> bool:
        """Attempt to claim/lease a work item for processing.

        Should atomically transition from READY to IN_PROGRESS and
        record the claiming worker. Only one worker should succeed.

        Args:
            work_id: The work item identifier.
            worker_id: Identifier of the claiming worker.
            lease_duration_seconds: How long the lease is valid.

        Returns:
            True if claim succeeded, False if item was already claimed.

        TODO: Implement with atomic claim semantics.
        """
        pass

    @abstractmethod
    async def release_work_item(
        self,
        work_id: str,
        worker_id: str,
        new_status: WorkItemStatus = WorkItemStatus.READY,
    ) -> bool:
        """Release a claimed work item.

        Should only succeed if the specified worker holds the claim.

        Args:
            work_id: The work item identifier.
            worker_id: Identifier of the releasing worker.
            new_status: Status to transition to (READY for retry, FAILED, etc.).

        Returns:
            True if release succeeded, False if worker doesn't hold claim.

        TODO: Implement with claim ownership verification.
        """
        pass

    @abstractmethod
    async def query_by_status(
        self,
        status: WorkItemStatus,
        work_type: WorkItemType | None = None,
        job_id: str | None = None,
        limit: int = 100,
    ) -> list[WorkItem]:
        """Query work items by status.

        Args:
            status: Filter by this status.
            work_type: Optional filter by work type.
            job_id: Optional filter by job ID.
            limit: Maximum number of results.

        Returns:
            List of matching work items.

        TODO: Implement with efficient querying.
        """
        pass

    @abstractmethod
    async def query_by_job(
        self,
        job_id: str,
        work_type: WorkItemType | None = None,
    ) -> list[WorkItem]:
        """Query all work items for a job.

        Args:
            job_id: The job identifier.
            work_type: Optional filter by work type.

        Returns:
            List of work items for the job.

        TODO: Implement with efficient querying.
        """
        pass

    @abstractmethod
    async def get_ready_work_items(
        self,
        work_type: WorkItemType | None = None,
        limit: int = 10,
    ) -> list[WorkItem]:
        """Get work items ready for processing.

        Convenience method for workers polling for work.

        Args:
            work_type: Optional filter by work type.
            limit: Maximum number of results.

        Returns:
            List of READY work items.

        TODO: Implement with efficient polling support.
        """
        pass

    async def stream_work_items(
        self,
        job_id: str,
        work_type: WorkItemType | None = None,
    ) -> AsyncIterator[WorkItem]:
        """Stream work items for a job.

        Default implementation queries and yields. Override for
        systems with native streaming support.

        Args:
            job_id: The job identifier.
            work_type: Optional filter by work type.

        Yields:
            Work items for the job.
        """
        items = await self.query_by_job(job_id, work_type)
        for item in items:
            yield item

    @abstractmethod
    async def check_dependencies_done(
        self,
        work_id: str,
    ) -> bool:
        """Check if all dependencies of a work item are DONE.

        Used to determine when BLOCKED items can become READY.

        Args:
            work_id: The work item identifier.

        Returns:
            True if all dependencies are DONE, False otherwise.

        TODO: Implement with efficient dependency checking.
        """
        pass

    @abstractmethod
    async def find_by_idempotency_key(
        self,
        idempotency_key: str,
    ) -> WorkItem | None:
        """Find a work item by its idempotency key.

        Used to prevent duplicate work item creation.

        Args:
            idempotency_key: The idempotency key to search for.

        Returns:
            The WorkItem if found, None otherwise.

        TODO: Implement with indexed lookup for efficiency.
        """
        pass
