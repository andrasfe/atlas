"""In-memory ticket system adapter for testing.

This adapter provides a fully-functional in-memory implementation of the
TicketSystemAdapter interface. It supports all operations including:
- Work item creation with idempotency key checking
- Status transitions with optimistic locking
- Claim/release with worker_id tracking
- Query by status, job, and type

This is suitable for unit testing and development but not for production
use as it does not persist data between restarts.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from atlas.adapters.ticket_system import TicketSystemAdapter
from atlas.models.enums import WorkItemStatus, WorkItemType
from atlas.models.work_item import WorkItem


class WorkItemNotFoundError(Exception):
    """Raised when a work item is not found."""
    pass


class InvalidStatusTransitionError(Exception):
    """Raised when a status transition is not allowed."""
    pass


class DuplicateWorkItemError(Exception):
    """Raised when attempting to create a duplicate work item."""
    pass


class MemoryTicketSystem(TicketSystemAdapter):
    """In-memory implementation of the ticket system adapter.

    This implementation stores all work items in memory using dictionaries.
    It provides full support for:
    - Idempotent work item creation via idempotency_key
    - Optimistic locking via expected_status parameter
    - Claim/release with lease tracking
    - Efficient querying by status, job, and type

    Thread Safety:
        This implementation is NOT thread-safe. For concurrent access,
        wrap operations with appropriate locking mechanisms.

    Example:
        >>> ticket_system = MemoryTicketSystem()
        >>> work_id = await ticket_system.create_work_item(work_item)
        >>> claimed = await ticket_system.claim_work_item(work_id, "worker-1")
        >>> await ticket_system.update_status(work_id, WorkItemStatus.DONE)
    """

    def __init__(self) -> None:
        """Initialize the in-memory ticket system."""
        # Main storage: work_id -> WorkItem
        self._items: dict[str, WorkItem] = {}

        # Claim tracking: work_id -> claim info
        self._claims: dict[str, dict[str, Any]] = {}

        # Idempotency key index: idempotency_key -> work_id
        self._idempotency_index: dict[str, str] = {}

        # Counter for auto-generating work IDs
        self._id_counter: int = 0

    def _generate_work_id(self) -> str:
        """Generate a unique work ID.

        Returns:
            A unique work ID string.
        """
        self._id_counter += 1
        return f"work-{self._id_counter:06d}"

    def _now(self) -> str:
        """Get current UTC timestamp as ISO string.

        Returns:
            ISO format timestamp string.
        """
        return datetime.now(timezone.utc).isoformat()

    async def create_work_item(
        self,
        work_item: WorkItem,
    ) -> str:
        """Create a new work item in the in-memory store.

        If idempotency_key is set and a matching item exists, returns the
        existing item's ID without creating a duplicate.

        Args:
            work_item: The work item to create.

        Returns:
            The work ID (existing if idempotency match, new otherwise).

        Raises:
            DuplicateWorkItemError: If work_id already exists without matching
                idempotency_key.
        """
        # Check idempotency key first
        if work_item.idempotency_key:
            existing_id = self._idempotency_index.get(work_item.idempotency_key)
            if existing_id:
                return existing_id

        # Generate work_id if not provided
        work_id = work_item.work_id or self._generate_work_id()

        # Check for duplicate work_id
        if work_id in self._items:
            raise DuplicateWorkItemError(f"Work item already exists: {work_id}")

        # Set timestamps
        now = self._now()
        work_item.work_id = work_id
        work_item.created_at = now
        work_item.updated_at = now

        # Store the work item
        self._items[work_id] = work_item

        # Index by idempotency key if present
        if work_item.idempotency_key:
            self._idempotency_index[work_item.idempotency_key] = work_id

        return work_id

    async def get_work_item(self, work_id: str) -> WorkItem | None:
        """Retrieve a work item by ID.

        Args:
            work_id: The work item identifier.

        Returns:
            The WorkItem if found, None otherwise.
        """
        return self._items.get(work_id)

    async def update_status(
        self,
        work_id: str,
        new_status: WorkItemStatus,
        *,
        expected_status: WorkItemStatus | None = None,
    ) -> bool:
        """Update the status of a work item.

        Implements optimistic locking when expected_status is provided.
        Also validates that the status transition is allowed.

        Args:
            work_id: The work item identifier.
            new_status: The target status.
            expected_status: If provided, only update if current status matches.

        Returns:
            True if update succeeded, False if status didn't match expected.

        Raises:
            WorkItemNotFoundError: If work item doesn't exist.
            InvalidStatusTransitionError: If transition is not allowed.
        """
        item = self._items.get(work_id)
        if item is None:
            raise WorkItemNotFoundError(f"Work item not found: {work_id}")

        # Check expected status (optimistic locking)
        if expected_status is not None and item.status != expected_status:
            return False

        # Validate transition
        if not item.status.can_transition_to(new_status):
            raise InvalidStatusTransitionError(
                f"Cannot transition from {item.status.value} to {new_status.value}"
            )

        # Update status and timestamp
        item.status = new_status
        item.updated_at = self._now()

        return True

    async def claim_work_item(
        self,
        work_id: str,
        worker_id: str,
        lease_duration_seconds: int = 300,
    ) -> bool:
        """Attempt to claim/lease a work item for processing.

        Atomically transitions from READY to IN_PROGRESS and records
        the claiming worker. Only one worker can hold a claim.

        Args:
            work_id: The work item identifier.
            worker_id: Identifier of the claiming worker.
            lease_duration_seconds: How long the lease is valid.

        Returns:
            True if claim succeeded, False if item was already claimed
            or not in READY status.
        """
        item = self._items.get(work_id)
        if item is None:
            return False

        # Can only claim READY items
        if item.status != WorkItemStatus.READY:
            return False

        # Check if already claimed (and lease not expired)
        if work_id in self._claims:
            claim_info = self._claims[work_id]
            expires_at = claim_info.get("expires_at")
            if expires_at and datetime.fromisoformat(expires_at) > datetime.now(timezone.utc):
                return False
            # Lease expired, remove old claim
            del self._claims[work_id]

        # Create the claim
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=lease_duration_seconds)

        self._claims[work_id] = {
            "worker_id": worker_id,
            "claimed_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
        }

        # Update status
        item.status = WorkItemStatus.IN_PROGRESS
        item.updated_at = self._now()

        return True

    async def release_work_item(
        self,
        work_id: str,
        worker_id: str,
        new_status: WorkItemStatus = WorkItemStatus.READY,
    ) -> bool:
        """Release a claimed work item.

        Only succeeds if the specified worker holds the claim.

        Args:
            work_id: The work item identifier.
            worker_id: Identifier of the releasing worker.
            new_status: Status to transition to (READY for retry, FAILED, etc.).

        Returns:
            True if release succeeded, False if worker doesn't hold claim.
        """
        # Check claim ownership
        claim_info = self._claims.get(work_id)
        if claim_info is None:
            return False

        if claim_info.get("worker_id") != worker_id:
            return False

        # Release the claim
        del self._claims[work_id]

        # Update status
        item = self._items.get(work_id)
        if item:
            item.status = new_status
            item.updated_at = self._now()

        return True

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
        """
        results: list[WorkItem] = []

        for item in self._items.values():
            # Filter by status
            if item.status != status:
                continue

            # Filter by work type
            if work_type is not None and item.work_type != work_type:
                continue

            # Filter by job ID
            if job_id is not None and item.payload.job_id != job_id:
                continue

            results.append(item)

            # Respect limit
            if len(results) >= limit:
                break

        return results

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
        """
        results: list[WorkItem] = []

        for item in self._items.values():
            # Filter by job ID
            if item.payload.job_id != job_id:
                continue

            # Filter by work type
            if work_type is not None and item.work_type != work_type:
                continue

            results.append(item)

        return results

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
        """
        return await self.query_by_status(
            WorkItemStatus.READY,
            work_type=work_type,
            limit=limit,
        )

    async def check_dependencies_done(
        self,
        work_id: str,
    ) -> bool:
        """Check if all dependencies of a work item are DONE.

        Used to determine when BLOCKED items can become READY.

        Args:
            work_id: The work item identifier.

        Returns:
            True if all dependencies are DONE (or no dependencies),
            False otherwise.
        """
        item = self._items.get(work_id)
        if item is None:
            return False

        # No dependencies means ready
        if not item.depends_on:
            return True

        # Check each dependency
        for dep_id in item.depends_on:
            dep = self._items.get(dep_id)
            if dep is None or dep.status != WorkItemStatus.DONE:
                return False

        return True

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
        """
        work_id = self._idempotency_index.get(idempotency_key)
        if work_id is None:
            return None

        return self._items.get(work_id)

    # Additional utility methods for testing

    def clear(self) -> None:
        """Clear all data from the ticket system.

        Useful for test setup/teardown.
        """
        self._items.clear()
        self._claims.clear()
        self._idempotency_index.clear()
        self._id_counter = 0

    def get_claim_info(self, work_id: str) -> dict[str, Any] | None:
        """Get claim information for a work item.

        Useful for testing claim behavior.

        Args:
            work_id: The work item identifier.

        Returns:
            Claim info dict if claimed, None otherwise.
        """
        return self._claims.get(work_id)

    @property
    def item_count(self) -> int:
        """Get the total number of work items.

        Returns:
            Number of work items stored.
        """
        return len(self._items)
