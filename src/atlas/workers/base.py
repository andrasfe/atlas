"""Abstract base class for all workers.

Workers are agents that claim work items and produce output artifacts.
This base class provides common functionality for all worker types.

Atlas does NOT call LLMs directly. The integrating system implements
concrete workers that wrap their own agents. Those agents handle all
LLM interactions internally.

Key Requirements:
- Claim work items atomically (only one worker succeeds)
- Write outputs to deterministic URIs
- Handle errors and retries gracefully
- Record open questions when context is insufficient
"""

from abc import ABC, abstractmethod
from typing import Any

from atlas.adapters.ticket_system import TicketSystemAdapter
from atlas.adapters.artifact_store import ArtifactStoreAdapter
from atlas.models.work_item import WorkItem
from atlas.models.enums import WorkItemStatus, WorkItemType


class Worker(ABC):
    """Abstract base class for all analysis workers.

    Workers implement the actual analysis logic for different work types.
    The integrating system provides concrete implementations that wrap
    their agents. Atlas workers do NOT call LLMs directly.

    Design Principles:
        - Idempotent: If output exists and is valid, mark DONE without recomputation
        - Explicit uncertainty: Record "unknowns" / "open questions" rather than guessing
        - Bounded context: Never exceed context budget
        - Integrator owns LLM: Atlas orchestrates; your agents call LLMs

    Example Implementation:
        >>> class MyScribeWorker(Worker):
        ...     def __init__(self, my_scribe_agent, ticket_system, artifact_store):
        ...         super().__init__("scribe-1", ticket_system, artifact_store)
        ...         self.scribe = my_scribe_agent  # Your agent handles LLM
        ...
        ...     async def process(self, work_item):
        ...         content = await self.artifact_store.read(...)
        ...         result = await self.scribe.analyze(content)  # Delegate to your agent
        ...         await self.artifact_store.write_json(output_uri, result)
        ...         return result
    """

    def __init__(
        self,
        worker_id: str,
        ticket_system: TicketSystemAdapter,
        artifact_store: ArtifactStoreAdapter,
    ):
        """Initialize the worker.

        Args:
            worker_id: Unique identifier for this worker instance.
            ticket_system: Ticket system adapter.
            artifact_store: Artifact store adapter.
        """
        self.worker_id = worker_id
        self.ticket_system = ticket_system
        self.artifact_store = artifact_store

    @property
    @abstractmethod
    def supported_work_types(self) -> list[WorkItemType]:
        """Get the work types this worker can process.

        Returns:
            List of supported WorkItemType values.
        """
        pass

    @abstractmethod
    async def process(self, work_item: WorkItem) -> Any:
        """Process a work item and produce output.

        Args:
            work_item: The work item to process.

        Returns:
            The result artifact (type depends on work type).

        Raises:
            WorkerError: If processing fails.

        TODO: Implement work type-specific processing.
        """
        pass

    async def claim_and_process(self, work_id: str) -> bool:
        """Attempt to claim a work item and process it.

        Atomically claims the work item, processes it, and updates
        status to DONE or FAILED.

        Args:
            work_id: The work item to claim and process.

        Returns:
            True if processing succeeded, False if claim failed or error.
        """
        # Attempt to claim
        claimed = await self.ticket_system.claim_work_item(
            work_id,
            self.worker_id,
            lease_duration_seconds=300,
        )

        if not claimed:
            return False

        try:
            # Get work item details
            work_item = await self.ticket_system.get_work_item(work_id)
            if work_item is None:
                return False

            # Check if output already exists (idempotency)
            if await self._output_exists(work_item):
                await self.ticket_system.update_status(
                    work_id,
                    WorkItemStatus.DONE,
                )
                return True

            # Process the work item
            await self.process(work_item)

            # Mark as done
            await self.ticket_system.update_status(
                work_id,
                WorkItemStatus.DONE,
            )
            return True

        except Exception as e:
            # Mark as failed and release
            await self.ticket_system.update_status(
                work_id,
                WorkItemStatus.FAILED,
            )
            # TODO: Log error details
            return False

    async def poll_and_process(
        self,
        limit: int = 1,
    ) -> int:
        """Poll for available work and process it.

        Args:
            limit: Maximum number of items to process.

        Returns:
            Number of items successfully processed.
        """
        processed = 0

        for work_type in self.supported_work_types:
            if processed >= limit:
                break

            items = await self.ticket_system.get_ready_work_items(
                work_type=work_type,
                limit=limit - processed,
            )

            for item in items:
                if await self.claim_and_process(item.work_id):
                    processed += 1

        return processed

    async def _output_exists(self, work_item: WorkItem) -> bool:
        """Check if output already exists for idempotency.

        Args:
            work_item: The work item to check.

        Returns:
            True if valid output exists.

        TODO: Override in subclasses to check specific output URIs.
        """
        return False

    def _get_output_uri(self, work_item: WorkItem) -> str | None:
        """Get the output URI from a work item payload.

        Args:
            work_item: The work item.

        Returns:
            Output URI if available, None otherwise.
        """
        payload = work_item.payload
        # Check various payload types for output URI
        if hasattr(payload, "result_uri"):
            return payload.result_uri
        if hasattr(payload, "output_uri"):
            return payload.output_uri
        if hasattr(payload, "output_doc_uri"):
            return payload.output_doc_uri
        return None
