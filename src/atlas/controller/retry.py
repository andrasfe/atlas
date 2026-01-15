"""Retry policy and failure handling for Atlas workflow.

This module implements exponential backoff retry policies and dead letter queue
handling for work items that fail permanently.

Key Features:
- Configurable exponential backoff with jitter
- Maximum retry count enforcement
- Dead letter queue for permanent failures
- Retry state tracking and persistence
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, TypeVar, Awaitable

from atlas.models.enums import WorkItemStatus
from atlas.models.work_item import WorkItem

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FailureReason(str, Enum):
    """Classification of failure reasons for retry decisions.

    Some failures are transient and should be retried, while others
    are permanent and should go to the dead letter queue.
    """

    TRANSIENT = "transient"
    """Temporary failure (network, rate limit, timeout). Retry."""

    VALIDATION = "validation"
    """Input validation failed. Do not retry."""

    DEPENDENCY = "dependency"
    """Missing dependency or resource. May retry after fix."""

    LOGIC = "logic"
    """Logic/programming error. Do not retry."""

    TIMEOUT = "timeout"
    """Operation timed out. Retry with backoff."""

    RATE_LIMIT = "rate_limit"
    """Rate limited by external service. Retry with backoff."""

    UNKNOWN = "unknown"
    """Unknown failure reason. Retry cautiously."""


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Implements exponential backoff with jitter to prevent thundering herd.

    Attributes:
        max_retries: Maximum number of retry attempts.
        initial_delay_seconds: Initial delay before first retry.
        max_delay_seconds: Maximum delay between retries.
        exponential_base: Base for exponential backoff calculation.
        jitter_factor: Random jitter factor (0.0 to 1.0).
        retryable_reasons: Set of failure reasons that should be retried.
    """

    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter_factor: float = 0.2
    retryable_reasons: set[FailureReason] = field(
        default_factory=lambda: {
            FailureReason.TRANSIENT,
            FailureReason.TIMEOUT,
            FailureReason.RATE_LIMIT,
            FailureReason.UNKNOWN,
        }
    )

    def compute_delay(self, attempt: int) -> float:
        """Compute delay for a given retry attempt.

        Uses exponential backoff with jitter:
        delay = min(initial * base^attempt, max) * (1 +/- jitter)

        Args:
            attempt: The current attempt number (0-indexed).

        Returns:
            Delay in seconds before next retry.
        """
        base_delay = min(
            self.initial_delay_seconds * (self.exponential_base**attempt),
            self.max_delay_seconds,
        )

        # Add jitter
        jitter = base_delay * self.jitter_factor * (2 * random.random() - 1)
        delay = max(0.0, base_delay + jitter)

        return delay

    def should_retry(self, reason: FailureReason, attempt: int) -> bool:
        """Determine if a failed operation should be retried.

        Args:
            reason: The failure reason classification.
            attempt: The current attempt number (0-indexed).

        Returns:
            True if operation should be retried.
        """
        if attempt >= self.max_retries:
            return False

        return reason in self.retryable_reasons


@dataclass
class RetryState:
    """State tracking for retry attempts.

    Maintains the history of retry attempts for a work item.

    Attributes:
        work_id: The work item being retried.
        attempt_count: Number of attempts made.
        last_attempt_at: Timestamp of last attempt.
        last_failure_reason: Classification of last failure.
        last_error: Error message from last failure.
        next_retry_at: When the next retry should occur.
        is_exhausted: True if max retries exceeded.
    """

    work_id: str
    attempt_count: int = 0
    last_attempt_at: str | None = None
    last_failure_reason: FailureReason = FailureReason.UNKNOWN
    last_error: str | None = None
    next_retry_at: str | None = None
    is_exhausted: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)

    def record_attempt(
        self,
        success: bool,
        failure_reason: FailureReason | None = None,
        error: str | None = None,
    ) -> None:
        """Record a retry attempt.

        Args:
            success: Whether the attempt succeeded.
            failure_reason: Classification of failure if not successful.
            error: Error message if not successful.
        """
        now = datetime.now(timezone.utc).isoformat()
        self.attempt_count += 1
        self.last_attempt_at = now

        if not success:
            self.last_failure_reason = failure_reason or FailureReason.UNKNOWN
            self.last_error = error

        # Add to history
        self.history.append({
            "attempt": self.attempt_count,
            "timestamp": now,
            "success": success,
            "failure_reason": failure_reason.value if failure_reason else None,
            "error": error,
        })

    def schedule_retry(self, delay_seconds: float) -> None:
        """Schedule the next retry attempt.

        Args:
            delay_seconds: Seconds until next retry.
        """
        retry_time = datetime.now(timezone.utc).timestamp() + delay_seconds
        self.next_retry_at = datetime.fromtimestamp(
            retry_time, tz=timezone.utc
        ).isoformat()

    def mark_exhausted(self) -> None:
        """Mark this work item as having exhausted all retries."""
        self.is_exhausted = True
        self.next_retry_at = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of retry state.
        """
        return {
            "work_id": self.work_id,
            "attempt_count": self.attempt_count,
            "last_attempt_at": self.last_attempt_at,
            "last_failure_reason": self.last_failure_reason.value,
            "last_error": self.last_error,
            "next_retry_at": self.next_retry_at,
            "is_exhausted": self.is_exhausted,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetryState":
        """Create from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            RetryState instance.
        """
        failure_reason = FailureReason.UNKNOWN
        if data.get("last_failure_reason"):
            try:
                failure_reason = FailureReason(data["last_failure_reason"])
            except ValueError:
                pass

        return cls(
            work_id=data["work_id"],
            attempt_count=data.get("attempt_count", 0),
            last_attempt_at=data.get("last_attempt_at"),
            last_failure_reason=failure_reason,
            last_error=data.get("last_error"),
            next_retry_at=data.get("next_retry_at"),
            is_exhausted=data.get("is_exhausted", False),
            history=data.get("history", []),
        )


class DeadLetterQueue:
    """Queue for work items that have permanently failed.

    Work items that exceed max retries or fail with non-retryable errors
    are moved to the dead letter queue for manual inspection.

    The DLQ supports:
    - Adding failed items with failure context
    - Listing items for inspection
    - Reprocessing items after fixes
    - Purging old items
    """

    def __init__(self) -> None:
        """Initialize the dead letter queue."""
        self._items: dict[str, dict[str, Any]] = {}

    def add(
        self,
        work_item: WorkItem,
        retry_state: RetryState,
        reason: str,
    ) -> None:
        """Add a work item to the dead letter queue.

        Args:
            work_item: The failed work item.
            retry_state: The retry state with attempt history.
            reason: Human-readable reason for permanent failure.
        """
        entry = {
            "work_item": work_item.model_dump(),
            "retry_state": retry_state.to_dict(),
            "reason": reason,
            "added_at": datetime.now(timezone.utc).isoformat(),
            "reprocessed": False,
            "reprocessed_at": None,
        }
        self._items[work_item.work_id] = entry
        logger.warning(
            f"Work item {work_item.work_id} added to dead letter queue: {reason}"
        )

    def get(self, work_id: str) -> dict[str, Any] | None:
        """Get a dead letter queue entry.

        Args:
            work_id: The work item ID.

        Returns:
            DLQ entry if found, None otherwise.
        """
        return self._items.get(work_id)

    def list_items(
        self,
        include_reprocessed: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List dead letter queue items.

        Args:
            include_reprocessed: Include items that have been reprocessed.
            limit: Maximum items to return.

        Returns:
            List of DLQ entries.
        """
        items = []
        for entry in self._items.values():
            if not include_reprocessed and entry.get("reprocessed"):
                continue
            items.append(entry)
            if len(items) >= limit:
                break
        return items

    def mark_reprocessed(self, work_id: str) -> bool:
        """Mark an item as reprocessed.

        Args:
            work_id: The work item ID.

        Returns:
            True if item was found and marked.
        """
        if work_id not in self._items:
            return False

        self._items[work_id]["reprocessed"] = True
        self._items[work_id]["reprocessed_at"] = datetime.now(timezone.utc).isoformat()
        return True

    def remove(self, work_id: str) -> bool:
        """Remove an item from the DLQ.

        Args:
            work_id: The work item ID.

        Returns:
            True if item was found and removed.
        """
        if work_id in self._items:
            del self._items[work_id]
            return True
        return False

    def purge(self, before: datetime | None = None) -> int:
        """Purge old items from the DLQ.

        Args:
            before: Purge items added before this time.
                    If None, purge all items.

        Returns:
            Number of items purged.
        """
        if before is None:
            count = len(self._items)
            self._items.clear()
            return count

        before_iso = before.isoformat()
        to_remove = [
            work_id
            for work_id, entry in self._items.items()
            if entry["added_at"] < before_iso
        ]

        for work_id in to_remove:
            del self._items[work_id]

        return len(to_remove)

    def __len__(self) -> int:
        """Get number of items in the DLQ."""
        return len(self._items)


class RetryManager:
    """Manages retry logic and dead letter queue for work items.

    Coordinates retry attempts, tracks state, and handles permanent
    failures by moving them to the dead letter queue.

    Example:
        >>> config = RetryConfig(max_retries=3)
        >>> manager = RetryManager(config)
        >>> async with manager.with_retry(work_item) as ctx:
        ...     result = await process_work_item(work_item)
        ...     ctx.success = True
        >>> if ctx.should_dlq:
        ...     # Handle permanent failure
        ...     pass
    """

    def __init__(
        self,
        config: RetryConfig | None = None,
        dlq: DeadLetterQueue | None = None,
    ) -> None:
        """Initialize the retry manager.

        Args:
            config: Retry configuration.
            dlq: Dead letter queue instance.
        """
        self.config = config or RetryConfig()
        self.dlq = dlq or DeadLetterQueue()
        self._states: dict[str, RetryState] = {}

    def get_state(self, work_id: str) -> RetryState:
        """Get or create retry state for a work item.

        Args:
            work_id: The work item ID.

        Returns:
            RetryState for the work item.
        """
        if work_id not in self._states:
            self._states[work_id] = RetryState(work_id=work_id)
        return self._states[work_id]

    def clear_state(self, work_id: str) -> None:
        """Clear retry state for a work item.

        Args:
            work_id: The work item ID.
        """
        self._states.pop(work_id, None)

    async def execute_with_retry(
        self,
        work_item: WorkItem,
        operation: Callable[[WorkItem], Awaitable[T]],
        classify_error: Callable[[Exception], FailureReason] | None = None,
    ) -> tuple[T | None, RetryState]:
        """Execute an operation with retry logic.

        Automatically retries transient failures with exponential backoff.

        Args:
            work_item: The work item being processed.
            operation: Async function to execute.
            classify_error: Function to classify exceptions into FailureReason.

        Returns:
            Tuple of (result or None, final RetryState).
        """
        state = self.get_state(work_item.work_id)
        classifier = classify_error or self._default_error_classifier

        while True:
            try:
                result = await operation(work_item)
                state.record_attempt(success=True)
                return result, state

            except Exception as e:
                reason = classifier(e)
                state.record_attempt(
                    success=False,
                    failure_reason=reason,
                    error=str(e),
                )

                logger.warning(
                    f"Work item {work_item.work_id} failed "
                    f"(attempt {state.attempt_count}): {e}"
                )

                if not self.config.should_retry(reason, state.attempt_count):
                    # Max retries exceeded or non-retryable error
                    state.mark_exhausted()
                    self.dlq.add(
                        work_item,
                        state,
                        f"Exhausted retries ({state.attempt_count}) "
                        f"or non-retryable error: {reason.value}",
                    )
                    return None, state

                # Schedule and wait for retry
                delay = self.config.compute_delay(state.attempt_count - 1)
                state.schedule_retry(delay)
                logger.info(
                    f"Scheduling retry for {work_item.work_id} "
                    f"in {delay:.2f}s (attempt {state.attempt_count + 1})"
                )
                await asyncio.sleep(delay)

    def should_retry(self, work_item: WorkItem, error: Exception) -> bool:
        """Check if a failed work item should be retried.

        Args:
            work_item: The failed work item.
            error: The exception that caused failure.

        Returns:
            True if the work item should be retried.
        """
        state = self.get_state(work_item.work_id)
        reason = self._default_error_classifier(error)
        return self.config.should_retry(reason, state.attempt_count)

    def record_failure(
        self,
        work_item: WorkItem,
        error: Exception,
        reason: FailureReason | None = None,
    ) -> RetryState:
        """Record a failure for a work item.

        Args:
            work_item: The failed work item.
            error: The exception that caused failure.
            reason: Optional explicit failure reason.

        Returns:
            Updated RetryState.
        """
        state = self.get_state(work_item.work_id)
        actual_reason = reason or self._default_error_classifier(error)
        state.record_attempt(
            success=False,
            failure_reason=actual_reason,
            error=str(error),
        )

        if not self.config.should_retry(actual_reason, state.attempt_count):
            state.mark_exhausted()
            self.dlq.add(
                work_item,
                state,
                f"Exhausted retries: {actual_reason.value} - {error}",
            )
        else:
            delay = self.config.compute_delay(state.attempt_count - 1)
            state.schedule_retry(delay)

        return state

    def record_success(self, work_item: WorkItem) -> RetryState:
        """Record a successful completion for a work item.

        Args:
            work_item: The successfully completed work item.

        Returns:
            Updated RetryState.
        """
        state = self.get_state(work_item.work_id)
        state.record_attempt(success=True)
        return state

    def get_ready_for_retry(self) -> list[str]:
        """Get work IDs that are ready for retry.

        Returns:
            List of work IDs whose retry time has passed.
        """
        now = datetime.now(timezone.utc).isoformat()
        ready = []

        for work_id, state in self._states.items():
            if state.is_exhausted:
                continue
            if state.next_retry_at and state.next_retry_at <= now:
                ready.append(work_id)

        return ready

    def _default_error_classifier(self, error: Exception) -> FailureReason:
        """Default error classification logic.

        Args:
            error: The exception to classify.

        Returns:
            FailureReason classification.
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # Timeout errors
        if "timeout" in error_str or "timeout" in error_type:
            return FailureReason.TIMEOUT

        # Rate limiting
        if "rate" in error_str and "limit" in error_str:
            return FailureReason.RATE_LIMIT
        if "429" in error_str or "too many requests" in error_str:
            return FailureReason.RATE_LIMIT

        # Validation errors
        if "validation" in error_str or "invalid" in error_str:
            return FailureReason.VALIDATION
        if isinstance(error, (ValueError, TypeError)):
            return FailureReason.VALIDATION

        # Connection/network errors
        if any(
            term in error_str
            for term in ["connection", "network", "socket", "refused"]
        ):
            return FailureReason.TRANSIENT

        # Dependency errors
        if "not found" in error_str or "missing" in error_str:
            return FailureReason.DEPENDENCY

        return FailureReason.UNKNOWN


# Module-level default instances
_default_config = RetryConfig()
_default_dlq = DeadLetterQueue()
_default_manager: RetryManager | None = None


def get_retry_manager() -> RetryManager:
    """Get the default retry manager.

    Returns:
        The global RetryManager instance.
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = RetryManager(_default_config, _default_dlq)
    return _default_manager


def get_dlq() -> DeadLetterQueue:
    """Get the default dead letter queue.

    Returns:
        The global DeadLetterQueue instance.
    """
    return _default_dlq


def configure_retry(config: RetryConfig) -> None:
    """Configure the default retry behavior.

    Args:
        config: New retry configuration.
    """
    global _default_config, _default_manager
    _default_config = config
    _default_manager = RetryManager(config, _default_dlq)
