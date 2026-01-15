"""Event emission for workflow transitions.

Provides event emission for workflow state changes, enabling integrations
with external systems (webhooks, message queues, etc.).

Design Principles:
- Async-safe event emission
- Pluggable event handlers
- Type-safe event definitions
- Event history for debugging

Usage:
    >>> emitter = get_event_emitter()
    >>> emitter.on("chunk_completed", lambda e: print(e))
    >>> emit_event("chunk_completed", job_id="job-123", chunk_id="chunk-001")
"""

import asyncio
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable
from collections import defaultdict


class EventType(str, Enum):
    """Types of workflow events.

    Events are emitted at key workflow transitions.
    """

    # Job lifecycle
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"

    # Phase transitions
    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"

    # Work item lifecycle
    WORK_ITEM_CREATED = "work_item_created"
    WORK_ITEM_CLAIMED = "work_item_claimed"
    WORK_ITEM_COMPLETED = "work_item_completed"
    WORK_ITEM_FAILED = "work_item_failed"
    WORK_ITEM_UNBLOCKED = "work_item_unblocked"

    # Specific work types
    CHUNK_COMPLETED = "chunk_completed"
    MERGE_COMPLETED = "merge_completed"
    CHALLENGE_COMPLETED = "challenge_completed"
    FOLLOWUP_COMPLETED = "followup_completed"
    PATCH_MERGE_COMPLETED = "patch_merge_completed"

    # Challenger events
    ISSUES_RAISED = "issues_raised"
    FOLLOWUPS_DISPATCHED = "followups_dispatched"

    # Progress events
    PROGRESS_UPDATE = "progress_update"

    # Error events
    ERROR_OCCURRED = "error_occurred"
    RETRY_SCHEDULED = "retry_scheduled"


@dataclass
class Event:
    """An event emitted by the workflow system.

    Attributes:
        event_type: Type of event.
        timestamp: When the event occurred.
        job_id: Associated job identifier.
        work_id: Associated work item identifier (if applicable).
        data: Event-specific data.
        source: Source component that emitted the event.
    """

    event_type: EventType | str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    job_id: str | None = None
    work_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary.

        Returns:
            Dictionary representation of event.
        """
        event_type_value = (
            self.event_type.value
            if isinstance(self.event_type, EventType)
            else self.event_type
        )
        return {
            "event_type": event_type_value,
            "timestamp": self.timestamp,
            "job_id": self.job_id,
            "work_id": self.work_id,
            "data": self.data,
            "source": self.source,
        }


# Type aliases for handlers
SyncHandler = Callable[[Event], None]
AsyncHandler = Callable[[Event], Awaitable[None]]
Handler = SyncHandler | AsyncHandler


class EventEmitter:
    """Emits and distributes workflow events.

    Thread-safe event emission with support for both sync and async handlers.
    Supports pattern-based subscriptions and event history.

    Example:
        >>> emitter = EventEmitter()
        >>> emitter.on("chunk_completed", lambda e: print(f"Chunk done: {e.work_id}"))
        >>> emitter.emit("chunk_completed", job_id="job-123", work_id="chunk-001")
    """

    def __init__(self, max_history: int = 1000) -> None:
        """Initialize event emitter.

        Args:
            max_history: Maximum number of events to keep in history.
        """
        self._lock = threading.Lock()
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._global_handlers: list[Handler] = []
        self._history: list[Event] = []
        self._max_history = max_history

    def on(
        self,
        event_type: EventType | str,
        handler: Handler,
    ) -> Callable[[], None]:
        """Register a handler for an event type.

        Args:
            event_type: Event type to handle (or "*" for all events).
            handler: Handler function (sync or async).

        Returns:
            Function to unregister the handler.
        """
        event_key = (
            event_type.value if isinstance(event_type, EventType) else event_type
        )

        with self._lock:
            if event_key == "*":
                self._global_handlers.append(handler)
            else:
                self._handlers[event_key].append(handler)

        def unregister() -> None:
            with self._lock:
                if event_key == "*":
                    if handler in self._global_handlers:
                        self._global_handlers.remove(handler)
                else:
                    if handler in self._handlers[event_key]:
                        self._handlers[event_key].remove(handler)

        return unregister

    def off(
        self,
        event_type: EventType | str,
        handler: Handler,
    ) -> None:
        """Unregister a handler for an event type.

        Args:
            event_type: Event type.
            handler: Handler to remove.
        """
        event_key = (
            event_type.value if isinstance(event_type, EventType) else event_type
        )

        with self._lock:
            if event_key == "*":
                if handler in self._global_handlers:
                    self._global_handlers.remove(handler)
            else:
                if handler in self._handlers[event_key]:
                    self._handlers[event_key].remove(handler)

    def emit(
        self,
        event_type: EventType | str,
        job_id: str | None = None,
        work_id: str | None = None,
        source: str | None = None,
        **data: Any,
    ) -> Event:
        """Emit an event.

        Args:
            event_type: Type of event.
            job_id: Associated job identifier.
            work_id: Associated work item identifier.
            source: Source component.
            **data: Event-specific data.

        Returns:
            The emitted event.
        """
        event = Event(
            event_type=event_type,
            job_id=job_id,
            work_id=work_id,
            data=data,
            source=source,
        )

        # Add to history
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        # Get handlers
        event_key = (
            event_type.value if isinstance(event_type, EventType) else event_type
        )
        handlers: list[Handler] = []
        with self._lock:
            handlers = (
                self._handlers.get(event_key, []).copy()
                + self._global_handlers.copy()
            )

        # Invoke handlers
        for handler in handlers:
            try:
                result = handler(event)
                # Handle async handlers
                if asyncio.iscoroutine(result):
                    # Try to get running loop
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(result)
                    except RuntimeError:
                        # No running loop, run synchronously
                        asyncio.run(result)
            except Exception:
                pass  # Don't let handler errors break event emission

        return event

    async def emit_async(
        self,
        event_type: EventType | str,
        job_id: str | None = None,
        work_id: str | None = None,
        source: str | None = None,
        **data: Any,
    ) -> Event:
        """Emit an event asynchronously.

        Awaits async handlers instead of creating tasks.

        Args:
            event_type: Type of event.
            job_id: Associated job identifier.
            work_id: Associated work item identifier.
            source: Source component.
            **data: Event-specific data.

        Returns:
            The emitted event.
        """
        event = Event(
            event_type=event_type,
            job_id=job_id,
            work_id=work_id,
            data=data,
            source=source,
        )

        # Add to history
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        # Get handlers
        event_key = (
            event_type.value if isinstance(event_type, EventType) else event_type
        )
        handlers: list[Handler] = []
        with self._lock:
            handlers = (
                self._handlers.get(event_key, []).copy()
                + self._global_handlers.copy()
            )

        # Invoke handlers
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass  # Don't let handler errors break event emission

        return event

    def get_history(
        self,
        event_type: EventType | str | None = None,
        job_id: str | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        """Get event history.

        Args:
            event_type: Filter by event type.
            job_id: Filter by job ID.
            limit: Maximum number of events to return.

        Returns:
            List of events matching filters.
        """
        with self._lock:
            events = self._history.copy()

        # Apply filters
        if event_type is not None:
            event_key = (
                event_type.value if isinstance(event_type, EventType) else event_type
            )
            events = [
                e for e in events
                if (e.event_type.value if isinstance(e.event_type, EventType) else e.event_type) == event_key
            ]

        if job_id is not None:
            events = [e for e in events if e.job_id == job_id]

        # Apply limit (from most recent)
        if limit is not None:
            events = events[-limit:]

        return events

    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._history.clear()

    def clear_handlers(self) -> None:
        """Clear all handlers."""
        with self._lock:
            self._handlers.clear()
            self._global_handlers.clear()


# Global event emitter
_event_emitter: EventEmitter | None = None
_emitter_lock = threading.Lock()


def get_event_emitter() -> EventEmitter:
    """Get the global event emitter.

    Returns:
        Global EventEmitter instance.
    """
    global _event_emitter
    if _event_emitter is None:
        with _emitter_lock:
            if _event_emitter is None:
                _event_emitter = EventEmitter()
    return _event_emitter


def emit_event(
    event_type: EventType | str,
    job_id: str | None = None,
    work_id: str | None = None,
    source: str | None = None,
    **data: Any,
) -> Event:
    """Emit an event using the global emitter.

    Convenience function for quick event emission.

    Args:
        event_type: Type of event.
        job_id: Associated job identifier.
        work_id: Associated work item identifier.
        source: Source component.
        **data: Event-specific data.

    Returns:
        The emitted event.
    """
    return get_event_emitter().emit(
        event_type=event_type,
        job_id=job_id,
        work_id=work_id,
        source=source,
        **data,
    )


async def emit_event_async(
    event_type: EventType | str,
    job_id: str | None = None,
    work_id: str | None = None,
    source: str | None = None,
    **data: Any,
) -> Event:
    """Emit an event asynchronously using the global emitter.

    Args:
        event_type: Type of event.
        job_id: Associated job identifier.
        work_id: Associated work item identifier.
        source: Source component.
        **data: Event-specific data.

    Returns:
        The emitted event.
    """
    return await get_event_emitter().emit_async(
        event_type=event_type,
        job_id=job_id,
        work_id=work_id,
        source=source,
        **data,
    )
