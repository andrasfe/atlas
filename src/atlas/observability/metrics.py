"""Metrics collection for workflow monitoring.

Provides metrics tracking for work items processed, durations, and job progress.

Design Principles:
- Lightweight in-memory metrics by default
- Pluggable backends for production (Prometheus, StatsD, etc.)
- Thread-safe counters and histograms
- Per-job and aggregate metrics

Usage:
    >>> metrics = get_metrics()
    >>> metrics.record_work_item_processed("DOC_CHUNK", duration_ms=150)
    >>> metrics.record_job_phase_transition("job-123", "chunk", "merge")
    >>> print(metrics.get_job_metrics("job-123"))
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from collections import defaultdict


@dataclass
class WorkItemMetrics:
    """Metrics for a specific work item type.

    Attributes:
        work_type: Type of work item.
        total_processed: Total items processed.
        total_failed: Total items that failed.
        total_duration_ms: Sum of all processing durations.
        min_duration_ms: Minimum processing duration.
        max_duration_ms: Maximum processing duration.
        last_processed_at: Timestamp of last processed item.
    """

    work_type: str
    total_processed: int = 0
    total_failed: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float("inf")
    max_duration_ms: float = 0.0
    last_processed_at: str | None = None

    @property
    def avg_duration_ms(self) -> float:
        """Calculate average processing duration.

        Returns:
            Average duration in milliseconds, or 0 if no items processed.
        """
        if self.total_processed == 0:
            return 0.0
        return self.total_duration_ms / self.total_processed

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary.

        Returns:
            Dictionary representation of metrics.
        """
        return {
            "work_type": self.work_type,
            "total_processed": self.total_processed,
            "total_failed": self.total_failed,
            "avg_duration_ms": self.avg_duration_ms,
            "min_duration_ms": self.min_duration_ms if self.total_processed > 0 else 0,
            "max_duration_ms": self.max_duration_ms,
            "last_processed_at": self.last_processed_at,
        }


@dataclass
class JobMetrics:
    """Metrics for a specific job.

    Attributes:
        job_id: Job identifier.
        phase: Current workflow phase.
        chunks_total: Total number of chunks.
        chunks_done: Number of completed chunks.
        merges_total: Total number of merge nodes.
        merges_done: Number of completed merges.
        challenges_total: Total challenge iterations.
        challenges_done: Completed challenge iterations.
        followups_total: Total follow-up items.
        followups_done: Completed follow-up items.
        started_at: Job start timestamp.
        last_activity_at: Last activity timestamp.
        phase_transitions: List of phase transitions with timestamps.
    """

    job_id: str
    phase: str = "unknown"
    chunks_total: int = 0
    chunks_done: int = 0
    merges_total: int = 0
    merges_done: int = 0
    challenges_total: int = 0
    challenges_done: int = 0
    followups_total: int = 0
    followups_done: int = 0
    started_at: str | None = None
    last_activity_at: str | None = None
    phase_transitions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def chunks_progress(self) -> float:
        """Calculate chunk completion progress.

        Returns:
            Progress percentage (0.0 to 100.0).
        """
        if self.chunks_total == 0:
            return 100.0
        return (self.chunks_done / self.chunks_total) * 100.0

    @property
    def merges_progress(self) -> float:
        """Calculate merge completion progress.

        Returns:
            Progress percentage (0.0 to 100.0).
        """
        if self.merges_total == 0:
            return 100.0
        return (self.merges_done / self.merges_total) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary.

        Returns:
            Dictionary representation of metrics.
        """
        return {
            "job_id": self.job_id,
            "phase": self.phase,
            "chunks_total": self.chunks_total,
            "chunks_done": self.chunks_done,
            "chunks_progress": self.chunks_progress,
            "merges_total": self.merges_total,
            "merges_done": self.merges_done,
            "merges_progress": self.merges_progress,
            "challenges_total": self.challenges_total,
            "challenges_done": self.challenges_done,
            "followups_total": self.followups_total,
            "followups_done": self.followups_done,
            "started_at": self.started_at,
            "last_activity_at": self.last_activity_at,
            "phase_transitions": self.phase_transitions,
        }


class MetricsCollector:
    """Collects and aggregates workflow metrics.

    Thread-safe metrics collection with support for work item processing
    times and job progress tracking.

    Example:
        >>> collector = MetricsCollector()
        >>> collector.record_work_item_processed("DOC_CHUNK", duration_ms=150)
        >>> collector.record_job_progress("job-123", chunks_done=5, chunks_total=10)
        >>> print(collector.get_summary())
    """

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self._lock = threading.Lock()
        self._work_item_metrics: dict[str, WorkItemMetrics] = {}
        self._job_metrics: dict[str, JobMetrics] = {}
        self._event_handlers: list[Callable[[str, dict[str, Any]], None]] = []

    def record_work_item_processed(
        self,
        work_type: str,
        duration_ms: float,
        success: bool = True,
        job_id: str | None = None,
    ) -> None:
        """Record a processed work item.

        Args:
            work_type: Type of work item (DOC_CHUNK, DOC_MERGE, etc.).
            duration_ms: Processing duration in milliseconds.
            success: Whether processing succeeded.
            job_id: Optional job identifier.
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            # Update work item type metrics
            if work_type not in self._work_item_metrics:
                self._work_item_metrics[work_type] = WorkItemMetrics(work_type=work_type)

            metrics = self._work_item_metrics[work_type]
            if success:
                metrics.total_processed += 1
                metrics.total_duration_ms += duration_ms
                metrics.min_duration_ms = min(metrics.min_duration_ms, duration_ms)
                metrics.max_duration_ms = max(metrics.max_duration_ms, duration_ms)
            else:
                metrics.total_failed += 1
            metrics.last_processed_at = now

            # Update job-specific metrics
            if job_id:
                if job_id not in self._job_metrics:
                    self._job_metrics[job_id] = JobMetrics(job_id=job_id)
                self._job_metrics[job_id].last_activity_at = now

                # Update type-specific counts
                job_metrics = self._job_metrics[job_id]
                if success:
                    if work_type == "DOC_CHUNK":
                        job_metrics.chunks_done += 1
                    elif work_type == "DOC_MERGE":
                        job_metrics.merges_done += 1
                    elif work_type == "DOC_CHALLENGE":
                        job_metrics.challenges_done += 1
                    elif work_type == "DOC_FOLLOWUP":
                        job_metrics.followups_done += 1

        # Notify handlers
        self._notify("work_item_processed", {
            "work_type": work_type,
            "duration_ms": duration_ms,
            "success": success,
            "job_id": job_id,
        })

    def record_job_started(
        self,
        job_id: str,
        chunks_total: int = 0,
        merges_total: int = 0,
    ) -> None:
        """Record a job start.

        Args:
            job_id: Job identifier.
            chunks_total: Total number of chunks.
            merges_total: Total number of merge nodes.
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            if job_id not in self._job_metrics:
                self._job_metrics[job_id] = JobMetrics(job_id=job_id)

            metrics = self._job_metrics[job_id]
            metrics.started_at = now
            metrics.last_activity_at = now
            metrics.chunks_total = chunks_total
            metrics.merges_total = merges_total
            metrics.phase = "plan"
            metrics.phase_transitions.append({
                "from": None,
                "to": "plan",
                "timestamp": now,
            })

        self._notify("job_started", {
            "job_id": job_id,
            "chunks_total": chunks_total,
            "merges_total": merges_total,
        })

    def record_job_phase_transition(
        self,
        job_id: str,
        from_phase: str | None,
        to_phase: str,
    ) -> None:
        """Record a job phase transition.

        Args:
            job_id: Job identifier.
            from_phase: Previous phase.
            to_phase: New phase.
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            if job_id not in self._job_metrics:
                self._job_metrics[job_id] = JobMetrics(job_id=job_id)

            metrics = self._job_metrics[job_id]
            metrics.phase = to_phase
            metrics.last_activity_at = now
            metrics.phase_transitions.append({
                "from": from_phase,
                "to": to_phase,
                "timestamp": now,
            })

        self._notify("phase_transition", {
            "job_id": job_id,
            "from_phase": from_phase,
            "to_phase": to_phase,
        })

    def record_job_progress(
        self,
        job_id: str,
        chunks_total: int | None = None,
        chunks_done: int | None = None,
        merges_total: int | None = None,
        merges_done: int | None = None,
        challenges_total: int | None = None,
        challenges_done: int | None = None,
        followups_total: int | None = None,
        followups_done: int | None = None,
    ) -> None:
        """Update job progress metrics.

        Args:
            job_id: Job identifier.
            chunks_total: Total chunks (if known).
            chunks_done: Completed chunks.
            merges_total: Total merges (if known).
            merges_done: Completed merges.
            challenges_total: Total challenges (if known).
            challenges_done: Completed challenges.
            followups_total: Total follow-ups (if known).
            followups_done: Completed follow-ups.
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            if job_id not in self._job_metrics:
                self._job_metrics[job_id] = JobMetrics(job_id=job_id)

            metrics = self._job_metrics[job_id]
            metrics.last_activity_at = now

            if chunks_total is not None:
                metrics.chunks_total = chunks_total
            if chunks_done is not None:
                metrics.chunks_done = chunks_done
            if merges_total is not None:
                metrics.merges_total = merges_total
            if merges_done is not None:
                metrics.merges_done = merges_done
            if challenges_total is not None:
                metrics.challenges_total = challenges_total
            if challenges_done is not None:
                metrics.challenges_done = challenges_done
            if followups_total is not None:
                metrics.followups_total = followups_total
            if followups_done is not None:
                metrics.followups_done = followups_done

    def record_job_completed(
        self,
        job_id: str,
        success: bool = True,
    ) -> None:
        """Record job completion.

        Args:
            job_id: Job identifier.
            success: Whether job completed successfully.
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            if job_id not in self._job_metrics:
                self._job_metrics[job_id] = JobMetrics(job_id=job_id)

            metrics = self._job_metrics[job_id]
            metrics.phase = "complete" if success else "failed"
            metrics.last_activity_at = now
            metrics.phase_transitions.append({
                "from": metrics.phase_transitions[-1]["to"] if metrics.phase_transitions else None,
                "to": metrics.phase,
                "timestamp": now,
            })

        self._notify("job_completed", {
            "job_id": job_id,
            "success": success,
        })

    def get_work_item_metrics(self, work_type: str) -> WorkItemMetrics | None:
        """Get metrics for a work item type.

        Args:
            work_type: Type of work item.

        Returns:
            WorkItemMetrics if available, None otherwise.
        """
        with self._lock:
            return self._work_item_metrics.get(work_type)

    def get_job_metrics(self, job_id: str) -> JobMetrics | None:
        """Get metrics for a job.

        Args:
            job_id: Job identifier.

        Returns:
            JobMetrics if available, None otherwise.
        """
        with self._lock:
            return self._job_metrics.get(job_id)

    def get_all_work_item_metrics(self) -> dict[str, WorkItemMetrics]:
        """Get all work item metrics.

        Returns:
            Dictionary of work type to metrics.
        """
        with self._lock:
            return self._work_item_metrics.copy()

    def get_all_job_metrics(self) -> dict[str, JobMetrics]:
        """Get all job metrics.

        Returns:
            Dictionary of job ID to metrics.
        """
        with self._lock:
            return self._job_metrics.copy()

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all metrics.

        Returns:
            Summary dictionary with work item and job metrics.
        """
        with self._lock:
            return {
                "work_items": {
                    work_type: metrics.to_dict()
                    for work_type, metrics in self._work_item_metrics.items()
                },
                "jobs": {
                    job_id: metrics.to_dict()
                    for job_id, metrics in self._job_metrics.items()
                },
                "totals": {
                    "total_items_processed": sum(
                        m.total_processed for m in self._work_item_metrics.values()
                    ),
                    "total_items_failed": sum(
                        m.total_failed for m in self._work_item_metrics.values()
                    ),
                    "total_jobs": len(self._job_metrics),
                },
            }

    def add_handler(self, handler: Callable[[str, dict[str, Any]], None]) -> None:
        """Add a metrics event handler.

        Args:
            handler: Callback function(event_name, data).
        """
        with self._lock:
            self._event_handlers.append(handler)

    def remove_handler(self, handler: Callable[[str, dict[str, Any]], None]) -> None:
        """Remove a metrics event handler.

        Args:
            handler: Handler to remove.
        """
        with self._lock:
            if handler in self._event_handlers:
                self._event_handlers.remove(handler)

    def _notify(self, event_name: str, data: dict[str, Any]) -> None:
        """Notify all handlers of a metrics event.

        Args:
            event_name: Name of the event.
            data: Event data.
        """
        handlers = []
        with self._lock:
            handlers = self._event_handlers.copy()

        for handler in handlers:
            try:
                handler(event_name, data)
            except Exception:
                pass  # Don't let handler errors break metrics

    def reset(self) -> None:
        """Reset all metrics.

        Useful for testing.
        """
        with self._lock:
            self._work_item_metrics.clear()
            self._job_metrics.clear()


# Global metrics collector
_metrics_collector: MetricsCollector | None = None
_metrics_lock = threading.Lock()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector.

    Returns:
        Global MetricsCollector instance.
    """
    global _metrics_collector
    if _metrics_collector is None:
        with _metrics_lock:
            if _metrics_collector is None:
                _metrics_collector = MetricsCollector()
    return _metrics_collector


class timed_operation:
    """Context manager for timing operations.

    Records the duration of an operation as a metric.

    Example:
        >>> with timed_operation("DOC_CHUNK", job_id="job-123"):
        ...     process_chunk()
    """

    def __init__(
        self,
        work_type: str,
        job_id: str | None = None,
        metrics: MetricsCollector | None = None,
    ):
        """Initialize timed operation.

        Args:
            work_type: Type of work being timed.
            job_id: Optional job identifier.
            metrics: Optional metrics collector (uses global if None).
        """
        self.work_type = work_type
        self.job_id = job_id
        self.metrics = metrics or get_metrics()
        self._start_time: float = 0.0
        self._success = True

    def __enter__(self) -> "timed_operation":
        """Start timing.

        Returns:
            This context manager.
        """
        self._start_time = time.monotonic()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any,
    ) -> None:
        """Stop timing and record metric."""
        duration_ms = (time.monotonic() - self._start_time) * 1000
        # Only override success if there was an exception
        # (preserve manually set failure via mark_failed())
        if exc_type is not None:
            self._success = False
        self.metrics.record_work_item_processed(
            self.work_type,
            duration_ms=duration_ms,
            success=self._success,
            job_id=self.job_id,
        )

    def mark_failed(self) -> None:
        """Mark the operation as failed (even without exception)."""
        self._success = False
