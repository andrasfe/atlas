"""Observability framework for Atlas workflow orchestration.

This module provides structured logging, metrics tracking, and event emission
for workflow monitoring and debugging.

Components:
- StructuredLogger: Context-aware logging with job_id and work_item_id
- MetricsCollector: Tracks workflow metrics (items processed, durations)
- EventEmitter: Emits workflow transition events

Usage:
    >>> from atlas.observability import get_logger, get_metrics, emit_event
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing chunk", job_id="job-123", chunk_id="chunk-001")
    >>> get_metrics().record_work_item_processed("DOC_CHUNK", duration_ms=150)
    >>> emit_event("chunk_completed", job_id="job-123", chunk_id="chunk-001")
"""

from atlas.observability.logging import (
    StructuredLogger,
    get_logger,
    configure_logging,
    LogContext,
    with_context,
)
from atlas.observability.metrics import (
    MetricsCollector,
    get_metrics,
    WorkItemMetrics,
    JobMetrics,
)
from atlas.observability.events import (
    EventEmitter,
    Event,
    EventType,
    get_event_emitter,
    emit_event,
)

__all__ = [
    # Logging
    "StructuredLogger",
    "get_logger",
    "configure_logging",
    "LogContext",
    "with_context",
    # Metrics
    "MetricsCollector",
    "get_metrics",
    "WorkItemMetrics",
    "JobMetrics",
    # Events
    "EventEmitter",
    "Event",
    "EventType",
    "get_event_emitter",
    "emit_event",
]
