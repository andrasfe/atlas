"""Tests for metrics collection."""

import pytest

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit
import time

from atlas.observability.metrics import (
    MetricsCollector,
    get_metrics,
    WorkItemMetrics,
    JobMetrics,
    timed_operation,
)


class TestWorkItemMetrics:
    """Tests for work item metrics dataclass."""

    def test_avg_duration_empty(self):
        """Test average duration with no items."""
        metrics = WorkItemMetrics(work_type="DOC_CHUNK")
        assert metrics.avg_duration_ms == 0.0

    def test_avg_duration(self):
        """Test average duration calculation."""
        metrics = WorkItemMetrics(
            work_type="DOC_CHUNK",
            total_processed=10,
            total_duration_ms=1000.0,
        )
        assert metrics.avg_duration_ms == 100.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = WorkItemMetrics(
            work_type="DOC_CHUNK",
            total_processed=5,
            total_failed=1,
            total_duration_ms=500.0,
            min_duration_ms=50.0,
            max_duration_ms=200.0,
        )
        data = metrics.to_dict()

        assert data["work_type"] == "DOC_CHUNK"
        assert data["total_processed"] == 5
        assert data["total_failed"] == 1
        assert data["avg_duration_ms"] == 100.0
        assert data["min_duration_ms"] == 50.0
        assert data["max_duration_ms"] == 200.0


class TestJobMetrics:
    """Tests for job metrics dataclass."""

    def test_chunks_progress_empty(self):
        """Test progress with no chunks."""
        metrics = JobMetrics(job_id="job-123")
        assert metrics.chunks_progress == 100.0

    def test_chunks_progress(self):
        """Test chunk progress calculation."""
        metrics = JobMetrics(
            job_id="job-123",
            chunks_total=10,
            chunks_done=5,
        )
        assert metrics.chunks_progress == 50.0

    def test_merges_progress(self):
        """Test merge progress calculation."""
        metrics = JobMetrics(
            job_id="job-123",
            merges_total=4,
            merges_done=2,
        )
        assert metrics.merges_progress == 50.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = JobMetrics(
            job_id="job-123",
            phase="chunk",
            chunks_total=10,
            chunks_done=5,
        )
        data = metrics.to_dict()

        assert data["job_id"] == "job-123"
        assert data["phase"] == "chunk"
        assert data["chunks_total"] == 10
        assert data["chunks_done"] == 5
        assert data["chunks_progress"] == 50.0


class TestMetricsCollector:
    """Tests for metrics collector."""

    def setup_method(self):
        """Create fresh collector for each test."""
        self.collector = MetricsCollector()

    def test_record_work_item_processed(self):
        """Test recording processed work item."""
        self.collector.record_work_item_processed(
            "DOC_CHUNK", duration_ms=100.0
        )

        metrics = self.collector.get_work_item_metrics("DOC_CHUNK")
        assert metrics is not None
        assert metrics.total_processed == 1
        assert metrics.total_duration_ms == 100.0

    def test_record_multiple_items(self):
        """Test recording multiple work items."""
        self.collector.record_work_item_processed("DOC_CHUNK", duration_ms=50.0)
        self.collector.record_work_item_processed("DOC_CHUNK", duration_ms=150.0)
        self.collector.record_work_item_processed("DOC_CHUNK", duration_ms=100.0)

        metrics = self.collector.get_work_item_metrics("DOC_CHUNK")
        assert metrics.total_processed == 3
        assert metrics.avg_duration_ms == 100.0
        assert metrics.min_duration_ms == 50.0
        assert metrics.max_duration_ms == 150.0

    def test_record_failed_item(self):
        """Test recording failed work item."""
        self.collector.record_work_item_processed(
            "DOC_CHUNK", duration_ms=100.0, success=False
        )

        metrics = self.collector.get_work_item_metrics("DOC_CHUNK")
        assert metrics.total_failed == 1
        assert metrics.total_processed == 0

    def test_record_job_started(self):
        """Test recording job start."""
        self.collector.record_job_started(
            "job-123", chunks_total=10, merges_total=3
        )

        metrics = self.collector.get_job_metrics("job-123")
        assert metrics is not None
        assert metrics.chunks_total == 10
        assert metrics.merges_total == 3
        assert metrics.phase == "plan"
        assert metrics.started_at is not None

    def test_record_phase_transition(self):
        """Test recording phase transition."""
        self.collector.record_job_started("job-123")
        self.collector.record_job_phase_transition("job-123", "plan", "chunk")

        metrics = self.collector.get_job_metrics("job-123")
        assert metrics.phase == "chunk"
        assert len(metrics.phase_transitions) == 2

    def test_record_job_progress(self):
        """Test recording job progress."""
        self.collector.record_job_started("job-123", chunks_total=10)
        self.collector.record_job_progress("job-123", chunks_done=5)

        metrics = self.collector.get_job_metrics("job-123")
        assert metrics.chunks_done == 5
        assert metrics.chunks_progress == 50.0

    def test_record_job_completed(self):
        """Test recording job completion."""
        self.collector.record_job_started("job-123")
        self.collector.record_job_completed("job-123", success=True)

        metrics = self.collector.get_job_metrics("job-123")
        assert metrics.phase == "complete"

    def test_record_job_failed(self):
        """Test recording job failure."""
        self.collector.record_job_started("job-123")
        self.collector.record_job_completed("job-123", success=False)

        metrics = self.collector.get_job_metrics("job-123")
        assert metrics.phase == "failed"

    def test_get_summary(self):
        """Test getting metrics summary."""
        self.collector.record_job_started("job-123")
        self.collector.record_work_item_processed("DOC_CHUNK", duration_ms=100.0)
        self.collector.record_work_item_processed("DOC_MERGE", duration_ms=200.0)

        summary = self.collector.get_summary()
        assert "work_items" in summary
        assert "jobs" in summary
        assert "totals" in summary
        assert summary["totals"]["total_items_processed"] == 2

    def test_event_handler(self):
        """Test event handler registration."""
        events = []

        def handler(event_name, data):
            events.append((event_name, data))

        self.collector.add_handler(handler)
        self.collector.record_work_item_processed("DOC_CHUNK", duration_ms=100.0)

        assert len(events) == 1
        assert events[0][0] == "work_item_processed"

    def test_reset(self):
        """Test resetting metrics."""
        self.collector.record_job_started("job-123")
        self.collector.record_work_item_processed("DOC_CHUNK", duration_ms=100.0)

        self.collector.reset()

        assert self.collector.get_job_metrics("job-123") is None
        assert self.collector.get_work_item_metrics("DOC_CHUNK") is None


class TestTimedOperation:
    """Tests for timed_operation context manager."""

    def test_basic_timing(self):
        """Test basic operation timing."""
        collector = MetricsCollector()

        with timed_operation("DOC_CHUNK", metrics=collector):
            time.sleep(0.01)  # 10ms

        metrics = collector.get_work_item_metrics("DOC_CHUNK")
        assert metrics.total_processed == 1
        assert metrics.total_duration_ms >= 10.0

    def test_timing_with_exception(self):
        """Test timing when exception occurs."""
        collector = MetricsCollector()

        with pytest.raises(ValueError):
            with timed_operation("DOC_CHUNK", metrics=collector):
                raise ValueError("Test error")

        metrics = collector.get_work_item_metrics("DOC_CHUNK")
        assert metrics.total_failed == 1

    def test_timing_with_job_id(self):
        """Test timing with job ID."""
        collector = MetricsCollector()

        with timed_operation("DOC_CHUNK", job_id="job-123", metrics=collector):
            pass

        job_metrics = collector.get_job_metrics("job-123")
        assert job_metrics is not None
        assert job_metrics.chunks_done == 1

    def test_mark_failed(self):
        """Test manually marking operation as failed."""
        collector = MetricsCollector()

        with timed_operation("DOC_CHUNK", metrics=collector) as op:
            op.mark_failed()

        metrics = collector.get_work_item_metrics("DOC_CHUNK")
        assert metrics.total_failed == 1


class TestGlobalMetrics:
    """Tests for global metrics functions."""

    def test_get_metrics_singleton(self):
        """Test that get_metrics returns singleton."""
        metrics1 = get_metrics()
        metrics2 = get_metrics()
        assert metrics1 is metrics2
