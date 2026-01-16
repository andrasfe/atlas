"""Tests for event emission."""

import asyncio
import pytest

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit

from atlas.observability.events import (
    EventEmitter,
    Event,
    EventType,
    get_event_emitter,
    emit_event,
    emit_event_async,
)


class TestEvent:
    """Tests for Event dataclass."""

    def test_basic_event(self):
        """Test basic event creation."""
        event = Event(
            event_type=EventType.JOB_STARTED,
            job_id="job-123",
        )

        assert event.event_type == EventType.JOB_STARTED
        assert event.job_id == "job-123"
        assert event.timestamp is not None

    def test_event_with_data(self):
        """Test event with additional data."""
        event = Event(
            event_type=EventType.CHUNK_COMPLETED,
            job_id="job-123",
            work_id="chunk-001",
            data={"duration_ms": 150},
        )

        assert event.data["duration_ms"] == 150

    def test_to_dict(self):
        """Test event conversion to dictionary."""
        event = Event(
            event_type=EventType.JOB_STARTED,
            job_id="job-123",
            data={"chunks_total": 10},
        )
        data = event.to_dict()

        assert data["event_type"] == "job_started"
        assert data["job_id"] == "job-123"
        assert data["data"]["chunks_total"] == 10


class TestEventEmitter:
    """Tests for EventEmitter."""

    def setup_method(self):
        """Create fresh emitter for each test."""
        self.emitter = EventEmitter()

    def test_emit_event(self):
        """Test basic event emission."""
        received = []

        def handler(event):
            received.append(event)

        self.emitter.on(EventType.JOB_STARTED, handler)
        self.emitter.emit(EventType.JOB_STARTED, job_id="job-123")

        assert len(received) == 1
        assert received[0].job_id == "job-123"

    def test_emit_string_event_type(self):
        """Test emitting with string event type."""
        received = []

        def handler(event):
            received.append(event)

        self.emitter.on("custom_event", handler)
        self.emitter.emit("custom_event", job_id="job-123")

        assert len(received) == 1

    def test_emit_with_data(self):
        """Test emitting event with data."""
        received = []

        def handler(event):
            received.append(event)

        self.emitter.on(EventType.CHUNK_COMPLETED, handler)
        self.emitter.emit(
            EventType.CHUNK_COMPLETED,
            job_id="job-123",
            chunk_id="chunk-001",
            duration_ms=150,
        )

        assert received[0].data["chunk_id"] == "chunk-001"
        assert received[0].data["duration_ms"] == 150

    def test_multiple_handlers(self):
        """Test multiple handlers for same event."""
        received1 = []
        received2 = []

        def handler1(event):
            received1.append(event)

        def handler2(event):
            received2.append(event)

        self.emitter.on(EventType.JOB_STARTED, handler1)
        self.emitter.on(EventType.JOB_STARTED, handler2)
        self.emitter.emit(EventType.JOB_STARTED, job_id="job-123")

        assert len(received1) == 1
        assert len(received2) == 1

    def test_global_handler(self):
        """Test global handler for all events."""
        received = []

        def handler(event):
            received.append(event)

        self.emitter.on("*", handler)
        self.emitter.emit(EventType.JOB_STARTED, job_id="job-123")
        self.emitter.emit(EventType.CHUNK_COMPLETED, job_id="job-123")

        assert len(received) == 2

    def test_unregister_handler(self):
        """Test unregistering handler."""
        received = []

        def handler(event):
            received.append(event)

        unregister = self.emitter.on(EventType.JOB_STARTED, handler)
        self.emitter.emit(EventType.JOB_STARTED, job_id="job-123")
        unregister()
        self.emitter.emit(EventType.JOB_STARTED, job_id="job-456")

        assert len(received) == 1

    def test_off_method(self):
        """Test off method for removing handlers."""
        received = []

        def handler(event):
            received.append(event)

        self.emitter.on(EventType.JOB_STARTED, handler)
        self.emitter.off(EventType.JOB_STARTED, handler)
        self.emitter.emit(EventType.JOB_STARTED, job_id="job-123")

        assert len(received) == 0

    def test_event_history(self):
        """Test event history."""
        self.emitter.emit(EventType.JOB_STARTED, job_id="job-123")
        self.emitter.emit(EventType.CHUNK_COMPLETED, job_id="job-123")
        self.emitter.emit(EventType.JOB_COMPLETED, job_id="job-123")

        history = self.emitter.get_history()
        assert len(history) == 3

    def test_history_filter_by_type(self):
        """Test filtering history by event type."""
        self.emitter.emit(EventType.JOB_STARTED, job_id="job-123")
        self.emitter.emit(EventType.CHUNK_COMPLETED, job_id="job-123")
        self.emitter.emit(EventType.CHUNK_COMPLETED, job_id="job-123")

        history = self.emitter.get_history(event_type=EventType.CHUNK_COMPLETED)
        assert len(history) == 2

    def test_history_filter_by_job(self):
        """Test filtering history by job ID."""
        self.emitter.emit(EventType.CHUNK_COMPLETED, job_id="job-123")
        self.emitter.emit(EventType.CHUNK_COMPLETED, job_id="job-456")
        self.emitter.emit(EventType.CHUNK_COMPLETED, job_id="job-123")

        history = self.emitter.get_history(job_id="job-123")
        assert len(history) == 2

    def test_history_limit(self):
        """Test history limit."""
        for i in range(5):
            self.emitter.emit(EventType.CHUNK_COMPLETED, job_id=f"job-{i}")

        history = self.emitter.get_history(limit=3)
        assert len(history) == 3

    def test_history_max_size(self):
        """Test history max size."""
        emitter = EventEmitter(max_history=5)

        for i in range(10):
            emitter.emit(EventType.CHUNK_COMPLETED, job_id=f"job-{i}")

        history = emitter.get_history()
        assert len(history) == 5

    def test_clear_history(self):
        """Test clearing history."""
        self.emitter.emit(EventType.JOB_STARTED, job_id="job-123")
        self.emitter.clear_history()

        assert len(self.emitter.get_history()) == 0

    def test_clear_handlers(self):
        """Test clearing handlers."""
        received = []

        def handler(event):
            received.append(event)

        self.emitter.on(EventType.JOB_STARTED, handler)
        self.emitter.clear_handlers()
        self.emitter.emit(EventType.JOB_STARTED, job_id="job-123")

        assert len(received) == 0

    def test_handler_exception_does_not_break_emission(self):
        """Test that handler exceptions don't break other handlers."""
        received = []

        def bad_handler(event):
            raise ValueError("Handler error")

        def good_handler(event):
            received.append(event)

        self.emitter.on(EventType.JOB_STARTED, bad_handler)
        self.emitter.on(EventType.JOB_STARTED, good_handler)
        self.emitter.emit(EventType.JOB_STARTED, job_id="job-123")

        assert len(received) == 1


class TestAsyncEventEmitter:
    """Tests for async event emission."""

    def setup_method(self):
        """Create fresh emitter for each test."""
        self.emitter = EventEmitter()

    @pytest.mark.asyncio
    async def test_emit_async(self):
        """Test async event emission."""
        received = []

        async def async_handler(event):
            received.append(event)

        self.emitter.on(EventType.JOB_STARTED, async_handler)
        await self.emitter.emit_async(EventType.JOB_STARTED, job_id="job-123")

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_emit_async_mixed_handlers(self):
        """Test async emission with sync and async handlers."""
        sync_received = []
        async_received = []

        def sync_handler(event):
            sync_received.append(event)

        async def async_handler(event):
            async_received.append(event)

        self.emitter.on(EventType.JOB_STARTED, sync_handler)
        self.emitter.on(EventType.JOB_STARTED, async_handler)
        await self.emitter.emit_async(EventType.JOB_STARTED, job_id="job-123")

        assert len(sync_received) == 1
        assert len(async_received) == 1


class TestGlobalEmitter:
    """Tests for global emitter functions."""

    def test_get_event_emitter_singleton(self):
        """Test that get_event_emitter returns singleton."""
        emitter1 = get_event_emitter()
        emitter2 = get_event_emitter()
        assert emitter1 is emitter2

    def test_emit_event_function(self):
        """Test emit_event convenience function."""
        received = []
        emitter = get_event_emitter()

        def handler(event):
            received.append(event)

        unregister = emitter.on("test_event", handler)
        emit_event("test_event", job_id="job-123")
        unregister()

        assert len(received) >= 1

    @pytest.mark.asyncio
    async def test_emit_event_async_function(self):
        """Test emit_event_async convenience function."""
        received = []
        emitter = get_event_emitter()

        async def handler(event):
            received.append(event)

        unregister = emitter.on("test_async_event", handler)
        await emit_event_async("test_async_event", job_id="job-123")
        unregister()

        assert len(received) >= 1
