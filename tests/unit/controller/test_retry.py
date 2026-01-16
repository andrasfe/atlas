"""Unit tests for retry policy and failure handling.

Tests cover:
- RetryConfig delay computation
- RetryState tracking
- DeadLetterQueue operations
- RetryManager execution with retry
- Error classification
"""

import asyncio
import pytest

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from atlas.controller.retry import (
    DeadLetterQueue,
    FailureReason,
    RetryConfig,
    RetryManager,
    RetryState,
    configure_retry,
    get_dlq,
    get_retry_manager,
)
from atlas.models.enums import WorkItemStatus, WorkItemType
from atlas.models.work_item import WorkItem, DocChunkPayload, ChunkLocator
from atlas.models.artifact import ArtifactRef


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_work_item() -> WorkItem:
    """Create a sample work item for testing."""
    return WorkItem(
        work_id="test-work-001",
        work_type=WorkItemType.DOC_CHUNK,
        status=WorkItemStatus.READY,
        payload=DocChunkPayload(
            job_id="test-job-001",
            artifact_ref=ArtifactRef(
                artifact_id="TEST.cbl",
                artifact_type="cobol",
                artifact_version="abc123",
                artifact_uri="s3://bucket/TEST.cbl",
            ),
            manifest_uri="s3://bucket/manifest.json",
            chunk_id="chunk-001",
            chunk_locator=ChunkLocator(start_line=1, end_line=100),
            result_uri="s3://bucket/results/chunk-001.json",
        ),
    )


@pytest.fixture
def retry_config() -> RetryConfig:
    """Create a test retry configuration."""
    return RetryConfig(
        max_retries=3,
        initial_delay_seconds=0.01,  # Fast for testing
        max_delay_seconds=0.1,
        exponential_base=2.0,
        jitter_factor=0.1,
    )


# ============================================================================
# RetryConfig Tests
# ============================================================================


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay_seconds == 1.0
        assert config.max_delay_seconds == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter_factor == 0.2
        assert FailureReason.TRANSIENT in config.retryable_reasons
        assert FailureReason.TIMEOUT in config.retryable_reasons
        assert FailureReason.VALIDATION not in config.retryable_reasons

    def test_compute_delay_exponential(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(
            initial_delay_seconds=1.0,
            exponential_base=2.0,
            jitter_factor=0.0,  # No jitter for predictable tests
            max_delay_seconds=100.0,
        )

        # Attempt 0: 1 * 2^0 = 1
        assert config.compute_delay(0) == 1.0

        # Attempt 1: 1 * 2^1 = 2
        assert config.compute_delay(1) == 2.0

        # Attempt 2: 1 * 2^2 = 4
        assert config.compute_delay(2) == 4.0

    def test_compute_delay_capped_at_max(self):
        """Test delay is capped at max_delay_seconds."""
        config = RetryConfig(
            initial_delay_seconds=1.0,
            exponential_base=2.0,
            jitter_factor=0.0,
            max_delay_seconds=5.0,
        )

        # Attempt 10: Would be 1024, but capped at 5
        assert config.compute_delay(10) == 5.0

    def test_compute_delay_with_jitter(self):
        """Test delay has jitter applied."""
        config = RetryConfig(
            initial_delay_seconds=10.0,
            jitter_factor=0.2,  # +/- 20%
        )

        delays = [config.compute_delay(0) for _ in range(100)]

        # All delays should be within jitter range: 8.0 to 12.0
        assert all(8.0 <= d <= 12.0 for d in delays)

        # Delays should vary (not all the same)
        assert len(set(delays)) > 1

    def test_should_retry_under_max(self):
        """Test should_retry returns True when under max attempts."""
        config = RetryConfig(max_retries=3)

        assert config.should_retry(FailureReason.TRANSIENT, attempt=0)
        assert config.should_retry(FailureReason.TRANSIENT, attempt=1)
        assert config.should_retry(FailureReason.TRANSIENT, attempt=2)

    def test_should_retry_at_max(self):
        """Test should_retry returns False at max attempts."""
        config = RetryConfig(max_retries=3)

        assert not config.should_retry(FailureReason.TRANSIENT, attempt=3)
        assert not config.should_retry(FailureReason.TRANSIENT, attempt=4)

    def test_should_retry_non_retryable(self):
        """Test should_retry returns False for non-retryable reasons."""
        config = RetryConfig(max_retries=3)

        assert not config.should_retry(FailureReason.VALIDATION, attempt=0)
        assert not config.should_retry(FailureReason.LOGIC, attempt=0)


# ============================================================================
# RetryState Tests
# ============================================================================


class TestRetryState:
    """Tests for RetryState."""

    def test_initial_state(self):
        """Test initial retry state."""
        state = RetryState(work_id="test-001")

        assert state.work_id == "test-001"
        assert state.attempt_count == 0
        assert state.last_attempt_at is None
        assert state.last_error is None
        assert not state.is_exhausted
        assert state.history == []

    def test_record_success_attempt(self):
        """Test recording a successful attempt."""
        state = RetryState(work_id="test-001")
        state.record_attempt(success=True)

        assert state.attempt_count == 1
        assert state.last_attempt_at is not None
        assert len(state.history) == 1
        assert state.history[0]["success"] is True

    def test_record_failure_attempt(self):
        """Test recording a failed attempt."""
        state = RetryState(work_id="test-001")
        state.record_attempt(
            success=False,
            failure_reason=FailureReason.TIMEOUT,
            error="Request timed out",
        )

        assert state.attempt_count == 1
        assert state.last_failure_reason == FailureReason.TIMEOUT
        assert state.last_error == "Request timed out"
        assert state.history[0]["success"] is False
        assert state.history[0]["failure_reason"] == "timeout"

    def test_schedule_retry(self):
        """Test scheduling a retry."""
        state = RetryState(work_id="test-001")
        state.schedule_retry(5.0)

        assert state.next_retry_at is not None
        # Should be approximately 5 seconds in the future
        retry_time = datetime.fromisoformat(state.next_retry_at)
        now = datetime.now(timezone.utc)
        diff = (retry_time - now).total_seconds()
        assert 4.5 <= diff <= 5.5

    def test_mark_exhausted(self):
        """Test marking state as exhausted."""
        state = RetryState(work_id="test-001")
        state.schedule_retry(5.0)
        state.mark_exhausted()

        assert state.is_exhausted
        assert state.next_retry_at is None

    def test_to_dict(self):
        """Test serialization to dictionary."""
        state = RetryState(work_id="test-001")
        state.record_attempt(
            success=False,
            failure_reason=FailureReason.TIMEOUT,
            error="Timeout",
        )

        data = state.to_dict()

        assert data["work_id"] == "test-001"
        assert data["attempt_count"] == 1
        assert data["last_failure_reason"] == "timeout"
        assert data["last_error"] == "Timeout"
        assert len(data["history"]) == 1

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "work_id": "test-001",
            "attempt_count": 2,
            "last_attempt_at": "2024-01-15T10:00:00+00:00",
            "last_failure_reason": "rate_limit",
            "last_error": "Too many requests",
            "next_retry_at": "2024-01-15T10:01:00+00:00",
            "is_exhausted": False,
            "history": [],
        }

        state = RetryState.from_dict(data)

        assert state.work_id == "test-001"
        assert state.attempt_count == 2
        assert state.last_failure_reason == FailureReason.RATE_LIMIT
        assert not state.is_exhausted


# ============================================================================
# DeadLetterQueue Tests
# ============================================================================


class TestDeadLetterQueue:
    """Tests for DeadLetterQueue."""

    def test_add_item(self, sample_work_item):
        """Test adding item to DLQ."""
        dlq = DeadLetterQueue()
        state = RetryState(work_id=sample_work_item.work_id)
        state.mark_exhausted()

        dlq.add(sample_work_item, state, "Max retries exceeded")

        assert len(dlq) == 1
        entry = dlq.get(sample_work_item.work_id)
        assert entry is not None
        assert entry["reason"] == "Max retries exceeded"
        assert not entry["reprocessed"]

    def test_get_nonexistent(self):
        """Test getting nonexistent item."""
        dlq = DeadLetterQueue()
        assert dlq.get("nonexistent") is None

    def test_list_items(self, sample_work_item):
        """Test listing DLQ items."""
        dlq = DeadLetterQueue()
        state = RetryState(work_id=sample_work_item.work_id)

        dlq.add(sample_work_item, state, "Error")

        items = dlq.list_items()
        assert len(items) == 1

    def test_list_items_excludes_reprocessed(self, sample_work_item):
        """Test listing excludes reprocessed items by default."""
        dlq = DeadLetterQueue()
        state = RetryState(work_id=sample_work_item.work_id)

        dlq.add(sample_work_item, state, "Error")
        dlq.mark_reprocessed(sample_work_item.work_id)

        items = dlq.list_items(include_reprocessed=False)
        assert len(items) == 0

        items = dlq.list_items(include_reprocessed=True)
        assert len(items) == 1

    def test_mark_reprocessed(self, sample_work_item):
        """Test marking item as reprocessed."""
        dlq = DeadLetterQueue()
        state = RetryState(work_id=sample_work_item.work_id)

        dlq.add(sample_work_item, state, "Error")
        result = dlq.mark_reprocessed(sample_work_item.work_id)

        assert result is True
        entry = dlq.get(sample_work_item.work_id)
        assert entry["reprocessed"] is True
        assert entry["reprocessed_at"] is not None

    def test_mark_reprocessed_nonexistent(self):
        """Test marking nonexistent item."""
        dlq = DeadLetterQueue()
        assert dlq.mark_reprocessed("nonexistent") is False

    def test_remove_item(self, sample_work_item):
        """Test removing item from DLQ."""
        dlq = DeadLetterQueue()
        state = RetryState(work_id=sample_work_item.work_id)

        dlq.add(sample_work_item, state, "Error")
        assert len(dlq) == 1

        result = dlq.remove(sample_work_item.work_id)
        assert result is True
        assert len(dlq) == 0

    def test_purge_all(self, sample_work_item):
        """Test purging all items."""
        dlq = DeadLetterQueue()
        state = RetryState(work_id=sample_work_item.work_id)

        dlq.add(sample_work_item, state, "Error")
        count = dlq.purge()

        assert count == 1
        assert len(dlq) == 0


# ============================================================================
# RetryManager Tests
# ============================================================================


class TestRetryManager:
    """Tests for RetryManager."""

    def test_get_state_creates_new(self):
        """Test get_state creates new state if not exists."""
        manager = RetryManager()
        state = manager.get_state("work-001")

        assert state.work_id == "work-001"
        assert state.attempt_count == 0

    def test_get_state_returns_existing(self):
        """Test get_state returns existing state."""
        manager = RetryManager()
        state1 = manager.get_state("work-001")
        state1.record_attempt(success=False, error="test")

        state2 = manager.get_state("work-001")
        assert state2.attempt_count == 1

    def test_clear_state(self):
        """Test clearing state."""
        manager = RetryManager()
        manager.get_state("work-001")
        manager.clear_state("work-001")

        state = manager.get_state("work-001")
        assert state.attempt_count == 0

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(
        self, sample_work_item, retry_config
    ):
        """Test execute_with_retry on immediate success."""
        manager = RetryManager(retry_config)

        async def operation(item):
            return "success"

        result, state = await manager.execute_with_retry(
            sample_work_item, operation
        )

        assert result == "success"
        assert state.attempt_count == 1
        assert not state.is_exhausted

    @pytest.mark.asyncio
    async def test_execute_with_retry_transient_failure(
        self, sample_work_item, retry_config
    ):
        """Test execute_with_retry with transient failures then success."""
        manager = RetryManager(retry_config)
        call_count = 0

        async def operation(item):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"

        result, state = await manager.execute_with_retry(
            sample_work_item, operation
        )

        assert result == "success"
        assert call_count == 3
        assert state.attempt_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_retry_exhausted(
        self, sample_work_item, retry_config
    ):
        """Test execute_with_retry when retries are exhausted."""
        manager = RetryManager(retry_config)

        async def operation(item):
            raise ConnectionError("Persistent network error")

        result, state = await manager.execute_with_retry(
            sample_work_item, operation
        )

        assert result is None
        assert state.is_exhausted
        # Initial attempt + max_retries - 1 retry attempts = max_retries total
        # (first attempt is counted, then max_retries-1 more retries)
        # But our logic: attempt 0 fails, check should_retry(0) -> True
        # attempt 1 fails, check should_retry(1) -> True
        # attempt 2 fails, check should_retry(2) -> True
        # attempt 3 fails, check should_retry(3) -> False (>= max_retries)
        # So we expect max_retries + 1 attempts, but should_retry checks attempt
        # before incrementing, so max_retries attempts total
        assert state.attempt_count == retry_config.max_retries
        assert len(manager.dlq) == 1

    @pytest.mark.asyncio
    async def test_execute_with_retry_non_retryable(
        self, sample_work_item, retry_config
    ):
        """Test execute_with_retry with non-retryable error."""
        manager = RetryManager(retry_config)

        async def operation(item):
            raise ValueError("Invalid input")

        result, state = await manager.execute_with_retry(
            sample_work_item, operation
        )

        assert result is None
        assert state.is_exhausted
        assert state.attempt_count == 1  # Only one attempt
        assert len(manager.dlq) == 1

    def test_should_retry_transient(self, sample_work_item):
        """Test should_retry for transient errors."""
        manager = RetryManager()
        error = ConnectionError("Network error")

        assert manager.should_retry(sample_work_item, error)

    def test_should_retry_validation(self, sample_work_item):
        """Test should_retry for validation errors."""
        manager = RetryManager()
        error = ValueError("Invalid input")

        assert not manager.should_retry(sample_work_item, error)

    def test_record_failure(self, sample_work_item):
        """Test recording a failure."""
        manager = RetryManager()
        error = TimeoutError("Request timed out")

        state = manager.record_failure(sample_work_item, error)

        assert state.attempt_count == 1
        assert state.last_failure_reason == FailureReason.TIMEOUT
        assert state.next_retry_at is not None  # Retry scheduled

    def test_record_success(self, sample_work_item):
        """Test recording a success."""
        manager = RetryManager()

        state = manager.record_success(sample_work_item)

        assert state.attempt_count == 1
        assert not state.is_exhausted

    def test_get_ready_for_retry(self, sample_work_item):
        """Test getting work items ready for retry."""
        manager = RetryManager()

        # Record a failure and schedule immediate retry
        state = manager.get_state(sample_work_item.work_id)
        state.record_attempt(
            success=False,
            failure_reason=FailureReason.TRANSIENT,
            error="test",
        )
        state.schedule_retry(-1.0)  # Already past

        ready = manager.get_ready_for_retry()
        assert sample_work_item.work_id in ready

    def test_default_error_classifier_timeout(self):
        """Test error classifier for timeout errors."""
        manager = RetryManager()

        reason = manager._default_error_classifier(TimeoutError("Timed out"))
        assert reason == FailureReason.TIMEOUT

    def test_default_error_classifier_rate_limit(self):
        """Test error classifier for rate limit errors."""
        manager = RetryManager()

        reason = manager._default_error_classifier(
            Exception("HTTP 429: Too Many Requests")
        )
        assert reason == FailureReason.RATE_LIMIT

    def test_default_error_classifier_validation(self):
        """Test error classifier for validation errors."""
        manager = RetryManager()

        reason = manager._default_error_classifier(ValueError("Invalid value"))
        assert reason == FailureReason.VALIDATION

    def test_default_error_classifier_connection(self):
        """Test error classifier for connection errors."""
        manager = RetryManager()

        reason = manager._default_error_classifier(
            ConnectionError("Connection refused")
        )
        assert reason == FailureReason.TRANSIENT


# ============================================================================
# Module Functions Tests
# ============================================================================


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_retry_manager_singleton(self):
        """Test get_retry_manager returns singleton."""
        manager1 = get_retry_manager()
        manager2 = get_retry_manager()

        assert manager1 is manager2

    def test_get_dlq_singleton(self):
        """Test get_dlq returns singleton."""
        dlq1 = get_dlq()
        dlq2 = get_dlq()

        assert dlq1 is dlq2

    def test_configure_retry(self):
        """Test configuring retry behavior."""
        config = RetryConfig(max_retries=5)
        configure_retry(config)

        manager = get_retry_manager()
        assert manager.config.max_retries == 5
