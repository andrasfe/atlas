"""Tests for structured logging."""

import json
import logging
import pytest

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit
from io import StringIO

from atlas.observability.logging import (
    StructuredLogger,
    get_logger,
    configure_logging,
    with_context,
    get_context,
    set_context,
    clear_context,
    StructuredFormatter,
    HumanReadableFormatter,
)


class TestWithContext:
    """Tests for context manager."""

    def setup_method(self):
        """Clear context before each test."""
        clear_context()

    def test_basic_context(self):
        """Test basic context setting."""
        with with_context(job_id="job-123"):
            ctx = get_context()
            assert ctx["job_id"] == "job-123"

    def test_context_cleared_after_exit(self):
        """Test context is cleared after exiting context manager."""
        with with_context(job_id="job-123"):
            pass
        ctx = get_context()
        assert "job_id" not in ctx

    def test_nested_context(self):
        """Test nested context managers."""
        with with_context(job_id="job-123"):
            assert get_context()["job_id"] == "job-123"

            with with_context(work_id="work-456"):
                ctx = get_context()
                assert ctx["job_id"] == "job-123"
                assert ctx["work_id"] == "work-456"

            # Inner context should be restored
            ctx = get_context()
            assert ctx["job_id"] == "job-123"
            assert "work_id" not in ctx

    def test_context_override(self):
        """Test that inner context can override outer values."""
        with with_context(job_id="job-123", phase="chunk"):
            with with_context(phase="merge"):
                ctx = get_context()
                assert ctx["job_id"] == "job-123"
                assert ctx["phase"] == "merge"


class TestSetContext:
    """Tests for direct context setting."""

    def setup_method(self):
        """Clear context before each test."""
        clear_context()

    def test_set_context(self):
        """Test setting context directly."""
        set_context(job_id="job-123")
        assert get_context()["job_id"] == "job-123"

    def test_set_multiple_values(self):
        """Test setting multiple values."""
        set_context(job_id="job-123", work_id="work-456")
        ctx = get_context()
        assert ctx["job_id"] == "job-123"
        assert ctx["work_id"] == "work-456"


class TestStructuredFormatter:
    """Tests for JSON formatter."""

    def test_basic_format(self):
        """Test basic JSON formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_format_with_extras(self):
        """Test formatting with extra attributes."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.job_id = "job-123"
        output = formatter.format(record)
        data = json.loads(output)

        assert data["extra"]["job_id"] == "job-123"


class TestHumanReadableFormatter:
    """Tests for human-readable formatter."""

    def test_basic_format(self):
        """Test basic human-readable formatting."""
        formatter = HumanReadableFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)

        assert "INFO" in output
        assert "test" in output
        assert "Test message" in output

    def test_format_with_extras(self):
        """Test formatting with extra attributes."""
        formatter = HumanReadableFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.job_id = "job-123"
        output = formatter.format(record)

        assert "job_id=job-123" in output


class TestStructuredLogger:
    """Tests for structured logger."""

    def setup_method(self):
        """Clear context before each test."""
        clear_context()

    def test_get_logger(self):
        """Test logger retrieval."""
        logger = get_logger("test.module")
        assert isinstance(logger, StructuredLogger)

    def test_logger_caching(self):
        """Test that loggers are cached."""
        logger1 = get_logger("test.module")
        logger2 = get_logger("test.module")
        assert logger1 is logger2

    def test_log_levels(self):
        """Test different log levels."""
        logger = get_logger("test.levels")
        # Just verify methods exist and don't raise
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

    def test_log_with_context(self):
        """Test logging with context."""
        logger = get_logger("test.context")
        with with_context(job_id="job-123"):
            # This should include job_id in the log output
            logger.info("Processing", work_id="work-456")
