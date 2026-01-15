"""Structured logging with context support.

Provides structured logging that automatically includes context like
job_id, work_item_id, and other relevant identifiers.

Design Principles:
- Structured logs for machine parsing
- Context propagation through the call stack
- Minimal performance overhead
- Compatible with standard Python logging

Usage:
    >>> logger = get_logger(__name__)
    >>> with with_context(job_id="job-123"):
    ...     logger.info("Starting job")  # Automatically includes job_id
"""

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

# Context variable for structured logging context
_log_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "log_context", default={}
)

# Global logging configuration
_configured = False

T = TypeVar("T")


@dataclass
class LogContext:
    """Context holder for structured logging.

    Stores context values that will be automatically included in log messages.

    Attributes:
        job_id: Current job identifier.
        work_id: Current work item identifier.
        work_type: Type of work being processed.
        phase: Current workflow phase.
        worker_id: Identifier of the worker.
        extra: Additional context values.
    """

    job_id: str | None = None
    work_id: str | None = None
    work_type: str | None = None
    phase: str | None = None
    worker_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary, excluding None values.

        Returns:
            Dictionary of non-None context values.
        """
        result = {}
        if self.job_id:
            result["job_id"] = self.job_id
        if self.work_id:
            result["work_id"] = self.work_id
        if self.work_type:
            result["work_type"] = self.work_type
        if self.phase:
            result["phase"] = self.phase
        if self.worker_id:
            result["worker_id"] = self.worker_id
        result.update(self.extra)
        return result


class with_context:
    """Context manager for adding structured logging context.

    Adds context values that will be included in all log messages
    within the context manager's scope.

    Example:
        >>> with with_context(job_id="job-123", phase="chunk"):
        ...     logger.info("Processing")  # Includes job_id and phase
    """

    def __init__(self, **kwargs: Any):
        """Initialize context with given values.

        Args:
            **kwargs: Context key-value pairs.
        """
        self._context = kwargs
        self._token: contextvars.Token[dict[str, Any]] | None = None

    def __enter__(self) -> "with_context":
        """Enter context and set values.

        Returns:
            This context manager.
        """
        current = _log_context.get().copy()
        current.update(self._context)
        self._token = _log_context.set(current)
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any,
    ) -> None:
        """Exit context and restore previous values."""
        if self._token is not None:
            _log_context.reset(self._token)


def get_context() -> dict[str, Any]:
    """Get current logging context.

    Returns:
        Dictionary of current context values.
    """
    return _log_context.get().copy()


def set_context(**kwargs: Any) -> None:
    """Set values in the current logging context.

    This directly modifies the context without using a context manager.
    Use with_context() for scoped context.

    Args:
        **kwargs: Context key-value pairs to set.
    """
    current = _log_context.get().copy()
    current.update(kwargs)
    _log_context.set(current)


def clear_context() -> None:
    """Clear all values from the logging context."""
    _log_context.set({})


class StructuredFormatter(logging.Formatter):
    """Log formatter that produces structured JSON output.

    Includes timestamp, level, message, and context in JSON format.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record to format.

        Returns:
            JSON-formatted log string.
        """
        # Build base log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add context from context variable
        context = get_context()
        if context:
            log_entry["context"] = context

        # Add context from record extras
        extras = {}
        skip_attrs = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "exc_info", "exc_text", "thread", "threadName",
            "message", "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in skip_attrs and not key.startswith("_"):
                extras[key] = value
        if extras:
            log_entry["extra"] = extras

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class HumanReadableFormatter(logging.Formatter):
    """Log formatter that produces human-readable output with context.

    Includes timestamp, level, message, and context in readable format.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for human reading.

        Args:
            record: Log record to format.

        Returns:
            Human-readable log string.
        """
        # Get timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # Build base message
        message = f"{timestamp} [{record.levelname:8}] {record.name}: {record.getMessage()}"

        # Add context
        context = get_context()
        extras = {}
        skip_attrs = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "exc_info", "exc_text", "thread", "threadName",
            "message", "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in skip_attrs and not key.startswith("_"):
                extras[key] = value

        all_context = {**context, **extras}
        if all_context:
            context_str = " ".join(f"{k}={v}" for k, v in all_context.items())
            message = f"{message} | {context_str}"

        # Add exception info if present
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"

        return message


class StructuredLogger:
    """Logger wrapper that adds structured context to log messages.

    Wraps a standard Python logger to automatically include context
    from the context variable and any extra kwargs.

    Attributes:
        logger: Underlying Python logger.
    """

    def __init__(self, name: str):
        """Initialize structured logger.

        Args:
            name: Logger name (typically __name__).
        """
        self.logger = logging.getLogger(name)

    def _log(
        self,
        level: int,
        msg: str,
        *args: Any,
        exc_info: Any = None,
        **kwargs: Any,
    ) -> None:
        """Log message with context.

        Args:
            level: Log level.
            msg: Log message.
            *args: Message format arguments.
            exc_info: Exception info.
            **kwargs: Extra context values.
        """
        # Get current context
        context = get_context()
        context.update(kwargs)

        # Log with extras as record attributes
        self.logger.log(level, msg, *args, exc_info=exc_info, extra=kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message.

        Args:
            msg: Log message.
            *args: Message format arguments.
            **kwargs: Extra context values.
        """
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log info message.

        Args:
            msg: Log message.
            *args: Message format arguments.
            **kwargs: Extra context values.
        """
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message.

        Args:
            msg: Log message.
            *args: Message format arguments.
            **kwargs: Extra context values.
        """
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log error message.

        Args:
            msg: Log message.
            *args: Message format arguments.
            **kwargs: Extra context values.
        """
        self._log(logging.ERROR, msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log exception with traceback.

        Args:
            msg: Log message.
            *args: Message format arguments.
            **kwargs: Extra context values.
        """
        self._log(logging.ERROR, msg, *args, exc_info=True, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log critical message.

        Args:
            msg: Log message.
            *args: Message format arguments.
            **kwargs: Extra context values.
        """
        self._log(logging.CRITICAL, msg, *args, **kwargs)


# Logger cache
_loggers: dict[str, StructuredLogger] = {}


def get_logger(name: str) -> StructuredLogger:
    """Get or create a structured logger.

    Args:
        name: Logger name (typically __name__).

    Returns:
        StructuredLogger instance.
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)
    return _loggers[name]


def configure_logging(
    level: int = logging.INFO,
    format: str = "human",  # "human" or "json"
    stream: Any = None,
) -> None:
    """Configure logging for the Atlas application.

    Args:
        level: Minimum log level.
        format: Output format ("human" or "json").
        stream: Output stream (defaults to stderr).
    """
    global _configured
    if _configured:
        return

    # Set up root logger
    root_logger = logging.getLogger("atlas")
    root_logger.setLevel(level)

    # Create handler
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level)

    # Set formatter
    if format == "json":
        formatter = StructuredFormatter()
    else:
        formatter = HumanReadableFormatter()
    handler.setFormatter(formatter)

    # Add handler
    root_logger.addHandler(handler)

    _configured = True
