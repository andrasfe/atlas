"""Controller interfaces for workflow orchestration.

The Controller is the orchestration core that coordinates the analysis
workflow. It operates as a reconcile loop: observe state, compute
missing work, create it, gate/advance.

Design Principles:
- Reconcile loop pattern (observe -> compute -> create -> gate)
- Idempotent operations (safe to re-run after crashes)
- Gates merge tickets until prerequisites complete
- Routes challenger issues to bounded follow-ups
"""

from atlas.controller.base import Controller, ControllerConfig, ReconcileResult
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

__all__ = [
    "Controller",
    "ControllerConfig",
    "ReconcileResult",
    "DeadLetterQueue",
    "FailureReason",
    "RetryConfig",
    "RetryManager",
    "RetryState",
    "configure_retry",
    "get_dlq",
    "get_retry_manager",
]
