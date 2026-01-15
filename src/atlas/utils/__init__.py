"""Shared utilities for Atlas.

This module provides common utilities used across the codebase,
including hashing for content addressing and idempotency key generation.
"""

from atlas.utils.hashing import (
    compute_content_hash,
    compute_idempotency_key,
    compute_work_item_idempotency_key,
    compute_work_item_key_from_work_item,
    compute_artifact_uri,
    compute_result_uri,
    content_hash_short,
)

__all__ = [
    "compute_content_hash",
    "compute_idempotency_key",
    "compute_work_item_idempotency_key",
    "compute_work_item_key_from_work_item",
    "compute_artifact_uri",
    "compute_result_uri",
    "content_hash_short",
]
