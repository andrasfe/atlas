"""Configuration management for Atlas.

This module provides configuration handling using pydantic-settings
for environment variable support and validation.

Settings are organized into logical groups:
- LoggingSettings: Logging configuration
- ChunkingSettings: Chunking and tokenization
- LLMSettings: LLM provider settings
- StorageSettings: Artifact and ticket storage
- WorkerSettings: Worker process configuration
- ControllerSettings: Orchestration settings
- RetrySettings: Retry policy configuration

Each settings class loads from environment variables with its own prefix:
- ATLAS_LOG_*: Logging settings
- ATLAS_CHUNK_*: Chunking settings
- ATLAS_LLM_*: LLM settings
- ATLAS_STORAGE_*: Storage settings
- ATLAS_WORKER_*: Worker settings
- ATLAS_CONTROLLER_*: Controller settings
- ATLAS_RETRY_*: Retry settings
"""

from atlas.config.settings import (
    Settings,
    LoggingSettings,
    ChunkingSettings,
    LLMSettings,
    StorageSettings,
    WorkerSettings,
    ControllerSettings,
    RetrySettings,
    get_settings,
    get_logging_settings,
    get_chunking_settings,
    get_llm_settings,
    get_storage_settings,
    get_worker_settings,
    get_controller_settings,
    get_retry_settings,
    reload_settings,
)

__all__ = [
    "Settings",
    "LoggingSettings",
    "ChunkingSettings",
    "LLMSettings",
    "StorageSettings",
    "WorkerSettings",
    "ControllerSettings",
    "RetrySettings",
    "get_settings",
    "get_logging_settings",
    "get_chunking_settings",
    "get_llm_settings",
    "get_storage_settings",
    "get_worker_settings",
    "get_controller_settings",
    "get_retry_settings",
    "reload_settings",
]
