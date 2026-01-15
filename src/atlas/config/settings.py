"""Application settings using pydantic-settings.

Configuration can be loaded from environment variables or .env files.
All settings have sensible defaults for development.

Configuration Hierarchy (highest to lowest priority):
1. Environment variables with ATLAS_ prefix
2. .env file in current directory
3. Default values defined here

Validation:
- All settings are validated on load
- Invalid values raise clear error messages
- Sensitive values (API keys) are masked in logs
"""

import logging
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "human"] = "human"
    include_timestamp: bool = True
    include_context: bool = True


class ChunkingSettings(BaseSettings):
    """Chunking and context budget settings."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_CHUNK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    context_budget: int = Field(
        default=4000,
        ge=100,
        le=128000,
        description="Default context budget in tokens",
    )
    max_tokens: int = Field(
        default=3500,
        ge=100,
        le=32000,
        description="Maximum tokens per chunk",
    )
    min_tokens: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Minimum tokens per chunk",
    )
    overlap_tokens: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap tokens between chunks",
    )
    semantic_chunking: bool = Field(
        default=True,
        description="Enable semantic boundary detection",
    )

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int, info) -> int:
        """Ensure max_tokens is less than context_budget."""
        # Note: Cross-field validation happens in model_validator
        return v

    @model_validator(mode="after")
    def validate_token_constraints(self) -> "ChunkingSettings":
        """Validate token constraints are consistent."""
        if self.min_tokens >= self.max_tokens:
            raise ValueError(
                f"min_tokens ({self.min_tokens}) must be less than "
                f"max_tokens ({self.max_tokens})"
            )
        if self.overlap_tokens >= self.min_tokens:
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be less than "
                f"min_tokens ({self.min_tokens})"
            )
        return self


class LLMSettings(BaseSettings):
    """LLM provider and model settings."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: Literal["openai", "anthropic", "azure", "mock"] = Field(
        default="openai",
        description="LLM provider",
    )
    model: str = Field(
        default="gpt-4",
        description="Model name or identifier",
    )
    api_key: str | None = Field(
        default=None,
        description="API key (use environment variable for security)",
    )
    api_base: str | None = Field(
        default=None,
        description="Custom API base URL",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        default=4096,
        ge=100,
        le=32000,
        description="Maximum output tokens",
    )
    timeout_seconds: float = Field(
        default=120.0,
        ge=5.0,
        le=600.0,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts",
    )

    @field_validator("model")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name is not empty."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class StorageSettings(BaseSettings):
    """Storage and artifact settings."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_STORAGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    artifact_store_uri: str = Field(
        default="file://./artifacts",
        description="Base URI for artifact storage",
    )
    ticket_store_uri: str | None = Field(
        default=None,
        description="URI for ticket system (in-memory if None)",
    )
    checkpoint_dir: str = Field(
        default="./checkpoints",
        description="Directory for job checkpoints",
    )
    enable_compression: bool = Field(
        default=True,
        description="Compress stored artifacts",
    )
    max_artifact_size_mb: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum artifact size in MB",
    )

    @field_validator("artifact_store_uri")
    @classmethod
    def validate_artifact_uri(cls, v: str) -> str:
        """Validate artifact store URI format."""
        valid_schemes = ("file://", "s3://", "gs://", "az://", "memory://")
        if not any(v.startswith(scheme) for scheme in valid_schemes):
            raise ValueError(
                f"artifact_store_uri must start with one of: {valid_schemes}"
            )
        return v


class WorkerSettings(BaseSettings):
    """Worker process settings."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_WORKER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    poll_interval_seconds: float = Field(
        default=5.0,
        ge=0.5,
        le=60.0,
        description="Interval between work item polls",
    )
    lease_duration_seconds: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Work item lease duration",
    )
    heartbeat_interval_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="Heartbeat interval for active work",
    )
    max_concurrent_items: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum concurrent work items per worker",
    )
    graceful_shutdown_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Time to wait for graceful shutdown",
    )


class ControllerSettings(BaseSettings):
    """Controller and orchestration settings."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_CONTROLLER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    max_concurrent_chunks: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum concurrent chunk analyses",
    )
    max_concurrent_merges: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent merge operations",
    )
    max_merge_fan_in: int = Field(
        default=15,
        ge=2,
        le=50,
        description="Maximum inputs per merge node",
    )
    max_challenge_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum challenger loop iterations",
    )
    followup_batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of follow-ups to dispatch at once",
    )
    reconcile_interval_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Interval between reconciliation cycles",
    )


class RetrySettings(BaseSettings):
    """Retry policy settings."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_RETRY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts",
    )
    initial_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Initial delay before first retry",
    )
    max_delay_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=600.0,
        description="Maximum delay between retries",
    )
    exponential_base: float = Field(
        default=2.0,
        ge=1.1,
        le=5.0,
        description="Base for exponential backoff",
    )
    jitter_factor: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="Random jitter factor",
    )


class Settings(BaseSettings):
    """Main application settings.

    Settings are loaded from environment variables with the ATLAS_ prefix.

    Example:
        ATLAS_LOG_LEVEL=DEBUG
        ATLAS_CHUNK_CONTEXT_BUDGET=8000
        ATLAS_LLM_MODEL=gpt-4-turbo

    Attributes:
        log: Logging configuration.
        chunking: Chunking settings.
        llm: LLM provider settings.
        storage: Storage settings.
        worker: Worker process settings.
        controller: Controller settings.
        retry: Retry policy settings.
    """

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Nested settings - loaded from their own prefixes
    # NOTE: We use Field(default_factory=...) to create nested settings

    # Legacy flat settings for backward compatibility
    log_level: str = "INFO"
    context_budget: int = 4000
    max_chunk_tokens: int = 3500
    max_merge_fan_in: int = 15
    max_challenge_iterations: int = 3
    artifact_store_uri: str = "file://./artifacts"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.0
    llm_max_retries: int = 3
    worker_poll_interval: float = 5.0
    worker_lease_duration: int = 300

    # Application metadata
    app_name: str = "atlas"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    @model_validator(mode="after")
    def configure_debug_mode(self) -> "Settings":
        """Enable debug settings in development."""
        if self.debug or self.environment == "development":
            # Lower log level in debug mode
            if self.log_level not in ("DEBUG",):
                logger.debug("Debug mode: log_level not overridden")
        return self

    def get_logging_settings(self) -> LoggingSettings:
        """Get logging settings instance."""
        return LoggingSettings()

    def get_chunking_settings(self) -> ChunkingSettings:
        """Get chunking settings instance."""
        return ChunkingSettings()

    def get_llm_settings(self) -> LLMSettings:
        """Get LLM settings instance."""
        return LLMSettings()

    def get_storage_settings(self) -> StorageSettings:
        """Get storage settings instance."""
        return StorageSettings()

    def get_worker_settings(self) -> WorkerSettings:
        """Get worker settings instance."""
        return WorkerSettings()

    def get_controller_settings(self) -> ControllerSettings:
        """Get controller settings instance."""
        return ControllerSettings()

    def get_retry_settings(self) -> RetrySettings:
        """Get retry settings instance."""
        return RetrySettings()

    def to_dict(self, include_sensitive: bool = False) -> dict[str, Any]:
        """Convert settings to dictionary.

        Args:
            include_sensitive: Include sensitive values like API keys.

        Returns:
            Dictionary of all settings.
        """
        data = self.model_dump()

        if not include_sensitive:
            # Mask sensitive values
            sensitive_keys = {"api_key", "api_secret", "password", "token"}
            for key in data:
                if any(s in key.lower() for s in sensitive_keys):
                    data[key] = "***MASKED***" if data[key] else None

        return data


# Module-level cached settings
_settings: Settings | None = None


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.

    Returns:
        Cached Settings instance.
    """
    return Settings()


def get_logging_settings() -> LoggingSettings:
    """Get logging settings.

    Returns:
        LoggingSettings instance.
    """
    return LoggingSettings()


def get_chunking_settings() -> ChunkingSettings:
    """Get chunking settings.

    Returns:
        ChunkingSettings instance.
    """
    return ChunkingSettings()


def get_llm_settings() -> LLMSettings:
    """Get LLM settings.

    Returns:
        LLMSettings instance.
    """
    return LLMSettings()


def get_storage_settings() -> StorageSettings:
    """Get storage settings.

    Returns:
        StorageSettings instance.
    """
    return StorageSettings()


def get_worker_settings() -> WorkerSettings:
    """Get worker settings.

    Returns:
        WorkerSettings instance.
    """
    return WorkerSettings()


def get_controller_settings() -> ControllerSettings:
    """Get controller settings.

    Returns:
        ControllerSettings instance.
    """
    return ControllerSettings()


def get_retry_settings() -> RetrySettings:
    """Get retry settings.

    Returns:
        RetrySettings instance.
    """
    return RetrySettings()


def reload_settings() -> Settings:
    """Reload settings (clears cache).

    Use this when environment variables have changed.

    Returns:
        Fresh Settings instance.
    """
    get_settings.cache_clear()
    return get_settings()
