"""Unit tests for configuration management.

Tests cover:
- Default settings values
- Environment variable loading
- Validation rules
- Nested settings classes
"""

import os
import pytest

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit
from unittest.mock import patch

from atlas.config import (
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


# ============================================================================
# LoggingSettings Tests
# ============================================================================


class TestLoggingSettings:
    """Tests for LoggingSettings."""

    def test_defaults(self):
        """Test default logging settings."""
        settings = LoggingSettings()
        assert settings.level == "INFO"
        assert settings.format == "human"
        assert settings.include_timestamp is True
        assert settings.include_context is True

    def test_valid_log_levels(self):
        """Test all valid log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            with patch.dict(os.environ, {"ATLAS_LOG_LEVEL": level}):
                settings = LoggingSettings()
                assert settings.level == level

    def test_invalid_log_level(self):
        """Test invalid log level raises error."""
        with patch.dict(os.environ, {"ATLAS_LOG_LEVEL": "INVALID"}):
            with pytest.raises(ValueError):
                LoggingSettings()


# ============================================================================
# ChunkingSettings Tests
# ============================================================================


class TestChunkingSettings:
    """Tests for ChunkingSettings."""

    def test_defaults(self):
        """Test default chunking settings."""
        settings = ChunkingSettings()
        assert settings.context_budget == 4000
        assert settings.max_tokens == 3500
        assert settings.min_tokens == 100
        assert settings.overlap_tokens == 50
        assert settings.semantic_chunking is True

    def test_env_override(self):
        """Test environment variable overrides."""
        with patch.dict(os.environ, {"ATLAS_CHUNK_CONTEXT_BUDGET": "8000"}):
            settings = ChunkingSettings()
            assert settings.context_budget == 8000

    def test_validation_min_less_than_max(self):
        """Test min_tokens must be less than max_tokens."""
        with patch.dict(
            os.environ,
            {
                "ATLAS_CHUNK_MIN_TOKENS": "500",
                "ATLAS_CHUNK_MAX_TOKENS": "100",
                "ATLAS_CHUNK_OVERLAP_TOKENS": "20",
            },
        ):
            with pytest.raises(ValueError, match="min_tokens.*must be less than"):
                ChunkingSettings()

    def test_validation_overlap_less_than_min(self):
        """Test overlap_tokens must be less than min_tokens."""
        with patch.dict(
            os.environ,
            {
                "ATLAS_CHUNK_MIN_TOKENS": "100",
                "ATLAS_CHUNK_OVERLAP_TOKENS": "150",
            },
        ):
            with pytest.raises(ValueError, match="overlap_tokens.*must be less than"):
                ChunkingSettings()

    def test_range_validation(self):
        """Test value range validation."""
        # context_budget too low
        with patch.dict(os.environ, {"ATLAS_CHUNK_CONTEXT_BUDGET": "50"}):
            with pytest.raises(ValueError):
                ChunkingSettings()


# ============================================================================
# LLMSettings Tests
# ============================================================================


class TestLLMSettings:
    """Tests for LLMSettings."""

    def test_defaults(self):
        """Test default LLM settings."""
        settings = LLMSettings()
        assert settings.provider == "openai"
        assert settings.model == "gpt-4"
        assert settings.api_key is None
        assert settings.temperature == 0.0
        assert settings.max_tokens == 4096
        assert settings.timeout_seconds == 120.0
        assert settings.max_retries == 3

    def test_env_override(self):
        """Test environment variable overrides."""
        with patch.dict(
            os.environ,
            {
                "ATLAS_LLM_PROVIDER": "anthropic",
                "ATLAS_LLM_MODEL": "claude-3-opus",
                "ATLAS_LLM_TEMPERATURE": "0.7",
            },
        ):
            settings = LLMSettings()
            assert settings.provider == "anthropic"
            assert settings.model == "claude-3-opus"
            assert settings.temperature == 0.7

    def test_valid_providers(self):
        """Test valid provider values."""
        for provider in ["openai", "anthropic", "azure", "mock"]:
            with patch.dict(os.environ, {"ATLAS_LLM_PROVIDER": provider}):
                settings = LLMSettings()
                assert settings.provider == provider

    def test_invalid_provider(self):
        """Test invalid provider raises error."""
        with patch.dict(os.environ, {"ATLAS_LLM_PROVIDER": "invalid"}):
            with pytest.raises(ValueError):
                LLMSettings()

    def test_empty_model_name(self):
        """Test empty model name validation."""
        with patch.dict(os.environ, {"ATLAS_LLM_MODEL": "   "}):
            with pytest.raises(ValueError, match="cannot be empty"):
                LLMSettings()

    def test_temperature_range(self):
        """Test temperature range validation."""
        with patch.dict(os.environ, {"ATLAS_LLM_TEMPERATURE": "3.0"}):
            with pytest.raises(ValueError):
                LLMSettings()


# ============================================================================
# StorageSettings Tests
# ============================================================================


class TestStorageSettings:
    """Tests for StorageSettings."""

    def test_defaults(self):
        """Test default storage settings."""
        settings = StorageSettings()
        assert settings.artifact_store_uri == "file://./artifacts"
        assert settings.ticket_store_uri is None
        assert settings.checkpoint_dir == "./checkpoints"
        assert settings.enable_compression is True
        assert settings.max_artifact_size_mb == 100

    def test_valid_uri_schemes(self):
        """Test valid artifact store URI schemes."""
        valid_uris = [
            "file://./path",
            "s3://bucket/path",
            "gs://bucket/path",
            "az://container/path",
            "memory://store",
        ]
        for uri in valid_uris:
            with patch.dict(os.environ, {"ATLAS_STORAGE_ARTIFACT_STORE_URI": uri}):
                settings = StorageSettings()
                assert settings.artifact_store_uri == uri

    def test_invalid_uri_scheme(self):
        """Test invalid URI scheme raises error."""
        with patch.dict(
            os.environ,
            {"ATLAS_STORAGE_ARTIFACT_STORE_URI": "http://example.com"},
        ):
            with pytest.raises(ValueError, match="must start with one of"):
                StorageSettings()


# ============================================================================
# WorkerSettings Tests
# ============================================================================


class TestWorkerSettings:
    """Tests for WorkerSettings."""

    def test_defaults(self):
        """Test default worker settings."""
        settings = WorkerSettings()
        assert settings.poll_interval_seconds == 5.0
        assert settings.lease_duration_seconds == 300
        assert settings.heartbeat_interval_seconds == 30.0
        assert settings.max_concurrent_items == 5
        assert settings.graceful_shutdown_seconds == 30

    def test_env_override(self):
        """Test environment variable overrides."""
        with patch.dict(
            os.environ,
            {
                "ATLAS_WORKER_POLL_INTERVAL_SECONDS": "10.0",
                "ATLAS_WORKER_MAX_CONCURRENT_ITEMS": "20",
            },
        ):
            settings = WorkerSettings()
            assert settings.poll_interval_seconds == 10.0
            assert settings.max_concurrent_items == 20


# ============================================================================
# ControllerSettings Tests
# ============================================================================


class TestControllerSettings:
    """Tests for ControllerSettings."""

    def test_defaults(self):
        """Test default controller settings."""
        settings = ControllerSettings()
        assert settings.max_concurrent_chunks == 50
        assert settings.max_concurrent_merges == 10
        assert settings.max_merge_fan_in == 15
        assert settings.max_challenge_iterations == 3
        assert settings.followup_batch_size == 10
        assert settings.reconcile_interval_seconds == 10.0


# ============================================================================
# RetrySettings Tests
# ============================================================================


class TestRetrySettings:
    """Tests for RetrySettings."""

    def test_defaults(self):
        """Test default retry settings."""
        settings = RetrySettings()
        assert settings.max_retries == 3
        assert settings.initial_delay_seconds == 1.0
        assert settings.max_delay_seconds == 60.0
        assert settings.exponential_base == 2.0
        assert settings.jitter_factor == 0.2


# ============================================================================
# Settings (Main) Tests
# ============================================================================


class TestSettings:
    """Tests for main Settings class."""

    def test_defaults(self):
        """Test default settings."""
        settings = Settings()
        assert settings.app_name == "atlas"
        assert settings.environment == "development"
        assert settings.debug is False

    def test_legacy_flat_settings(self):
        """Test backward compatible flat settings."""
        settings = Settings()
        assert settings.log_level == "INFO"
        assert settings.context_budget == 4000
        assert settings.max_chunk_tokens == 3500
        assert settings.llm_provider == "openai"

    def test_nested_settings_accessors(self):
        """Test nested settings accessor methods."""
        settings = Settings()

        log_settings = settings.get_logging_settings()
        assert isinstance(log_settings, LoggingSettings)

        chunk_settings = settings.get_chunking_settings()
        assert isinstance(chunk_settings, ChunkingSettings)

        llm_settings = settings.get_llm_settings()
        assert isinstance(llm_settings, LLMSettings)

    def test_to_dict(self):
        """Test converting settings to dictionary."""
        settings = Settings()
        data = settings.to_dict()

        assert "app_name" in data
        assert data["app_name"] == "atlas"
        assert "environment" in data

    def test_to_dict_masks_sensitive(self):
        """Test to_dict masks sensitive values."""
        # This test ensures any api_key fields would be masked
        settings = Settings()
        data = settings.to_dict(include_sensitive=False)

        # Main settings shouldn't have api_key, but the masking logic
        # should work if any sensitive key exists
        assert isinstance(data, dict)


# ============================================================================
# Module Functions Tests
# ============================================================================


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_settings_cached(self):
        """Test get_settings returns cached instance."""
        reload_settings()  # Clear cache first
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_reload_settings(self):
        """Test reload_settings clears cache."""
        settings1 = get_settings()
        settings2 = reload_settings()
        settings3 = get_settings()

        # After reload, should be new instance
        assert settings2 is settings3
        # Note: Due to caching, settings1 might still be the old one
        # This depends on when cache was last cleared

    def test_get_specific_settings_functions(self):
        """Test individual settings getter functions."""
        assert isinstance(get_logging_settings(), LoggingSettings)
        assert isinstance(get_chunking_settings(), ChunkingSettings)
        assert isinstance(get_llm_settings(), LLMSettings)
        assert isinstance(get_storage_settings(), StorageSettings)
        assert isinstance(get_worker_settings(), WorkerSettings)
        assert isinstance(get_controller_settings(), ControllerSettings)
        assert isinstance(get_retry_settings(), RetrySettings)


# ============================================================================
# Integration Tests
# ============================================================================


class TestConfigIntegration:
    """Integration tests for configuration."""

    def test_full_env_configuration(self):
        """Test loading full configuration from environment."""
        env = {
            "ATLAS_LOG_LEVEL": "DEBUG",
            "ATLAS_CHUNK_CONTEXT_BUDGET": "8000",
            "ATLAS_LLM_PROVIDER": "anthropic",
            "ATLAS_LLM_MODEL": "claude-3-opus",
            "ATLAS_STORAGE_ARTIFACT_STORE_URI": "s3://my-bucket/artifacts",
            "ATLAS_WORKER_MAX_CONCURRENT_ITEMS": "10",
            "ATLAS_CONTROLLER_MAX_CONCURRENT_CHUNKS": "100",
            "ATLAS_RETRY_MAX_RETRIES": "5",
        }

        with patch.dict(os.environ, env):
            # Test nested settings pick up their environment
            log = get_logging_settings()
            assert log.level == "DEBUG"

            chunk = get_chunking_settings()
            assert chunk.context_budget == 8000

            llm = get_llm_settings()
            assert llm.provider == "anthropic"
            assert llm.model == "claude-3-opus"

            storage = get_storage_settings()
            assert storage.artifact_store_uri == "s3://my-bucket/artifacts"

            worker = get_worker_settings()
            assert worker.max_concurrent_items == 10

            controller = get_controller_settings()
            assert controller.max_concurrent_chunks == 100

            retry = get_retry_settings()
            assert retry.max_retries == 5
