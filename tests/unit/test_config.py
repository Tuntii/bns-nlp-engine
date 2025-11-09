"""
Unit tests for configuration management.
"""

import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from bnsnlp.core.config import (
    Config,
    EmbedConfig,
    LoggingConfig,
    PreprocessConfig,
    SearchConfig,
    TelemetryConfig,
)
from bnsnlp.core.exceptions import ConfigurationError


class TestLoggingConfig:
    """Tests for LoggingConfig."""

    def test_default_values(self):
        """Test default logging configuration values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.format == "json"
        assert config.output == "stdout"

    def test_custom_values(self):
        """Test custom logging configuration values."""
        config = LoggingConfig(level="DEBUG", format="text", output="file.log")
        assert config.level == "DEBUG"
        assert config.format == "text"
        assert config.output == "file.log"

    def test_invalid_level(self):
        """Test validation error for invalid log level."""
        with pytest.raises(ValidationError) as exc_info:
            LoggingConfig(level="INVALID")
        # Pydantic's Literal validation provides the error message
        assert "Input should be" in str(exc_info.value)


class TestTelemetryConfig:
    """Tests for TelemetryConfig."""

    def test_default_values(self):
        """Test default telemetry configuration values."""
        config = TelemetryConfig()
        assert config.enabled is False
        assert config.endpoint is None

    def test_enabled_without_endpoint(self):
        """Test validation error when telemetry is enabled without endpoint."""
        with pytest.raises(ValidationError) as exc_info:
            TelemetryConfig(enabled=True, endpoint=None)
        assert "endpoint is required" in str(exc_info.value)

    def test_enabled_with_endpoint(self):
        """Test valid telemetry configuration with endpoint."""
        config = TelemetryConfig(enabled=True, endpoint="https://telemetry.example.com")
        assert config.enabled is True
        assert config.endpoint == "https://telemetry.example.com"


class TestPreprocessConfig:
    """Tests for PreprocessConfig."""

    def test_default_values(self):
        """Test default preprocessing configuration values."""
        config = PreprocessConfig()
        assert config.lowercase is True
        assert config.remove_punctuation is True
        assert config.remove_stopwords is True
        assert config.lemmatize is True
        assert config.batch_size == 32

    def test_custom_values(self):
        """Test custom preprocessing configuration values."""
        config = PreprocessConfig(
            lowercase=False,
            remove_punctuation=False,
            remove_stopwords=False,
            lemmatize=False,
            batch_size=64,
        )
        assert config.lowercase is False
        assert config.batch_size == 64

    def test_invalid_batch_size(self):
        """Test validation error for invalid batch size."""
        with pytest.raises(ValidationError):
            PreprocessConfig(batch_size=0)

        with pytest.raises(ValidationError):
            PreprocessConfig(batch_size=-1)


class TestEmbedConfig:
    """Tests for EmbedConfig."""

    def test_default_values(self):
        """Test default embedding configuration values."""
        config = EmbedConfig()
        assert config.provider == "openai"
        assert config.model == "text-embedding-3-small"
        assert config.batch_size == 16
        assert config.use_gpu is True
        assert config.api_key is None

    def test_custom_values(self):
        """Test custom embedding configuration values."""
        config = EmbedConfig(
            provider="cohere",
            model="embed-multilingual-v3.0",
            batch_size=32,
            use_gpu=False,
            api_key="test-key",
        )
        assert config.provider == "cohere"
        assert config.model == "embed-multilingual-v3.0"
        assert config.batch_size == 32
        assert config.use_gpu is False
        assert config.api_key == "test-key"

    def test_invalid_provider(self):
        """Test validation error for invalid provider."""
        with pytest.raises(ValidationError) as exc_info:
            EmbedConfig(provider="invalid")
        assert "Invalid embedding provider" in str(exc_info.value)

    def test_invalid_batch_size(self):
        """Test validation error for invalid batch size."""
        with pytest.raises(ValidationError):
            EmbedConfig(batch_size=0)


class TestSearchConfig:
    """Tests for SearchConfig."""

    def test_default_values(self):
        """Test default search configuration values."""
        config = SearchConfig()
        assert config.provider == "faiss"
        assert config.top_k == 10
        assert config.similarity_threshold == 0.7

    def test_custom_values(self):
        """Test custom search configuration values."""
        config = SearchConfig(provider="qdrant", top_k=20, similarity_threshold=0.8)
        assert config.provider == "qdrant"
        assert config.top_k == 20
        assert config.similarity_threshold == 0.8

    def test_invalid_provider(self):
        """Test validation error for invalid provider."""
        with pytest.raises(ValidationError) as exc_info:
            SearchConfig(provider="invalid")
        assert "Invalid search provider" in str(exc_info.value)

    def test_invalid_top_k(self):
        """Test validation error for invalid top_k."""
        with pytest.raises(ValidationError):
            SearchConfig(top_k=0)

        with pytest.raises(ValidationError):
            SearchConfig(top_k=-1)

    def test_invalid_similarity_threshold(self):
        """Test validation error for invalid similarity threshold."""
        with pytest.raises(ValidationError):
            SearchConfig(similarity_threshold=-0.1)

        with pytest.raises(ValidationError):
            SearchConfig(similarity_threshold=1.5)


class TestConfig:
    """Tests for main Config class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = Config()
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.telemetry, TelemetryConfig)
        assert isinstance(config.preprocess, PreprocessConfig)
        assert isinstance(config.embed, EmbedConfig)
        assert isinstance(config.search, SearchConfig)

    def test_custom_values(self):
        """Test custom configuration values."""
        config = Config(
            logging=LoggingConfig(level="DEBUG"),
            embed=EmbedConfig(provider="cohere"),
        )
        assert config.logging.level == "DEBUG"
        assert config.embed.provider == "cohere"

    def test_from_yaml(self, temp_config_dir: Path):
        """Test loading configuration from YAML file."""
        config_file = temp_config_dir / "config.yaml"
        config_data = {
            "logging": {"level": "DEBUG", "format": "text"},
            "embed": {"provider": "cohere", "batch_size": 32},
            "search": {"provider": "qdrant", "top_k": 20},
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = Config.from_yaml(config_file)
        assert config.logging.level == "DEBUG"
        assert config.logging.format == "text"
        assert config.embed.provider == "cohere"
        assert config.embed.batch_size == 32
        assert config.search.provider == "qdrant"
        assert config.search.top_k == 20

    def test_from_yaml_file_not_found(self):
        """Test error when YAML file is not found."""
        with pytest.raises(ConfigurationError) as exc_info:
            Config.from_yaml(Path("nonexistent.yaml"))
        assert "not found" in str(exc_info.value)

    def test_from_yaml_invalid_yaml(self, temp_config_dir: Path):
        """Test error when YAML file is invalid."""
        config_file = temp_config_dir / "invalid.yaml"
        with open(config_file, "w") as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(ConfigurationError) as exc_info:
            Config.from_yaml(config_file)
        assert "Invalid YAML" in str(exc_info.value)

    def test_from_yaml_empty_file(self, temp_config_dir: Path):
        """Test loading from empty YAML file uses defaults."""
        config_file = temp_config_dir / "empty.yaml"
        with open(config_file, "w") as f:
            f.write("")

        config = Config.from_yaml(config_file)
        assert config.logging.level == "INFO"
        assert config.embed.provider == "openai"

    def test_from_env(self, monkeypatch):
        """Test loading configuration from environment variables."""
        monkeypatch.setenv("BNSNLP_LOGGING__LEVEL", "DEBUG")
        monkeypatch.setenv("BNSNLP_LOGGING__FORMAT", "text")
        monkeypatch.setenv("BNSNLP_EMBED__PROVIDER", "cohere")
        monkeypatch.setenv("BNSNLP_EMBED__BATCH_SIZE", "32")
        monkeypatch.setenv("BNSNLP_EMBED__USE_GPU", "false")
        monkeypatch.setenv("BNSNLP_SEARCH__TOP_K", "20")
        monkeypatch.setenv("BNSNLP_SEARCH__SIMILARITY_THRESHOLD", "0.8")

        config = Config.from_env()
        assert config.logging.level == "DEBUG"
        assert config.logging.format == "text"
        assert config.embed.provider == "cohere"
        assert config.embed.batch_size == 32
        assert config.embed.use_gpu is False
        assert config.search.top_k == 20
        assert config.search.similarity_threshold == 0.8

    def test_from_env_boolean_parsing(self, monkeypatch):
        """Test boolean value parsing from environment variables."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("yes", True),
            ("1", True),
            ("on", True),
            ("false", False),
            ("False", False),
            ("no", False),
            ("0", False),
            ("off", False),
        ]

        for env_value, expected in test_cases:
            monkeypatch.setenv("BNSNLP_EMBED__USE_GPU", env_value)
            config = Config.from_env()
            assert config.embed.use_gpu == expected

    def test_from_env_numeric_parsing(self, monkeypatch):
        """Test numeric value parsing from environment variables."""
        monkeypatch.setenv("BNSNLP_EMBED__BATCH_SIZE", "64")
        monkeypatch.setenv("BNSNLP_SEARCH__SIMILARITY_THRESHOLD", "0.85")

        config = Config.from_env()
        assert config.embed.batch_size == 64
        assert isinstance(config.embed.batch_size, int)
        assert config.search.similarity_threshold == 0.85
        assert isinstance(config.search.similarity_threshold, float)

    def test_from_yaml_with_env_override(self, temp_config_dir: Path, monkeypatch):
        """Test environment variables override YAML values."""
        config_file = temp_config_dir / "config.yaml"
        config_data = {
            "logging": {"level": "INFO", "format": "json"},
            "embed": {"provider": "openai", "batch_size": 16},
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Set environment variables that should override YAML
        monkeypatch.setenv("BNSNLP_LOGGING__LEVEL", "DEBUG")
        monkeypatch.setenv("BNSNLP_EMBED__BATCH_SIZE", "32")

        config = Config.from_yaml_with_env_override(config_file)

        # Environment variables should override YAML
        assert config.logging.level == "DEBUG"
        assert config.embed.batch_size == 32

        # YAML values should be used where no env override exists
        assert config.logging.format == "json"
        assert config.embed.provider == "openai"

    def test_parse_env_value(self):
        """Test environment value parsing."""
        assert Config._parse_env_value("true") is True
        assert Config._parse_env_value("false") is False
        assert Config._parse_env_value("123") == 123
        assert Config._parse_env_value("3.14") == 3.14
        assert Config._parse_env_value("text") == "text"

    def test_deep_merge(self):
        """Test deep merge of dictionaries."""
        base = {
            "logging": {"level": "INFO", "format": "json"},
            "embed": {"provider": "openai"},
        }
        override = {
            "logging": {"level": "DEBUG"},
            "search": {"top_k": 20},
        }

        result = Config._deep_merge(base, override)

        assert result["logging"]["level"] == "DEBUG"
        assert result["logging"]["format"] == "json"
        assert result["embed"]["provider"] == "openai"
        assert result["search"]["top_k"] == 20
