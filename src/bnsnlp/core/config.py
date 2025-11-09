"""
Configuration management for bns-nlp-engine.

This module provides Pydantic-based configuration models with support for
YAML files and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )
    format: Literal["json", "text"] = Field(default="json", description="Log format")
    output: str = Field(default="stdout", description="Log output destination")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v


class TelemetryConfig(BaseModel):
    """Telemetry configuration."""

    enabled: bool = Field(default=False, description="Enable telemetry (opt-in)")
    endpoint: Optional[str] = Field(default=None, description="Telemetry endpoint URL")

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, v: Optional[str], info) -> Optional[str]:
        """Validate telemetry endpoint."""
        if info.data.get("enabled") and not v:
            raise ValueError("Telemetry endpoint is required when telemetry is enabled")
        return v


class PreprocessConfig(BaseModel):
    """Preprocessing configuration."""

    lowercase: bool = Field(default=True, description="Convert text to lowercase")
    remove_punctuation: bool = Field(default=True, description="Remove punctuation marks")
    remove_stopwords: bool = Field(default=True, description="Remove Turkish stop words")
    lemmatize: bool = Field(default=True, description="Apply lemmatization")
    batch_size: int = Field(default=32, description="Batch size for processing", gt=0)

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size is positive."""
        if v <= 0:
            raise ValueError("Batch size must be greater than 0")
        return v


class EmbedConfig(BaseModel):
    """Embedding configuration."""

    provider: str = Field(default="openai", description="Embedding provider")
    model: str = Field(default="text-embedding-3-small", description="Model name for embeddings")
    batch_size: int = Field(default=16, description="Batch size for embedding", gt=0)
    use_gpu: bool = Field(default=True, description="Use GPU if available")
    api_key: Optional[str] = Field(default=None, description="API key for embedding provider")

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size is positive."""
        if v <= 0:
            raise ValueError("Batch size must be greater than 0")
        return v

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate embedding provider."""
        valid_providers = ["openai", "cohere", "huggingface"]
        if v not in valid_providers:
            raise ValueError(f"Invalid embedding provider: {v}. Must be one of {valid_providers}")
        return v


class SearchConfig(BaseModel):
    """Search configuration."""

    provider: str = Field(default="faiss", description="Search backend provider")
    top_k: int = Field(default=10, description="Number of top results to return", gt=0)
    similarity_threshold: float = Field(
        default=0.7, description="Minimum similarity threshold", ge=0.0, le=1.0
    )

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        """Validate top_k is positive."""
        if v <= 0:
            raise ValueError("top_k must be greater than 0")
        return v

    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v: float) -> float:
        """Validate similarity threshold is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        return v

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate search provider."""
        valid_providers = ["faiss", "qdrant", "pinecone"]
        if v not in valid_providers:
            raise ValueError(f"Invalid search provider: {v}. Must be one of {valid_providers}")
        return v


class Config(BaseModel):
    """Main configuration for bns-nlp-engine."""

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Config instance

        Raises:
            ConfigurationError: If file cannot be read or parsed
        """
        from bnsnlp.core.exceptions import ConfigurationError

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return cls(**(data or {}))
        except FileNotFoundError as e:
            raise ConfigurationError(
                f"Configuration file not found: {path}",
                context={"path": str(path)},
            ) from e
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in configuration file: {path}",
                context={"path": str(path), "error": str(e)},
            ) from e
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration from {path}: {str(e)}",
                context={"path": str(path), "error": str(e)},
            ) from e

    @classmethod
    def from_env(cls, prefix: str = "BNSNLP") -> "Config":
        """
        Load configuration from environment variables.

        Environment variables should be prefixed with the given prefix and use
        double underscores to separate nested keys. For example:
        - BNSNLP_LOGGING__LEVEL=DEBUG
        - BNSNLP_EMBED__PROVIDER=openai
        - BNSNLP_EMBED__API_KEY=sk-...

        Args:
            prefix: Environment variable prefix (default: "BNSNLP")

        Returns:
            Config instance with values from environment variables
        """
        config_dict: Dict[str, Any] = {}

        # Parse environment variables
        for key, value in os.environ.items():
            if not key.startswith(f"{prefix}_"):
                continue

            # Remove prefix and split by double underscore
            key_parts = key[len(prefix) + 1 :].lower().split("__")

            if len(key_parts) == 2:
                section, field = key_parts
                if section not in config_dict:
                    config_dict[section] = {}
                # Convert string values to appropriate types
                config_dict[section][field] = cls._parse_env_value(value)

        return cls(**config_dict)

    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """
        Parse environment variable value to appropriate type.

        Args:
            value: String value from environment variable

        Returns:
            Parsed value (bool, int, float, or str)
        """
        # Boolean values
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        if value.lower() in ("false", "no", "0", "off"):
            return False

        # Numeric values
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # String value
        return value

    @classmethod
    def from_yaml_with_env_override(cls, path: Path, prefix: str = "BNSNLP") -> "Config":
        """
        Load configuration from YAML file with environment variable overrides.

        Environment variables take priority over YAML values.

        Args:
            path: Path to YAML configuration file
            prefix: Environment variable prefix (default: "BNSNLP")

        Returns:
            Config instance with merged values
        """
        # Load from YAML first
        yaml_config = cls.from_yaml(path)
        yaml_dict = yaml_config.model_dump()

        # Load from environment
        env_config = cls.from_env(prefix)
        env_dict = env_config.model_dump()

        # Merge dictionaries (env overrides yaml)
        merged_dict = cls._deep_merge(yaml_dict, env_dict)

        return cls(**merged_dict)

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                # Only override if the value is not the default
                if value != {}:
                    result[key] = value

        return result
