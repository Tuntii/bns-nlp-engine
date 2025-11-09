"""
Core module for bns-nlp-engine.

This module contains the core components including pipeline orchestrator,
plugin registry, configuration management, and common utilities.
"""

from bnsnlp.core.config import (
    Config,
    EmbedConfig,
    LoggingConfig,
    PreprocessConfig,
    SearchConfig,
    TelemetryConfig,
)
from bnsnlp.core.exceptions import (
    AdapterError,
    BNSNLPError,
    ConfigurationError,
    PluginError,
    ProcessingError,
    ValidationError,
)
from bnsnlp.core.pipeline import Pipeline, PipelineStep
from bnsnlp.core.registry import PluginRegistry
from bnsnlp.core.types import (
    ClassifierInterface,
    ConfigDict,
    EmbedderInterface,
    EmbeddingVector,
    MetadataDict,
    PluginInterface,
    PreprocessorInterface,
    SearchInterface,
    TextInput,
)

__all__ = [
    # Exceptions
    "BNSNLPError",
    "ConfigurationError",
    "PluginError",
    "ProcessingError",
    "AdapterError",
    "ValidationError",
    # Types
    "ConfigDict",
    "MetadataDict",
    "EmbeddingVector",
    "TextInput",
    "PluginInterface",
    "PreprocessorInterface",
    "EmbedderInterface",
    "SearchInterface",
    "ClassifierInterface",
    # Configuration
    "Config",
    "LoggingConfig",
    "TelemetryConfig",
    "PreprocessConfig",
    "EmbedConfig",
    "SearchConfig",
    # Registry
    "PluginRegistry",
    # Pipeline
    "Pipeline",
    "PipelineStep",
]
