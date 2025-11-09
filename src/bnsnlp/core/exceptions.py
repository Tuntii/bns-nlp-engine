"""
Exception hierarchy for bns-nlp-engine.

This module defines all custom exceptions used throughout the library,
providing structured error handling with error codes and context.
"""

from typing import Any, Dict, Optional


class BNSNLPError(Exception):
    """
    Base exception for all bns-nlp-engine errors.

    All custom exceptions in the library inherit from this base class,
    providing consistent error handling with error codes and context.

    Attributes:
        message: Human-readable error message
        code: Error code for programmatic error handling
        context: Additional context information about the error
    """

    def __init__(
        self, message: str, code: str = "BNSNLP_ERROR", context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the base exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic error handling
            context: Additional context information about the error
        """
        self.message = message
        self.code = code
        self.context = context or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.context:
            return f"[{self.code}] {self.message} (context: {self.context})"
        return f"[{self.code}] {self.message}"

    def __repr__(self) -> str:
        """Return detailed representation of the error."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"code={self.code!r}, "
            f"context={self.context!r})"
        )


class ConfigurationError(BNSNLPError):
    """
    Configuration related errors.

    Raised when there are issues with configuration loading, validation,
    or when required configuration values are missing.

    Examples:
        - Invalid YAML syntax in config file
        - Missing required environment variables
        - Invalid configuration values
        - Schema validation failures
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize configuration error.

        Args:
            message: Human-readable error message
            context: Additional context about the configuration error
        """
        super().__init__(message, "CONFIG_ERROR", context)


class PluginError(BNSNLPError):
    """
    Plugin loading and registration errors.

    Raised when there are issues with plugin discovery, loading,
    registration, or validation.

    Examples:
        - Plugin not found in registry
        - Plugin interface validation failure
        - Duplicate plugin registration
        - Entry point loading failure
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize plugin error.

        Args:
            message: Human-readable error message
            context: Additional context about the plugin error
        """
        super().__init__(message, "PLUGIN_ERROR", context)


class ProcessingError(BNSNLPError):
    """
    Text processing errors.

    Raised when there are issues during text processing operations
    such as preprocessing, embedding, search, or classification.

    Examples:
        - Text normalization failure
        - Tokenization error
        - Embedding generation failure
        - Search operation failure
        - Classification error
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize processing error.

        Args:
            message: Human-readable error message
            context: Additional context about the processing error
        """
        super().__init__(message, "PROCESSING_ERROR", context)


class AdapterError(BNSNLPError):
    """
    External service adapter errors.

    Raised when there are issues communicating with external services
    such as OpenAI, Cohere, Qdrant, or Pinecone.

    Examples:
        - API authentication failure
        - Network connection error
        - Rate limit exceeded
        - Invalid API response
        - Service unavailable
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize adapter error.

        Args:
            message: Human-readable error message
            context: Additional context about the adapter error
        """
        super().__init__(message, "ADAPTER_ERROR", context)


class ValidationError(BNSNLPError):
    """
    Data validation errors.

    Raised when input data fails validation checks, such as
    invalid data types, missing required fields, or constraint violations.

    Examples:
        - Invalid input data type
        - Missing required field
        - Value out of acceptable range
        - Schema validation failure
        - Pydantic validation error
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize validation error.

        Args:
            message: Human-readable error message
            context: Additional context about the validation error
        """
        super().__init__(message, "VALIDATION_ERROR", context)


__all__ = [
    "BNSNLPError",
    "ConfigurationError",
    "PluginError",
    "ProcessingError",
    "AdapterError",
    "ValidationError",
]
