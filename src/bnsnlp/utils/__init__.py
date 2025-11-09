"""
Utility module for common functionality.

This module provides logging, async helpers, performance utilities, security, and telemetry.
"""

from bnsnlp.utils.logging import (
    CorrelationLoggerAdapter,
    JSONFormatter,
    clear_correlation_id,
    generate_correlation_id,
    get_correlation_id,
    get_logger,
    set_correlation_id,
    setup_logging,
)
from bnsnlp.utils.performance import (
    BatchProcessor,
    CacheManager,
    ConnectionPool,
    GPUAccelerator,
    MultiprocessingExecutor,
    StreamProcessor,
)
from bnsnlp.utils.security import SecureConfig
from bnsnlp.utils.telemetry import (
    Telemetry,
    get_telemetry,
    initialize_telemetry,
    set_telemetry,
    track_event,
)

__all__ = [
    # Logging
    "JSONFormatter",
    "setup_logging",
    "get_logger",
    "set_correlation_id",
    "get_correlation_id",
    "generate_correlation_id",
    "clear_correlation_id",
    "CorrelationLoggerAdapter",
    # Performance
    "BatchProcessor",
    "StreamProcessor",
    "MultiprocessingExecutor",
    "GPUAccelerator",
    "ConnectionPool",
    "CacheManager",
    # Security
    "SecureConfig",
    # Telemetry
    "Telemetry",
    "get_telemetry",
    "set_telemetry",
    "initialize_telemetry",
    "track_event",
]
