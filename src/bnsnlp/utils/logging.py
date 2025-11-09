"""Logging utilities for structured logging with JSON format and correlation tracking."""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Format logs as JSON with timestamp, level, context, and correlation IDs.
    
    This formatter creates structured JSON logs that include:
    - timestamp: ISO format UTC timestamp
    - level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - logger: Logger name
    - message: Log message
    - module: Module name where log was called
    - function: Function name where log was called
    - line: Line number where log was called
    - correlation_id: Optional correlation ID for tracking async operations
    - context: Optional additional context dictionary
    - exception: Optional exception traceback
    
    Example:
        >>> handler = logging.StreamHandler()
        >>> handler.setFormatter(JSONFormatter())
        >>> logger = logging.getLogger('bnsnlp')
        >>> logger.addHandler(handler)
        >>> logger.info('Processing text', extra={'correlation_id': 'abc-123', 'context': {'length': 100}})
    """
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the JSON formatter."""
        super().__init__(*args, **kwargs)
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.
        
        Args:
            record: The log record to format
            
        Returns:
            JSON string representation of the log record
        """
        log_data: Dict[str, Any] = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add correlation ID if present
        if hasattr(record, 'correlation_id'):
            log_data['correlation_id'] = record.correlation_id
        
        # Add context if present
        if hasattr(record, 'context'):
            log_data['context'] = record.context
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Filter out sensitive data from the log
        log_data = self._filter_sensitive_data(log_data)
        
        return json.dumps(log_data, ensure_ascii=False)
    
    def _filter_sensitive_data(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sensitive data from log entries.
        
        This method ensures that API keys, passwords, and other sensitive
        information are not logged.
        
        Args:
            log_data: The log data dictionary
            
        Returns:
            Filtered log data dictionary
        """
        sensitive_keys = {'api_key', 'password', 'token', 'secret', 'authorization'}
        
        # Filter message
        message = log_data.get('message', '')
        for key in sensitive_keys:
            if key in message.lower():
                # Don't log the actual message if it contains sensitive keys
                log_data['message'] = f"[REDACTED: message contains sensitive data ({key})]"
                break
        
        # Filter context
        if 'context' in log_data and isinstance(log_data['context'], dict):
            filtered_context = {}
            for key, value in log_data['context'].items():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    filtered_context[key] = '[REDACTED]'
                else:
                    filtered_context[key] = value
            log_data['context'] = filtered_context
        
        return log_data



def setup_logging(config: Any) -> logging.Logger:
    """Setup logging configuration for the application.
    
    This function configures the root logger for the bnsnlp package with
    the specified configuration. It supports both JSON and text formats,
    configurable log levels, and custom output destinations.
    
    Args:
        config: LoggingConfig instance or dict with logging configuration.
                Expected keys:
                - level: Log level (DEBUG, INFO, WARNING, ERROR)
                - format: Log format ('json' or 'text')
                - output: Output destination ('stdout', 'stderr', or file path)
    
    Returns:
        Configured logger instance for 'bnsnlp' package
        
    Example:
        >>> from bnsnlp.core.config import LoggingConfig
        >>> config = LoggingConfig(level='INFO', format='json', output='stdout')
        >>> logger = setup_logging(config)
        >>> logger.info('Application started')
    """
    # Get configuration values
    if hasattr(config, 'level'):
        level = config.level
        log_format = config.format
        output = config.output
    else:
        level = config.get('level', 'INFO')
        log_format = config.get('format', 'json')
        output = config.get('output', 'stdout')
    
    # Get or create logger
    logger = logging.getLogger('bnsnlp')
    logger.setLevel(getattr(logging, level))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create handler based on output destination
    if output == 'stdout':
        import sys
        handler = logging.StreamHandler(sys.stdout)
    elif output == 'stderr':
        import sys
        handler = logging.StreamHandler(sys.stderr)
    else:
        # Assume it's a file path
        handler = logging.FileHandler(output, encoding='utf-8')
    
    # Set formatter based on format type
    if log_format == 'json':
        handler.setFormatter(JSONFormatter())
    else:
        # Text format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str = 'bnsnlp') -> logging.Logger:
    """Get a logger instance.
    
    This is a convenience function to get a logger instance. If the logger
    hasn't been configured yet, it will return a basic logger.
    
    Args:
        name: Logger name (default: 'bnsnlp')
        
    Returns:
        Logger instance
        
    Example:
        >>> logger = get_logger()
        >>> logger.info('Processing started')
    """
    return logging.getLogger(name)



# Correlation ID context management using contextvars
import contextvars
import uuid
from typing import Optional

# Context variable for storing correlation ID
correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'correlation_id', default=None
)


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for the current context.
    
    This function sets a correlation ID that will be automatically included
    in all log messages within the current async context. This is useful for
    tracking related operations across async calls.
    
    Args:
        correlation_id: Unique identifier for correlating related operations
        
    Example:
        >>> set_correlation_id('abc-123-def-456')
        >>> logger = get_logger()
        >>> logger.info('Processing started')  # Will include correlation_id in logs
    """
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get correlation ID from the current context.
    
    Returns:
        Current correlation ID or None if not set
        
    Example:
        >>> set_correlation_id('abc-123')
        >>> correlation_id = get_correlation_id()
        >>> print(correlation_id)
        'abc-123'
    """
    return correlation_id_var.get()


def generate_correlation_id() -> str:
    """Generate a new unique correlation ID.
    
    This function generates a UUID-based correlation ID that can be used
    to track related operations.
    
    Returns:
        New correlation ID as a string
        
    Example:
        >>> correlation_id = generate_correlation_id()
        >>> set_correlation_id(correlation_id)
    """
    return str(uuid.uuid4())


def clear_correlation_id() -> None:
    """Clear the correlation ID from the current context.
    
    This function removes the correlation ID from the current context.
    Useful for cleanup after processing is complete.
    
    Example:
        >>> set_correlation_id('abc-123')
        >>> clear_correlation_id()
        >>> assert get_correlation_id() is None
    """
    correlation_id_var.set(None)


class CorrelationLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that automatically adds correlation ID to log records.
    
    This adapter wraps a logger and automatically includes the correlation ID
    from the current context in all log messages.
    
    Example:
        >>> logger = get_logger()
        >>> adapter = CorrelationLoggerAdapter(logger)
        >>> set_correlation_id('abc-123')
        >>> adapter.info('Processing started')  # Includes correlation_id automatically
    """
    
    def process(self, msg: str, kwargs: Any) -> tuple[str, Dict[str, Any]]:
        """Process the log message to add correlation ID.
        
        Args:
            msg: Log message
            kwargs: Additional keyword arguments
            
        Returns:
            Tuple of (message, kwargs) with correlation ID added to extra
        """
        correlation_id = get_correlation_id()
        
        if correlation_id:
            # Add correlation_id to extra
            if 'extra' not in kwargs:
                kwargs['extra'] = {}
            kwargs['extra']['correlation_id'] = correlation_id
        
        return msg, kwargs
