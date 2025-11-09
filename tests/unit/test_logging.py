"""
Unit tests for logging utilities.
"""

import json
import logging
from io import StringIO
from pathlib import Path

import pytest

from bnsnlp.core.config import LoggingConfig
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


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_basic_json_formatting(self):
        """Test basic JSON log formatting."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.module = "test_module"
        record.funcName = "test_function"

        result = formatter.format(record)
        log_data = json.loads(result)

        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "test_module"
        assert log_data["function"] == "test_function"
        assert log_data["line"] == 10
        assert "timestamp" in log_data
        # Timestamp should be in ISO format with timezone
        assert "+00:00" in log_data["timestamp"] or log_data["timestamp"].endswith("Z")

    def test_json_formatting_with_correlation_id(self):
        """Test JSON formatting includes correlation ID."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.module = "test_module"
        record.funcName = "test_function"
        record.correlation_id = "abc-123-def-456"

        result = formatter.format(record)
        log_data = json.loads(result)

        assert log_data["correlation_id"] == "abc-123-def-456"

    def test_json_formatting_with_context(self):
        """Test JSON formatting includes context."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.module = "test_module"
        record.funcName = "test_function"
        record.context = {"user_id": "123", "operation": "process"}

        result = formatter.format(record)
        log_data = json.loads(result)

        assert log_data["context"]["user_id"] == "123"
        assert log_data["context"]["operation"] == "process"

    def test_json_formatting_with_exception(self):
        """Test JSON formatting includes exception information."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )
        record.module = "test_module"
        record.funcName = "test_function"

        result = formatter.format(record)
        log_data = json.loads(result)

        assert "exception" in log_data
        assert "ValueError: Test error" in log_data["exception"]
        assert "Traceback" in log_data["exception"]

    def test_sensitive_data_filtering_in_message(self):
        """Test that sensitive data is filtered from log messages."""
        formatter = JSONFormatter()
        sensitive_messages = [
            "User api_key is sk-12345",
            "Password: secret123",
            "Token: bearer xyz",
            "Secret value: abc",
            "Authorization header set",
        ]

        for msg in sensitive_messages:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=msg,
                args=(),
                exc_info=None,
            )
            record.module = "test_module"
            record.funcName = "test_function"

            result = formatter.format(record)
            log_data = json.loads(result)

            assert "[REDACTED:" in log_data["message"]
            assert "sk-12345" not in log_data["message"]
            assert "secret123" not in log_data["message"]

    def test_sensitive_data_filtering_in_context(self):
        """Test that sensitive data is filtered from context."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Processing request",
            args=(),
            exc_info=None,
        )
        record.module = "test_module"
        record.funcName = "test_function"
        record.context = {
            "user_id": "123",
            "api_key": "sk-secret",
            "password": "secret123",
            "operation": "embed",
        }

        result = formatter.format(record)
        log_data = json.loads(result)

        assert log_data["context"]["user_id"] == "123"
        assert log_data["context"]["operation"] == "embed"
        assert log_data["context"]["api_key"] == "[REDACTED]"
        assert log_data["context"]["password"] == "[REDACTED]"


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_with_json_format(self):
        """Test logging setup with JSON format."""
        config = LoggingConfig(level="INFO", format="json", output="stdout")
        logger = setup_logging(config)

        assert logger.name == "bnsnlp"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0].formatter, JSONFormatter)

    def test_setup_logging_with_text_format(self):
        """Test logging setup with text format."""
        config = LoggingConfig(level="DEBUG", format="text", output="stdout")
        logger = setup_logging(config)

        assert logger.name == "bnsnlp"
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 1
        assert not isinstance(logger.handlers[0].formatter, JSONFormatter)

    def test_setup_logging_with_file_output(self, tmp_path: Path):
        """Test logging setup with file output."""
        log_file = tmp_path / "test.log"
        config = LoggingConfig(level="INFO", format="json", output=str(log_file))
        logger = setup_logging(config)

        assert logger.name == "bnsnlp"
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.FileHandler)

        # Test that logging to file works
        logger.info("Test message")
        logger.handlers[0].flush()

        assert log_file.exists()
        content = log_file.read_text()
        log_data = json.loads(content)
        assert log_data["message"] == "Test message"

    def test_setup_logging_with_different_levels(self):
        """Test logging setup with different log levels."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR"]

        for level in levels:
            config = LoggingConfig(level=level, format="json", output="stdout")
            logger = setup_logging(config)
            assert logger.level == getattr(logging, level)

    def test_setup_logging_removes_existing_handlers(self):
        """Test that setup_logging removes existing handlers."""
        config = LoggingConfig(level="INFO", format="json", output="stdout")

        # Setup logging twice
        logger1 = setup_logging(config)
        logger2 = setup_logging(config)

        # Should only have one handler
        assert len(logger2.handlers) == 1
        assert logger1 is logger2

    def test_setup_logging_with_dict_config(self):
        """Test logging setup with dictionary configuration."""
        config_dict = {"level": "DEBUG", "format": "text", "output": "stdout"}
        logger = setup_logging(config_dict)

        assert logger.name == "bnsnlp"
        assert logger.level == logging.DEBUG

    def test_logger_does_not_propagate(self):
        """Test that logger does not propagate to root logger."""
        config = LoggingConfig(level="INFO", format="json", output="stdout")
        logger = setup_logging(config)

        assert logger.propagate is False


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_default_name(self):
        """Test getting logger with default name."""
        logger = get_logger()
        assert logger.name == "bnsnlp"

    def test_get_logger_custom_name(self):
        """Test getting logger with custom name."""
        logger = get_logger("custom")
        assert logger.name == "custom"

    def test_get_logger_returns_same_instance(self):
        """Test that get_logger returns the same instance."""
        logger1 = get_logger()
        logger2 = get_logger()
        assert logger1 is logger2


class TestCorrelationID:
    """Tests for correlation ID management."""

    def test_set_and_get_correlation_id(self):
        """Test setting and getting correlation ID."""
        correlation_id = "test-correlation-id"
        set_correlation_id(correlation_id)

        assert get_correlation_id() == correlation_id

        # Clean up
        clear_correlation_id()

    def test_get_correlation_id_when_not_set(self):
        """Test getting correlation ID when not set."""
        clear_correlation_id()
        assert get_correlation_id() is None

    def test_generate_correlation_id(self):
        """Test generating correlation ID."""
        correlation_id = generate_correlation_id()

        assert correlation_id is not None
        assert isinstance(correlation_id, str)
        assert len(correlation_id) > 0

        # Should generate unique IDs
        correlation_id2 = generate_correlation_id()
        assert correlation_id != correlation_id2

    def test_clear_correlation_id(self):
        """Test clearing correlation ID."""
        set_correlation_id("test-id")
        assert get_correlation_id() == "test-id"

        clear_correlation_id()
        assert get_correlation_id() is None

    def test_correlation_id_isolation_between_contexts(self):
        """Test that correlation IDs are isolated between contexts."""
        import asyncio

        async def task1():
            set_correlation_id("task1-id")
            await asyncio.sleep(0.01)
            return get_correlation_id()

        async def task2():
            set_correlation_id("task2-id")
            await asyncio.sleep(0.01)
            return get_correlation_id()

        async def run_test():
            results = await asyncio.gather(task1(), task2())
            return results

        results = asyncio.run(run_test())
        assert "task1-id" in results
        assert "task2-id" in results


class TestCorrelationLoggerAdapter:
    """Tests for CorrelationLoggerAdapter."""

    def test_adapter_adds_correlation_id(self):
        """Test that adapter adds correlation ID to logs."""
        # Setup logger with string stream
        logger = logging.getLogger("test_adapter")
        logger.setLevel(logging.INFO)
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

        # Create adapter
        adapter = CorrelationLoggerAdapter(logger, {})

        # Set correlation ID and log
        set_correlation_id("adapter-test-id")
        adapter.info("Test message")

        # Check output
        stream.seek(0)
        log_output = stream.read()
        log_data = json.loads(log_output)

        assert log_data["correlation_id"] == "adapter-test-id"
        assert log_data["message"] == "Test message"

        # Clean up
        clear_correlation_id()
        logger.removeHandler(handler)

    def test_adapter_without_correlation_id(self):
        """Test that adapter works without correlation ID."""
        logger = logging.getLogger("test_adapter_no_id")
        logger.setLevel(logging.INFO)
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

        adapter = CorrelationLoggerAdapter(logger, {})

        clear_correlation_id()
        adapter.info("Test message")

        stream.seek(0)
        log_output = stream.read()
        log_data = json.loads(log_output)

        assert "correlation_id" not in log_data
        assert log_data["message"] == "Test message"

        # Clean up
        logger.removeHandler(handler)


class TestLoggingIntegration:
    """Integration tests for logging system."""

    def test_complete_logging_workflow(self, tmp_path: Path):
        """Test complete logging workflow with correlation tracking."""
        log_file = tmp_path / "integration.log"
        config = LoggingConfig(level="INFO", format="json", output=str(log_file))

        # Setup logging
        logger = setup_logging(config)

        # Generate and set correlation ID
        correlation_id = generate_correlation_id()
        set_correlation_id(correlation_id)

        # Log messages with context
        logger.info(
            "Processing started",
            extra={
                "correlation_id": correlation_id,
                "context": {"operation": "embed", "batch_size": 10},
            },
        )

        logger.info(
            "Processing completed",
            extra={
                "correlation_id": correlation_id,
                "context": {"operation": "embed", "duration_ms": 150},
            },
        )

        # Flush and read logs
        logger.handlers[0].flush()
        log_content = log_file.read_text()
        log_lines = log_content.strip().split("\n")

        assert len(log_lines) == 2

        # Verify first log
        log1 = json.loads(log_lines[0])
        assert log1["message"] == "Processing started"
        assert log1["correlation_id"] == correlation_id
        assert log1["context"]["operation"] == "embed"
        assert log1["context"]["batch_size"] == 10

        # Verify second log
        log2 = json.loads(log_lines[1])
        assert log2["message"] == "Processing completed"
        assert log2["correlation_id"] == correlation_id
        assert log2["context"]["duration_ms"] == 150

        # Clean up
        clear_correlation_id()

    def test_logging_with_adapter_workflow(self, tmp_path: Path):
        """Test logging workflow using CorrelationLoggerAdapter."""
        log_file = tmp_path / "adapter_integration.log"
        config = LoggingConfig(level="INFO", format="json", output=str(log_file))

        # Setup logging
        base_logger = setup_logging(config)
        logger = CorrelationLoggerAdapter(base_logger, {})

        # Set correlation ID
        correlation_id = generate_correlation_id()
        set_correlation_id(correlation_id)

        # Log with adapter (correlation ID added automatically)
        logger.info("Automatic correlation tracking")

        # Flush and read logs
        base_logger.handlers[0].flush()
        log_content = log_file.read_text()
        log_data = json.loads(log_content.strip())

        assert log_data["message"] == "Automatic correlation tracking"
        assert log_data["correlation_id"] == correlation_id

        # Clean up
        clear_correlation_id()
