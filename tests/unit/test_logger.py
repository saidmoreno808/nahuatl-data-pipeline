"""
Unit tests for logging utilities.

Run with:
    pytest tests/unit/test_logger.py -v
"""

import json
import logging
import sys
import uuid
from pathlib import Path

import pytest

from src.utils.logger import (
    StructuredFormatter,
    TextFormatter,
    get_logger,
)


def unique_logger_name(prefix="test"):
    """Generate unique logger name to avoid caching issues."""
    return f"{prefix}.{uuid.uuid4().hex[:8]}"


class TestStructuredFormatter:
    """Test JSON structured formatter."""

    def test_format_basic_message(self):
        """Test formatting a basic log message."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["logger"] == "test_logger"
        assert "timestamp" in data
        assert data["line"] == 10

    def test_format_with_extra_fields(self):
        """Test formatting with extra fields."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Add custom fields
        record.user_id = "user123"
        record.action = "login"

        output = formatter.format(record)
        data = json.loads(output)

        assert data["user_id"] == "user123"
        assert data["action"] == "login"

    def test_format_with_exception(self):
        """Test formatting with exception info."""
        formatter = StructuredFormatter()

        try:
            1 / 0
        except ZeroDivisionError:
            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Error occurred",
                args=(),
                exc_info=exc_info,
            )

            output = formatter.format(record)
            data = json.loads(output)

            assert "exception" in data
            assert "ZeroDivisionError" in data["exception"]


class TestTextFormatter:
    """Test human-readable text formatter."""

    def test_format_basic_message(self):
        """Test formatting a basic log message."""
        formatter = TextFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        assert "INFO" in output
        assert "test_logger" in output
        assert "Test message" in output

    def test_format_includes_colors(self):
        """Test that output includes ANSI color codes."""
        formatter = TextFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        # Should contain ANSI escape codes
        assert "\033[" in output  # ANSI escape sequence

    def test_format_with_extra_fields(self):
        """Test formatting with extra fields."""
        formatter = TextFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        record.user_id = "user123"
        record.count = 42

        output = formatter.format(record)

        assert "user_id=user123" in output
        assert "count=42" in output


class TestGetLogger:
    """Test logger creation."""

    def test_get_logger_creates_logger(self):
        """Test that get_logger creates a logger."""
        logger = get_logger("test.module")

        assert logger.name == "test.module"
        assert isinstance(logger, logging.Logger)

    def test_get_logger_is_idempotent(self):
        """Test that calling get_logger twice returns same logger."""
        logger1 = get_logger("test.module2")
        logger2 = get_logger("test.module2")

        # Should be the same instance
        assert logger1 is logger2

    def test_logger_logs_to_file(self, tmp_path):
        """Test logging to file."""
        log_file = tmp_path / "test.log"
        logger = get_logger(unique_logger_name("file"), log_file=log_file)

        logger.info("Test message")

        # File should exist and contain message
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content

        # Should be JSON format
        lines = content.strip().split("\n")
        data = json.loads(lines[0])
        assert data["message"] == "Test message"

    def test_logger_respects_level(self):
        """Test that logger respects log level."""
        logger = get_logger(unique_logger_name("level"), level="ERROR")

        # Should be ERROR level
        assert logger.level == logging.ERROR

    def test_logger_with_extra_fields(self, tmp_path):
        """Test logging with extra fields."""
        log_file = tmp_path / "extra.log"
        logger = get_logger(unique_logger_name("extra"), log_file=log_file)

        logger.info(
            "Processing data",
            extra={"records": 1000, "stage": "bronze"},
        )

        # Check that extra fields are in the JSON output
        content = log_file.read_text()
        data = json.loads(content.strip())
        assert data["records"] == 1000
        assert data["stage"] == "bronze"


@pytest.mark.slow
class TestLoggingIntegration:
    """Integration tests for logging system."""

    def test_logging_to_multiple_handlers(self, tmp_path):
        """Test logging to both console and file."""
        log_file = tmp_path / "integration.log"
        logger = get_logger(unique_logger_name("integration"), log_file=log_file)

        logger.info("Integration test message")
        logger.warning("Warning message")
        logger.error("Error message")

        # File should contain all messages
        content = log_file.read_text()
        assert "Integration test message" in content
        assert "Warning message" in content
        assert "Error message" in content

        # Each line should be valid JSON
        for line in content.strip().split("\n"):
            data = json.loads(line)
            assert "level" in data
            assert "message" in data
            assert "timestamp" in data
