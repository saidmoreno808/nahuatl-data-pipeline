"""
Structured Logging

Provides JSON-formatted logging for easy parsing and analysis.
Integrates with ELK stack, CloudWatch, or other log aggregation systems.

Example:
    >>> from src.utils.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Pipeline started", extra={"records": 1000, "stage": "bronze"})
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.config import get_settings


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured JSON logging.

    Outputs logs in JSON format for easy parsing by log aggregators.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            str: JSON-formatted log line
        """
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from logger.info(..., extra={...})
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add custom fields from record.__dict__
        # (anything not in standard LogRecord)
        standard_fields = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "thread", "threadName", "exc_info", "exc_text", "stack_info",
        }

        for key, value in record.__dict__.items():
            if key not in standard_fields and not key.startswith("_"):
                # Convert non-JSON-serializable types
                try:
                    json.dumps(value)
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)

        return json.dumps(log_data, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """
    Human-readable text formatter for development.

    Outputs logs in a clean, colored format for terminal viewing.
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",   # Green
        "WARNING": "\033[33m", # Yellow
        "ERROR": "\033[31m",   # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors.

        Args:
            record: Log record to format

        Returns:
            str: Colored log line
        """
        # Add color
        levelname = record.levelname
        color = self.COLORS.get(levelname, "")
        colored_levelname = f"{color}{levelname:8s}{self.RESET}"

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Format message
        message = record.getMessage()

        # Add extra fields if present
        extra = []
        standard_fields = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "thread", "threadName", "exc_info", "exc_text", "stack_info",
        }

        for key, value in record.__dict__.items():
            if key not in standard_fields and not key.startswith("_"):
                extra.append(f"{key}={value}")

        extra_str = f" [{', '.join(extra)}]" if extra else ""

        # Combine
        log_line = f"{timestamp} {colored_levelname} {record.name:20s} {message}{extra_str}"

        # Add exception if present
        if record.exc_info:
            log_line += f"\n{self.formatException(record.exc_info)}"

        return log_line


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Get or create a logger with structured formatting.

    Args:
        name: Logger name (typically __name__)
        level: Log level (overrides settings)
        log_file: Optional file to write logs to

    Returns:
        logging.Logger: Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started", extra={"records": 1000})
    """
    settings = get_settings()

    # Get or create logger
    logger = logging.getLogger(name)

    # Prevent duplicate handlers (check own handlers only, not inherited)
    if logger.handlers:
        return logger

    # Set level
    log_level = level or settings.log_level
    logger.setLevel(getattr(logging, log_level))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))

    # Choose formatter based on settings
    if settings.log_format == "json":
        formatter = StructuredFormatter()
    else:
        formatter = TextFormatter()

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(StructuredFormatter())  # Always JSON for files
        logger.addHandler(file_handler)

    # Don't propagate to root logger (avoid duplicate logs)
    logger.propagate = False

    return logger


def configure_root_logger(level: str = None) -> None:
    """
    Configure the root logger for the application.

    Args:
        level: Override log level (e.g. "DEBUG", "INFO", "ERROR").
               Falls back to settings.log_level if not provided.

    Call this once at application startup.
    """
    settings = get_settings()
    effective_level = level or settings.log_level

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, effective_level))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, effective_level))

    if settings.log_format == "json":
        formatter = StructuredFormatter()
    else:
        formatter = TextFormatter()

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("yt_dlp").setLevel(logging.WARNING)


# Example usage for module testing
if __name__ == "__main__":
    configure_root_logger()
    logger = get_logger(__name__)

    logger.debug("Debug message")
    logger.info("Info message", extra={"user": "admin", "action": "login"})
    logger.warning("Warning message")
    logger.error("Error message")

    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("Exception occurred")
