"""Pantropic - Structured Logging.

Provides a unified logging interface with:
- Structured JSON output for production
- Pretty console output for development
- Request correlation via context variables
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any

# Context variable for request correlation
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class PantropicFormatter(logging.Formatter):
    """Custom formatter with optional JSON output."""

    LEVEL_COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, *, json_output: bool = False, include_timestamp: bool = True) -> None:
        super().__init__()
        self.json_output = json_output
        self.include_timestamp = include_timestamp

    def format(self, record: logging.LogRecord) -> str:
        """Format log record."""
        request_id = request_id_var.get()

        if self.json_output:
            import json
            log_dict: dict[str, Any] = {
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name,
            }
            if self.include_timestamp:
                log_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
            if request_id:
                log_dict["request_id"] = request_id
            if record.exc_info:
                log_dict["exception"] = self.formatException(record.exc_info)
            return json.dumps(log_dict)

        # Pretty console format
        timestamp = datetime.now().strftime("%H:%M:%S")
        level = record.levelname
        color = self.LEVEL_COLORS.get(level, "")

        prefix = f"{timestamp} | {color}{level:>7}{self.RESET} |"
        if request_id:
            prefix = f"{prefix} [{request_id[:8]}]"

        message = record.getMessage()

        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        return f"{prefix} {message}"


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure Pantropic logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: Use JSON format for structured logging
        log_file: Optional file path for log output

    Returns:
        Root logger configured for Pantropic
    """
    root_logger = logging.getLogger("pantropic")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(PantropicFormatter(json_output=json_output))
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(PantropicFormatter(json_output=True))
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Optional module name (automatically prefixed with 'pantropic.')

    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"pantropic.{name}")
    return logging.getLogger("pantropic")
