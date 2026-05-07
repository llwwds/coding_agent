"""
Logging module.

Provides both file and console logging with configurable log level.
Log files are written to {WORKSPACE_DIR}/logs/agent.log.
"""

import logging
import os
import sys
from config import settings


def _setup_logger() -> None:
    """Initialize the root logger with file and console handlers."""
    log_dir = os.path.join(settings.WORKSPACE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, date_format)

    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    file_handler = logging.FileHandler(
        os.path.join(log_dir, "agent.log"), encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """Factory function to get a named logger.

    Args:
        name: The name for the logger, typically __name__ of the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    _setup_logger()
    return logging.getLogger(name)
