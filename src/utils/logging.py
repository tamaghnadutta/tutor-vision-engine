"""
Standard logging configuration
"""

import logging
import sys
from typing import Any, Dict

from src.config.settings import get_settings


def setup_logging() -> None:
    """Configure standard logging"""
    settings = get_settings()

    # Configure standard library logging
    format_string = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        if settings.structured_logs
        else "%(levelname)s - %(name)s - %(message)s"
    )

    logging.basicConfig(
        format=format_string,
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str = None) -> logging.Logger:
    """Get a configured logger"""
    return logging.getLogger(name)