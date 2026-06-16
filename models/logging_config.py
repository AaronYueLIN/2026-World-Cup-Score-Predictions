"""Structured logging config — structlog (Enterprise)

Usage:
    from models.logging_config import logger
    logger.info("predict", home="Spain", away="Cape Verde", prob_home=0.81)

Compatible: existing logging.getLogger warning/error still work normally.
"""
from __future__ import annotations

import logging
import sys

import structlog

from db.settings import settings


def setup_logging() -> None:
    """Global logging configuration (call once at application entry)."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            # Production JSON format, development readable format
            structlog.processors.JSONRenderer()
            if settings.is_production
            else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set root logger to specified level, avoid duplicate output
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        root.addHandler(handler)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Quick access to structured logger."""
    return structlog.get_logger(name or __name__)


# Automatically initialize on application startup
setup_logging()
