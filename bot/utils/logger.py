"""Structured logging for the trading bot."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logger(name: str = "bot", level: str = "INFO") -> logging.Logger:
    """Create a consistently formatted logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(name)-12s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


log: Optional[logging.Logger] = None


def get_logger(name: str = "bot", level: str = "INFO") -> logging.Logger:
    """Get or create the global logger instance."""
    global log
    if log is None:
        log = setup_logger(name, level)
    return log
