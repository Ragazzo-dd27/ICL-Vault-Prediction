"""Minimal logging helpers for V2."""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str = "icl_vault") -> logging.Logger:
    """Return a basic logger instance."""
    return logging.getLogger(name)


def setup_experiment_logger(log_path: str | Path, name: str = "icl_vault") -> logging.Logger:
    """Create a lightweight console + file logger for experiment scripts."""
    resolved_log_path = Path(log_path)
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(resolved_log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
