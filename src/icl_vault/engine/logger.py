"""Minimal logging helpers for V2."""

import logging


def get_logger(name: str = "icl_vault") -> logging.Logger:
    """Return a basic logger instance."""
    return logging.getLogger(name)
