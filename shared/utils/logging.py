"""Standardized logging setup for all projects.

Usage:
    from shared.utils.logging import get_logger
    logger = get_logger("p3_itc")
    logger.info("Processing tile %s", tile_name)
"""

import logging
import sys


def get_logger(
    name: str,
    level: str = "INFO",
    fmt: str = "%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s",
) -> logging.Logger:
    """Create a configured logger.

    Parameters
    ----------
    name : str
        Logger name (typically project identifier).
    level : str
        Logging level string (DEBUG, INFO, WARNING, ERROR).
    fmt : str
        Log message format string.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

    return logger
