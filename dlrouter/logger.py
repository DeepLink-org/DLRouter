"""Logging utilities for DLRouter."""

import logging
import sys


def get_logger(name: str = 'dlrouter', level: str = 'INFO') -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name.
        level: Log level string.

    Returns:
        Configured logger.
    """
    # Only add handler to the root 'dlrouter' logger;
    # child loggers (e.g. 'dlrouter.health') inherit it via propagation.
    root = logging.getLogger('dlrouter')
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    return logging.getLogger(name)
