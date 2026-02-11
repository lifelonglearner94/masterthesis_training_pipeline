"""Python logger utilities."""

import logging
from typing import Final

from lightning.pytorch.utilities.rank_zero import rank_zero_only


def get_pylogger(name: str = __name__) -> logging.Logger:
    """Initialize a multi-GPU-friendly python command line logger.

    Args:
        name: Name of the logger, defaults to __name__.

    Returns:
        A configured logger instance that only logs on rank 0 in
        multi-GPU setups, preventing log multiplication across processes.
    """
    logger = logging.getLogger(name)

    # Decorate all logging levels with rank_zero_only to prevent
    # log multiplication in multi-GPU training
    logging_levels: Final = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )

    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
