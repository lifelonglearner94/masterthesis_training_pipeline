"""Logging utilities."""

import logging
from typing import Optional

from pytorch_lightning.utilities.rank_zero import rank_zero_only
from rich.logging import RichHandler

from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Log hyperparameters to all loggers.

    This method controls which parameters from Hydra config are saved by lightning loggers.

    Args:
        object_dict: A dictionary containing the following objects:
            - cfg: Hydra config
            - model: PyTorch Lightning model
            - trainer: PyTorch Lightning trainer
    """
    hparams = {}

    cfg = object_dict.get("cfg")
    if cfg:
        hparams["cfg"] = cfg

    model = object_dict.get("model")
    if model:
        hparams["model"] = model

    trainer = object_dict.get("trainer")
    if trainer:
        # Send hparams to all loggers
        for logger in trainer.loggers:
            logger.log_hyperparams(hparams)


def setup_rich_logging(level: int = logging.INFO) -> None:
    """Set up rich logging handler.

    Args:
        level: Logging level.
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
