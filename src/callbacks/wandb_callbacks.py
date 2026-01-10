"""Custom Weights & Biases callbacks for PyTorch Lightning."""

from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class WatchModelCallback(Callback):
    """Callback to watch model with W&B.

    Logs model topology and gradients to W&B.
    """

    def __init__(
        self,
        log: str = "gradients",
        log_freq: int = 100,
        log_graph: bool = True,
    ) -> None:
        """Initialize the callback.

        Args:
            log: Type of logging. Options: "gradients", "parameters", "all", or None.
            log_freq: Frequency of logging in training steps.
            log_graph: Whether to log the model graph.
        """
        self.log = log
        self.log_freq = log_freq
        self.log_graph = log_graph

    @rank_zero_only
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Watch model at the start of training."""
        for logger in trainer.loggers:
            if hasattr(logger, "experiment") and hasattr(logger.experiment, "watch"):
                logger.experiment.watch(
                    pl_module,
                    log=self.log,
                    log_freq=self.log_freq,
                    log_graph=self.log_graph,
                )


class LogPredictionsCallback(Callback):
    """Callback to log sample predictions to W&B.

    Override this class to implement custom prediction logging.
    """

    def __init__(
        self,
        num_samples: int = 8,
        log_every_n_epochs: int = 1,
    ) -> None:
        """Initialize the callback.

        Args:
            num_samples: Number of samples to log.
            log_every_n_epochs: Log predictions every N epochs.
        """
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs

    @rank_zero_only
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Log predictions at the end of validation epoch."""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        # Override this method to implement custom logging
        # Example: Log images, text, or other predictions to W&B
        pass
