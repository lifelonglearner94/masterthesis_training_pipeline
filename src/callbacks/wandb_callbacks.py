"""Custom Weights & Biases callbacks for PyTorch Lightning."""

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.rank_zero import rank_zero_only

try:
    import wandb
except ImportError:
    wandb = None


class WatchModelCallback(Callback):
    """Callback to watch model with W&B.

    Logs model topology and gradients to W&B.
    """

    def __init__(
        self,
        log: Optional[str] = "gradients",
        log_freq: int = 100,
        log_graph: bool = True,
    ) -> None:
        """Initialize the callback.

        Args:
            log: Type of logging. Options: "gradients", "parameters", "all", or None.
            log_freq: Frequency of logging in training steps.
            log_graph: Whether to log the model graph.
        """
        # Handle string "None" from YAML config - convert to actual None
        if log is None or (isinstance(log, str) and log.lower() == "none"):
            self.log = None
        elif log in ("gradients", "parameters", "all"):
            self.log = log
        else:
            raise ValueError(
                f"Invalid log value: {log!r}. Must be one of 'gradients', 'parameters', 'all', or None"
            )
        self.log_freq = log_freq
        self.log_graph = log_graph

    @rank_zero_only
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Watch model at the start of training."""
        for logger in trainer.loggers:
            if hasattr(logger, "experiment") and hasattr(logger.experiment, "watch"):
                try:
                    logger.experiment.watch(
                        pl_module,
                        log=self.log,
                        log_freq=self.log_freq,
                        log_graph=self.log_graph,
                    )
                except Exception as e:
                    # Log warning but don't fail training if watch fails
                    import warnings
                    warnings.warn(f"Failed to watch model with W&B: {e}")


class LogPredictionsCallback(Callback):
    """Callback to log sample predictions and prediction quality to W&B.

    Logs prediction error histograms and sample prediction comparisons
    for action-conditioned predictor models.
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
        self._val_outputs: list[Dict[str, Any]] = []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Store validation outputs for later logging."""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        # Only collect from first few batches
        if batch_idx < self.num_samples and outputs is not None:
            self._val_outputs.append(outputs)

    @rank_zero_only
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Log predictions at the end of validation epoch."""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            self._val_outputs.clear()
            return

        if wandb is None or not self._val_outputs:
            self._val_outputs.clear()
            return

        # Find W&B logger
        wandb_logger = None
        for logger in trainer.loggers:
            if hasattr(logger, "experiment") and hasattr(logger.experiment, "log"):
                wandb_logger = logger
                break

        if wandb_logger is None:
            self._val_outputs.clear()
            return

        try:
            # Collect prediction errors from stored outputs
            errors = []
            for output in self._val_outputs:
                if isinstance(output, dict):
                    # Handle different output formats
                    if "loss" in output:
                        errors.append(float(output["loss"]))
                    elif "val_loss" in output:
                        errors.append(float(output["val_loss"]))
                elif isinstance(output, torch.Tensor):
                    errors.append(float(output.mean()))

            if errors:
                # Log error distribution histogram
                wandb_logger.experiment.log({
                    "val/prediction_error_histogram": wandb.Histogram(errors),
                    "val/mean_batch_error": sum(errors) / len(errors),
                    "val/max_batch_error": max(errors),
                    "val/min_batch_error": min(errors),
                    "epoch": trainer.current_epoch,
                })

        except Exception as e:
            # Silently handle logging errors to not interrupt training
            pass

        finally:
            self._val_outputs.clear()
