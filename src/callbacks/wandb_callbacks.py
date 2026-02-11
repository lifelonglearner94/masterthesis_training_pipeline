"""Custom Weights & Biases callbacks for PyTorch Lightning."""

import logging
import warnings
from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.rank_zero import rank_zero_only

try:
    import wandb
except ImportError:
    wandb = None

# Constants
DEFAULT_LOG_FREQ = 100

# Initialize module logger
logger = logging.getLogger(__name__)


class InvalidLogValueError(ValueError):
    """Raised when an invalid log value is provided to WatchModelCallback."""

    def __init__(self, value: str) -> None:
        self.value = value
        super().__init__(
            f"Invalid log value: {value!r}. Must be one of 'gradients', 'parameters', 'all', or None"
        )


class WatchModelCallback(Callback):
    """Callback to watch model with W&B.

    Logs model topology and gradients to W&B.
    """

    def __init__(
        self,
        log: str | None = "gradients",
        log_freq: int = DEFAULT_LOG_FREQ,
        log_graph: bool = True,
    ) -> None:
        """Initialize the callback.

        Args:
            log: Type of logging. Options: "gradients", "parameters", "all", or None.
            log_freq: Frequency of logging in training steps.
            log_graph: Whether to log the model graph.

        Raises:
            InvalidLogValueError: If log value is not one of the allowed options.
        """
        self.log = self._normalize_log_value(log)
        self.log_freq = log_freq
        self.log_graph = log_graph

    def _normalize_log_value(self, value: str | None) -> str | None:
        """Normalize log parameter value.

        Args:
            value: The raw log value from config.

        Returns:
            Normalized log value or None.

        Raises:
            InvalidLogValueError: If value is not valid.
        """
        if value is None or (isinstance(value, str) and value.lower() == "none"):
            return None
        if value in ("gradients", "parameters", "all"):
            return value
        raise InvalidLogValueError(value)

    @rank_zero_only
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Watch model at the start of training."""
        for logger_obj in trainer.loggers:
            if hasattr(logger_obj, "experiment") and hasattr(
                logger_obj.experiment, "watch"
            ):
                try:
                    logger_obj.experiment.watch(
                        pl_module,
                        log=self.log,
                        log_freq=self.log_freq,
                        log_graph=self.log_graph,
                    )
                except Exception as e:
                    warnings.warn(
                        f"Failed to watch model with W&B: {e}",
                        UserWarning,
                        stacklevel=2,
                    )


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
        self._val_outputs: list[dict[str, Any]] = []

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

        wandb_logger = self._find_wandb_logger(trainer)
        if wandb_logger is None:
            self._val_outputs.clear()
            return

        try:
            self._log_prediction_metrics(wandb_logger, trainer.current_epoch)
        except Exception as e:
            logger.exception("Failed to log predictions to W&B")
        finally:
            self._val_outputs.clear()

    def _find_wandb_logger(self, trainer: Trainer) -> Any:
        """Find the W&B logger from trainer's loggers.

        Args:
            trainer: The Lightning Trainer instance.

        Returns:
            The W&B logger instance or None.
        """
        for logger_obj in trainer.loggers:
            if hasattr(logger_obj, "experiment") and hasattr(
                logger_obj.experiment, "log"
            ):
                return logger_obj
        return None

    def _log_prediction_metrics(self, wandb_logger: Any, epoch: int) -> None:
        """Log prediction error metrics to W&B.

        Args:
            wandb_logger: The W&B logger instance.
            epoch: Current training epoch.
        """
        errors = self._extract_errors_from_outputs()

        if not errors:
            return

        wandb_logger.experiment.log(
            {
                "val/prediction_error_histogram": wandb.Histogram(errors),
                "val/mean_batch_error": sum(errors) / len(errors),
                "val/max_batch_error": max(errors),
                "val/min_batch_error": min(errors),
                "epoch": epoch,
            }
        )

    def _extract_errors_from_outputs(self) -> list[float]:
        """Extract prediction errors from stored validation outputs.

        Returns:
            List of error values as floats.
        """
        errors: list[float] = []
        for output in self._val_outputs:
            if isinstance(output, dict):
                if "loss" in output:
                    errors.append(float(output["loss"]))
                elif "val_loss" in output:
                    errors.append(float(output["val_loss"]))
            elif isinstance(output, torch.Tensor):
                errors.append(float(output.mean()))
        return errors
