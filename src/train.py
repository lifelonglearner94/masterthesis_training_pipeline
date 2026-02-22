import pyrootutils

# Finds the root of the repo automatically — MUST run before any `src.*` imports
root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

import logging
from typing import Any

import hydra
import omegaconf
import torch
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import instantiators
from src.utils.device_utils import log_device_info
from src.utils.logging_utils import log_hyperparameters

# Fix for PyTorch 2.6+ security update: allow OmegaConf objects in checkpoints
torch.serialization.add_safe_globals([
    omegaconf.listconfig.ListConfig,
    omegaconf.dictconfig.DictConfig,
    omegaconf.base.ContainerMetadata,
    Any,  # typing.Any used in checkpoint hyperparameters
])

# Constants
DEFAULT_CHECKPOINT_PATH = "best"


def setup_verbose_logging(cfg: DictConfig) -> None:
    """Configure logging level based on config.

    Checks if verbose logging is enabled by examining the callbacks config
    for any enabled verbose logging flags.

    Args:
        cfg: Hydra configuration dictionary.
    """
    verbose = False

    if cfg.get("callbacks"):
        callbacks_cfg = cfg.callbacks
        if hasattr(callbacks_cfg, "verbose_logging") or "verbose_logging" in callbacks_cfg:
            vl_cfg = callbacks_cfg.get("verbose_logging", {})
            # Enable verbose logging if at least one flag is True
            verbose = any([
                vl_cfg.get("log_memory", False),
                vl_cfg.get("log_data", False),
                vl_cfg.get("log_forward_pass", False),
                vl_cfg.get("log_backward_pass", False),
                vl_cfg.get("log_gradients", False),
                vl_cfg.get("log_weights", False),
            ])

    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S',
        )
        logging.getLogger("src").setLevel(logging.DEBUG)
        logging.getLogger("src.models").setLevel(logging.DEBUG)
        logging.getLogger("src.callbacks").setLevel(logging.DEBUG)
        logging.getLogger(__name__).info("VERBOSE LOGGING ENABLED - All DEBUG messages will be shown")
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S',
        )


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> float | None:
    """Main training pipeline entry point.

    Orchestrates the complete training workflow including:
    - Logging configuration
    - Hardware info logging
    - Seed setting for reproducibility
    - DataModule instantiation
    - Model instantiation
    - Callback and logger setup
    - Trainer configuration
    - Training and testing execution

    Args:
        cfg: Hydra configuration dictionary containing all experiment settings.

    Returns:
        Validation loss metric from the best checkpoint, or None if not available.
    """
    # Setup logging first
    setup_verbose_logging(cfg)

    # 0. Log hardware info at startup
    log_device_info()

    # 1. Gold Standard: Determinism
    if cfg.get("seed"):
        import lightning as L
        L.seed_everything(cfg.seed, workers=True)

    # 2. Instantiate DataModule
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # 3. Instantiate Model
    # Architecture-agnostic: Reads from conf/model/titans.yaml or vit.yaml
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # 4. Callbacks & Loggers
    callbacks: list[Callback] = instantiators.instantiate_callbacks(cfg.get("callbacks"))
    loggers: list[Logger] = instantiators.instantiate_loggers(cfg.get("logger"))

    # 5. Trainer
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
        _convert_="partial",
    )

    # 6. Log hyperparameters
    log_hyperparameters({"cfg": cfg, "model": model, "trainer": trainer})

    # 7. Train
    if cfg.get("train"):
        ckpt_path = cfg.get("ckpt_path")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # 8. Test
    if cfg.get("test"):
        trainer.test(model=model, datamodule=datamodule, ckpt_path=DEFAULT_CHECKPOINT_PATH)

    return trainer.callback_metrics.get("val/loss")


if __name__ == "__main__":
    main()
