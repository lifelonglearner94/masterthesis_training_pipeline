import pyrootutils

# Finds the root of the repo automatically
root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from typing import List, Optional

from src.utils import instantiators
from src.utils.logging_utils import log_hyperparameters
from src.utils.device_utils import log_device_info

@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # 0. Log hardware info at startup
    log_device_info()

    # 1. Gold Standard: Determinism
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # 2. Instantiate DataModule
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # 3. Instantiate Model
    # Architecture-agnostic: Reads from conf/model/titans.yaml or vit.yaml
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # 4. Callbacks & Loggers
    callbacks: List[Callback] = instantiators.instantiate_callbacks(cfg.get("callbacks"))
    logger: List[Logger] = instantiators.instantiate_loggers(cfg.get("logger"))

    # 5. Trainer
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial"
    )

    # 6. Log hyperparameters
    log_hyperparameters({"cfg": cfg, "model": model, "trainer": trainer})

    # 7. Train
    if cfg.get("train"):
        trainer.fit(model=model, datamodule=datamodule)

    # 8. Test
    if cfg.get("test"):
        trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    return trainer.callback_metrics.get("val/loss")

if __name__ == "__main__":
    main()
