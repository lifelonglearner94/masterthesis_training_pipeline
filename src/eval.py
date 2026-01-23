import pyrootutils

# Finds the root of the repo automatically
root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from typing import List, Optional

import hydra
import lightning as L
import omegaconf
import torch
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

# Fix for PyTorch 2.6+ security update: allow OmegaConf objects in checkpoints
torch.serialization.add_safe_globals([
    omegaconf.listconfig.ListConfig,
    omegaconf.dictconfig.DictConfig,
    omegaconf.base.ContainerMetadata,
])

from src.utils import instantiators
from src.utils.device_utils import log_device_info


def print_test_config(cfg: DictConfig) -> None:
    """Print test configuration summary."""
    print("\n" + "=" * 70)
    print("                    TEST CONFIGURATION")
    print("=" * 70)
    print(f"\n📁 CHECKPOINT:")
    print(f"   {cfg.get('ckpt_path', 'None (random weights)')}")

    print(f"\n📊 DATA:")
    print(f"   Data dir:    {cfg.data.get('data_dir', cfg.paths.get('data_dir', 'default'))}")
    print(f"   Clip range:  {cfg.data.get('clip_start', 0)} - {cfg.data.get('clip_end', 'end')}")
    print(f"   Batch size:  {cfg.data.get('batch_size', 32)}")

    print(f"\n🔧 MODEL:")
    print(f"   Context frames: {cfg.model.get('context_frames', 3)}")
    print(f"   T_rollout:      {cfg.model.get('T_rollout', 4)}")
    print(f"   T_teacher:      {cfg.model.get('T_teacher', 7)}")

    print(f"\n🏷️  METADATA:")
    print(f"   Task:  {cfg.get('task_name', 'unknown')}")
    print(f"   Seed:  {cfg.get('seed', 'None')}")
    print("=" * 70 + "\n")


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Evaluate a trained model on test data.

    This script loads a checkpoint and runs evaluation on a test dataset,
    computing rollout losses with per-timestep breakdown.

    Usage:
        # Basic usage with default test config
        uv run src/eval.py experiment=test_ac_predictor ckpt_path=/path/to/checkpoint.ckpt

        # Custom test range
        uv run src/eval.py experiment=test_ac_predictor \\
            ckpt_path=/path/to/checkpoint.ckpt \\
            data.clip_start=15000 data.clip_end=16000

        # With custom data directory
        uv run src/eval.py experiment=test_ac_predictor \\
            ckpt_path=/path/to/checkpoint.ckpt \\
            paths.data_dir=/path/to/data
    """
    # 0. Log hardware info at startup
    log_device_info()

    # 1. Gold Standard: Determinism
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # 2. Print test configuration
    print_test_config(cfg)

    # 3. Validate checkpoint path
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Please provide a valid checkpoint path via: ckpt_path=/path/to/checkpoint.ckpt"
            )
        print(f"✅ Checkpoint found: {ckpt_path}")
        print(f"   Size: {ckpt_path.stat().st_size / 1e6:.1f} MB")
    else:
        print("⚠️  No checkpoint specified - testing with random weights!")

    # 4. Instantiate DataModule
    print("\n📦 Loading data module...")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # 5. Instantiate Model
    print("🧠 Instantiating model...")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # 6. Callbacks & Loggers
    callbacks: List[Callback] = instantiators.instantiate_callbacks(cfg.get("callbacks"))
    logger: List[Logger] = instantiators.instantiate_loggers(cfg.get("logger"))

    # 7. Trainer
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial"
    )

    # 8. Run test
    print("\n🚀 Starting test evaluation...")
    if ckpt_path:
        trainer.test(model=model, datamodule=datamodule, ckpt_path=str(ckpt_path))
    else:
        trainer.test(model=model, datamodule=datamodule)

    # 9. Return final metric
    final_loss = trainer.callback_metrics.get("test/loss")
    if final_loss is not None:
        print(f"\n📊 Final test loss: {final_loss:.6f}")

    return final_loss.item() if final_loss is not None else None


if __name__ == "__main__":
    main()
