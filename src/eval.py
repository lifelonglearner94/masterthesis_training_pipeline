import pyrootutils

# Finds the root of the repo automatically
root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

import hydra
import lightning as L
import torch
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import instantiators
from src.utils.device_utils import log_device_info


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when a checkpoint file cannot be found at the specified path."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(
            f"Checkpoint not found: {path}\n"
            f"Please provide a valid checkpoint path via: ckpt_path=/path/to/checkpoint.ckpt"
        )


def print_test_config(cfg: DictConfig) -> None:
    """Print test configuration summary.

    Args:
        cfg: Hydra configuration dictionary.
    """
    print("\n" + "=" * 70)
    print("                    TEST CONFIGURATION")
    print("=" * 70)
    print(f"\n📁 CHECKPOINT:")
    print(f"   {cfg.get('ckpt_path', 'None (random weights)')}")

    print(f"\n📊 DATA:")
    print(
        f"   Data dir:    {cfg.data.get('data_dir', cfg.paths.get('data_dir', 'default'))}"
    )
    print(f"   Clip range:  {cfg.data.get('clip_start', 0)} - {cfg.data.get('clip_end', 'end')}")
    print(f"   Batch size:  {cfg.data.get('batch_size', 32)}")

    print(f"\n🔧 MODEL:")
    print(f"   Jump K:         {cfg.model.get('jump_k', 3)}")
    print(f"   T_teacher:      {cfg.model.get('T_teacher', 7)}")

    print(f"\n🏷️  METADATA:")
    print(f"   Task:  {cfg.get('task_name', 'unknown')}")
    print(f"   Seed:  {cfg.get('seed', 'None')}")
    print("=" * 70 + "\n")


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> float | None:
    """Evaluate a trained model on test data.

    This script loads a checkpoint and runs evaluation on a test dataset,
    computing jump losses with per-timestep breakdown.

    Args:
        cfg: Hydra configuration dictionary containing model, data, and
            training parameters.

    Returns:
        Final test loss if available, None otherwise.

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
            raise CheckpointNotFoundError(ckpt_path)
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

    # 6. Load checkpoint weights manually (avoids PyTorch 2.6 weights_only issues)
    if ckpt_path:
        print("📥 Loading checkpoint weights...")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        print("   ✅ Weights loaded successfully")

    # 7. Callbacks & Loggers
    callbacks: list[Callback] = instantiators.instantiate_callbacks(cfg.get("callbacks"))
    logger: list[Logger] = instantiators.instantiate_loggers(cfg.get("logger"))

    # 8. Trainer
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial",
    )

    # 9. Run test (no ckpt_path since weights are already loaded)
    print("\n🚀 Starting test evaluation...")
    trainer.test(model=model, datamodule=datamodule)

    # 10. Return final metric
    final_loss = trainer.callback_metrics.get("test/loss")
    if final_loss is not None:
        print(f"\n📊 Final test loss: {final_loss:.6f}")

    return final_loss.item() if final_loss is not None else None


if __name__ == "__main__":
    main()
