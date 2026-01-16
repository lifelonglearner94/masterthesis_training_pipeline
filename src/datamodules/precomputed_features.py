"""DataModule for pre-computed V-JEPA2 encoder features.

This module loads pre-computed encoder features and actions from numpy files
for training the AC Predictor model. States are zero-filled by default.

Expected data format:
    - features: .npy files with shape [T+1, N, D] or [T+1, H, W, D]
      where T+1 is the number of frames, N = H*W is patches per frame, D is embed_dim
    - actions: .npy files with shape [T+1, action_dim] (will be sliced to [T, action_dim])
    - states: Created as zeros with shape [T, action_dim]
    - (optional) extrinsics: .npy files with shape [T, action_dim-1]

Directory structure:
    data_dir/
    ├── clip_00001/
    │   ├── feature_maps/
    │   │   └── vjepa2_vitl16.npy   # [T+1, N, D]
    │   └── actions_states/
    │       └── actions.npy         # [T+1, action_dim]
    ├── clip_00002/
    └── ...
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.utils.device_utils import should_pin_memory


class PrecomputedFeaturesDataset(Dataset):
    """Dataset for pre-computed V-JEPA2 encoder features."""

    def __init__(
        self,
        data_dir: str | Path,
        num_frames: int = 16,
        patches_per_frame: int | None = None,
        use_extrinsics: bool = False,
        feature_map_name: str = "vjepa2_vitl16",
        clip_prefix: str = "clip_",
    ) -> None:
        """Initialize the dataset.

        Args:
            data_dir: Path to directory containing clip subdirectories
            num_frames: Number of frames to load (T in the loss formulation)
            patches_per_frame: Number of patches per frame (H*W). If None, inferred from data.
            use_extrinsics: Whether to load extrinsics data
            feature_map_name: Name of the feature map file (without .npy extension)
            clip_prefix: Prefix for clip directories (e.g., "clip_")
        """
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.patches_per_frame = patches_per_frame
        self.use_extrinsics = use_extrinsics
        self.feature_map_name = feature_map_name
        self.clip_prefix = clip_prefix

        # Find all clip directories with the expected structure
        self.episode_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir()
            and d.name.startswith(clip_prefix)
            and (d / "feature_maps" / f"{feature_map_name}.npy").exists()
        ])

        if len(self.episode_dirs) == 0:
            raise ValueError(
                f"No valid clips found in {data_dir}. "
                f"Expected structure: {clip_prefix}XXXXX/feature_maps/{feature_map_name}.npy"
            )

        # Infer action_dim from first clip
        first_actions = np.load(self.episode_dirs[0] / "actions_states" / "actions.npy")
        self.action_dim = first_actions.shape[-1]

    def __len__(self) -> int:
        return len(self.episode_dirs)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Load a single clip.

        Returns:
            Dictionary with keys:
                - features: [T+1, N, D] tensor
                - actions: [T, action_dim] tensor
                - states: [T, action_dim] tensor (zeros)
                - extrinsics (optional): [T, action_dim-1] tensor
        """
        episode_dir = self.episode_dirs[idx]

        # Load features from clip_XXXXX/feature_maps/{feature_map_name}.npy
        features = np.load(episode_dir / "feature_maps" / f"{self.feature_map_name}.npy")

        # Handle different feature shapes: [T+1, N, D] or [T+1, H, W, D]
        if features.ndim == 4:
            T_plus_1, H, W, D = features.shape
            features = features.reshape(T_plus_1, H * W, D)

        # Limit to num_frames + 1 (for target)
        T_plus_1 = min(features.shape[0], self.num_frames + 1)
        features = features[:T_plus_1]
        T = T_plus_1 - 1

        # Load actions from clip_XXXXX/actions_states/actions.npy
        # Actions array has T+1 rows, slice off the last row to get T actions
        actions = np.load(episode_dir / "actions_states" / "actions.npy")
        actions = actions[:-1]  # Drop last row: [T+1, action_dim] -> [T, action_dim]
        actions = actions[:T]   # Limit to num_frames

        # Create zero-filled states array (same shape as actions)
        states = np.zeros((T, self.action_dim), dtype=np.float32)

        result = {
            "features": torch.from_numpy(features).float(),
            "actions": torch.from_numpy(actions).float(),
            "states": torch.from_numpy(states).float(),
        }

        # Optionally load extrinsics
        if self.use_extrinsics:
            extrinsics_path = episode_dir / "actions_states" / "extrinsics.npy"
            if extrinsics_path.exists():
                extrinsics = np.load(extrinsics_path)[:T]
                result["extrinsics"] = torch.from_numpy(extrinsics).float()

        return result


def collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Custom collate function to handle variable-length sequences.

    Pads sequences to the maximum length in the batch.
    Infers action_dim from the batch data.
    """
    # Find max lengths
    max_T_plus_1 = max(item["features"].shape[0] for item in batch)
    max_T = max_T_plus_1 - 1

    # Get dimensions from first item
    N = batch[0]["features"].shape[1]
    D = batch[0]["features"].shape[2]
    # Infer action_dim from actions tensor shape
    action_dim = batch[0]["actions"].shape[-1]

    # Check for extrinsics
    has_extrinsics = "extrinsics" in batch[0]

    # Initialize padded tensors
    B = len(batch)
    features = torch.zeros(B, max_T_plus_1, N, D)
    actions = torch.zeros(B, max_T, action_dim)
    states = torch.zeros(B, max_T, action_dim)
    if has_extrinsics:
        extrinsics_dim = batch[0]["extrinsics"].shape[-1]
        extrinsics = torch.zeros(B, max_T, extrinsics_dim)

    # Fill in data
    for i, item in enumerate(batch):
        T_plus_1 = item["features"].shape[0]
        T = T_plus_1 - 1
        features[i, :T_plus_1] = item["features"]
        actions[i, :T] = item["actions"]
        states[i, :T] = item["states"]
        if has_extrinsics and "extrinsics" in item:
            extrinsics[i, :T] = item["extrinsics"]

    result = {
        "features": features,
        "actions": actions,
        "states": states,
    }
    if has_extrinsics:
        result["extrinsics"] = extrinsics

    return result


class PrecomputedFeaturesDataModule(pl.LightningDataModule):
    """Lightning DataModule for pre-computed V-JEPA2 encoder features."""

    def __init__(
        self,
        data_dir: str = "data",
        num_frames: int = 16,
        patches_per_frame: int | None = None,
        use_extrinsics: bool = False,
        feature_map_name: str = "vjepa2_vitl16",
        clip_prefix: str = "clip_",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DataModule.

        Args:
            data_dir: Root directory containing clip subdirectories
            num_frames: Number of frames per sample (T in the loss formulation)
            patches_per_frame: Number of patches per frame (H*W)
            use_extrinsics: Whether to load extrinsics data
            feature_map_name: Name of the feature map file (without .npy extension)
            clip_prefix: Prefix for clip directories (e.g., "clip_")
            batch_size: Batch size for DataLoaders
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer.
                       If None, auto-detects based on CUDA availability.
        """
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.patches_per_frame = patches_per_frame
        self.use_extrinsics = use_extrinsics
        self.feature_map_name = feature_map_name
        self.clip_prefix = clip_prefix
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Auto-detect pin_memory if not specified (only beneficial for CUDA)
        self.pin_memory = pin_memory if pin_memory is not None else should_pin_memory()

        self.train_dataset: PrecomputedFeaturesDataset | None = None
        self.val_dataset: PrecomputedFeaturesDataset | None = None
        self.test_dataset: PrecomputedFeaturesDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage.

        Uses data_dir directly (flat structure with clip_* directories).
        All clips are used for training - no train/val split.
        """
        dataset_kwargs = {
            "data_dir": self.data_dir,
            "num_frames": self.num_frames,
            "patches_per_frame": self.patches_per_frame,
            "use_extrinsics": self.use_extrinsics,
            "feature_map_name": self.feature_map_name,
            "clip_prefix": self.clip_prefix,
        }

        if stage == "fit" or stage is None:
            # Use all clips for training (no split)
            self.train_dataset = PrecomputedFeaturesDataset(**dataset_kwargs)
            # No validation dataset for now
            self.val_dataset = None

        if stage == "test" or stage is None:
            # Use same clips for testing
            self.test_dataset = PrecomputedFeaturesDataset(**dataset_kwargs)

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )
