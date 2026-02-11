"""DataModule for pre-computed V-JEPA2 encoder features.

This module loads pre-computed encoder features and actions from numpy files
for training the AC Predictor model. States are zero-filled by default.

Temporal Handling:
    Features are pre-encoded with V-JEPA2 tubelet encoding (tubelet_size=2), so they
    already have half the temporal resolution of the original video frames:
    - num_timesteps = 8 (pre-encoded from 16 original frames with tubelet_size=2)
    - T_actions = num_timesteps - 1 (actions have one less timestep than features)

    IMPORTANT: The `num_timesteps` parameter refers to ENCODED timesteps in the
    precomputed .npy files, NOT original video frames.

    For actions, only the value at index 1 (second timestep) is preserved and
    moved to index 0 in the output array. This handles the case where the important
    action occurs at the second timestep in the original data.

Expected data format:
    - features: .npy files with shape [T*N, D] (flattened) or [T, N, D] or [T, H, W, D]
      where T is the encoded temporal dim, N = H*W is patches per frame, D is embed_dim
    - actions: .npy files with shape [T_original, action_dim] (will be resampled)
    - states: Created as zeros with shape [T_actions, action_dim]
    - (optional) extrinsics: .npy files with shape [T, action_dim-1]

Directory structure:
    data_dir/
    ├── clip_00001/
    │   ├── feature_maps/
    │   │   └── vjepa2_vitl16.npy   # [T*N, D] or [T, N, D]
    │   └── actions_states/
    │       └── actions.npy         # [T_original, action_dim]
    ├── clip_00002/
    └── ...
"""

from pathlib import Path
from typing import Any, Final

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.utils.device_utils import should_pin_memory

# Constants
MIN_NUM_TIMESTEPS: Final = 2
FEATURE_MAPS_DIR = "feature_maps"
ACTIONS_STATES_DIR = "actions_states"
ACTIONS_FILE = "actions.npy"
EXTRINSICS_FILE = "extrinsics.npy"
IMPORTANT_ACTION_INDEX = 1


class DatasetNotFoundError(ValueError):
    """Raised when no valid clip directories are found."""

    def __init__(self, data_dir: Path, clip_prefix: str, feature_map_name: str) -> None:
        self.data_dir = data_dir
        self.clip_prefix = clip_prefix
        self.feature_map_name = feature_map_name
        expected_structure = f"{clip_prefix}XXXXX/{FEATURE_MAPS_DIR}/{feature_map_name}.npy"
        super().__init__(
            f"No valid clips found in {data_dir}. "
            f"Expected structure: {expected_structure}"
        )


class InvalidClipRangeError(ValueError):
    """Raised when clip range configuration is invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class TimestepsValidationError(ValueError):
    """Raised when num_timesteps is invalid."""

    def __init__(self, num_timesteps: int) -> None:
        self.num_timesteps = num_timesteps
        super().__init__(
            f"num_timesteps must be >= {MIN_NUM_TIMESTEPS} (got {num_timesteps}). "
            f"Need at least {MIN_NUM_TIMESTEPS} timesteps for context + target."
        )


type FeatureArray = np.ndarray
type SampleDict = dict[str, Tensor | str]
type BatchDict = dict[str, Tensor]


class PrecomputedFeaturesDataset(Dataset):
    """Dataset for pre-computed V-JEPA2 encoder features.

    Supports optional clip range filtering via clip_start and clip_end parameters.
    This allows selecting a subset of clips for training (e.g., clips 0-5000).
    """

    def __init__(
        self,
        data_dir: str | Path,
        num_timesteps: int = 8,
        patches_per_frame: int = 256,
        use_extrinsics: bool = False,
        feature_map_name: str = "vjepa2_vitl16",
        clip_prefix: str = "clip_",
        clip_start: int | None = None,
        clip_end: int | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_dir: Path to directory containing clip subdirectories.
            num_timesteps: Number of encoded timesteps to load from precomputed features.
                This is T in the loss formulation. For V-JEPA2 with tubelet_size=2,
                this equals original_frames // 2 (e.g., 16 frames -> 8 timesteps).
            patches_per_frame: Number of patches per frame (H*W). Required for reshaping
                flattened features. Default 256 (16x16 patches for ViT-L).
            use_extrinsics: Whether to load extrinsics data.
            feature_map_name: Name of the feature map file (without .npy extension).
            clip_prefix: Prefix for clip directories (e.g., "clip_").
            clip_start: Start of clip range (inclusive). If None, no lower bound.
                Clips are filtered by their numeric ID (e.g., clip_00042 -> 42).
            clip_end: End of clip range (exclusive). If None, no upper bound.
                Example: clip_start=0, clip_end=5000 selects clips 0-4999.

        Raises:
            TimestepsValidationError: If num_timesteps < 2.
            DatasetNotFoundError: If no valid clips are found.
        """
        self.data_dir = Path(data_dir)
        self.num_timesteps = num_timesteps
        self.patches_per_frame = patches_per_frame
        self.use_extrinsics = use_extrinsics
        self.feature_map_name = feature_map_name
        self.clip_prefix = clip_prefix
        self.clip_start = clip_start
        self.clip_end = clip_end

        # Validate num_timesteps
        if num_timesteps < MIN_NUM_TIMESTEPS:
            raise TimestepsValidationError(num_timesteps)

        # Find all clip directories with the expected structure
        feature_map_path = f"{FEATURE_MAPS_DIR}/{feature_map_name}.npy"
        self.episode_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir()
            and d.name.startswith(clip_prefix)
            and (d / feature_map_path).exists()
        ])

        # Filter by clip range if specified
        if clip_start is not None or clip_end is not None:
            self.episode_dirs = self._filter_by_clip_range(self.episode_dirs)

        if len(self.episode_dirs) == 0:
            raise DatasetNotFoundError(self.data_dir, clip_prefix, feature_map_name)

        # Infer action_dim from first clip
        first_actions = np.load(
            self.episode_dirs[0] / ACTIONS_STATES_DIR / ACTIONS_FILE
        )
        self.action_dim = first_actions.shape[-1]

    def _filter_by_clip_range(self, directories: list[Path]) -> list[Path]:
        """Filter directories by numeric clip ID range.

        Args:
            directories: List of clip directory paths.

        Returns:
            Filtered list of directories within the specified range.
        """
        filtered = []
        for directory in directories:
            try:
                clip_id = int(directory.name.replace(self.clip_prefix, ""))
            except ValueError:
                continue

            if self.clip_start is not None and clip_id < self.clip_start:
                continue
            if self.clip_end is not None and clip_id >= self.clip_end:
                continue

            filtered.append(directory)

        return filtered

    def __len__(self) -> int:
        return len(self.episode_dirs)

    def __getitem__(self, idx: int) -> SampleDict:
        """Load a single clip.

        Handles temporal alignment for V-JEPA2 tubelet encoding:
        - Features are reshaped from [T*N, D] to [T, N, D] if flattened.
        - T_actions = T - 1 (actions have one less timestep than features).
        - Actions: only value at index 1 is preserved, moved to index 0.
        - States: zero-filled array with shape [T_actions, action_dim].

        Args:
            idx: Index of the clip to load.

        Returns:
            Dictionary with keys:
                - features: [T, N, D] tensor (T encoded timesteps).
                - actions: [T-1, action_dim] tensor (resampled).
                - states: [T-1, action_dim] tensor (zeros).
                - extrinsics (optional): [T-1, action_dim-1] tensor.
                - clip_name: str (name of the clip directory).
        """
        episode_dir = self.episode_dirs[idx]

        # Load and reshape features
        features = self._load_features(episode_dir)

        # Calculate T_actions = T_encoded - 1
        T_actions = features.shape[0] - 1

        # Load and process actions
        actions = self._load_actions(episode_dir, T_actions)

        # Create zero-filled states array
        states = np.zeros((T_actions, self.action_dim), dtype=np.float32)

        result = {
            "features": torch.from_numpy(features).float(),
            "actions": torch.from_numpy(actions).float(),
            "states": torch.from_numpy(states).float(),
            "clip_name": episode_dir.name,
        }

        # Optionally load extrinsics
        if self.use_extrinsics:
            extrinsics = self._load_extrinsics(episode_dir, T_actions)
            if extrinsics is not None:
                result["extrinsics"] = extrinsics

        return result

    def _load_features(self, episode_dir: Path) -> FeatureArray:
        """Load and reshape feature array from disk.

        Args:
            episode_dir: Path to the clip directory.

        Returns:
            Feature array with shape [T, N, D].
        """
        features = np.load(
            episode_dir / FEATURE_MAPS_DIR / f"{self.feature_map_name}.npy"
        )

        # Handle different feature shapes:
        # - [T*N, D] flattened -> reshape to [T, N, D]
        # - [T, N, D] already correct shape
        # - [T, H, W, D] -> reshape to [T, H*W, D]
        if features.ndim == 2:
            total_tokens, D = features.shape
            T_encoded = total_tokens // self.patches_per_frame
            features = features.reshape(T_encoded, self.patches_per_frame, D)
        elif features.ndim == 4:
            T_encoded, H, W, D = features.shape
            features = features.reshape(T_encoded, H * W, D)
        else:
            T_encoded = features.shape[0]

        # Limit to num_timesteps + 1 (T context + 1 target)
        T_encoded = min(T_encoded, self.num_timesteps + 1)
        return features[:T_encoded]

    def _load_actions(self, episode_dir: Path, T_actions: int) -> FeatureArray:
        """Load and process action array from disk.

        Args:
            episode_dir: Path to the clip directory.
            T_actions: Number of action timesteps to output.

        Returns:
            Action array with shape [T_actions, action_dim].
        """
        actions_original = np.load(episode_dir / ACTIONS_STATES_DIR / ACTIONS_FILE)
        actions = np.zeros((T_actions, self.action_dim), dtype=np.float32)

        # Preserve the important action value from index 1 -> index 0
        if actions_original.shape[0] > IMPORTANT_ACTION_INDEX:
            actions[0] = actions_original[IMPORTANT_ACTION_INDEX]

        return actions

    def _load_extrinsics(
        self, episode_dir: Path, T_actions: int
    ) -> Tensor | None:
        """Load extrinsics array from disk if available.

        Args:
            episode_dir: Path to the clip directory.
            T_actions: Number of timesteps to slice.

        Returns:
            Extrinsics tensor with shape [T_actions, extrinsics_dim], or None.
        """
        extrinsics_path = episode_dir / ACTIONS_STATES_DIR / EXTRINSICS_FILE
        if not extrinsics_path.exists():
            return None

        extrinsics = np.load(extrinsics_path)[:T_actions]
        return torch.from_numpy(extrinsics).float()


def collate_fn(batch: list[SampleDict]) -> BatchDict:
    """Custom collate function to handle variable-length sequences.

    Pads sequences to the maximum length in the batch.
    Infers action_dim from the batch data.

    Args:
        batch: List of sample dictionaries from the dataset.

    Returns:
        Dictionary with padded tensors and optional clip_names.
    """
    # Find max lengths
    max_T_plus_1 = max(item["features"].shape[0] for item in batch)
    max_T = max_T_plus_1 - 1

    # Get dimensions from first item
    N = batch[0]["features"].shape[1]
    D = batch[0]["features"].shape[2]
    action_dim = batch[0]["actions"].shape[-1]

    # Check for extrinsics
    has_extrinsics = "extrinsics" in batch[0]

    # Initialize padded tensors
    B = len(batch)
    features = torch.zeros(B, max_T_plus_1, N, D)
    actions = torch.zeros(B, max_T, action_dim)
    states = torch.zeros(B, max_T, action_dim)

    result: dict[str, Tensor | list[str]] = {
        "features": features,
        "actions": actions,
        "states": states,
    }

    if has_extrinsics:
        extrinsics_dim = batch[0]["extrinsics"].shape[-1]
        extrinsics = torch.zeros(B, max_T, extrinsics_dim)
        result["extrinsics"] = extrinsics

    # Fill in data
    for i, item in enumerate(batch):
        T_plus_1 = item["features"].shape[0]
        T = T_plus_1 - 1
        features[i, :T_plus_1] = item["features"]
        actions[i, :T] = item["actions"]
        states[i, :T] = item["states"]
        if has_extrinsics and "extrinsics" in item:
            extrinsics[i, :T] = item["extrinsics"]

    # Include clip_names if available in batch items
    if "clip_name" in batch[0]:
        result["clip_names"] = [item["clip_name"] for item in batch]

    return result


class PrecomputedFeaturesDataModule(L.LightningDataModule):
    """Lightning DataModule for pre-computed V-JEPA2 encoder features."""

    def __init__(
        self,
        data_dir: str = "data",
        num_timesteps: int = 8,
        patches_per_frame: int | None = None,
        use_extrinsics: bool = False,
        feature_map_name: str = "vjepa2_vitl16",
        clip_prefix: str = "clip_",
        clip_start: int | None = None,
        clip_end: int | None = None,
        val_split: float = 0.0,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool | None = None,
        persistent_workers: bool = True,
        shuffle_test: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the DataModule.

        Args:
            data_dir: Root directory containing clip subdirectories.
            num_timesteps: Number of encoded timesteps per sample. This is T in the
                loss formulation. For V-JEPA2 with tubelet_size=2, this equals
                original_frames // 2 (e.g., 16 frames -> 8 timesteps).
            patches_per_frame: Number of patches per frame (H*W).
            use_extrinsics: Whether to load extrinsics data.
            feature_map_name: Name of the feature map file (without .npy extension).
            clip_prefix: Prefix for clip directories (e.g., "clip_").
            clip_start: Start of clip range (inclusive). If None, no lower bound.
            clip_end: End of clip range (exclusive). If None, no upper bound.
            val_split: Fraction of clips to use for validation (e.g., 0.1 = 10%).
                The last N% of clips (by ID) are used for validation.
                Must be in range [0.0, 1.0). If 0.0, no validation set is created.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of worker processes for data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
                If None, auto-detects based on CUDA availability.
            persistent_workers: Whether to keep worker processes alive between epochs.
                Speeds up dataloader initialization. Requires num_workers > 0.
            shuffle_test: Whether to shuffle the test dataloader. Default False.
                Set to False for TTA to ensure deterministic clip ordering.

        Raises:
            InvalidClipRangeError: If val_split would leave no clips for training.
        """
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.num_timesteps = num_timesteps
        self.patches_per_frame = patches_per_frame
        self.use_extrinsics = use_extrinsics
        self.feature_map_name = feature_map_name
        self.clip_prefix = clip_prefix
        self.clip_start = clip_start
        self.clip_end = clip_end
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Auto-detect pin_memory if not specified (only beneficial for CUDA)
        self.pin_memory = pin_memory if pin_memory is not None else should_pin_memory()

        # persistent_workers requires num_workers > 0
        self.persistent_workers = persistent_workers and num_workers > 0

        # shuffle_test for TTA compatibility
        self.shuffle_test = shuffle_test

        self.train_dataset: PrecomputedFeaturesDataset | None = None
        self.val_dataset: PrecomputedFeaturesDataset | None = None
        self.test_dataset: PrecomputedFeaturesDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage.

        Uses data_dir directly (flat structure with clip_* directories).
        If val_split > 0, the last N% of clips (by ID) are used for validation.

        Args:
            stage: One of "fit", "test", or None for all stages.

        Raises:
            InvalidClipRangeError: If val_split configuration is invalid.
        """
        # Compute effective clip range
        effective_start = self.clip_start if self.clip_start is not None else 0
        effective_end = self.clip_end

        # Calculate train/val split point if val_split is specified
        if self.val_split > 0.0:
            val_start, train_end = self._calculate_split_ranges(
                effective_start, effective_end
            )
        else:
            train_end = effective_end
            val_start = None

        # Base dataset kwargs (without clip range)
        base_kwargs = {
            "data_dir": self.data_dir,
            "num_timesteps": self.num_timesteps,
            "patches_per_frame": self.patches_per_frame,
            "use_extrinsics": self.use_extrinsics,
            "feature_map_name": self.feature_map_name,
            "clip_prefix": self.clip_prefix,
        }

        if stage == "fit" or stage is None:
            self._setup_train_val_datasets(base_kwargs, effective_start, train_end, val_start, effective_end)

        if stage == "test" or stage is None:
            self.test_dataset = PrecomputedFeaturesDataset(
                **base_kwargs,
                clip_start=effective_start,
                clip_end=train_end,
            )

    def _calculate_split_ranges(
        self, effective_start: int, effective_end: int | None
    ) -> tuple[int | None, int | None]:
        """Calculate train/val split ranges.

        Args:
            effective_start: Start of the clip range.
            effective_end: End of the clip range.

        Returns:
            Tuple of (val_start, train_end).

        Raises:
            InvalidClipRangeError: If val_split configuration is invalid.
        """
        if self.val_split >= 1.0:
            raise InvalidClipRangeError(
                f"val_split must be in range [0.0, 1.0), got {self.val_split}"
            )

        if effective_end is None:
            raise InvalidClipRangeError(
                "clip_end must be specified when using val_split to determine "
                "the validation range. Set clip_end to the total number of clips."
            )

        total_clips = effective_end - effective_start
        val_clips = int(total_clips * self.val_split)
        val_clips = max(1, val_clips)  # Ensure at least 1 clip in validation

        # Ensure at least 1 clip remains for training
        if val_clips >= total_clips:
            raise InvalidClipRangeError(
                f"val_split={self.val_split} would leave no clips for training. "
                f"Total clips: {total_clips}, val clips: {val_clips}"
            )

        val_start = effective_end - val_clips
        train_end = val_start

        return val_start, train_end

    def _setup_train_val_datasets(
        self,
        base_kwargs: dict[str, Any],
        effective_start: int,
        train_end: int | None,
        val_start: int | None,
        effective_end: int | None,
    ) -> None:
        """Set up training and validation datasets.

        Args:
            base_kwargs: Base dataset keyword arguments.
            effective_start: Start of the clip range.
            train_end: End of training clip range.
            val_start: Start of validation clip range.
            effective_end: End of the total clip range.
        """
        # Training dataset: clips from start to split point
        self.train_dataset = PrecomputedFeaturesDataset(
            **base_kwargs,
            clip_start=effective_start,
            clip_end=train_end,
        )

        # Validation dataset: clips from split point to end (if val_split > 0)
        if val_start is not None:
            self.val_dataset = PrecomputedFeaturesDataset(
                **base_kwargs,
                clip_start=val_start,
                clip_end=effective_end,
            )
        else:
            self.val_dataset = None

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader.

        Returns:
            DataLoader for training data.

        Raises:
            RuntimeError: If train_dataset is not initialized.
        """
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader | None:
        """Create validation DataLoader.

        Returns None if no validation dataset is configured, which signals
        to PyTorch Lightning to skip validation entirely.

        Returns:
            DataLoader for validation data, or None if no validation set.
        """
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader.

        Uses shuffle_test parameter to control shuffling (default: False).
        For TTA, shuffle should be disabled to ensure deterministic ordering.

        Returns:
            DataLoader for test data.

        Raises:
            RuntimeError: If test_dataset is not initialized.
        """
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_test,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
        )
