"""Unit tests for PrecomputedFeaturesDataModule.

Tests cover:
1. Dataset initialization and validation
2. Data loading with correct shapes
3. Edge cases (clip range filtering)
4. Collate function behavior
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.datamodules.precomputed_features import (
    PrecomputedFeaturesDataset,
    PrecomputedFeaturesDataModule,
    collate_fn,
)


class TestPrecomputedFeaturesDataset:
    """Tests for the PrecomputedFeaturesDataset class."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary data directory with sample clips."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Create sample clips
            for clip_id in range(5):
                clip_dir = data_dir / f"clip_{clip_id:05d}"
                (clip_dir / "feature_maps").mkdir(parents=True)
                (clip_dir / "actions_states").mkdir(parents=True)

                # Create feature maps: [T, N, D]
                T, N, D = 8, 256, 768
                features = np.random.randn(T, N, D).astype(np.float32)
                np.save(clip_dir / "feature_maps" / "vjepa2_vitl16.npy", features)

                # Create actions: [T_original, action_dim]
                action_dim = 7
                actions = np.random.randn(T * 2, action_dim).astype(np.float32)
                np.save(clip_dir / "actions_states" / "actions.npy", actions)

            yield data_dir

    def test_dataset_initialization(self, temp_data_dir):
        """Test that dataset initializes correctly."""
        dataset = PrecomputedFeaturesDataset(
            data_dir=temp_data_dir,
            num_timesteps=8,
            patches_per_frame=256,
        )

        assert len(dataset) == 5
        assert dataset.action_dim == 7

    def test_dataset_num_timesteps_validation(self, temp_data_dir):
        """Test that num_timesteps < 2 raises ValueError."""
        with pytest.raises(ValueError, match="num_timesteps must be >= 2"):
            PrecomputedFeaturesDataset(
                data_dir=temp_data_dir,
                num_timesteps=1,
            )

    def test_dataset_empty_dir_raises(self):
        """Test that empty/invalid directory raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No valid clips found"):
                PrecomputedFeaturesDataset(
                    data_dir=tmpdir,
                    num_timesteps=8,
                )

    def test_getitem_returns_correct_keys(self, temp_data_dir):
        """Test that __getitem__ returns correct dictionary keys."""
        dataset = PrecomputedFeaturesDataset(
            data_dir=temp_data_dir,
            num_timesteps=8,
            patches_per_frame=256,
        )

        item = dataset[0]

        assert "features" in item
        assert "actions" in item
        assert "states" in item

    def test_getitem_returns_correct_shapes(self, temp_data_dir):
        """Test that __getitem__ returns tensors with correct shapes."""
        dataset = PrecomputedFeaturesDataset(
            data_dir=temp_data_dir,
            num_timesteps=7,  # Request 7 timesteps, will get 8 (T+1)
            patches_per_frame=256,
        )

        item = dataset[0]

        # Features: [T+1, N, D] where T = min(encoded_T, num_timesteps)
        assert item["features"].dim() == 3
        T_plus_1, N, D = item["features"].shape
        assert N == 256
        assert D == 768

        # Actions: [T, action_dim]
        T_actions = T_plus_1 - 1
        assert item["actions"].shape == (T_actions, 7)

        # States: [T, action_dim] (zero-filled)
        assert item["states"].shape == (T_actions, 7)
        assert torch.allclose(item["states"], torch.zeros_like(item["states"]))

    def test_clip_range_filtering(self, temp_data_dir):
        """Test that clip_start and clip_end filter correctly."""
        # Filter to clips 1-3 (exclusive end)
        dataset = PrecomputedFeaturesDataset(
            data_dir=temp_data_dir,
            num_timesteps=8,
            patches_per_frame=256,
            clip_start=1,
            clip_end=4,
        )

        assert len(dataset) == 3  # Clips 1, 2, 3

    def test_clip_range_start_only(self, temp_data_dir):
        """Test filtering with only clip_start."""
        dataset = PrecomputedFeaturesDataset(
            data_dir=temp_data_dir,
            num_timesteps=8,
            patches_per_frame=256,
            clip_start=3,
        )

        assert len(dataset) == 2  # Clips 3, 4

    def test_clip_range_end_only(self, temp_data_dir):
        """Test filtering with only clip_end."""
        dataset = PrecomputedFeaturesDataset(
            data_dir=temp_data_dir,
            num_timesteps=8,
            patches_per_frame=256,
            clip_end=2,
        )

        assert len(dataset) == 2  # Clips 0, 1


class TestPrecomputedFeaturesDatasetWithExtrinsics:
    """Tests for dataset with extrinsics loading."""

    @pytest.fixture
    def temp_data_dir_with_extrinsics(self):
        """Create data directory with extrinsics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            clip_dir = data_dir / "clip_00000"
            (clip_dir / "feature_maps").mkdir(parents=True)
            (clip_dir / "actions_states").mkdir(parents=True)

            T, N, D = 8, 256, 768
            action_dim = 7

            features = np.random.randn(T, N, D).astype(np.float32)
            np.save(clip_dir / "feature_maps" / "vjepa2_vitl16.npy", features)

            actions = np.random.randn(T * 2, action_dim).astype(np.float32)
            np.save(clip_dir / "actions_states" / "actions.npy", actions)

            # Create extrinsics: [T, action_dim-1]
            extrinsics = np.random.randn(T, action_dim - 1).astype(np.float32)
            np.save(clip_dir / "actions_states" / "extrinsics.npy", extrinsics)

            yield data_dir

    def test_extrinsics_loading(self, temp_data_dir_with_extrinsics):
        """Test that extrinsics are loaded when use_extrinsics=True."""
        dataset = PrecomputedFeaturesDataset(
            data_dir=temp_data_dir_with_extrinsics,
            num_timesteps=7,
            patches_per_frame=256,
            use_extrinsics=True,
        )

        item = dataset[0]

        assert "extrinsics" in item
        # Extrinsics should have T_actions timesteps (using the fixed T_actions variable)
        T_actions = item["features"].shape[0] - 1
        assert item["extrinsics"].shape[0] == T_actions
        assert item["extrinsics"].shape[1] == 6  # action_dim - 1


class TestCollateFn:
    """Tests for the custom collate function."""

    def test_collate_fn_batches_correctly(self):
        """Test that collate_fn creates proper batched tensors."""
        batch = [
            {
                "features": torch.randn(8, 256, 768),
                "actions": torch.randn(7, 7),
                "states": torch.randn(7, 7),
            },
            {
                "features": torch.randn(8, 256, 768),
                "actions": torch.randn(7, 7),
                "states": torch.randn(7, 7),
            },
        ]

        result = collate_fn(batch)

        assert result["features"].shape == (2, 8, 256, 768)
        assert result["actions"].shape == (2, 7, 7)
        assert result["states"].shape == (2, 7, 7)

    def test_collate_fn_pads_variable_length(self):
        """Test that collate_fn pads variable-length sequences."""
        batch = [
            {
                "features": torch.randn(6, 256, 768),  # Shorter
                "actions": torch.randn(5, 7),
                "states": torch.randn(5, 7),
            },
            {
                "features": torch.randn(8, 256, 768),  # Longer
                "actions": torch.randn(7, 7),
                "states": torch.randn(7, 7),
            },
        ]

        result = collate_fn(batch)

        # Should pad to max length
        assert result["features"].shape == (2, 8, 256, 768)
        assert result["actions"].shape == (2, 7, 7)

    def test_collate_fn_with_extrinsics(self):
        """Test collate_fn with extrinsics."""
        batch = [
            {
                "features": torch.randn(8, 256, 768),
                "actions": torch.randn(7, 7),
                "states": torch.randn(7, 7),
                "extrinsics": torch.randn(7, 6),
            },
            {
                "features": torch.randn(8, 256, 768),
                "actions": torch.randn(7, 7),
                "states": torch.randn(7, 7),
                "extrinsics": torch.randn(7, 6),
            },
        ]

        result = collate_fn(batch)

        assert "extrinsics" in result
        assert result["extrinsics"].shape == (2, 7, 6)


class TestPrecomputedFeaturesDataModule:
    """Tests for the Lightning DataModule wrapper."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary data directory with sample clips."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            for clip_id in range(10):
                clip_dir = data_dir / f"clip_{clip_id:05d}"
                (clip_dir / "feature_maps").mkdir(parents=True)
                (clip_dir / "actions_states").mkdir(parents=True)

                T, N, D = 8, 256, 768
                features = np.random.randn(T, N, D).astype(np.float32)
                np.save(clip_dir / "feature_maps" / "vjepa2_vitl16.npy", features)

                actions = np.random.randn(T * 2, 7).astype(np.float32)
                np.save(clip_dir / "actions_states" / "actions.npy", actions)

            yield data_dir

    def test_datamodule_setup(self, temp_data_dir):
        """Test that DataModule sets up correctly."""
        datamodule = PrecomputedFeaturesDataModule(
            data_dir=str(temp_data_dir),
            num_timesteps=8,
            patches_per_frame=256,
            batch_size=2,
            val_split=0.2,
            clip_end=10,  # Required when using val_split
        )

        datamodule.setup(stage="fit")

        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None
        assert len(datamodule.train_dataset) == 8
        assert len(datamodule.val_dataset) == 2

    def test_datamodule_no_val_split(self, temp_data_dir):
        """Test DataModule with no validation split."""
        datamodule = PrecomputedFeaturesDataModule(
            data_dir=str(temp_data_dir),
            num_timesteps=8,
            patches_per_frame=256,
            batch_size=2,
            val_split=0.0,
        )

        datamodule.setup(stage="fit")

        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is None
        assert len(datamodule.train_dataset) == 10

    def test_train_dataloader(self, temp_data_dir):
        """Test that train_dataloader works correctly."""
        datamodule = PrecomputedFeaturesDataModule(
            data_dir=str(temp_data_dir),
            num_timesteps=8,
            patches_per_frame=256,
            batch_size=2,
        )

        datamodule.setup(stage="fit")
        loader = datamodule.train_dataloader()

        batch = next(iter(loader))

        assert "features" in batch
        assert batch["features"].shape[0] == 2  # Batch size
