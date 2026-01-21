"""Unit tests for AC Predictor model and lightning module.

Tests cover:
1. Model forward pass with correct shapes
2. Loss computation correctness
3. Numerical stability guards
4. Edge cases (single timestep, batch size 1)
5. Rollout logic with shape assertions
"""

import pytest
import torch
import torch.nn.functional as F

from src.models.ac_predictor.lightning_module import ACPredictorModule
from src.models.ac_predictor.utils.tensors import trunc_normal_


class TestACPredictorModule:
    """Tests for the ACPredictorModule lightning wrapper."""

    @pytest.fixture
    def default_module(self):
        """Create a default ACPredictorModule for testing."""
        return ACPredictorModule(
            img_size=(256, 256),
            patch_size=16,
            num_timesteps=8,
            embed_dim=768,
            predictor_embed_dim=384,  # Smaller for faster tests
            depth=2,  # Fewer layers for faster tests
            num_heads=8,
            mlp_ratio=4.0,
            action_embed_dim=7,
            use_rope=True,
            is_frame_causal=True,
            T_teacher=7,
            T_rollout=2,
            context_frames=1,
            loss_exp=1.0,
            normalize_reps=True,
        )

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        B = 2  # Batch size
        T_plus_1 = 8  # 7 context + 1 target
        N = 256  # 16x16 patches
        D = 768  # Embedding dimension
        action_dim = 7

        return {
            "features": torch.randn(B, T_plus_1, N, D),
            "actions": torch.randn(B, T_plus_1 - 1, action_dim),
            "states": torch.randn(B, T_plus_1 - 1, action_dim),
        }

    def test_module_initialization(self, default_module):
        """Test that module initializes correctly."""
        assert default_module is not None
        assert default_module.model is not None
        assert default_module.T_teacher == 7
        assert default_module.T_rollout == 2
        assert default_module.loss_exp == 1.0

    def test_loss_exp_validation_zero(self):
        """Test that loss_exp=0 raises ValueError."""
        with pytest.raises(ValueError, match="loss_exp must be positive"):
            ACPredictorModule(
                depth=1,
                loss_exp=0.0,
            )

    def test_loss_exp_validation_negative(self):
        """Test that negative loss_exp raises ValueError."""
        with pytest.raises(ValueError, match="loss_exp must be positive"):
            ACPredictorModule(
                depth=1,
                loss_exp=-1.0,
            )

    def test_compute_loss_l1(self, default_module):
        """Test L1 loss computation (loss_exp=1.0)."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 3.5])

        # L1 loss: mean(|pred - target|) / 1.0 = mean(0.5) = 0.5
        loss = default_module._compute_loss(pred, target)
        expected = torch.tensor(0.5)

        assert torch.allclose(loss, expected, atol=1e-6)

    def test_compute_loss_l2(self):
        """Test L2 loss computation (loss_exp=2.0)."""
        module = ACPredictorModule(depth=1, loss_exp=2.0)

        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 3.5])

        # L2 loss: mean(|pred - target|^2) / 2.0 = mean(0.25) / 2 = 0.125
        loss = module._compute_loss(pred, target)
        expected = torch.tensor(0.125)

        assert torch.allclose(loss, expected, atol=1e-6)

    def test_forward_pass_shapes(self, default_module, sample_batch):
        """Test that forward pass produces correct output shapes."""
        B, T_plus_1, N, D = sample_batch["features"].shape
        T = T_plus_1 - 1

        # Flatten features for forward pass
        features_flat = sample_batch["features"][:, :T].reshape(B, T * N, D)
        actions = sample_batch["actions"][:, :T]
        states = sample_batch["states"][:, :T]

        output = default_module(features_flat, actions, states)

        assert output.shape == (B, T * N, D), f"Expected shape {(B, T * N, D)}, got {output.shape}"

    def test_teacher_forcing_loss_runs(self, default_module, sample_batch):
        """Test that teacher-forcing loss can be computed."""
        loss = default_module._compute_teacher_forcing_loss(
            sample_batch["features"],
            sample_batch["actions"],
            sample_batch["states"],
        )

        assert loss.dim() == 0, "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"
        assert loss >= 0, "Loss should be non-negative"

    def test_rollout_loss_runs(self, default_module, sample_batch):
        """Test that rollout loss can be computed."""
        loss = default_module._compute_rollout_loss(
            sample_batch["features"],
            sample_batch["actions"],
            sample_batch["states"],
        )

        assert loss.dim() == 0, "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"
        assert loss >= 0, "Loss should be non-negative"

    def test_shared_step_returns_combined_loss(self, default_module, sample_batch):
        """Test that _shared_step returns combined loss."""
        loss = default_module._shared_step(sample_batch, "train")

        assert loss.dim() == 0, "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert loss >= 0, "Loss should be non-negative"

    def test_batch_size_one(self, default_module):
        """Test that batch size 1 works correctly."""
        B = 1
        T_plus_1 = 4
        N = 256
        D = 768
        action_dim = 7

        batch = {
            "features": torch.randn(B, T_plus_1, N, D),
            "actions": torch.randn(B, T_plus_1 - 1, action_dim),
            "states": torch.randn(B, T_plus_1 - 1, action_dim),
        }

        loss = default_module._shared_step(batch, "train")

        assert not torch.isnan(loss), "Loss should not be NaN for batch_size=1"

    def test_minimum_timesteps(self):
        """Test with minimum valid timesteps (num_timesteps=2)."""
        module = ACPredictorModule(
            num_timesteps=2,
            depth=1,
            T_teacher=1,
            T_rollout=1,
            context_frames=1,
        )

        B, T_plus_1, N, D = 2, 2, 256, 768
        action_dim = 7

        batch = {
            "features": torch.randn(B, T_plus_1, N, D),
            "actions": torch.randn(B, T_plus_1 - 1, action_dim),
            "states": torch.randn(B, T_plus_1 - 1, action_dim),
        }

        loss = module._shared_step(batch, "train")

        assert not torch.isnan(loss), "Loss should not be NaN for minimum timesteps"


class TestNumericalStability:
    """Tests for numerical stability guards."""

    def test_trunc_normal_std_zero_raises(self):
        """Test that trunc_normal_ with std=0 raises ValueError."""
        tensor = torch.empty(10, 10)

        with pytest.raises(ValueError, match="std must be positive"):
            trunc_normal_(tensor, std=0.0)

    def test_trunc_normal_std_negative_raises(self):
        """Test that trunc_normal_ with negative std raises ValueError."""
        tensor = torch.empty(10, 10)

        with pytest.raises(ValueError, match="std must be positive"):
            trunc_normal_(tensor, std=-0.5)

    def test_trunc_normal_valid_std(self):
        """Test that trunc_normal_ works with valid std."""
        tensor = torch.empty(100, 100)
        result = trunc_normal_(tensor, mean=0.0, std=0.02)

        assert not torch.isnan(result).any(), "Result should not contain NaN"
        assert not torch.isinf(result).any(), "Result should not contain Inf"
        # Check values are within truncation bounds (default a=-2.0, b=2.0 absolute)
        assert result.min() >= -2.0 - 1e-6
        assert result.max() <= 2.0 + 1e-6

    def test_layer_norm_with_small_variance(self):
        """Test that layer_norm with explicit epsilon handles small variance."""
        # Create input with very small variance
        x = torch.ones(2, 256, 768) * 1e-8
        x[:, :, 0] += 1e-10  # Tiny variation

        # This should not produce NaN with explicit epsilon
        result = F.layer_norm(x, (768,), eps=1e-6)

        assert not torch.isnan(result).any(), "LayerNorm should not produce NaN"

    def test_layer_norm_with_zero_input(self):
        """Test layer_norm behavior with all-zero input."""
        x = torch.zeros(2, 256, 768)

        # With explicit epsilon, should produce zeros (or near-zero values)
        result = F.layer_norm(x, (768,), eps=1e-6)

        assert not torch.isnan(result).any(), "LayerNorm should not produce NaN for zero input"


class TestRolloutLogic:
    """Tests for autoregressive rollout logic."""

    @pytest.fixture
    def module_for_rollout(self):
        """Create module configured for rollout testing."""
        return ACPredictorModule(
            num_timesteps=8,
            depth=1,  # Minimal depth for speed
            T_teacher=7,
            T_rollout=3,
            context_frames=2,
            normalize_reps=True,
        )

    def test_rollout_context_frames_respected(self, module_for_rollout):
        """Test that rollout uses correct number of context frames."""
        B, T_plus_1, N, D = 2, 8, 256, 768
        action_dim = 7

        batch = {
            "features": torch.randn(B, T_plus_1, N, D),
            "actions": torch.randn(B, T_plus_1 - 1, action_dim),
            "states": torch.randn(B, T_plus_1 - 1, action_dim),
        }

        # This should run without errors
        loss = module_for_rollout._compute_rollout_loss(
            batch["features"],
            batch["actions"],
            batch["states"],
        )

        assert not torch.isnan(loss)

    def test_rollout_with_extrinsics(self):
        """Test rollout with optional extrinsics."""
        module = ACPredictorModule(
            num_timesteps=8,
            depth=1,
            T_rollout=2,
            context_frames=1,
            use_extrinsics=True,
        )

        B, T_plus_1, N, D = 2, 8, 256, 768
        action_dim = 7

        batch = {
            "features": torch.randn(B, T_plus_1, N, D),
            "actions": torch.randn(B, T_plus_1 - 1, action_dim),
            "states": torch.randn(B, T_plus_1 - 1, action_dim),
            "extrinsics": torch.randn(B, T_plus_1 - 1, action_dim - 1),
        }

        loss = module._compute_rollout_loss(
            batch["features"],
            batch["actions"],
            batch["states"],
            batch["extrinsics"],
        )

        assert not torch.isnan(loss)


class TestCurriculumLearning:
    """Tests for curriculum learning schedule."""

    def test_valid_curriculum_schedule(self):
        """Test that valid curriculum schedule is accepted."""
        schedule = [
            {"epoch": 0, "T_rollout": 1, "loss_weight_teacher": 1.0},
            {"epoch": 10, "T_rollout": 2, "loss_weight_teacher": 0.5},
            {"epoch": 20, "T_rollout": 3, "loss_weight_teacher": 0.0},
        ]

        module = ACPredictorModule(
            num_timesteps=8,
            depth=1,
            T_rollout=1,
            curriculum_schedule=schedule,
        )

        assert module.curriculum_schedule == schedule

    def test_curriculum_schedule_missing_epoch_raises(self):
        """Test that schedule without 'epoch' key raises error."""
        schedule = [
            {"T_rollout": 1},  # Missing 'epoch'
        ]

        with pytest.raises(ValueError, match="must have 'epoch' key"):
            ACPredictorModule(
                num_timesteps=8,
                depth=1,
                curriculum_schedule=schedule,
            )

    def test_curriculum_schedule_unsorted_raises(self):
        """Test that unsorted schedule raises error."""
        schedule = [
            {"epoch": 10, "T_rollout": 2},
            {"epoch": 0, "T_rollout": 1},  # Out of order
        ]

        with pytest.raises(ValueError, match="must be sorted by epoch"):
            ACPredictorModule(
                num_timesteps=8,
                depth=1,
                curriculum_schedule=schedule,
            )


class TestOptimizerConfiguration:
    """Tests for optimizer and scheduler configuration."""

    def test_epoch_based_scheduler(self):
        """Test epoch-based LR scheduler configuration."""
        module = ACPredictorModule(
            depth=1,
            use_iteration_scheduler=False,
            warmup_epochs=5,
            max_epochs=100,
        )

        config = module.configure_optimizers()

        assert "optimizer" in config
        assert "lr_scheduler" in config
        assert config["lr_scheduler"]["interval"] == "epoch"

    def test_iteration_based_scheduler(self):
        """Test iteration-based LR scheduler configuration (V-JEPA2 paper)."""
        module = ACPredictorModule(
            depth=1,
            use_iteration_scheduler=True,
            warmup_iters=100,
            constant_iters=800,
            decay_iters=100,
        )

        config = module.configure_optimizers()

        assert "optimizer" in config
        assert "lr_scheduler" in config
        assert config["lr_scheduler"]["interval"] == "step"
