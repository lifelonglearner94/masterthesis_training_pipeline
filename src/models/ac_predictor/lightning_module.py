# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Lightning Module wrapper for AC Predictor.

Implements teacher-forcing and jump prediction losses for training the action-conditioned
vision transformer predictor.

Loss formulations (configurable via loss_type: "l1", "l2", or "huber"):
- Teacher-Forcing: L_tf computed over single forward pass with full context.
- Jump Prediction: L_jump — single forward pass from z₀ + a₀ with RoPE(τ) to predict z_τ.
- Combined: L = (1 - λ) * L_tf + λ * L_jump

Key features:
- Optional layer normalization after each predictor step (normalize_reps)
- Configurable loss type: "l1" (MAE), "l2" (MSE), or "huber" (smooth L1)
- RoPE-conditioned jump prediction replacing jump prediction
- Stochastic target frame sampling for jump loss
"""

import json
import logging
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.ac_predictor.ac_predictor import vit_ac_predictor
from src.models.mixins import ACPredictorLossMixin, TTAMixin

# Constants
DEFAULT_LAYER_NORM_EPS: Final = 1e-6
ROLLING_AVG_WINDOW_SIZE: Final = 50
JSON_INDENT: Final = 2
NUM_WORST_CLIPS: Final = 10
NUM_BEST_CLIPS: Final = 5

# Type alias for test results
type TestResultsDict = dict[str, Any]


class ACPredictorModule(TTAMixin, ACPredictorLossMixin, L.LightningModule):
    """PyTorch Lightning module for Action-Conditioned Vision Transformer Predictor.

    This module wraps the VisionTransformerPredictorAC and implements:
    - Teacher-forcing loss: predicts next latent from context, averaged over T steps
    - Jump prediction loss: predicts future frame z_τ from z₀ + a₀ via RoPE conditioning
    - Test Time Adaptation (TTA): online adaptation using jump prediction error

    Inherits loss computation methods from ACPredictorLossMixin for code reuse
    with baseline models. TTAMixin provides TTA capabilities for test-time adaptation.

    Expected batch format:
        - features: [B, T+1, N, D] - Pre-computed V-JEPA2 encoder features for T+1 timesteps
        - actions: [B, T, action_dim] - 7D end-effector state changes
        - states: [B, T, action_dim] - 7D end-effector states

    Where:
        - T is the number of encoded timesteps (e.g., 8 for 16 original frames with tubelet_size=2)
        - N is the number of patches per frame (H*W)
        - D is the embedding dimension

    TTA Mode (enabled via tta_enabled=True):
        - Adapts LayerNorm parameters online during testing
        - Uses jump prediction loss as self-supervised signal
        - Per-clip reset restores original weights between clips

    Note: num_timesteps refers to the ENCODED temporal dimension of precomputed features,
    NOT the original video frame count.
    """

    def __init__(
        self,
        # Model architecture
        img_size: tuple[int, int] = (256, 256),
        patch_size: int = 16,
        num_timesteps: int = 8,
        embed_dim: int = 768,
        predictor_embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        action_embed_dim: int = 7,
        use_rope: bool = True,
        is_frame_causal: bool = True,
        use_activation_checkpointing: bool = False,
        use_extrinsics: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        # Loss settings
        T_teacher: int = 7,
        jump_k: int = 3,
        loss_weight_teacher: float = 1.0,
        loss_weight_jump: float = 1.0,
        normalize_reps: bool = True,
        loss_type: str = "l1",
        huber_delta: float = 1.0,
        # Optimizer settings
        learning_rate: float = 4.25e-4,
        weight_decay: float = 0.04,
        betas: tuple[float, float] = (0.9, 0.999),
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        # Iteration-based LR schedule (V-JEPA2 paper)
        use_iteration_scheduler: bool = False,
        # Curriculum learning schedule for dynamic loss adjustment
        curriculum_schedule: list[dict] | None = None,
        # Percentage-based schedule (percentages of total training iterations, must sum to 1.0)
        warmup_pct: float = 0.085,
        constant_pct: float = 0.83,
        decay_pct: float = 0.085,
        warmup_start_lr: float = 7.5e-5,
        # Test Time Adaptation (TTA) settings
        tta_enabled: bool = False,
        tta_lr: float = 1e-4,
        tta_grad_clip: float = 1.0,
        tta_reset_per_clip: bool = True,
        tta_adaptation_horizon: int = 1,
        tta_optimizer_type: str = "adam",
        tta_optimizer_betas: tuple[float, float] = (0.9, 0.999),
        tta_mode: str = "jump",
        tta_num_adaptation_steps: int = 1,
        tta_adapt_layers: str = "layernorm",  # Which layers to adapt
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Initialize TTA configuration (from TTAMixin)
        self._init_tta(
            tta_enabled=tta_enabled,
            tta_lr=tta_lr,
            tta_grad_clip=tta_grad_clip,
            tta_reset_per_clip=tta_reset_per_clip,
            tta_adaptation_horizon=tta_adaptation_horizon,
            tta_optimizer_type=tta_optimizer_type,
            tta_optimizer_betas=tta_optimizer_betas,
            tta_mode=tta_mode,
            tta_num_adaptation_steps=tta_num_adaptation_steps,
            tta_adapt_layers=tta_adapt_layers,
        )

        # Store num_timesteps for validation
        self.num_timesteps = num_timesteps

        # Build the predictor model
        self.model = vit_ac_predictor(
            img_size=img_size,
            patch_size=patch_size,
            num_timesteps=num_timesteps,
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            action_embed_dim=action_embed_dim,
            use_rope=use_rope,
            is_frame_causal=is_frame_causal,
            use_activation_checkpointing=use_activation_checkpointing,
            use_extrinsics=use_extrinsics,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # Loss hyperparameters
        self.T_teacher = T_teacher
        self.jump_k = jump_k

        # Validate T_teacher and jump_k against available timesteps
        max_prediction_steps = num_timesteps - 1  # Need at least 1 context + 1 target
        if T_teacher > max_prediction_steps:
            warnings.warn(
                f"T_teacher ({T_teacher}) exceeds available prediction steps ({max_prediction_steps}) "
                f"for num_timesteps={num_timesteps}. Will be clamped to {max_prediction_steps} at runtime.",
                UserWarning,
                stacklevel=2,
            )
        if jump_k > num_timesteps:
            warnings.warn(
                f"jump_k ({jump_k}) exceeds num_timesteps ({num_timesteps}). "
                f"Will be clamped at runtime.",
                UserWarning,
                stacklevel=2,
            )
        self.loss_weight_teacher = loss_weight_teacher
        self.loss_weight_jump = loss_weight_jump
        self.normalize_reps = normalize_reps

        # Validate and store loss type
        from src.models.mixins.loss_mixin import VALID_LOSS_TYPES
        if loss_type not in VALID_LOSS_TYPES:
            raise ValueError(
                f"loss_type must be one of {VALID_LOSS_TYPES} (got '{loss_type}')."
            )
        self.loss_type = loss_type
        self.huber_delta = huber_delta

        # Optimizer hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        # Iteration-based LR schedule (V-JEPA2 paper)
        self.use_iteration_scheduler = use_iteration_scheduler
        self.warmup_pct = warmup_pct
        self.constant_pct = constant_pct
        self.decay_pct = decay_pct
        self.warmup_start_lr = warmup_start_lr

        # Grid size for reshaping
        self.grid_height = img_size[0] // patch_size
        self.grid_width = img_size[1] // patch_size
        self.patches_per_frame = self.grid_height * self.grid_width

        # Curriculum learning: schedule for dynamic jump_k and loss weights
        # Format: [{"epoch": 0, "jump_k": 3, "loss_weight_jump": 0.2}, ...]
        self.curriculum_schedule = curriculum_schedule
        if curriculum_schedule:
            self._validate_curriculum_schedule(curriculum_schedule)

        # Storage for test results (populated during test_step, aggregated in on_test_epoch_end)
        self._test_results: list[TestResultsDict] = []

        # Logger instance
        self._log = logging.getLogger(__name__)

    def _validate_curriculum_schedule(self, schedule: list[dict]) -> None:
        """Validate curriculum schedule format and values.

        Args:
            schedule: List of dicts with keys: epoch, jump_k (optional),
                     loss_weight_teacher (optional), loss_weight_jump (optional)

        Raises:
            ValueError: If schedule is invalid
        """
        if not isinstance(schedule, Sequence) or isinstance(schedule, str) or len(schedule) == 0:
            raise ValueError("curriculum_schedule must be a non-empty list")

        for i, phase in enumerate(schedule):
            if not isinstance(phase, Mapping):
                raise ValueError(f"Phase {i} must be a dict, got {type(phase)}")
            if "epoch" not in phase:
                raise ValueError(f"Phase {i} must have 'epoch' key")
            if not isinstance(phase["epoch"], int) or phase["epoch"] < 0:
                raise ValueError(f"Phase {i} 'epoch' must be a non-negative integer")

            # Validate jump_k if present
            if "jump_k" in phase:
                if phase["jump_k"] > self.num_timesteps:
                    raise ValueError(
                        f"Phase {i}: jump_k ({phase['jump_k']}) exceeds "
                        f"num_timesteps ({self.num_timesteps})"
                    )

        # Ensure phases are sorted by epoch
        epochs = [p["epoch"] for p in schedule]
        if epochs != sorted(epochs):
            raise ValueError("curriculum_schedule phases must be sorted by epoch")

    def _get_curriculum_params_for_epoch(self, epoch: int) -> dict[str, Any]:
        """Get curriculum parameters for the given epoch.

        Finds the most recent phase whose epoch is <= current epoch.

        Args:
            epoch: Current training epoch

        Returns:
            Dict with jump_k, loss_weight_teacher, loss_weight_jump
        """
        if not self.curriculum_schedule:
            return {}

        # Find the applicable phase (last one where epoch <= current)
        applicable_phase = None
        for phase in self.curriculum_schedule:
            if phase["epoch"] <= epoch:
                applicable_phase = phase
            else:
                break

        if applicable_phase is None:
            return {}

        # Extract params (excluding 'epoch' key)
        return {k: v for k, v in applicable_phase.items() if k != "epoch"}

    def on_train_epoch_start(self) -> None:
        """Update curriculum parameters at the start of each epoch."""
        if not self.curriculum_schedule:
            return

        # Allow disabling curriculum during task fine-tuning phases
        if getattr(self, "disable_curriculum", False):
            return

        epoch = self.current_epoch
        params = self._get_curriculum_params_for_epoch(epoch)

        if not params:
            return

        # Track if anything changed for logging
        changes = []

        if "jump_k" in params and params["jump_k"] != self.jump_k:
            old_val = self.jump_k
            self.jump_k = params["jump_k"]
            changes.append(f"jump_k: {old_val} → {self.jump_k}")

        if "loss_weight_teacher" in params and params["loss_weight_teacher"] != self.loss_weight_teacher:
            old_val = self.loss_weight_teacher
            self.loss_weight_teacher = params["loss_weight_teacher"]
            changes.append(f"loss_weight_teacher: {old_val} → {self.loss_weight_teacher}")

        if "loss_weight_jump" in params and params["loss_weight_jump"] != self.loss_weight_jump:
            old_val = self.loss_weight_jump
            self.loss_weight_jump = params["loss_weight_jump"]
            changes.append(f"loss_weight_jump: {old_val} → {self.loss_weight_jump}")

        # Log changes
        if changes:
            self._log.info(f"[Curriculum] Epoch {epoch}: {', '.join(changes)}")

        # Log current curriculum state for tracking
        self.log("curriculum/jump_k", float(self.jump_k), sync_dist=True)
        self.log("curriculum/loss_weight_teacher", self.loss_weight_teacher, sync_dist=True)
        self.log("curriculum/loss_weight_jump", self.loss_weight_jump, sync_dist=True)

    def forward(
        self,
        features: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through the predictor.

        Args:
            features: Encoded video features [B, T*N, D] where N = H*W patches
            actions: Action sequences [B, T, action_dim]
            states: State sequences [B, T, action_dim]
            extrinsics: Optional extrinsic parameters [B, T, action_dim-1]

        Returns:
            Predicted features [B, T*N, D]
        """
        return self.model(features, actions, states, extrinsics)

    def _step_predictor(
        self,
        z: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
        target_timestep: int | None = None,
    ) -> Tensor:
        """Single predictor step with optional layer normalization.

        Args:
            z: Input features [B, T*N, D]
            actions: Actions [B, T, action_dim]
            states: States [B, T, action_dim]
            extrinsics: Optional extrinsics [B, T, action_dim-1]
            target_timestep: Optional target frame index for jump prediction.
                When set, RoPE positions are overridden to encode this target.

        Returns:
            Predicted features [B, T*N, D], optionally normalized
        """
        z_pred = self.model(z, actions, states, extrinsics, target_timestep=target_timestep)
        if self.normalize_reps:
            z_pred = F.layer_norm(z_pred, (z_pred.size(-1),), eps=DEFAULT_LAYER_NORM_EPS)
        return z_pred

    # Loss computation methods are inherited from ACPredictorLossMixin:
    # - _compute_loss
    # - _compute_teacher_forcing_loss
    # - _compute_jump_loss
    # - _compute_jump_loss_per_timestep
    # - _shared_step

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Validation step."""
        return self._shared_step(batch, "val")

    def _log_test_metrics(
        self,
        loss_teacher: Tensor,
        loss_jump: Tensor,
        loss: Tensor,
        per_timestep_losses: list[Tensor],
    ) -> None:
        """Log test metrics.

        Args:
            loss_teacher: Teacher-forcing loss
            loss_jump: Jump prediction loss
            loss: Combined loss
            per_timestep_losses: Per-timestep loss breakdown
        """
        # Log aggregate metrics
        self.log("test/loss_teacher", loss_teacher, prog_bar=True, sync_dist=True, batch_size=1)
        self.log("test/loss_jump", loss_jump, prog_bar=True, sync_dist=True, batch_size=1)
        self.log("test/loss", loss, prog_bar=True, sync_dist=True, batch_size=1)

        # Log per-timestep losses (target frames in the jump range)
        T = self.num_timesteps
        k = min(self.jump_k, T)
        tau_min = T + 1 - k
        for step, step_loss in enumerate(per_timestep_losses):
            target_frame = tau_min + step
            self.log(f"test/loss_jump_tau_{target_frame}", step_loss.mean(), sync_dist=True, batch_size=1)

    def _store_test_results(
        self,
        clip_names: list[str],
        batch_idx: int,
        B: int,
        per_sample_losses: list[list[Tensor]],
        per_timestep_losses: list[Tensor],
        loss_teacher: Tensor,
    ) -> None:
        """Store test results for later aggregation.

        Args:
            clip_names: List of clip identifiers
            batch_idx: Batch index
            B: Batch size
            per_sample_losses: Per-sample jump prediction losses
            per_timestep_losses: Per-timestep losses
            loss_teacher: Teacher-forcing loss
        """
        T = self.num_timesteps
        k = min(self.jump_k, T)
        tau_min = T + 1 - k
        per_sample_jump_losses = per_sample_losses[0]  # [B]
        for i in range(B):
            clip_result: TestResultsDict = {
                "clip_name": clip_names[i] if i < len(clip_names) else f"unknown_{batch_idx}_{i}",
                "loss_jump": per_sample_jump_losses[i].item(),
                "loss_teacher": loss_teacher.item(),  # Note: this is batch-level for teacher
                "per_timestep_losses": {
                    f"tau_{tau_min + s}": per_timestep_losses[s][i].item()
                    for s in range(len(per_timestep_losses))
                },
            }
            self._test_results.append(clip_result)

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Test step with detailed per-clip and per-timestep analysis.

        Computes jump losses with per-timestep breakdown and stores results
        for aggregation in on_test_epoch_end.

        If TTA is enabled (tta_enabled=True), performs online adaptation using
        the look-back scheme before computing final predictions.
        """
        features = batch["features"]
        actions = batch["actions"]
        states = batch["states"]
        extrinsics = batch.get("extrinsics", None)
        clip_names = batch.get("clip_names", [f"clip_{batch_idx}_{i}" for i in range(features.shape[0])])

        # Reshape features if needed: [B, (T+1)*N, D] -> [B, T+1, N, D]
        if features.dim() == 3:
            B, total_tokens, D = features.shape
            T_plus_1 = total_tokens // self.patches_per_frame
            features = features.reshape(B, T_plus_1, self.patches_per_frame, D)
        else:
            B = features.shape[0]

        # TTA-enabled test step: process each sample in batch individually for per-clip adaptation
        if self.tta_enabled:
            return self._test_step_with_tta(
                features, actions, states, extrinsics, clip_names, batch_idx
            )

        # Standard test step (no TTA)
        # Compute teacher forcing loss (for comparison)
        loss_teacher = self._compute_teacher_forcing_loss(features, actions, states, extrinsics)

        # Compute jump prediction loss with per-timestep breakdown
        loss_jump, per_timestep_losses, per_sample_losses = self._compute_jump_loss_per_timestep(
            features, actions, states, extrinsics
        )

        # Combined loss
        loss = self.loss_weight_teacher * loss_teacher + self.loss_weight_jump * loss_jump

        self._log_test_metrics(loss_teacher, loss_jump, loss, per_timestep_losses)
        self._store_test_results(
            clip_names, batch_idx, B, per_sample_losses, per_timestep_losses, loss_teacher
        )

        return loss

    def _process_single_sample_tta(
        self,
        sample_features: Tensor,
        sample_actions: Tensor,
        sample_states: Tensor,
        sample_extrinsics: Tensor | None,
        clip_name: str,
        batch_idx: int,
        sample_idx: int,
        T_pred: int,
        C: int,
        N: int,
    ) -> tuple[float, dict[str, float], dict[str, Any]]:
        """Process a single sample with TTA.

        Args:
            sample_features: Features for single sample [1, T+1, N, D]
            sample_actions: Actions for single sample [1, T, action_dim]
            sample_states: States for single sample [1, T, action_dim]
            sample_extrinsics: Optional extrinsics [1, T, action_dim-1]
            clip_name: Clip identifier
            batch_idx: Batch index for logging
            sample_idx: Sample index within batch
            T_pred: Number of target frames (jump_k)
            C: Not used (kept for interface compat)
            N: Patches per frame

        Returns:
            Tuple of (sample_loss, per_step_losses, tta_stats)
        """
        # Process with TTA (from TTAMixin)
        if self.tta_mode == "jump":
            predictions, targets, tta_stats = self._tta_process_clip_jump(
                sample_features, sample_actions, sample_states, sample_extrinsics,
                num_adaptation_steps=self.tta_num_adaptation_steps,
            )
        else:  # "sequential" mode
            predictions, targets, tta_stats = self._tta_process_clip_sequential(
                sample_features, sample_actions, sample_states, sample_extrinsics
            )

        # Compute loss for this sample
        sample_loss = self._compute_loss(predictions, targets)

        # Compute per-timestep losses for analysis
        per_step_losses: dict[str, float] = {}
        for step in range(T_pred):
            step_pred = predictions[:, step*N:(step+1)*N, :]
            step_target = targets[:, step*N:(step+1)*N, :]
            step_loss = self._compute_loss(step_pred, step_target)
            per_step_losses[f"step_{C + step}"] = step_loss.item()

        return sample_loss.item(), per_step_losses, tta_stats

    def _log_tta_sample_metrics(
        self,
        sample_loss: float,
        tta_stats: dict[str, Any],
        batch_idx: int,
        sample_idx: int,
    ) -> None:
        """Log TTA metrics for a single sample.

        Args:
            sample_loss: Loss for the sample
            tta_stats: TTA statistics dictionary
            batch_idx: Batch index
            sample_idx: Sample index within batch
        """
        # Core per-clip metrics (logged every clip)
        self.log("test/tta_loss_jump", sample_loss, on_step=True, on_epoch=False, sync_dist=True, batch_size=1)
        self.log("test/tta_improvement", tta_stats.get("improvement", 0.0), on_step=True, on_epoch=False, sync_dist=True, batch_size=1)

        # Track cumulative statistics for adaptation trend
        if not hasattr(self, '_tta_cumulative_losses'):
            self._tta_cumulative_losses: list[float] = []
        self._tta_cumulative_losses.append(sample_loss)

        # Log rolling average (last 50 clips) to show adaptation trend
        window_size = min(ROLLING_AVG_WINDOW_SIZE, len(self._tta_cumulative_losses))
        rolling_avg = sum(self._tta_cumulative_losses[-window_size:]) / window_size
        self.log("test/tta_loss_rolling_50", rolling_avg, on_step=True, on_epoch=False, sync_dist=True, batch_size=1)

        # Log gradient diagnostics if available
        if "grad_norm_before_clip" in tta_stats:
            self.log("test/tta_grad_norm_before", tta_stats["grad_norm_before_clip"], on_step=True, on_epoch=False, sync_dist=True, batch_size=1)
        if "grad_norm_after_clip" in tta_stats:
            self.log("test/tta_grad_norm_after", tta_stats["grad_norm_after_clip"], on_step=True, on_epoch=False, sync_dist=True, batch_size=1)
        if "param_delta_norm" in tta_stats:
            self.log("test/tta_param_delta", tta_stats["param_delta_norm"], on_step=True, on_epoch=False, sync_dist=True, batch_size=1)

    def _test_step_with_tta(
        self,
        features: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None,
        clip_names: list[str],
        batch_idx: int,
    ) -> Tensor:
        """Test step with Test Time Adaptation.

        Processes each sample in the batch individually to allow per-clip TTA.
        Uses the look-back scheme from TTAMixin to adapt LayerNorm parameters.

        Args:
            features: [B, T+1, N, D] - Full sequence features
            actions: [B, T, action_dim] - Actions
            states: [B, T, action_dim] - States
            extrinsics: [B, T, action_dim-1] - Optional
            clip_names: List of clip identifiers
            batch_idx: Batch index for logging

        Returns:
            Combined loss tensor
        """
        B, T_plus_1, N, D = features.shape
        T = T_plus_1 - 1
        k = min(self.jump_k, T)
        T_pred = k  # Number of target frames in jump range

        all_losses: list[float] = []
        all_per_timestep_losses: list[list[float]] = [[] for _ in range(T_pred)]

        # Process each sample individually for per-clip TTA
        for i in range(B):
            # Extract single sample
            sample_features = features[i:i+1]  # [1, T+1, N, D]
            sample_actions = actions[i:i+1]
            sample_states = states[i:i+1]
            sample_extrinsics = extrinsics[i:i+1] if extrinsics is not None else None
            clip_name = clip_names[i] if i < len(clip_names) else f"unknown_{batch_idx}_{i}"

            # Process single sample with TTA
            sample_loss, per_step_losses, tta_stats = self._process_single_sample_tta(
                sample_features, sample_actions, sample_states, sample_extrinsics,
                clip_name, batch_idx, i, T_pred, 0, N,
            )

            all_losses.append(sample_loss)

            # Collect per-timestep losses
            for step_idx, (step_key, step_loss) in enumerate(per_step_losses.items()):
                if step_idx < T_pred:
                    all_per_timestep_losses[step_idx].append(step_loss)

            # Store result
            clip_result: TestResultsDict = {
                "clip_name": clip_name,
                "loss_jump": sample_loss,
                "loss_teacher": 0.0,  # Not computed in TTA mode
                "per_timestep_losses": per_step_losses,
                "tta_stats": tta_stats,
            }
            self._test_results.append(clip_result)

            # Log TTA metrics per-clip
            self._log_tta_sample_metrics(sample_loss, tta_stats, batch_idx, i)

        # Compute batch-level metrics
        loss_jump = sum(all_losses) / len(all_losses)

        # Log metrics
        self.log("test/loss_jump", loss_jump, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("test/loss", loss_jump, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)

        # Log per-timestep losses
        tau_min = T + 1 - k
        for step in range(T_pred):
            if all_per_timestep_losses[step]:
                step_mean = sum(all_per_timestep_losses[step]) / len(all_per_timestep_losses[step])
                self.log(f"test/loss_jump_tau_{tau_min + step}", step_mean, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)

        return torch.tensor(loss_jump, device=features.device)

    def on_test_epoch_start(self) -> None:
        """Clear test results and configure TTA at the start of test epoch."""
        self._test_results = []

        # Reset cumulative tracking for new test epoch
        self._tta_cumulative_losses: list[float] = []

        # Configure TTA if enabled
        if self.tta_enabled:
            self._tta_configure_params()
            self._tta_original_ln_state = self._tta_save_ln_state()
            self._tta_all_clip_stats = []

            # Initialize optimizer once at epoch start
            # (will be reset per-clip if tta_reset_per_clip=True)
            self._tta_optimizer = self._tta_create_optimizer()
            self._tta_clip_stats = {
                "adaptation_losses": [],
                "num_adaptations": 0,
            }

            self._log.info(
                f"[TTA] Test epoch started with TTA enabled. "
                f"LR={self.tta_lr}, grad_clip={self.tta_grad_clip}, "
                f"reset_per_clip={self.tta_reset_per_clip}"
            )

    def _compute_test_statistics(self) -> dict[str, Any]:
        """Compute aggregate test statistics.

        Returns:
            Dictionary containing mean, median, std, min, max, and per-timestep stats
        """
        num_clips = len(self._test_results)
        jump_losses = [r["loss_jump"] for r in self._test_results]

        mean_loss = sum(jump_losses) / num_clips
        sorted_losses = sorted(jump_losses)
        median_loss = sorted_losses[num_clips // 2]
        min_loss = min(jump_losses)
        max_loss = max(jump_losses)
        std_loss = (sum((x - mean_loss) ** 2 for x in jump_losses) / num_clips) ** 0.5

        # Compute per-timestep statistics
        num_timesteps = len(self._test_results[0]["per_timestep_losses"])
        per_timestep_stats: dict[str, dict[str, float]] = {}
        for step_key in self._test_results[0]["per_timestep_losses"].keys():
            step_losses = [r["per_timestep_losses"][step_key] for r in self._test_results]
            per_timestep_stats[step_key] = {
                "mean": sum(step_losses) / num_clips,
                "min": min(step_losses),
                "max": max(step_losses),
            }

        # Find worst-performing and best-performing clips
        worst_clips = sorted(self._test_results, key=lambda x: x["loss_jump"], reverse=True)[:NUM_WORST_CLIPS]
        best_clips = sorted(self._test_results, key=lambda x: x["loss_jump"])[:NUM_BEST_CLIPS]

        return {
            "num_clips": num_clips,
            "mean_loss": mean_loss,
            "median_loss": median_loss,
            "std_loss": std_loss,
            "min_loss": min_loss,
            "max_loss": max_loss,
            "per_timestep_stats": per_timestep_stats,
            "worst_clips": worst_clips,
            "best_clips": best_clips,
        }

    def _print_test_summary(self, stats: dict[str, Any]) -> None:
        """Print formatted test results summary to console.

        Args:
            stats: Statistics dictionary from _compute_test_statistics
        """
        num_clips = stats["num_clips"]
        mean_loss = stats["mean_loss"]
        median_loss = stats["median_loss"]
        std_loss = stats["std_loss"]
        min_loss = stats["min_loss"]
        max_loss = stats["max_loss"]
        per_timestep_stats = stats["per_timestep_stats"]
        worst_clips = stats["worst_clips"]
        best_clips = stats["best_clips"]

        print("\n" + "=" * 70)
        print("                    TEST RESULTS SUMMARY")
        print("=" * 70)
        print(f"\n📊 AGGREGATE STATISTICS (over {num_clips} clips)")
        print("-" * 50)
        print(f"  Jump Prediction Loss ({self.loss_type}):")
        print(f"    Mean:   {mean_loss:.6f}")
        print(f"    Median: {median_loss:.6f}")
        print(f"    Std:    {std_loss:.6f}")
        print(f"    Min:    {min_loss:.6f}")
        print(f"    Max:    {max_loss:.6f}")

        print(f"\n📈 PER-TARGET-FRAME BREAKDOWN (jump_k={self.jump_k})")
        print("-" * 50)
        print(f"  {'Target τ':<10} {'Mean Loss':<12} {'Min':<12} {'Max':<12}")
        print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
        for step_key, step_stats in per_timestep_stats.items():
            frame_label = step_key.replace("tau_", "z_")
            print(f"  {frame_label:<10} {step_stats['mean']:<12.6f} {step_stats['min']:<12.6f} {step_stats['max']:<12.6f}")

        print(f"\n🔴 WORST-PERFORMING CLIPS (top {NUM_WORST_CLIPS})")
        print("-" * 50)
        for i, clip in enumerate(worst_clips, 1):
            print(f"  {i:2d}. {clip['clip_name']:<20} loss={clip['loss_jump']:.6f}")

        print(f"\n🟢 BEST-PERFORMING CLIPS (top {NUM_BEST_CLIPS})")
        print("-" * 50)
        for i, clip in enumerate(best_clips, 1):
            print(f"  {i:2d}. {clip['clip_name']:<20} loss={clip['loss_jump']:.6f}")

        # Print TTA statistics if enabled
        if self.tta_enabled:
            tta_epoch_stats = self._tta_get_epoch_stats()
            print(f"\n🔄 TEST TIME ADAPTATION (TTA) STATISTICS")
            print("-" * 50)
            print(f"  TTA Enabled: Yes")
            print(f"  TTA Mode: {self.tta_mode}")
            print(f"  Adaptation Steps: {self.tta_num_adaptation_steps}")
            print(f"  Learning Rate: {self.tta_lr}")
            print(f"  Gradient Clip: {self.tta_grad_clip}")
            print(f"  Reset Per Clip: {self.tta_reset_per_clip}")
            print(f"  Total Clips: {tta_epoch_stats.get('total_clips', 0)}")
            print(f"  Total Adaptations: {tta_epoch_stats.get('total_adaptations', 0)}")
            print(f"  Mean Pre-Adapt Loss: {tta_epoch_stats.get('mean_pre_adapt_loss', 0):.6f}")
            print(f"  Mean Post-Adapt Loss: {tta_epoch_stats.get('mean_post_adapt_loss', 0):.6f}")
            print(f"  Mean Improvement: {tta_epoch_stats.get('mean_improvement', 0):.6f}")

            # Log TTA epoch-level statistics to wandb
            self.log("test/tta_epoch_mean_pre_adapt_loss", tta_epoch_stats.get("mean_pre_adapt_loss", 0.0), sync_dist=True)
            self.log("test/tta_epoch_mean_post_adapt_loss", tta_epoch_stats.get("mean_post_adapt_loss", 0.0), sync_dist=True)
            self.log("test/tta_epoch_mean_improvement", tta_epoch_stats.get("mean_improvement", 0.0), sync_dist=True)
            self.log("test/tta_epoch_total_adaptations", float(tta_epoch_stats.get("total_adaptations", 0)), sync_dist=True)

        print("\n" + "=" * 70)

    def _export_test_results(self, stats: dict[str, Any], output_dir: Path) -> None:
        """Export test results to JSON file.

        Args:
            stats: Statistics dictionary from _compute_test_statistics
            output_dir: Directory to save results file
        """
        export_data = {
            "summary": {
                "num_clips": stats["num_clips"],
                "jump_k": self.jump_k,
                "tta_enabled": self.tta_enabled,
                "jump_loss": {
                    "mean": stats["mean_loss"],
                    "median": stats["median_loss"],
                    "std": stats["std_loss"],
                    "min": stats["min_loss"],
                    "max": stats["max_loss"],
                },
                "per_timestep": stats["per_timestep_stats"],
            },
            "worst_clips": stats["worst_clips"],
            "best_clips": stats["best_clips"],
            "all_clips": self._test_results,
        }

        # Add TTA statistics if enabled
        if self.tta_enabled:
            export_data["summary"]["tta_config"] = {
                "lr": self.tta_lr,
                "grad_clip": self.tta_grad_clip,
                "reset_per_clip": self.tta_reset_per_clip,
                "adaptation_horizon": self.tta_adaptation_horizon,
            }
            export_data["summary"]["tta_stats"] = self._tta_get_epoch_stats()

        try:
            results_file = output_dir / "test_results.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, "w") as f:
                json.dump(export_data, f, indent=JSON_INDENT)
            print(f"\n💾 Detailed results exported to: {results_file}")
        except OSError as e:
            self._log.warning(f"Failed to export results to JSON: {e}")

    def on_test_epoch_end(self) -> None:
        """Aggregate test results and output summary.

        Prints summary statistics and optionally exports detailed results to JSON.
        """
        if not self._test_results:
            self._log.warning("No test results to aggregate")
            return

        # Compute aggregate statistics
        stats = self._compute_test_statistics()

        # Print summary to console
        self._print_test_summary(stats)

        # Log final metrics
        self.log("test/final_mean_loss_jump", stats["mean_loss"], sync_dist=True)
        self.log("test/final_median_loss_jump", stats["median_loss"], sync_dist=True)

        # Export to JSON
        output_dir = Path(self.trainer.log_dir) if self.trainer and self.trainer.log_dir else Path(".")
        self._export_test_results(stats, output_dir)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler.

        Supports two scheduling modes:
        1. Epoch-based (default): Cosine annealing with linear warmup
        2. Iteration-based (V-JEPA2 paper): Warmup → Constant → Decay
           - Linear warmup from warmup_start_lr to learning_rate
           - Constant phase at peak learning_rate
           - Linear decay from learning_rate to 0

        Returns:
            Dictionary with optimizer and optional lr_scheduler config
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )

        if self.use_iteration_scheduler:
            return self._configure_iteration_scheduler(optimizer)
        else:
            return self._configure_epoch_scheduler(optimizer)

    def _configure_epoch_scheduler(self, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        """Configure epoch-based learning rate scheduler.

        Args:
            optimizer: The optimizer to schedule

        Returns:
            Dictionary with optimizer and scheduler config
        """
        def lr_lambda_epoch(epoch: int) -> float:
            if epoch < self.warmup_epochs:
                return epoch / self.warmup_epochs
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_epoch)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def _configure_iteration_scheduler(self, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        """Configure iteration-based learning rate scheduler (V-JEPA2 paper).

        Args:
            optimizer: The optimizer to schedule

        Returns:
            Dictionary with optimizer and scheduler config
        """
        # Compute total training iterations from trainer
        if self.trainer is not None and self.trainer.estimated_stepping_batches:
            total_iters = int(self.trainer.estimated_stepping_batches)
        else:
            # Fallback: estimate from dataloader if trainer not fully set up
            warnings.warn(
                "Could not get total iterations from trainer. "
                "LR schedule may not work correctly.",
                UserWarning,
                stacklevel=2,
            )
            total_iters = 10000  # Fallback default

        # Convert percentages to iteration counts
        warmup_iters = int(self.warmup_pct * total_iters)
        constant_iters = int(self.constant_pct * total_iters)
        decay_iters = int(self.decay_pct * total_iters)

        # Ensure we don't exceed total due to rounding
        computed_total = warmup_iters + constant_iters + decay_iters
        if computed_total != total_iters:
            constant_iters += (total_iters - computed_total)

        self._log.info(
            f"[LR Schedule] Total iters: {total_iters}, "
            f"warmup: {warmup_iters} ({self.warmup_pct*100:.1f}%), "
            f"constant: {constant_iters} ({self.constant_pct*100:.1f}%), "
            f"decay: {decay_iters} ({self.decay_pct*100:.1f}%)"
        )

        warmup_end = warmup_iters
        constant_end = warmup_end + constant_iters

        # Scale factor for warmup: start_lr / peak_lr
        warmup_start_factor = self.warmup_start_lr / self.learning_rate

        def lr_lambda_iter(step: int) -> float:
            if step < warmup_end:
                # Linear warmup from warmup_start_lr to learning_rate
                progress = step / warmup_iters
                return warmup_start_factor + (1.0 - warmup_start_factor) * progress
            elif step < constant_end:
                # Constant at peak learning_rate
                return 1.0
            elif step < total_iters:
                # Linear decay from learning_rate to 0
                progress = (step - constant_end) / decay_iters
                return 1.0 - progress
            else:
                # After total_iters, stay at 0
                return 0.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_iter)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Iteration-based
                "frequency": 1,
            },
        }
