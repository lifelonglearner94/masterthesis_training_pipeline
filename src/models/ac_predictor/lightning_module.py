# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Lightning Module wrapper for AC Predictor.

Implements teacher-forcing and rollout losses for training the action-conditioned
vision transformer predictor, matching the V-JEPA2 paper implementation.

Loss formulations (with configurable loss_exp, default=1.0 for L1):
- Teacher-Forcing: L_tf = mean(|P_φ(z_{0:T-1}, a, s) - z_{1:T}|^exp) / exp
  Single forward pass with full context, predicts all frames in parallel.
- Rollout: L_ar = mean(|z_ar - z_{1:T}|^exp) / exp
  Autoregressive rollout seeded with ground-truth z_0 + first TF prediction.
- Combined: L = w_tf * L_tf + w_ar * L_ar

Key features matching original paper:
- Optional layer normalization after each predictor step (normalize_reps)
- Configurable loss exponent (loss_exp=1.0 for L1, loss_exp=2.0 for L2)
- Cumulative action/state context for autoregressive rollout
- TF-seeded initialization for rollout stability
"""

from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.ac_predictor.ac_predictor import vit_ac_predictor


# Type alias for test results
TestResultsDict = dict[str, Any]


class ACPredictorModule(L.LightningModule):
    """PyTorch Lightning module for Action-Conditioned Vision Transformer Predictor.

    This module wraps the VisionTransformerPredictorAC and implements:
    - Teacher-forcing loss: predicts next latent from context, averaged over T steps
    - Rollout loss: enforces consistency over multiple recurrent prediction steps

    Expected batch format:
        - features: [B, T+1, N, D] - Pre-computed V-JEPA2 encoder features for T+1 timesteps
        - actions: [B, T, action_dim] - 7D end-effector state changes
        - states: [B, T, action_dim] - 7D end-effector states

    Where:
        - T is the number of encoded timesteps (e.g., 8 for 16 original frames with tubelet_size=2)
        - N is the number of patches per frame (H*W)
        - D is the embedding dimension

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
        T_rollout: int = 2,
        context_frames: int = 1,  # Number of ground-truth context frames for rollout
        loss_weight_teacher: float = 1.0,
        loss_weight_rollout: float = 1.0,
        normalize_reps: bool = True,
        loss_exp: float = 1.0,
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
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

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
        self.T_rollout = T_rollout
        self.context_frames = context_frames

        # Validate T_teacher and T_rollout against available timesteps
        max_prediction_steps = num_timesteps - 1  # Need at least 1 context + 1 target
        if T_teacher > max_prediction_steps:
            import warnings
            warnings.warn(
                f"T_teacher ({T_teacher}) exceeds available prediction steps ({max_prediction_steps}) "
                f"for num_timesteps={num_timesteps}. Will be clamped to {max_prediction_steps} at runtime.",
                UserWarning
            )
        if T_rollout > max_prediction_steps:
            import warnings
            warnings.warn(
                f"T_rollout ({T_rollout}) exceeds available prediction steps ({max_prediction_steps}) "
                f"for num_timesteps={num_timesteps}. Will be clamped to {max_prediction_steps} at runtime.",
                UserWarning
            )
        self.loss_weight_teacher = loss_weight_teacher
        self.loss_weight_rollout = loss_weight_rollout
        self.normalize_reps = normalize_reps

        # Validate loss_exp to prevent division by zero
        if loss_exp <= 0:
            raise ValueError(
                f"loss_exp must be positive (got {loss_exp}). "
                "Use loss_exp=1.0 for L1 loss, loss_exp=2.0 for L2 loss."
            )
        self.loss_exp = loss_exp

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

        # Curriculum learning: schedule for dynamic T_rollout and loss_weight_teacher
        # Format: [{"epoch": 0, "T_rollout": 2, "loss_weight_teacher": 1.0}, ...]
        self.curriculum_schedule = curriculum_schedule
        if curriculum_schedule:
            self._validate_curriculum_schedule(curriculum_schedule)

        # Storage for test results (populated during test_step, aggregated in on_test_epoch_end)
        self._test_results: list[TestResultsDict] = []

    def _validate_curriculum_schedule(self, schedule: list[dict]) -> None:
        """Validate curriculum schedule format and values.

        Args:
            schedule: List of dicts with keys: epoch, T_rollout (optional),
                     loss_weight_teacher (optional), loss_weight_rollout (optional)

        Raises:
            ValueError: If schedule is invalid
        """
        # Support both Python lists and OmegaConf ListConfig
        from collections.abc import Sequence
        if not isinstance(schedule, Sequence) or isinstance(schedule, str) or len(schedule) == 0:
            raise ValueError("curriculum_schedule must be a non-empty list")

        max_prediction_steps = self.num_timesteps - 1

        # Support both Python dicts and OmegaConf DictConfig
        from collections.abc import Mapping
        for i, phase in enumerate(schedule):
            if not isinstance(phase, Mapping):
                raise ValueError(f"Phase {i} must be a dict, got {type(phase)}")
            if "epoch" not in phase:
                raise ValueError(f"Phase {i} must have 'epoch' key")
            if not isinstance(phase["epoch"], int) or phase["epoch"] < 0:
                raise ValueError(f"Phase {i} 'epoch' must be a non-negative integer")

            # Validate T_rollout if present
            if "T_rollout" in phase:
                if phase["T_rollout"] > max_prediction_steps:
                    raise ValueError(
                        f"Phase {i}: T_rollout ({phase['T_rollout']}) exceeds "
                        f"max prediction steps ({max_prediction_steps})"
                    )

        # Ensure phases are sorted by epoch
        epochs = [p["epoch"] for p in schedule]
        if epochs != sorted(epochs):
            raise ValueError("curriculum_schedule phases must be sorted by epoch")

    def _get_curriculum_params_for_epoch(self, epoch: int) -> dict:
        """Get curriculum parameters for the given epoch.

        Finds the most recent phase whose epoch is <= current epoch.

        Args:
            epoch: Current training epoch

        Returns:
            Dict with T_rollout, loss_weight_teacher, loss_weight_rollout
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

        epoch = self.current_epoch
        params = self._get_curriculum_params_for_epoch(epoch)

        if not params:
            return

        # Track if anything changed for logging
        changes = []

        if "T_rollout" in params and params["T_rollout"] != self.T_rollout:
            old_val = self.T_rollout
            self.T_rollout = params["T_rollout"]
            changes.append(f"T_rollout: {old_val} → {self.T_rollout}")

        if "loss_weight_teacher" in params and params["loss_weight_teacher"] != self.loss_weight_teacher:
            old_val = self.loss_weight_teacher
            self.loss_weight_teacher = params["loss_weight_teacher"]
            changes.append(f"loss_weight_teacher: {old_val} → {self.loss_weight_teacher}")

        if "loss_weight_rollout" in params and params["loss_weight_rollout"] != self.loss_weight_rollout:
            old_val = self.loss_weight_rollout
            self.loss_weight_rollout = params["loss_weight_rollout"]
            changes.append(f"loss_weight_rollout: {old_val} → {self.loss_weight_rollout}")

        # Log changes
        if changes:
            import logging
            log = logging.getLogger(__name__)
            log.info(f"[Curriculum] Epoch {epoch}: {', '.join(changes)}")

        # Log current curriculum state for tracking
        self.log("curriculum/T_rollout", float(self.T_rollout), sync_dist=True)
        self.log("curriculum/loss_weight_teacher", self.loss_weight_teacher, sync_dist=True)
        self.log("curriculum/loss_weight_rollout", self.loss_weight_rollout, sync_dist=True)

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
    ) -> Tensor:
        """Single predictor step with optional layer normalization.

        Args:
            z: Input features [B, T*N, D]
            actions: Actions [B, T, action_dim]
            states: States [B, T, action_dim]
            extrinsics: Optional extrinsics [B, T, action_dim-1]

        Returns:
            Predicted features [B, T*N, D], optionally normalized
        """
        z_pred = self.model(z, actions, states, extrinsics)
        if self.normalize_reps:
            z_pred = F.layer_norm(z_pred, (z_pred.size(-1),), eps=1e-6)
        return z_pred

    def _compute_loss(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute loss with configurable exponent.

        loss = mean(|pred - target|^loss_exp) / loss_exp

        When loss_exp=1.0, this is equivalent to L1 loss.
        When loss_exp=2.0, this is equivalent to 0.5 * L2 loss.

        Args:
            pred: Predicted features
            target: Target features

        Returns:
            Scalar loss tensor
        """
        import logging
        log = logging.getLogger(__name__)

        # Compute element-wise absolute difference
        diff = torch.abs(pred - target)
        log.debug(f"    [LOSS] pred shape: {pred.shape}, target shape: {target.shape}")
        log.debug(f"    [LOSS] |pred - target|: min={diff.min().item():.6f}, max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")

        # Apply exponent
        diff_exp = diff ** self.loss_exp
        log.debug(f"    [LOSS] |diff|^{self.loss_exp}: min={diff_exp.min().item():.6f}, max={diff_exp.max().item():.6f}, mean={diff_exp.mean().item():.6f}")

        # Compute final loss
        loss = torch.mean(diff_exp) / self.loss_exp
        log.debug(f"    [LOSS] Final loss: {loss.item():.8f}")

        return loss

    def _compute_teacher_forcing_loss(
        self,
        features: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
    ) -> Tensor:
        """Compute teacher-forcing loss (single forward pass, matching V-JEPA2 paper).

        The predictor receives all context frames (0 to T-1) and predicts frames 1 to T
        in a single forward pass. Loss compares predictions against ground-truth targets.

        This matches the original paper's implementation:
            z_tf = predictor(z[:, :-tokens_per_frame], actions, states, extrinsics)
            loss = mean(|z_tf - h[:, tokens_per_frame:]|^loss_exp) / loss_exp

        Args:
            features: [B, T+1, N, D] - Features for T+1 frames (T inputs + 1 target)
            actions: [B, T, action_dim] - Actions for T steps
            states: [B, T, action_dim] - States for T steps
            extrinsics: [B, T, action_dim-1] - Optional extrinsics

        Returns:
            Scalar loss tensor
        """
        import logging
        log = logging.getLogger(__name__)

        B, T_plus_1, N, D = features.shape
        log.debug(f"  [TEACHER FORCING] Input features: B={B}, T+1={T_plus_1}, N={N}, D={D}")
        T = min(T_plus_1 - 1, self.T_teacher)
        log.debug(f"  [TEACHER FORCING] Using T={T} timesteps (T_teacher={self.T_teacher})")

        # Context: all frames except the last one (frames 0 to T-1)
        # Shape: [B, T*N, D]
        context_features = features[:, :T, :, :].reshape(B, T * N, D)
        log.debug(f"  [TEACHER FORCING] Context features reshaped to: {context_features.shape}")
        log.debug(f"  [TEACHER FORCING] Context stats: min={context_features.min().item():.6f}, max={context_features.max().item():.6f}, mean={context_features.mean().item():.6f}")

        # Apply layer norm to input if enabled (matching paper)
        if self.normalize_reps:
            context_features = F.layer_norm(context_features, (context_features.size(-1),), eps=1e-6)
            log.debug(f"  [TEACHER FORCING] After LayerNorm: min={context_features.min().item():.6f}, max={context_features.max().item():.6f}, mean={context_features.mean().item():.6f}")

        # Actions/states for T steps
        context_actions = actions[:, :T, :]
        context_states = states[:, :T, :]
        context_extrinsics = extrinsics[:, :T, :] if extrinsics is not None else None
        log.debug(f"  [TEACHER FORCING] Actions shape: {context_actions.shape}, States shape: {context_states.shape}")

        # Single forward pass - predicts all T frames
        log.debug(f"  [TEACHER FORCING] >>> Calling predictor forward pass >>>")
        z_tf = self._step_predictor(
            context_features, context_actions, context_states, context_extrinsics
        )
        log.debug(f"  [TEACHER FORCING] <<< Predictor output: shape={z_tf.shape}, min={z_tf.min().item():.6f}, max={z_tf.max().item():.6f}, mean={z_tf.mean().item():.6f}")

        # Target: frames 1 to T (shifted by one from context)
        # Shape: [B, T*N, D]
        target = features[:, 1 : T + 1, :, :].reshape(B, T * N, D)
        log.debug(f"  [TEACHER FORCING] Target features: shape={target.shape}")
        if self.normalize_reps:
            target = F.layer_norm(target, (target.size(-1),), eps=1e-6)
            log.debug(f"  [TEACHER FORCING] Target after LayerNorm: min={target.min().item():.6f}, max={target.max().item():.6f}")

        log.debug(f"  [TEACHER FORCING] Computing loss between prediction and target...")
        return self._compute_loss(z_tf, target)

    def _compute_rollout_loss(
        self,
        features: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
    ) -> Tensor:
        """Compute rollout loss with fixed context frames.

        Autoregressive rollout seeded with `context_frames` ground-truth frames,
        then predicts `T_rollout` steps autoregressively.

        Example with context_frames=3, T_rollout=5:
            - Context: z_0, z_1, z_2 (ground-truth)
            - Predict: z_3, z_4, z_5, z_6, z_7 (autoregressive)
            - Loss computed on predictions vs ground-truth frames 3-7

        Args:
            features: [B, T+1, N, D] - Features for T+1 frames
            actions: [B, T, action_dim] - Actions for T steps
            states: [B, T, action_dim] - States for T steps
            extrinsics: [B, T, action_dim-1] - Optional extrinsics

        Returns:
            Scalar loss tensor
        """
        import logging
        log = logging.getLogger(__name__)

        B, T_plus_1, N, D = features.shape
        C = self.context_frames  # Number of ground-truth context frames
        T_pred = min(T_plus_1 - C, self.T_rollout)  # Number of steps to predict
        log.debug(f"  [ROLLOUT] Input: B={B}, T+1={T_plus_1}, N={N}, D={D}")
        log.debug(f"  [ROLLOUT] Context frames C={C}, Prediction steps T_pred={T_pred}")

        # Normalize ground-truth features if enabled
        h = features
        if self.normalize_reps:
            h = F.layer_norm(features.reshape(B, -1, D), (D,), eps=1e-6).reshape(B, T_plus_1, N, D)
            log.debug(f"  [ROLLOUT] Features normalized")

        # Step 1: Initialize with C ground-truth context frames
        # z_ar contains frames 0, 1, ..., C-1
        z_ar = h[:, :C, :, :].reshape(B, C * N, D)  # [B, C*N, D]
        log.debug(f"  [ROLLOUT] Initial context z_ar shape: {z_ar.shape}")

        # Step 2: Autoregressive rollout for T_pred steps
        # First prediction is frame C, using actions/states 0 to C-1
        for step in range(T_pred):
            log.debug(f"  [ROLLOUT] >>> Autoregressive step {step+1}/{T_pred} >>>")
            # Current prediction target is frame (C + step)
            # Use actions/states from 0 to (C + step - 1)
            num_action_steps = C + step
            ar_actions = actions[:, :num_action_steps, :]
            ar_states = states[:, :num_action_steps, :]
            ar_extrinsics = extrinsics[:, :num_action_steps, :] if extrinsics is not None else None
            log.debug(f"  [ROLLOUT] Step {step+1}: Using {num_action_steps} action steps, z_ar shape: {z_ar.shape}")

            # Predict next frame using full context
            z_nxt = self._step_predictor(z_ar, ar_actions, ar_states, ar_extrinsics)
            log.debug(f"  [ROLLOUT] Step {step+1}: Predictor output shape: {z_nxt.shape}")
            # Extract only the last frame prediction
            z_nxt = z_nxt[:, -N:, :]  # [B, N, D]
            log.debug(f"  [ROLLOUT] Step {step+1}: Extracted last frame: shape={z_nxt.shape}, mean={z_nxt.mean().item():.6f}")

            # Append to autoregressive context
            z_ar = torch.cat([z_ar, z_nxt], dim=1)  # [B, (C+step+1)*N, D]
            log.debug(f"  [ROLLOUT] Step {step+1}: Updated z_ar shape: {z_ar.shape}")

        # Step 3: Compare autoregressive predictions with targets
        # z_ar[:, C*N:] contains predictions for frames C, C+1, ..., C+T_pred-1
        # Target: h[:, C:C+T_pred] contains ground-truth for same frames
        z_ar_pred = z_ar[:, C * N:]  # [B, T_pred*N, D]
        target = h[:, C : C + T_pred, :, :].reshape(B, T_pred * N, D)  # [B, T_pred*N, D]
        log.debug(f"  [ROLLOUT] Final prediction shape: {z_ar_pred.shape}, target shape: {target.shape}")
        log.debug(f"  [ROLLOUT] Computing rollout loss...")

        return self._compute_loss(z_ar_pred, target)

    def _compute_rollout_loss_per_timestep(
        self,
        features: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """Compute rollout loss with per-timestep breakdown for detailed analysis.

        Same as _compute_rollout_loss, but returns per-timestep losses for analysis.

        Args:
            features: [B, T+1, N, D] - Features for T+1 frames
            actions: [B, T, action_dim] - Actions for T steps
            states: [B, T, action_dim] - States for T steps
            extrinsics: [B, T, action_dim-1] - Optional extrinsics

        Returns:
            Tuple of:
                - total_loss: Scalar loss tensor (average over all timesteps)
                - per_timestep_losses: List of [B] tensors, one per predicted timestep
                - per_sample_losses: List of [B] tensors with per-sample total rollout loss
        """
        import logging
        log = logging.getLogger(__name__)

        B, T_plus_1, N, D = features.shape
        C = self.context_frames  # Number of ground-truth context frames
        T_pred = min(T_plus_1 - C, self.T_rollout)  # Number of steps to predict

        # Normalize ground-truth features if enabled
        h = features
        if self.normalize_reps:
            h = F.layer_norm(features.reshape(B, -1, D), (D,), eps=1e-6).reshape(B, T_plus_1, N, D)

        # Initialize with C ground-truth context frames
        z_ar = h[:, :C, :, :].reshape(B, C * N, D)  # [B, C*N, D]

        # Store per-timestep predictions
        predictions_per_step: list[Tensor] = []

        # Autoregressive rollout for T_pred steps
        for step in range(T_pred):
            num_action_steps = C + step
            ar_actions = actions[:, :num_action_steps, :]
            ar_states = states[:, :num_action_steps, :]
            ar_extrinsics = extrinsics[:, :num_action_steps, :] if extrinsics is not None else None

            # Predict next frame
            z_nxt = self._step_predictor(z_ar, ar_actions, ar_states, ar_extrinsics)
            z_nxt = z_nxt[:, -N:, :]  # [B, N, D] - last frame only

            predictions_per_step.append(z_nxt)

            # Append to autoregressive context
            z_ar = torch.cat([z_ar, z_nxt], dim=1)

        # Compute per-timestep losses
        per_timestep_losses: list[Tensor] = []
        per_sample_total_losses = torch.zeros(B, device=features.device)

        for step, pred in enumerate(predictions_per_step):
            # Target for this step: frame C + step
            target_frame = h[:, C + step, :, :]  # [B, N, D]

            # Per-sample loss for this timestep
            diff = torch.abs(pred - target_frame)  # [B, N, D]
            diff_exp = diff ** self.loss_exp
            per_sample_loss = diff_exp.mean(dim=(1, 2)) / self.loss_exp  # [B]

            per_timestep_losses.append(per_sample_loss)
            per_sample_total_losses += per_sample_loss

        # Average per-sample losses across timesteps
        per_sample_total_losses = per_sample_total_losses / T_pred

        # Total loss (scalar)
        total_loss = per_sample_total_losses.mean()

        return total_loss, per_timestep_losses, [per_sample_total_losses]

    def _shared_step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        """Shared step for training and validation.

        Args:
            batch: Dictionary containing:
                - features: [B, T+1, N, D] or [B, (T+1)*N, D]
                - actions: [B, T, action_dim]
                - states: [B, T, action_dim]
                - extrinsics (optional): [B, T, action_dim-1]
            stage: 'train' or 'val'

        Returns:
            Combined loss tensor
        """
        import logging
        log = logging.getLogger(__name__)

        features = batch["features"]
        actions = batch["actions"]
        states = batch["states"]
        extrinsics = batch.get("extrinsics", None)

        log.debug(f"\n{'='*60}")
        log.debug(f"[SHARED STEP] Stage: {stage}")
        log.debug(f"[SHARED STEP] Input features shape: {features.shape}")
        log.debug(f"[SHARED STEP] Actions shape: {actions.shape}, States shape: {states.shape}")

        # Reshape features if needed: [B, (T+1)*N, D] -> [B, T+1, N, D]
        if features.dim() == 3:
            B, total_tokens, D = features.shape
            T_plus_1 = total_tokens // self.patches_per_frame
            features = features.reshape(B, T_plus_1, self.patches_per_frame, D)
            log.debug(f"[SHARED STEP] Reshaped features to: {features.shape}")

        log.debug(f"\n--- COMPUTING TEACHER FORCING LOSS ---")
        # Compute losses
        loss_teacher = self._compute_teacher_forcing_loss(features, actions, states, extrinsics)
        log.debug(f"[SHARED STEP] Teacher forcing loss: {loss_teacher.item():.8f}")

        log.debug(f"\n--- COMPUTING ROLLOUT LOSS ---")
        loss_rollout = self._compute_rollout_loss(features, actions, states, extrinsics)
        log.debug(f"[SHARED STEP] Rollout loss: {loss_rollout.item():.8f}")

        # Combined loss
        loss = self.loss_weight_teacher * loss_teacher + self.loss_weight_rollout * loss_rollout
        log.debug(f"\n--- COMBINED LOSS ---")
        log.debug(f"[SHARED STEP] weight_teacher={self.loss_weight_teacher}, weight_rollout={self.loss_weight_rollout}")
        log.debug(f"[SHARED STEP] Combined loss = {self.loss_weight_teacher} * {loss_teacher.item():.8f} + {self.loss_weight_rollout} * {loss_rollout.item():.8f} = {loss.item():.8f}")

        # Log metrics
        self.log(f"{stage}/loss_teacher", loss_teacher, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/loss_rollout", loss_rollout, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Validation step."""
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Test step with detailed per-clip and per-timestep analysis.

        Computes rollout losses with per-timestep breakdown and stores results
        for aggregation in on_test_epoch_end.
        """
        import logging
        log = logging.getLogger(__name__)

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

        # Compute teacher forcing loss (for comparison)
        loss_teacher = self._compute_teacher_forcing_loss(features, actions, states, extrinsics)

        # Compute rollout loss with per-timestep breakdown
        loss_rollout, per_timestep_losses, per_sample_losses = self._compute_rollout_loss_per_timestep(
            features, actions, states, extrinsics
        )

        # Combined loss
        loss = self.loss_weight_teacher * loss_teacher + self.loss_weight_rollout * loss_rollout

        # Log aggregate metrics
        self.log("test/loss_teacher", loss_teacher, prog_bar=True, sync_dist=True)
        self.log("test/loss_rollout", loss_rollout, prog_bar=True, sync_dist=True)
        self.log("test/loss", loss, prog_bar=True, sync_dist=True)

        # Log per-timestep losses
        for step, step_loss in enumerate(per_timestep_losses):
            predicted_frame = self.context_frames + step  # e.g., 3, 4, 5, 6
            self.log(f"test/loss_step_{predicted_frame}", step_loss.mean(), sync_dist=True)

        # Store per-clip results for aggregation
        per_sample_rollout_losses = per_sample_losses[0]  # [B]
        for i in range(B):
            clip_result: TestResultsDict = {
                "clip_name": clip_names[i] if i < len(clip_names) else f"unknown_{batch_idx}_{i}",
                "loss_rollout": per_sample_rollout_losses[i].item(),
                "loss_teacher": loss_teacher.item(),  # Note: this is batch-level for teacher
                "per_timestep_losses": {
                    f"step_{self.context_frames + s}": per_timestep_losses[s][i].item()
                    for s in range(len(per_timestep_losses))
                },
            }
            self._test_results.append(clip_result)

        return loss

    def on_test_epoch_start(self) -> None:
        """Clear test results at the start of test epoch."""
        self._test_results = []

    def on_test_epoch_end(self) -> None:
        """Aggregate test results and output summary.

        Prints summary statistics and optionally exports detailed results to JSON.
        """
        import json
        import logging
        from pathlib import Path

        log = logging.getLogger(__name__)

        if not self._test_results:
            log.warning("No test results to aggregate")
            return

        # Compute aggregate statistics
        num_clips = len(self._test_results)
        rollout_losses = [r["loss_rollout"] for r in self._test_results]

        mean_loss = sum(rollout_losses) / num_clips
        sorted_losses = sorted(rollout_losses)
        median_loss = sorted_losses[num_clips // 2]
        min_loss = min(rollout_losses)
        max_loss = max(rollout_losses)
        std_loss = (sum((x - mean_loss) ** 2 for x in rollout_losses) / num_clips) ** 0.5

        # Compute per-timestep statistics
        num_timesteps = len(self._test_results[0]["per_timestep_losses"])
        per_timestep_stats = {}
        for step_key in self._test_results[0]["per_timestep_losses"].keys():
            step_losses = [r["per_timestep_losses"][step_key] for r in self._test_results]
            per_timestep_stats[step_key] = {
                "mean": sum(step_losses) / num_clips,
                "min": min(step_losses),
                "max": max(step_losses),
            }

        # Find worst-performing clips
        worst_clips = sorted(self._test_results, key=lambda x: x["loss_rollout"], reverse=True)[:10]
        best_clips = sorted(self._test_results, key=lambda x: x["loss_rollout"])[:5]

        # Print summary to console
        print("\n" + "=" * 70)
        print("                    TEST RESULTS SUMMARY")
        print("=" * 70)
        print(f"\n📊 AGGREGATE STATISTICS (over {num_clips} clips)")
        print("-" * 50)
        print(f"  Rollout Loss (L1):")
        print(f"    Mean:   {mean_loss:.6f}")
        print(f"    Median: {median_loss:.6f}")
        print(f"    Std:    {std_loss:.6f}")
        print(f"    Min:    {min_loss:.6f}")
        print(f"    Max:    {max_loss:.6f}")

        print(f"\n📈 PER-TIMESTEP BREAKDOWN (Context: {self.context_frames} frames)")
        print("-" * 50)
        print(f"  {'Frame':<10} {'Mean Loss':<12} {'Min':<12} {'Max':<12}")
        print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
        for step_key, stats in per_timestep_stats.items():
            frame_num = step_key.replace("step_", "z_")
            print(f"  {frame_num:<10} {stats['mean']:<12.6f} {stats['min']:<12.6f} {stats['max']:<12.6f}")

        print(f"\n🔴 WORST-PERFORMING CLIPS (top 10)")
        print("-" * 50)
        for i, clip in enumerate(worst_clips, 1):
            print(f"  {i:2d}. {clip['clip_name']:<20} loss={clip['loss_rollout']:.6f}")

        print(f"\n🟢 BEST-PERFORMING CLIPS (top 5)")
        print("-" * 50)
        for i, clip in enumerate(best_clips, 1):
            print(f"  {i:2d}. {clip['clip_name']:<20} loss={clip['loss_rollout']:.6f}")

        print("\n" + "=" * 70)

        # Export to JSON if configured
        # Check trainer for config or use default
        export_results = True
        output_dir = None

        if self.trainer and hasattr(self.trainer, "log_dir") and self.trainer.log_dir:
            output_dir = Path(self.trainer.log_dir)
        else:
            # Use current directory as fallback
            output_dir = Path(".")

        if export_results:
            results_file = output_dir / "test_results.json"
            export_data = {
                "summary": {
                    "num_clips": num_clips,
                    "context_frames": self.context_frames,
                    "T_rollout": self.T_rollout,
                    "rollout_loss": {
                        "mean": mean_loss,
                        "median": median_loss,
                        "std": std_loss,
                        "min": min_loss,
                        "max": max_loss,
                    },
                    "per_timestep": per_timestep_stats,
                },
                "worst_clips": worst_clips,
                "best_clips": best_clips,
                "all_clips": self._test_results,
            }

            try:
                results_file.parent.mkdir(parents=True, exist_ok=True)
                with open(results_file, "w") as f:
                    json.dump(export_data, f, indent=2)
                print(f"\n💾 Detailed results exported to: {results_file}")
            except Exception as e:
                log.warning(f"Failed to export results to JSON: {e}")

        # Log final metrics
        self.log("test/final_mean_loss_rollout", mean_loss, sync_dist=True)
        self.log("test/final_median_loss_rollout", median_loss, sync_dist=True)

    def configure_optimizers(self) -> dict:
        """Configure optimizer and learning rate scheduler.

        Supports two scheduling modes:
        1. Epoch-based (default): Cosine annealing with linear warmup
        2. Iteration-based (V-JEPA2 paper): Warmup → Constant → Decay
           - Linear warmup from warmup_start_lr to learning_rate
           - Constant phase at peak learning_rate
           - Linear decay from learning_rate to 0
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )

        if self.use_iteration_scheduler:
            # Iteration-based: Warmup → Constant → Decay (V-JEPA2 paper)
            # Compute total training iterations from trainer
            # total_iters = num_batches_per_epoch * max_epochs
            if self.trainer is not None and self.trainer.estimated_stepping_batches:
                total_iters = int(self.trainer.estimated_stepping_batches)
            else:
                # Fallback: estimate from dataloader if trainer not fully set up
                import warnings
                warnings.warn(
                    "Could not get total iterations from trainer. "
                    "LR schedule may not work correctly.",
                    UserWarning
                )
                total_iters = 10000  # Fallback default

            # Convert percentages to iteration counts
            warmup_iters = int(self.warmup_pct * total_iters)
            constant_iters = int(self.constant_pct * total_iters)
            decay_iters = int(self.decay_pct * total_iters)

            # Ensure we don't exceed total due to rounding
            # Adjust constant_iters to account for any rounding differences
            computed_total = warmup_iters + constant_iters + decay_iters
            if computed_total != total_iters:
                constant_iters += (total_iters - computed_total)

            import logging
            log = logging.getLogger(__name__)
            log.info(
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
        else:
            # Epoch-based: Cosine annealing with linear warmup (default)
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
