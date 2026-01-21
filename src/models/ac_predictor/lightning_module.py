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
        warmup_iters: int = 4500,
        constant_iters: int = 85500,
        decay_iters: int = 4500,
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
        self.warmup_iters = warmup_iters
        self.constant_iters = constant_iters
        self.decay_iters = decay_iters
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
        return torch.mean(torch.abs(pred - target) ** self.loss_exp) / self.loss_exp

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
        B, T_plus_1, N, D = features.shape
        T = min(T_plus_1 - 1, self.T_teacher)

        # Context: all frames except the last one (frames 0 to T-1)
        # Shape: [B, T*N, D]
        context_features = features[:, :T, :, :].reshape(B, T * N, D)

        # Apply layer norm to input if enabled (matching paper)
        if self.normalize_reps:
            context_features = F.layer_norm(context_features, (context_features.size(-1),), eps=1e-6)

        # Actions/states for T steps
        context_actions = actions[:, :T, :]
        context_states = states[:, :T, :]
        context_extrinsics = extrinsics[:, :T, :] if extrinsics is not None else None

        # Single forward pass - predicts all T frames
        z_tf = self._step_predictor(
            context_features, context_actions, context_states, context_extrinsics
        )

        # Target: frames 1 to T (shifted by one from context)
        # Shape: [B, T*N, D]
        target = features[:, 1 : T + 1, :, :].reshape(B, T * N, D)
        if self.normalize_reps:
            target = F.layer_norm(target, (target.size(-1),), eps=1e-6)

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
        B, T_plus_1, N, D = features.shape
        C = self.context_frames  # Number of ground-truth context frames
        T_pred = min(T_plus_1 - C, self.T_rollout)  # Number of steps to predict

        # Normalize ground-truth features if enabled
        h = features
        if self.normalize_reps:
            h = F.layer_norm(features.reshape(B, -1, D), (D,), eps=1e-6).reshape(B, T_plus_1, N, D)

        # Step 1: Initialize with C ground-truth context frames
        # z_ar contains frames 0, 1, ..., C-1
        z_ar = h[:, :C, :, :].reshape(B, C * N, D)  # [B, C*N, D]

        # Step 2: Autoregressive rollout for T_pred steps
        # First prediction is frame C, using actions/states 0 to C-1
        for step in range(T_pred):
            # Current prediction target is frame (C + step)
            # Use actions/states from 0 to (C + step - 1)
            num_action_steps = C + step
            ar_actions = actions[:, :num_action_steps, :]
            ar_states = states[:, :num_action_steps, :]
            ar_extrinsics = extrinsics[:, :num_action_steps, :] if extrinsics is not None else None

            # Predict next frame using full context
            z_nxt = self._step_predictor(z_ar, ar_actions, ar_states, ar_extrinsics)
            # Extract only the last frame prediction
            z_nxt = z_nxt[:, -N:, :]  # [B, N, D]

            # Append to autoregressive context
            z_ar = torch.cat([z_ar, z_nxt], dim=1)  # [B, (C+step+1)*N, D]

        # Step 3: Compare autoregressive predictions with targets
        # z_ar[:, C*N:] contains predictions for frames C, C+1, ..., C+T_pred-1
        # Target: h[:, C:C+T_pred] contains ground-truth for same frames
        z_ar_pred = z_ar[:, C * N:]  # [B, T_pred*N, D]
        target = h[:, C : C + T_pred, :, :].reshape(B, T_pred * N, D)  # [B, T_pred*N, D]

        return self._compute_loss(z_ar_pred, target)

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
        features = batch["features"]
        actions = batch["actions"]
        states = batch["states"]
        extrinsics = batch.get("extrinsics", None)

        # Reshape features if needed: [B, (T+1)*N, D] -> [B, T+1, N, D]
        if features.dim() == 3:
            B, total_tokens, D = features.shape
            T_plus_1 = total_tokens // self.patches_per_frame
            features = features.reshape(B, T_plus_1, self.patches_per_frame, D)

        # Compute losses
        loss_teacher = self._compute_teacher_forcing_loss(features, actions, states, extrinsics)
        loss_rollout = self._compute_rollout_loss(features, actions, states, extrinsics)

        # Combined loss
        loss = self.loss_weight_teacher * loss_teacher + self.loss_weight_rollout * loss_rollout

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
        """Test step."""
        return self._shared_step(batch, "test")

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
            # Total iterations = warmup_iters + constant_iters + decay_iters
            warmup_iters = self.warmup_iters
            constant_iters = self.constant_iters
            decay_iters = self.decay_iters
            warmup_end = warmup_iters
            constant_end = warmup_end + constant_iters
            total_iters = constant_end + decay_iters

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
