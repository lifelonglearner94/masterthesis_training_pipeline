# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Lightning Module wrapper for AC Predictor.

Implements teacher-forcing and rollout losses for training the action-conditioned
vision transformer predictor.

Loss formulations:
- Teacher-Forcing: L_tf = (1/T) Σ ||P_φ((a_t, s_t, z_t)_{t≤k}) - z_{k+1}||_1
- Rollout: L_rollout = ||P_φ(a_{1:T}, s_1, z_1) - z_{T+1}||_1
- Combined: L = L_teacher-forcing + L_rollout
"""

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.ac_predictor.ac_predictor import vit_ac_predictor


class ACPredictorModule(pl.LightningModule):
    """PyTorch Lightning module for Action-Conditioned Vision Transformer Predictor.

    This module wraps the VisionTransformerPredictorAC and implements:
    - Teacher-forcing loss: predicts next latent from context, averaged over T steps
    - Rollout loss: enforces consistency over multiple recurrent prediction steps

    Expected batch format:
        - features: [B, T+1, N, D] - Pre-computed V-JEPA2 encoder features for T+1 frames
        - actions: [B, T, action_dim] - 7D end-effector state changes
        - states: [B, T, action_dim] - 7D end-effector states

    Where:
        - T is the number of prediction steps (default: 15 for teacher-forcing)
        - N is the number of patches per frame (H*W)
        - D is the embedding dimension
    """

    def __init__(
        self,
        # Model architecture
        img_size: tuple[int, int] = (224, 224),
        patch_size: int = 16,
        num_frames: int = 16,
        tubelet_size: int = 2,
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
        T_teacher: int = 15,
        T_rollout: int = 2,
        loss_weight_teacher: float = 1.0,
        loss_weight_rollout: float = 1.0,
        # Optimizer settings
        learning_rate: float = 4.25e-4,
        weight_decay: float = 0.04,
        betas: tuple[float, float] = (0.9, 0.999),
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        # Iteration-based LR schedule (V-JEPA2 paper)
        use_iteration_scheduler: bool = False,
        warmup_iters: int = 4500,
        constant_iters: int = 85500,
        decay_iters: int = 4500,
        warmup_start_lr: float = 7.5e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Build the predictor model
        self.model = vit_ac_predictor(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
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
        self.loss_weight_teacher = loss_weight_teacher
        self.loss_weight_rollout = loss_weight_rollout

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

    def _compute_teacher_forcing_loss(
        self,
        features: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
    ) -> Tensor:
        """Compute teacher-forcing loss.

        L_teacher-forcing = (1/T) Σ_{k=1}^{T} ||P_φ((a_t, s_t, z_t)_{t≤k}) - z_{k+1}||_1

        The predictor receives context up to frame k and predicts the next frame.
        Loss is averaged over T prediction steps.

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

        total_loss = 0.0

        for k in range(1, T + 1):
            # Context: frames 0 to k-1 (k frames total)
            context_features = features[:, :k, :, :].reshape(B, k * N, D)
            context_actions = actions[:, :k, :]
            context_states = states[:, :k, :]
            context_extrinsics = extrinsics[:, :k, :] if extrinsics is not None else None

            # Predict
            pred = self.model(context_features, context_actions, context_states, context_extrinsics)

            # Target: frame k (the next frame after context)
            # pred has shape [B, k*N, D], we want the last frame's prediction
            pred_last_frame = pred[:, -N:, :]  # [B, N, D]
            target = features[:, k, :, :]  # [B, N, D]

            # L1 loss
            loss_k = F.l1_loss(pred_last_frame, target)
            total_loss = total_loss + loss_k

        # Average over T steps
        return total_loss / T

    def _compute_rollout_loss(
        self,
        features: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
    ) -> Tensor:
        """Compute rollout loss.

        L_rollout = ||P_φ(a_{1:T}, s_1, z_1) - z_{T+1}||_1

        Recurrently apply the predictor for T_rollout steps, using predicted
        features as input for subsequent steps.

        Args:
            features: [B, T+1, N, D] - Features for T+1 frames
            actions: [B, T, action_dim] - Actions for T steps
            states: [B, T, action_dim] - States for T steps
            extrinsics: [B, T, action_dim-1] - Optional extrinsics

        Returns:
            Scalar loss tensor
        """
        B, T_plus_1, N, D = features.shape
        T = min(T_plus_1 - 1, self.T_rollout)

        # Start with first frame
        current_features = features[:, 0:1, :, :].reshape(B, N, D)

        for t in range(T):
            # Get action/state for this step
            action_t = actions[:, t : t + 1, :]
            state_t = states[:, t : t + 1, :]
            extrinsics_t = extrinsics[:, t : t + 1, :] if extrinsics is not None else None

            # Predict next frame
            pred = self.model(current_features, action_t, state_t, extrinsics_t)

            # Use prediction as next input (autoregressive rollout)
            current_features = pred

        # Compare final prediction with target
        target = features[:, T, :, :].reshape(B, N, D)

        return F.l1_loss(current_features, target)

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
