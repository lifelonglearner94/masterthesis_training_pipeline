"""Baseline ConvLSTM model for action-conditioned prediction.

A simpler baseline model using ConvLSTM for temporal modeling, designed to be
scientifically comparable to the ViT AC Predictor while having fewer parameters.

Architecture:
    - 1×1 Conv Encoder: 1024 → hidden_dim (256)
    - Spatial Action Tiling: Tile 2D action to 16×16, concat with features
    - ConvLSTM Cell: Temporal modeling with spatial convolutions
    - Residual Decoder: hidden_dim → 1024, output = z_t + delta

Reference: docs/possible_Baseline.md
"""

from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.mixins import ACPredictorLossMixin


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell for spatiotemporal modeling.

    Uses convolutions instead of fully connected layers for state transitions,
    preserving spatial structure in the hidden and cell states.

    Args:
        input_dim: Number of input channels
        hidden_dim: Number of hidden channels
        kernel_size: Size of the convolutional kernel (default: 3)
        bias: Whether to use bias in convolutions
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Combined gates convolution (input, forget, cell, output)
        self.conv_gates = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias,
        )

    def forward(
        self,
        x: Tensor,
        hidden_state: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass through the ConvLSTM cell.

        Args:
            x: Input tensor [B, input_dim, H, W]
            hidden_state: Tuple of (h, c) each [B, hidden_dim, H, W], or None

        Returns:
            Tuple of (h_next, c_next) each [B, hidden_dim, H, W]
        """
        B, _, H, W = x.shape

        # Initialize hidden state if not provided
        if hidden_state is None:
            h = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        else:
            h, c = hidden_state

        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)  # [B, input_dim + hidden_dim, H, W]

        # Compute all gates in one convolution
        gates = self.conv_gates(combined)  # [B, 4 * hidden_dim, H, W]

        # Split into individual gates
        i, f, g, o = torch.split(gates, self.hidden_dim, dim=1)

        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)  # Cell gate
        o = torch.sigmoid(o)  # Output gate

        # Update cell and hidden state
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTMBaseline(nn.Module):
    """Action-Conditioned ConvLSTM baseline model.

    A simpler baseline architecture for action-conditioned prediction that:
    1. Compresses features with a 1×1 conv encoder
    2. Tiles actions spatially and concatenates with features
    3. Uses ConvLSTM for temporal modeling
    4. Predicts residuals with a 1×1 conv decoder

    The model predicts: z_{t+1} = z_t + delta_t

    Args:
        input_dim: Dimension of input features (1024 for V-JEPA2)
        hidden_dim: Hidden dimension for ConvLSTM (default: 256)
        action_dim: Action dimension (default: 2)
        spatial_size: Spatial grid size (default: 16 for 16×16)
        kernel_size: ConvLSTM kernel size (default: 3)
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        action_dim: int = 2,
        spatial_size: int = 16,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.spatial_size = spatial_size

        # 1×1 Conv Encoder: input_dim → hidden_dim
        self.encoder = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)

        # ConvLSTM input dim = hidden_dim + action_dim (after spatial tiling)
        self.conv_lstm = ConvLSTMCell(
            input_dim=hidden_dim + action_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
        )

        # 1×1 Conv Decoder: hidden_dim → input_dim (for residual)
        self.decoder = nn.Conv2d(hidden_dim, input_dim, kernel_size=1)

        # Initialize decoder to output small residuals
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def _tile_action(self, action: Tensor, H: int, W: int) -> Tensor:
        """Tile action vector spatially.

        Args:
            action: Action tensor [B, action_dim]
            H: Spatial height
            W: Spatial width

        Returns:
            Tiled action [B, action_dim, H, W]
        """
        B, A = action.shape
        # Reshape and expand: [B, A] -> [B, A, 1, 1] -> [B, A, H, W]
        return action.view(B, A, 1, 1).expand(B, A, H, W)

    def forward(
        self,
        z: Tensor,
        actions: Tensor,
        states: Tensor | None = None,
        extrinsics: Tensor | None = None,
    ) -> Tensor:
        """Forward pass - predict next frames given context and actions.

        This method handles both:
        1. Teacher-forcing: full sequence input, parallel prediction
        2. Single-step: for autoregressive rollout

        Args:
            z: Context features [B, T*N, D] where N = spatial_size^2, D = input_dim
            actions: Actions [B, T, action_dim]
            states: States (unused, for API compatibility)
            extrinsics: Extrinsics (unused, for API compatibility)

        Returns:
            Predicted features [B, T*N, D]
        """
        B = z.shape[0]
        N = self.spatial_size * self.spatial_size
        D = self.input_dim
        T = z.shape[1] // N

        # Reshape from [B, T*N, D] to [B, T, N, D] to [B, T, D, H, W]
        z_spatial = z.reshape(B, T, N, D)
        z_spatial = z_spatial.permute(0, 1, 3, 2)  # [B, T, D, N]
        z_spatial = z_spatial.reshape(B, T, D, self.spatial_size, self.spatial_size)

        # Initialize hidden state
        h, c = None, None

        # Store predictions
        predictions = []

        # Process each timestep
        for t in range(T):
            # Get current frame: [B, D, H, W]
            current_z = z_spatial[:, t]

            # Encode: [B, D, H, W] -> [B, hidden_dim, H, W]
            feat = self.encoder(current_z)

            # Get action for this timestep: [B, action_dim]
            action_t = actions[:, t]

            # Tile action spatially: [B, action_dim, H, W]
            action_tiled = self._tile_action(action_t, self.spatial_size, self.spatial_size)

            # Concatenate features and action: [B, hidden_dim + action_dim, H, W]
            lstm_input = torch.cat([feat, action_tiled], dim=1)

            # ConvLSTM step
            h, c = self.conv_lstm(lstm_input, (h, c) if h is not None else None)

            # Decode to residual: [B, hidden_dim, H, W] -> [B, D, H, W]
            delta = self.decoder(h)

            # Residual prediction: z_{t+1} = z_t + delta
            next_z = current_z + delta

            # Store prediction
            predictions.append(next_z)

        # Stack predictions: [B, T, D, H, W]
        pred_stack = torch.stack(predictions, dim=1)

        # Reshape back to [B, T*N, D]
        pred_stack = pred_stack.reshape(B, T, D, N)  # [B, T, D, N]
        pred_stack = pred_stack.permute(0, 1, 3, 2)  # [B, T, N, D]
        pred_stack = pred_stack.reshape(B, T * N, D)  # [B, T*N, D]

        return pred_stack


class BaselineLitModule(ACPredictorLossMixin, L.LightningModule):
    """PyTorch Lightning module for ConvLSTM Baseline.

    A simpler baseline for action-conditioned prediction, designed to be
    scientifically comparable to the ViT AC Predictor (ACPredictorModule).

    Uses the same loss computation, curriculum schedule, and evaluation
    protocol as ACPredictorModule via the ACPredictorLossMixin.

    Expected batch format:
        - features: [B, T+1, N, D] - Pre-computed V-JEPA2 encoder features
        - actions: [B, T, action_dim] - Action sequences
        - states: [B, T, action_dim] - State sequences (unused by baseline)

    Where:
        - T is the number of encoded timesteps (e.g., 8)
        - N is the number of patches per frame (256 = 16×16)
        - D is the embedding dimension (1024 for V-JEPA2)
    """

    def __init__(
        self,
        # Model architecture
        input_dim: int = 1024,
        hidden_dim: int = 256,
        action_dim: int = 2,
        spatial_size: int = 16,
        kernel_size: int = 3,
        num_timesteps: int = 8,
        # Loss settings (same as ACPredictorModule)
        T_teacher: int = 7,
        T_rollout: int = 7,
        context_frames: int = 1,
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
        # Iteration-based LR schedule
        use_iteration_scheduler: bool = False,
        curriculum_schedule: list[dict] | None = None,
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

        # Build the baseline model
        self.model = ConvLSTMBaseline(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            spatial_size=spatial_size,
            kernel_size=kernel_size,
        )

        # Loss hyperparameters (required by mixin)
        self.T_teacher = T_teacher
        self.T_rollout = T_rollout
        self.context_frames = context_frames
        self.loss_weight_teacher = loss_weight_teacher
        self.loss_weight_rollout = loss_weight_rollout
        self.normalize_reps = normalize_reps

        # Validate loss_exp
        if loss_exp <= 0:
            raise ValueError(f"loss_exp must be positive (got {loss_exp})")
        self.loss_exp = loss_exp

        # Optimizer hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        # Iteration-based LR schedule
        self.use_iteration_scheduler = use_iteration_scheduler
        self.warmup_pct = warmup_pct
        self.constant_pct = constant_pct
        self.decay_pct = decay_pct
        self.warmup_start_lr = warmup_start_lr

        # Grid size for reshaping (required by mixin)
        self.patches_per_frame = spatial_size * spatial_size

        # Curriculum learning schedule
        self.curriculum_schedule = curriculum_schedule
        if curriculum_schedule:
            self._validate_curriculum_schedule(curriculum_schedule)

        # Test results storage
        self._test_results: list[dict[str, Any]] = []

    def _validate_curriculum_schedule(self, schedule: list[dict]) -> None:
        """Validate curriculum schedule format."""
        from collections.abc import Mapping, Sequence

        if not isinstance(schedule, Sequence) or isinstance(schedule, str) or len(schedule) == 0:
            raise ValueError("curriculum_schedule must be a non-empty list")

        max_prediction_steps = self.num_timesteps - 1

        for i, phase in enumerate(schedule):
            if not isinstance(phase, Mapping):
                raise ValueError(f"Phase {i} must be a dict, got {type(phase)}")
            if "epoch" not in phase:
                raise ValueError(f"Phase {i} must have 'epoch' key")
            if not isinstance(phase["epoch"], int) or phase["epoch"] < 0:
                raise ValueError(f"Phase {i} 'epoch' must be a non-negative integer")

            if "T_rollout" in phase and phase["T_rollout"] > max_prediction_steps:
                raise ValueError(
                    f"Phase {i}: T_rollout ({phase['T_rollout']}) exceeds "
                    f"max prediction steps ({max_prediction_steps})"
                )

        epochs = [p["epoch"] for p in schedule]
        if epochs != sorted(epochs):
            raise ValueError("curriculum_schedule phases must be sorted by epoch")

    def _get_curriculum_params_for_epoch(self, epoch: int) -> dict:
        """Get curriculum parameters for the given epoch."""
        if not self.curriculum_schedule:
            return {}

        applicable_phase = None
        for phase in self.curriculum_schedule:
            if phase["epoch"] <= epoch:
                applicable_phase = phase
            else:
                break

        if applicable_phase is None:
            return {}

        return {k: v for k, v in applicable_phase.items() if k != "epoch"}

    def on_train_epoch_start(self) -> None:
        """Update curriculum parameters at the start of each epoch."""
        if not self.curriculum_schedule:
            return

        epoch = self.current_epoch
        params = self._get_curriculum_params_for_epoch(epoch)

        if not params:
            return

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

        if changes:
            import logging

            log = logging.getLogger(__name__)
            log.info(f"[Curriculum] Epoch {epoch}: {', '.join(changes)}")

        self.log("curriculum/T_rollout", float(self.T_rollout), sync_dist=True)
        self.log("curriculum/loss_weight_teacher", self.loss_weight_teacher, sync_dist=True)
        self.log("curriculum/loss_weight_rollout", self.loss_weight_rollout, sync_dist=True)

    def forward(
        self,
        features: Tensor,
        actions: Tensor,
        states: Tensor | None = None,
        extrinsics: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through the baseline model.

        Args:
            features: Encoded features [B, T*N, D]
            actions: Action sequences [B, T, action_dim]
            states: State sequences (unused)
            extrinsics: Extrinsics (unused)

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

        Required by ACPredictorLossMixin.

        Args:
            z: Input features [B, T*N, D]
            actions: Actions [B, T, action_dim]
            states: States [B, T, action_dim]
            extrinsics: Optional extrinsics

        Returns:
            Predicted features [B, T*N, D], optionally normalized
        """
        z_pred = self.model(z, actions, states, extrinsics)
        if self.normalize_reps:
            z_pred = F.layer_norm(z_pred, (z_pred.size(-1),), eps=1e-6)
        return z_pred

    # Loss methods are inherited from ACPredictorLossMixin:
    # - _compute_loss
    # - _compute_teacher_forcing_loss
    # - _compute_rollout_loss
    # - _compute_rollout_loss_per_timestep
    # - _shared_step

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Validation step."""
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Test step with detailed per-clip analysis."""
        features = batch["features"]
        actions = batch["actions"]
        states = batch["states"]
        extrinsics = batch.get("extrinsics", None)
        clip_names = batch.get(
            "clip_names", [f"clip_{batch_idx}_{i}" for i in range(features.shape[0])]
        )

        # Reshape features if needed
        if features.dim() == 3:
            B, total_tokens, D = features.shape
            T_plus_1 = total_tokens // self.patches_per_frame
            features = features.reshape(B, T_plus_1, self.patches_per_frame, D)
        else:
            B = features.shape[0]

        # Compute losses
        loss_teacher = self._compute_teacher_forcing_loss(features, actions, states, extrinsics)
        loss_rollout, per_timestep_losses, per_sample_losses = self._compute_rollout_loss_per_timestep(
            features, actions, states, extrinsics
        )

        loss = self.loss_weight_teacher * loss_teacher + self.loss_weight_rollout * loss_rollout

        # Log metrics
        self.log("test/loss_teacher", loss_teacher, prog_bar=True, sync_dist=True)
        self.log("test/loss_rollout", loss_rollout, prog_bar=True, sync_dist=True)
        self.log("test/loss", loss, prog_bar=True, sync_dist=True)

        for step, step_loss in enumerate(per_timestep_losses):
            predicted_frame = self.context_frames + step
            self.log(f"test/loss_step_{predicted_frame}", step_loss.mean(), sync_dist=True)

        # Store per-clip results
        per_sample_rollout_losses = per_sample_losses[0]
        for i in range(B):
            clip_result = {
                "clip_name": clip_names[i] if i < len(clip_names) else f"unknown_{batch_idx}_{i}",
                "loss_rollout": per_sample_rollout_losses[i].item(),
                "loss_teacher": loss_teacher.item(),
                "per_timestep_losses": {
                    f"step_{self.context_frames + s}": per_timestep_losses[s][i].item()
                    for s in range(len(per_timestep_losses))
                },
            }
            self._test_results.append(clip_result)

        return loss

    def on_test_epoch_start(self) -> None:
        """Clear test results."""
        self._test_results = []

    def on_test_epoch_end(self) -> None:
        """Aggregate and output test results."""
        import json
        import logging
        from pathlib import Path

        log = logging.getLogger(__name__)

        if not self._test_results:
            log.warning("No test results to aggregate")
            return

        num_clips = len(self._test_results)
        rollout_losses = [r["loss_rollout"] for r in self._test_results]

        mean_loss = sum(rollout_losses) / num_clips
        sorted_losses = sorted(rollout_losses)
        median_loss = sorted_losses[num_clips // 2]
        min_loss = min(rollout_losses)
        max_loss = max(rollout_losses)
        std_loss = (sum((x - mean_loss) ** 2 for x in rollout_losses) / num_clips) ** 0.5

        print("\n" + "=" * 70)
        print("            BASELINE TEST RESULTS SUMMARY")
        print("=" * 70)
        print(f"\n📊 AGGREGATE STATISTICS (over {num_clips} clips)")
        print("-" * 50)
        print(f"  Rollout Loss (L1):")
        print(f"    Mean:   {mean_loss:.6f}")
        print(f"    Median: {median_loss:.6f}")
        print(f"    Std:    {std_loss:.6f}")
        print(f"    Min:    {min_loss:.6f}")
        print(f"    Max:    {max_loss:.6f}")
        print("=" * 70)

        self.log("test/final_mean_loss_rollout", mean_loss, sync_dist=True)
        self.log("test/final_median_loss_rollout", median_loss, sync_dist=True)

    def configure_optimizers(self) -> dict:
        """Configure optimizer and learning rate scheduler.

        Supports epoch-based and iteration-based scheduling (matching ACPredictorModule).
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )

        if self.use_iteration_scheduler:
            if self.trainer is not None and self.trainer.estimated_stepping_batches:
                total_iters = int(self.trainer.estimated_stepping_batches)
            else:
                import warnings

                warnings.warn(
                    "Could not get total iterations from trainer. "
                    "LR schedule may not work correctly.",
                    UserWarning,
                )
                total_iters = 10000

            warmup_iters = int(self.warmup_pct * total_iters)
            constant_iters = int(self.constant_pct * total_iters)
            decay_iters = int(self.decay_pct * total_iters)

            computed_total = warmup_iters + constant_iters + decay_iters
            if computed_total != total_iters:
                constant_iters += total_iters - computed_total

            warmup_end = warmup_iters
            constant_end = warmup_end + constant_iters

            warmup_start_factor = self.warmup_start_lr / self.learning_rate

            def lr_lambda_iter(step: int) -> float:
                if step < warmup_end:
                    progress = step / warmup_iters
                    return warmup_start_factor + (1.0 - warmup_start_factor) * progress
                elif step < constant_end:
                    return 1.0
                elif step < total_iters:
                    progress = (step - constant_end) / decay_iters
                    return 1.0 - progress
                else:
                    return 0.0

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_iter)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
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
