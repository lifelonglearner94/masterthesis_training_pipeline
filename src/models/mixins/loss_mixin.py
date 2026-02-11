"""Loss computation mixin for AC predictor models.

Provides reusable loss computation methods for both the ViT AC Predictor
and baseline models, ensuring scientific comparability.
"""

import logging

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class ACPredictorLossMixin:
    """Mixin providing loss computation methods for AC predictors.

    This mixin expects the following attributes to be set by the inheriting class:
        - normalize_reps: bool - Whether to apply layer normalization
        - loss_exp: float - Loss exponent (1.0 for L1, 2.0 for L2)
        - T_teacher: int - Number of teacher-forcing steps
        - T_rollout: int - Number of rollout prediction steps
        - context_frames: int - Number of ground-truth context frames for rollout
        - patches_per_frame: int - Number of patches (tokens) per frame (N = H*W)

    The inheriting class must also implement:
        - _step_predictor(z, actions, states, extrinsics) -> Tensor
            Forward pass through the model with optional normalization
    """

    # Type hints for expected attributes (set by inheriting class)
    normalize_reps: bool
    loss_exp: float
    T_teacher: int
    T_rollout: int
    context_frames: int
    patches_per_frame: int
    loss_weight_teacher: float
    loss_weight_rollout: float

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
        # Compute element-wise absolute difference
        diff = torch.abs(pred - target)
        logger.debug(
            f"    [LOSS] pred shape: {pred.shape}, target shape: {target.shape}"
        )
        logger.debug(
            f"    [LOSS] |pred - target|: min={diff.min().item():.6f}, "
            f"max={diff.max().item():.6f}, mean={diff.mean().item():.6f}"
        )

        # Apply exponent
        diff_exp = diff**self.loss_exp
        logger.debug(
            f"    [LOSS] |diff|^{self.loss_exp}: min={diff_exp.min().item():.6f}, "
            f"max={diff_exp.max().item():.6f}, mean={diff_exp.mean().item():.6f}"
        )

        # Compute final loss
        loss = torch.mean(diff_exp) / self.loss_exp
        logger.debug(f"    [LOSS] Final loss: {loss.item():.8f}")

        return loss

    def _normalize_features(self, features: Tensor) -> Tensor:
        """Apply layer normalization if normalize_reps is enabled.

        Args:
            features: Input features of shape [..., D]

        Returns:
            Normalized features (if enabled) or original features
        """
        if self.normalize_reps:
            return F.layer_norm(features, (features.size(-1),), eps=1e-6)
        return features

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
        logger.debug(
            f"  [TEACHER FORCING] Input features: B={B}, T+1={T_plus_1}, N={N}, D={D}"
        )
        T = min(T_plus_1 - 1, self.T_teacher)
        logger.debug(
            f"  [TEACHER FORCING] Using T={T} timesteps (T_teacher={self.T_teacher})"
        )

        # Context: all frames except the last one (frames 0 to T-1)
        # Shape: [B, T*N, D]
        context_features = features[:, :T, :, :].reshape(B, T * N, D)
        logger.debug(
            f"  [TEACHER FORCING] Context features reshaped to: {context_features.shape}"
        )
        logger.debug(
            f"  [TEACHER FORCING] Context stats: min={context_features.min().item():.6f}, "
            f"max={context_features.max().item():.6f}, mean={context_features.mean().item():.6f}"
        )

        # Apply layer norm to input if enabled (matching paper)
        if self.normalize_reps:
            context_features = F.layer_norm(
                context_features, (context_features.size(-1),), eps=1e-6
            )
            logger.debug(
                f"  [TEACHER FORCING] After LayerNorm: min={context_features.min().item():.6f}, "
                f"max={context_features.max().item():.6f}, mean={context_features.mean().item():.6f}"
            )

        # Actions/states for T steps
        context_actions = actions[:, :T, :]
        context_states = states[:, :T, :]
        context_extrinsics = extrinsics[:, :T, :] if extrinsics is not None else None
        logger.debug(
            f"  [TEACHER FORCING] Actions shape: {context_actions.shape}, "
            f"States shape: {context_states.shape}"
        )

        # Single forward pass - predicts all T frames
        logger.debug("  [TEACHER FORCING] >>> Calling predictor forward pass >>>")
        z_tf = self._step_predictor(
            context_features, context_actions, context_states, context_extrinsics
        )
        logger.debug(
            f"  [TEACHER FORCING] <<< Predictor output: shape={z_tf.shape}, "
            f"min={z_tf.min().item():.6f}, max={z_tf.max().item():.6f}, "
            f"mean={z_tf.mean().item():.6f}"
        )

        # Target: frames 1 to T (shifted by one from context)
        # Shape: [B, T*N, D]
        target = features[:, 1 : T + 1, :, :].reshape(B, T * N, D)
        logger.debug(f"  [TEACHER FORCING] Target features: shape={target.shape}")
        if self.normalize_reps:
            target = F.layer_norm(target, (target.size(-1),), eps=1e-6)
            logger.debug(
                f"  [TEACHER FORCING] Target after LayerNorm: "
                f"min={target.min().item():.6f}, max={target.max().item():.6f}"
            )

        logger.debug("  [TEACHER FORCING] Computing loss between prediction and target...")
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
        logger.debug(f"  [ROLLOUT] Input: B={B}, T+1={T_plus_1}, N={N}, D={D}")
        logger.debug(
            f"  [ROLLOUT] Context frames C={C}, Prediction steps T_pred={T_pred}"
        )

        # Normalize ground-truth features if enabled
        h = features
        if self.normalize_reps:
            h = F.layer_norm(features.reshape(B, -1, D), (D,), eps=1e-6).reshape(
                B, T_plus_1, N, D
            )
            logger.debug("  [ROLLOUT] Features normalized")

        # Step 1: Initialize with C ground-truth context frames
        # z_ar contains frames 0, 1, ..., C-1
        z_ar = h[:, :C, :, :].reshape(B, C * N, D)  # [B, C*N, D]
        logger.debug(f"  [ROLLOUT] Initial context z_ar shape: {z_ar.shape}")

        # Step 2: Autoregressive rollout for T_pred steps
        # First prediction is frame C, using actions/states 0 to C-1
        for step in range(T_pred):
            logger.debug(f"  [ROLLOUT] >>> Autoregressive step {step+1}/{T_pred} >>>")
            # Current prediction target is frame (C + step)
            # Use actions/states from 0 to (C + step - 1)
            num_action_steps = C + step
            ar_actions = actions[:, :num_action_steps, :]
            ar_states = states[:, :num_action_steps, :]
            ar_extrinsics = (
                extrinsics[:, :num_action_steps, :] if extrinsics is not None else None
            )
            logger.debug(
                f"  [ROLLOUT] Step {step+1}: Using {num_action_steps} action steps, "
                f"z_ar shape: {z_ar.shape}"
            )

            # Predict next frame using full context
            z_nxt = self._step_predictor(z_ar, ar_actions, ar_states, ar_extrinsics)
            logger.debug(
                f"  [ROLLOUT] Step {step+1}: Predictor output shape: {z_nxt.shape}"
            )
            # Extract only the last frame prediction
            z_nxt = z_nxt[:, -N:, :]  # [B, N, D]
            logger.debug(
                f"  [ROLLOUT] Step {step+1}: Extracted last frame: shape={z_nxt.shape}, "
                f"mean={z_nxt.mean().item():.6f}"
            )

            # Append to autoregressive context
            z_ar = torch.cat([z_ar, z_nxt], dim=1)  # [B, (C+step+1)*N, D]
            logger.debug(f"  [ROLLOUT] Step {step+1}: Updated z_ar shape: {z_ar.shape}")

        # Step 3: Compare autoregressive predictions with targets
        # z_ar[:, C*N:] contains predictions for frames C, C+1, ..., C+T_pred-1
        # Target: h[:, C:C+T_pred] contains ground-truth for same frames
        z_ar_pred = z_ar[:, C * N :]  # [B, T_pred*N, D]
        target = h[:, C : C + T_pred, :, :].reshape(B, T_pred * N, D)  # [B, T_pred*N, D]
        logger.debug(
            f"  [ROLLOUT] Final prediction shape: {z_ar_pred.shape}, "
            f"target shape: {target.shape}"
        )
        logger.debug("  [ROLLOUT] Computing rollout loss...")

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
        B, T_plus_1, N, D = features.shape
        C = self.context_frames  # Number of ground-truth context frames
        T_pred = min(T_plus_1 - C, self.T_rollout)  # Number of steps to predict

        # Normalize ground-truth features if enabled
        h = features
        if self.normalize_reps:
            h = F.layer_norm(features.reshape(B, -1, D), (D,), eps=1e-6).reshape(
                B, T_plus_1, N, D
            )

        # Initialize with C ground-truth context frames
        z_ar = h[:, :C, :, :].reshape(B, C * N, D)  # [B, C*N, D]

        # Store per-timestep predictions
        predictions_per_step: list[Tensor] = []

        # Autoregressive rollout for T_pred steps
        for step in range(T_pred):
            num_action_steps = C + step
            ar_actions = actions[:, :num_action_steps, :]
            ar_states = states[:, :num_action_steps, :]
            ar_extrinsics = (
                extrinsics[:, :num_action_steps, :] if extrinsics is not None else None
            )

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
            diff_exp = diff**self.loss_exp
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
        features = batch["features"]
        actions = batch["actions"]
        states = batch["states"]
        extrinsics = batch.get("extrinsics", None)

        logger.debug(f"\n{'='*60}")
        logger.debug(f"[SHARED STEP] Stage: {stage}")
        logger.debug(f"[SHARED STEP] Input features shape: {features.shape}")
        logger.debug(
            f"[SHARED STEP] Actions shape: {actions.shape}, States shape: {states.shape}"
        )

        # Reshape features if needed: [B, (T+1)*N, D] -> [B, T+1, N, D]
        if features.dim() == 3:
            B, total_tokens, D = features.shape
            T_plus_1 = total_tokens // self.patches_per_frame
            features = features.reshape(B, T_plus_1, self.patches_per_frame, D)
            logger.debug(f"[SHARED STEP] Reshaped features to: {features.shape}")

        logger.debug("\n--- COMPUTING TEACHER FORCING LOSS ---")
        # Compute losses
        loss_teacher = self._compute_teacher_forcing_loss(
            features, actions, states, extrinsics
        )
        logger.debug(f"[SHARED STEP] Teacher forcing loss: {loss_teacher.item():.8f}")

        logger.debug("\n--- COMPUTING ROLLOUT LOSS ---")
        loss_rollout = self._compute_rollout_loss(
            features, actions, states, extrinsics
        )
        logger.debug(f"[SHARED STEP] Rollout loss: {loss_rollout.item():.8f}")

        # Combined loss
        loss = (
            self.loss_weight_teacher * loss_teacher
            + self.loss_weight_rollout * loss_rollout
        )
        logger.debug("\n--- COMBINED LOSS ---")
        logger.debug(
            f"[SHARED STEP] weight_teacher={self.loss_weight_teacher}, "
            f"weight_rollout={self.loss_weight_rollout}"
        )
        logger.debug(
            f"[SHARED STEP] Combined loss = {self.loss_weight_teacher} * "
            f"{loss_teacher.item():.8f} + {self.loss_weight_rollout} * "
            f"{loss_rollout.item():.8f} = {loss.item():.8f}"
        )

        # Log metrics - using self.log which is expected to be provided by LightningModule
        self.log(f"{stage}/loss_teacher", loss_teacher, prog_bar=True, sync_dist=True)  # type: ignore[attr-defined]
        self.log(f"{stage}/loss_rollout", loss_rollout, prog_bar=True, sync_dist=True)  # type: ignore[attr-defined]
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)  # type: ignore[attr-defined]

        return loss
