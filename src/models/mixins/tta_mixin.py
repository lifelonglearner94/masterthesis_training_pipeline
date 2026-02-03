"""Test Time Adaptation (TTA) mixin for Lightning modules.

Provides TTA capabilities that can be mixed into LightningModule subclasses.
Implements per-clip TTA with LayerNorm-only adaptation.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TTAMixin:
    """Mixin providing Test Time Adaptation capabilities.

    This mixin expects the following attributes from the inheriting class:
        - model: nn.Module - The predictor model
        - context_frames: int - Number of ground-truth context frames
        - T_rollout: int - Number of rollout steps
        - patches_per_frame: int - Number of patches per frame
        - normalize_reps: bool - Whether to normalize representations
        - loss_exp: float - Loss exponent (typically 1.0 for L1)

    TTA Configuration (set via __init__ or config):
        - tta_enabled: bool - Whether TTA is enabled
        - tta_lr: float - Learning rate for TTA updates
        - tta_grad_clip: float - Gradient clipping norm
        - tta_reset_per_clip: bool - Whether to reset after each clip
        - tta_adaptation_horizon: int - Number of steps for adaptation signal
    """

    # Type hints for expected attributes from inheriting class
    model: nn.Module
    context_frames: int
    T_rollout: int
    patches_per_frame: int
    normalize_reps: bool
    loss_exp: float

    # TTA configuration attributes
    tta_enabled: bool
    tta_lr: float
    tta_grad_clip: float
    tta_reset_per_clip: bool
    tta_adaptation_horizon: int
    tta_optimizer_type: str
    tta_optimizer_betas: tuple[float, float]
    tta_mode: str  # "full_rollout" or "sequential"
    tta_num_adaptation_steps: int  # Number of TTA updates per clip

    # Internal state
    _tta_optimizer: torch.optim.Optimizer | None
    _tta_original_ln_state: dict[str, Tensor] | None
    _tta_clip_stats: dict[str, Any]

    def _init_tta(
        self,
        tta_enabled: bool = False,
        tta_lr: float = 1e-4,
        tta_grad_clip: float = 1.0,
        tta_reset_per_clip: bool = True,
        tta_adaptation_horizon: int = 1,
        tta_optimizer_type: str = "adam",
        tta_optimizer_betas: tuple[float, float] = (0.9, 0.999),
        tta_mode: str = "full_rollout",
        tta_num_adaptation_steps: int = 1,
    ) -> None:
        """Initialize TTA configuration.

        Call this in the __init__ of the inheriting class.

        Args:
            tta_enabled: Whether TTA is enabled
            tta_lr: Learning rate for TTA updates
            tta_grad_clip: Maximum gradient norm for clipping
            tta_reset_per_clip: Whether to reset model state per clip
            tta_adaptation_horizon: Number of steps for adaptation (deprecated, use T_rollout)
            tta_optimizer_type: Type of optimizer ("adam" or "adamw")
            tta_optimizer_betas: Beta parameters for Adam
            tta_mode: TTA mode - "full_rollout" (recommended) or "sequential"
                - "full_rollout": Predict full sequence, compute loss, adapt, then evaluate
                - "sequential": Step-by-step look-back adaptation (original TENT-style)
            tta_num_adaptation_steps: Number of TTA update iterations per clip (default: 1)
        """
        self.tta_enabled = tta_enabled
        self.tta_lr = tta_lr
        self.tta_grad_clip = tta_grad_clip
        self.tta_reset_per_clip = tta_reset_per_clip
        self.tta_adaptation_horizon = tta_adaptation_horizon
        self.tta_optimizer_type = tta_optimizer_type
        self.tta_optimizer_betas = tta_optimizer_betas
        self.tta_mode = tta_mode
        self.tta_num_adaptation_steps = tta_num_adaptation_steps

        # Internal state (initialized in on_test_start)
        self._tta_optimizer = None
        self._tta_original_ln_state = None
        self._tta_clip_stats = {}
        self._tta_all_clip_stats: list[dict[str, Any]] = []

    def _tta_configure_params(self) -> None:
        """Configure trainable parameters for TTA (LayerNorm only)."""
        if not self.tta_enabled:
            return

        # Freeze all parameters
        self.model.requires_grad_(False)

        # Unfreeze LayerNorm parameters
        num_ln_params = 0
        for module in self.model.modules():
            if isinstance(module, nn.LayerNorm):
                for param in module.parameters():
                    param.requires_grad = True
                    num_ln_params += param.numel()

        total_params = sum(p.numel() for p in self.model.parameters())

        import logging
        log = logging.getLogger(__name__)
        log.info(
            f"[TTA] Configured: {num_ln_params:,} / {total_params:,} params trainable "
            f"({100*num_ln_params/total_params:.2f}%)"
        )

    def _tta_get_ln_params(self) -> list[nn.Parameter]:
        """Get list of LayerNorm parameters."""
        params = []
        for module in self.model.modules():
            if isinstance(module, nn.LayerNorm):
                params.extend(p for p in module.parameters() if p.requires_grad)
        return params

    def _tta_save_ln_state(self) -> dict[str, Tensor]:
        """Save current LayerNorm state."""
        state = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.LayerNorm):
                state[f"{name}.weight"] = module.weight.data.clone()
                state[f"{name}.bias"] = module.bias.data.clone()
        return state

    def _tta_restore_ln_state(self, state: dict[str, Tensor]) -> None:
        """Restore LayerNorm state from saved dict."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.LayerNorm):
                weight_key = f"{name}.weight"
                bias_key = f"{name}.bias"
                if weight_key in state:
                    module.weight.data.copy_(state[weight_key])
                if bias_key in state:
                    module.bias.data.copy_(state[bias_key])

    def _tta_set_ln_train_mode(self, training: bool) -> dict[str, bool]:
        """Set LayerNorm modules to train/eval mode.

        Args:
            training: Whether to set train mode (True) or eval mode (False)

        Returns:
            Dict mapping module names to their previous training state
        """
        prev_states = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.LayerNorm):
                prev_states[name] = module.training
                module.train(training)
        return prev_states

    def _tta_restore_ln_train_mode(self, states: dict[str, bool]) -> None:
        """Restore LayerNorm modules to their previous training state.

        Args:
            states: Dict mapping module names to their training state
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.LayerNorm):
                if name in states:
                    module.train(states[name])

    def _tta_is_mps_device(self) -> bool:
        """Check if the model is on an MPS (Apple Metal) device.

        MPS has known bugs with LayerNorm backward that produce NaN gradients,
        so we need to fall back to CPU during TTA adaptation.
        """
        try:
            # Check first parameter's device
            for param in self.model.parameters():
                return param.device.type == "mps"
        except StopIteration:
            pass
        return False

    def _tta_create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for TTA."""
        ln_params = self._tta_get_ln_params()

        if self.tta_optimizer_type.lower() == "adamw":
            return torch.optim.AdamW(
                ln_params,
                lr=self.tta_lr,
                betas=self.tta_optimizer_betas,
            )
        else:
            return torch.optim.Adam(
                ln_params,
                lr=self.tta_lr,
                betas=self.tta_optimizer_betas,
            )

    def _tta_reset_for_clip(self) -> None:
        """Reset TTA state for a new clip."""
        if not self.tta_enabled:
            return

        # Restore original LayerNorm parameters
        if self._tta_original_ln_state is not None:
            self._tta_restore_ln_state(self._tta_original_ln_state)

        # Reset optimizer
        self._tta_optimizer = self._tta_create_optimizer()

        # Reset clip statistics
        self._tta_clip_stats = {
            "adaptation_losses": [],
            "num_adaptations": 0,
        }

    def _tta_adapt(self, pred: Tensor, target: Tensor) -> float:
        """Perform single TTA adaptation step.

        Args:
            pred: Predicted features (with gradients)
            target: Ground-truth features (will be detached)

        Returns:
            Loss value for logging
        """
        if self._tta_optimizer is None:
            raise RuntimeError("TTA optimizer not initialized. Call _tta_reset_for_clip first.")

        # L1 loss with detached target
        loss = F.l1_loss(pred, target.detach())

        # Backward + clip + step
        self._tta_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._tta_get_ln_params(),
            max_norm=self.tta_grad_clip,
        )
        self._tta_optimizer.step()

        # Track
        loss_val = loss.item()
        self._tta_clip_stats["adaptation_losses"].append(loss_val)
        self._tta_clip_stats["num_adaptations"] += 1

        return loss_val

    def _tta_full_rollout(
        self,
        features: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
        detach: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Perform full autoregressive rollout.

        Args:
            features: [B, T+1, N, D] - Full sequence features
            actions: [B, T, action_dim] - Actions
            states: [B, T, action_dim] - States
            extrinsics: [B, T, action_dim-1] - Optional
            detach: Whether to detach predictions (for evaluation)

        Returns:
            Tuple of:
                - predictions: [B, T_pred*N, D] - Autoregressive predictions
                - targets: [B, T_pred*N, D] - Ground-truth targets
        """
        B, T_plus_1, N, D = features.shape
        C = self.context_frames
        T_pred = min(T_plus_1 - C, self.T_rollout)

        # Normalize features if enabled
        h = features
        if self.normalize_reps:
            h = F.layer_norm(features.reshape(B, -1, D), (D,), eps=1e-6).reshape(
                B, T_plus_1, N, D
            )

        # Initialize context with ground-truth frames
        z_ar = h[:, :C, :, :].reshape(B, C * N, D)

        # Store predictions
        predictions_list: list[Tensor] = []

        # Autoregressive rollout
        for step in range(T_pred):
            num_action_steps = C + step

            step_actions = actions[:, :num_action_steps, :]
            step_states = states[:, :num_action_steps, :]
            step_extrinsics = (
                extrinsics[:, :num_action_steps, :] if extrinsics is not None else None
            )

            # Forward pass
            z_pred_full = self._step_predictor(  # type: ignore[attr-defined]
                z_ar, step_actions, step_states, step_extrinsics
            )
            z_pred = z_pred_full[:, -N:, :]  # Last frame only

            predictions_list.append(z_pred)

            # Update context (autoregressive) - detach to prevent graph explosion
            z_ar = torch.cat([z_ar, z_pred.detach()], dim=1)

        # Stack predictions
        predictions = torch.cat(predictions_list, dim=1)  # [B, T_pred*N, D]
        targets = h[:, C:C + T_pred, :, :].reshape(B, T_pred * N, D)

        if detach:
            predictions = predictions.detach()

        return predictions, targets

    def _tta_process_clip_full_rollout(
        self,
        features: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
        num_adaptation_steps: int = 1,
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        """Process a clip with full-rollout TTA adaptation.

        This is the recommended TTA approach:
        1. Perform full autoregressive rollout (z₁→z₇)
        2. Compute rollout loss against ground-truth
        3. Update LayerNorm parameters
        4. (Optionally repeat for multiple adaptation steps)
        5. Evaluate with adapted model

        Args:
            features: [B, T+1, N, D] - Full sequence features
            actions: [B, T, action_dim] - Actions
            states: [B, T, action_dim] - States
            extrinsics: [B, T, action_dim-1] - Optional
            num_adaptation_steps: Number of TTA update iterations (default: 1)

        Returns:
            Tuple of:
                - final_predictions: [B, T_pred*N, D] - Predictions after TTA
                - targets: [B, T_pred*N, D] - Ground-truth targets
                - stats: Dict with adaptation statistics
        """
        # Reset for new clip
        if self.tta_reset_per_clip:
            self._tta_reset_for_clip()

        # Store pre-adaptation loss for comparison
        with torch.no_grad():
            pre_adapt_pred, targets = self._tta_full_rollout(
                features, actions, states, extrinsics, detach=True
            )
            pre_adapt_loss = F.l1_loss(pre_adapt_pred, targets).item()

        self._tta_clip_stats["pre_adapt_loss"] = pre_adapt_loss

        # TTA adaptation loop
        # NOTE: We need torch.inference_mode(False) because PyTorch Lightning uses
        # inference_mode during testing, which cannot be overridden by set_grad_enabled

        # MPS fallback: Move to CPU if on MPS to avoid NaN gradients from LayerNorm backward bug
        use_mps_fallback = self._tta_is_mps_device()
        original_device = None
        if use_mps_fallback:
            import logging
            log = logging.getLogger(__name__)
            log.debug("[TTA] MPS detected - moving to CPU for adaptation to avoid NaN gradients")
            original_device = next(self.model.parameters()).device
            self.model.to("cpu")
            features = features.to("cpu")
            actions = actions.to("cpu")
            states = states.to("cpu")
            if extrinsics is not None:
                extrinsics = extrinsics.to("cpu")
            # Recreate optimizer with CPU parameters
            self._tta_optimizer = self._tta_create_optimizer()

        try:
            for adapt_step in range(num_adaptation_steps):
                # Set LayerNorms to train mode and enable gradients for adaptation
                prev_ln_states = self._tta_set_ln_train_mode(True)
                try:
                    with torch.inference_mode(False):
                        with torch.set_grad_enabled(True):
                            # Clone inputs to detach from Lightning's inference context
                            features_clone = features.clone()
                            actions_clone = actions.clone()
                            states_clone = states.clone()
                            extrinsics_clone = extrinsics.clone() if extrinsics is not None else None

                            predictions, targets = self._tta_full_rollout(
                                features_clone, actions_clone, states_clone, extrinsics_clone, detach=False
                            )

                        # Perform TTA update using full rollout loss
                        self._tta_adapt(predictions, targets)
                finally:
                    # Restore LayerNorm training states
                    self._tta_restore_ln_train_mode(prev_ln_states)
        finally:
            # Move back to MPS if we used the fallback
            if use_mps_fallback and original_device is not None:
                self.model.to(original_device)
                # Recreate optimizer with MPS parameters
                self._tta_optimizer = self._tta_create_optimizer()

        # Final evaluation with adapted model (no gradients)
        with torch.no_grad():
            final_predictions, targets = self._tta_full_rollout(
                features, actions, states, extrinsics, detach=True
            )
            post_adapt_loss = F.l1_loss(final_predictions, targets).item()

        self._tta_clip_stats["post_adapt_loss"] = post_adapt_loss
        self._tta_clip_stats["improvement"] = pre_adapt_loss - post_adapt_loss

        # Compute statistics
        stats = self._tta_get_clip_stats()
        self._tta_all_clip_stats.append(stats)

        return final_predictions, targets, stats

    def _tta_process_clip_sequential(
        self,
        features: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        """Process a clip with sequential TTA adaptation.

        Implements the look-back adaptation scheme where we:
        1. Predict next frame
        2. Wait for ground-truth
        3. Adapt
        4. Predict next frame with updated model

        Args:
            features: [B, T+1, N, D] - Full sequence features
            actions: [B, T, action_dim] - Actions
            states: [B, T, action_dim] - States
            extrinsics: [B, T, action_dim-1] - Optional

        Returns:
            Tuple of:
                - adapted_predictions: [B, T_pred*N, D] - Predictions with TTA
                - targets: [B, T_pred*N, D] - Ground-truth targets
                - stats: Dict with adaptation statistics
        """
        B, T_plus_1, N, D = features.shape
        C = self.context_frames
        T_pred = min(T_plus_1 - C, self.T_rollout)

        # Reset for new clip
        if self.tta_reset_per_clip:
            self._tta_reset_for_clip()

        # Normalize features if enabled
        h = features
        if self.normalize_reps:
            h = F.layer_norm(features.reshape(B, -1, D), (D,), eps=1e-6).reshape(
                B, T_plus_1, N, D
            )

        # Initialize context
        z_ar = h[:, :C, :, :].reshape(B, C * N, D)

        # Store predictions
        predictions_list: list[Tensor] = []
        prev_pred: Tensor | None = None

        # Sequential processing with TTA
        for step in range(T_pred):
            target_idx = C + step
            num_action_steps = C + step

            step_actions = actions[:, :num_action_steps, :]
            step_states = states[:, :num_action_steps, :]
            step_extrinsics = (
                extrinsics[:, :num_action_steps, :] if extrinsics is not None else None
            )

            # Look-back adaptation (skip first step - no previous prediction)
            # NOTE: We need torch.inference_mode(False) because PyTorch Lightning uses
            # inference_mode during testing, which cannot be overridden by set_grad_enabled
            if prev_pred is not None and step > 0:
                prev_target = h[:, target_idx - 1, :, :].clone()  # Ground-truth for prev prediction
                with torch.inference_mode(False):
                    self._tta_adapt(prev_pred.clone(), prev_target)

            # Forward pass with gradients enabled for next adaptation
            # Set LayerNorms to train mode for gradient flow
            prev_ln_states = self._tta_set_ln_train_mode(True)
            try:
                with torch.inference_mode(False):
                    with torch.set_grad_enabled(True):
                        # Clone inputs to detach from Lightning's inference context
                        z_ar_clone = z_ar.clone()
                        step_actions_clone = step_actions.clone()
                        step_states_clone = step_states.clone()
                        step_extrinsics_clone = step_extrinsics.clone() if step_extrinsics is not None else None

                        z_pred_full = self._step_predictor(  # type: ignore[attr-defined]
                            z_ar_clone, step_actions_clone, step_states_clone, step_extrinsics_clone
                        )
                        z_pred = z_pred_full[:, -N:, :]  # Last frame only
            finally:
                self._tta_restore_ln_train_mode(prev_ln_states)

            # Store for next adaptation
            prev_pred = z_pred
            predictions_list.append(z_pred.detach())

            # Update context (autoregressive)
            z_ar = torch.cat([z_ar, z_pred.detach()], dim=1)

        # Final adaptation with last ground-truth
        if prev_pred is not None and T_pred > 0:
            final_target = h[:, C + T_pred - 1, :, :].clone()
            with torch.inference_mode(False):
                self._tta_adapt(prev_pred.clone(), final_target)

        # Stack predictions and targets
        predictions = torch.cat(predictions_list, dim=1)  # [B, T_pred*N, D]
        targets = h[:, C:C + T_pred, :, :].reshape(B, T_pred * N, D)

        # Compute statistics
        stats = self._tta_get_clip_stats()
        self._tta_all_clip_stats.append(stats)

        return predictions, targets, stats

    def _tta_get_clip_stats(self) -> dict[str, Any]:
        """Get statistics for current clip."""
        losses = self._tta_clip_stats.get("adaptation_losses", [])

        # For full-rollout mode, use pre/post adapt losses if available
        pre_adapt = self._tta_clip_stats.get("pre_adapt_loss", 0.0)
        post_adapt = self._tta_clip_stats.get("post_adapt_loss", 0.0)
        improvement = self._tta_clip_stats.get("improvement", 0.0)

        if not losses and pre_adapt == 0.0:
            return {
                "num_adaptations": 0,
                "mean_loss": 0.0,
                "first_loss": 0.0,
                "last_loss": 0.0,
                "pre_adapt_loss": 0.0,
                "post_adapt_loss": 0.0,
                "improvement": 0.0,
            }

        return {
            "num_adaptations": len(losses),
            "mean_loss": sum(losses) / len(losses) if losses else 0.0,
            "first_loss": losses[0] if losses else pre_adapt,
            "last_loss": losses[-1] if losses else post_adapt,
            "pre_adapt_loss": pre_adapt,
            "post_adapt_loss": post_adapt,
            "improvement": improvement if improvement != 0.0 else (losses[0] - losses[-1] if len(losses) > 1 else 0.0),
        }

    def _tta_get_epoch_stats(self) -> dict[str, Any]:
        """Get aggregated TTA statistics for the epoch."""
        if not self._tta_all_clip_stats:
            return {}

        total_adaptations = sum(s["num_adaptations"] for s in self._tta_all_clip_stats)
        mean_improvement = sum(s["improvement"] for s in self._tta_all_clip_stats) / len(self._tta_all_clip_stats)

        # Handle both sequential and full-rollout modes
        if "pre_adapt_loss" in self._tta_all_clip_stats[0]:
            mean_pre_adapt = sum(s["pre_adapt_loss"] for s in self._tta_all_clip_stats) / len(self._tta_all_clip_stats)
            mean_post_adapt = sum(s["post_adapt_loss"] for s in self._tta_all_clip_stats) / len(self._tta_all_clip_stats)
        else:
            mean_pre_adapt = sum(s["first_loss"] for s in self._tta_all_clip_stats) / len(self._tta_all_clip_stats)
            mean_post_adapt = sum(s["last_loss"] for s in self._tta_all_clip_stats) / len(self._tta_all_clip_stats)

        return {
            "total_clips": len(self._tta_all_clip_stats),
            "total_adaptations": total_adaptations,
            "mean_improvement": mean_improvement,
            "mean_pre_adapt_loss": mean_pre_adapt,
            "mean_post_adapt_loss": mean_post_adapt,
        }
