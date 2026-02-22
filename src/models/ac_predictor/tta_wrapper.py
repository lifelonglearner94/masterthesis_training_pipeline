"""Test Time Adaptation (TTA) wrapper for AC Predictor.

Implements self-supervised TTA using jump prediction error as the adaptation signal.
Only LayerNorm parameters are updated during adaptation, all other weights are frozen.

Reference:
    - TENT: Fully Test-Time Adaptation by Entropy Minimization (Wang et al., 2021)
    - Adapted for regression (feature prediction) using L1 loss instead of entropy.

TTA Loop ("Jump-Based" Update):
    1. Jump-predict z_τ from (z₀, a₀) with RoPE(τ)
    2. Wait for ground-truth z_τ
    3. Compute L1 loss between prediction and ground-truth
    4. Update LayerNorm parameters
    5. Jump-predict next τ with updated model
"""

import logging
from enum import StrEnum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Type aliases for better readability
type LayerNormState = dict[str, Tensor]
type AdaptationStats = dict[str, float | int]

# Constants
DEFAULT_TTA_LR = 1e-4
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_PATCHES_PER_FRAME = 256
DEFAULT_JUMP_K = 3
LAYER_NORM_EPS = 1e-6


class OptimizerType(StrEnum):
    """Supported optimizer types for TTA."""

    ADAM = "adam"
    ADAMW = "adamw"


logger = logging.getLogger(__name__)


class JumpTTAAgent(nn.Module):
    """Test Time Adaptation agent using jump prediction error.

    This agent wraps a predictor model and implements online adaptation
    by updating only LayerNorm parameters based on single-step prediction errors.

    The adaptation uses a "look-back" scheme:
        - Store prediction from step t
        - When ground-truth at t+1 is available, compute loss and update
        - Make new prediction for t+1

    Attributes:
        model: The wrapped predictor model (with frozen weights except LayerNorm)
        optimizer: Optimizer for LayerNorm parameters
        prev_prediction: Buffer storing the last prediction for look-back update
        tta_lr: Learning rate for TTA updates
        grad_clip_norm: Maximum gradient norm for clipping
        original_state: Original model state for per-clip reset
    """

    def __init__(
        self,
        predictor_model: nn.Module,
        tta_lr: float = DEFAULT_TTA_LR,
        grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM,
        optimizer_type: str = OptimizerType.ADAM,
        optimizer_betas: tuple[float, float] = (0.9, 0.999),
        reset_per_clip: bool = True,
    ) -> None:
        """Initialize the TTA agent.

        Args:
            predictor_model: The predictor model to wrap
            tta_lr: Learning rate for adaptation (default: 1e-4)
            grad_clip_norm: Maximum gradient norm for clipping (default: 1.0)
            optimizer_type: Type of optimizer ("adam" or "adamw")
            optimizer_betas: Beta parameters for Adam optimizer
            reset_per_clip: Whether to reset model state after each clip
        """
        super().__init__()
        self.model = predictor_model
        self.tta_lr = tta_lr
        self.grad_clip_norm = grad_clip_norm
        self.optimizer_type = optimizer_type
        self.optimizer_betas = optimizer_betas
        self.reset_per_clip = reset_per_clip

        # Configure model: freeze all, unfreeze LayerNorm only
        self._configure_trainable_params()

        # Create optimizer for LayerNorm parameters
        self.optimizer = self._create_optimizer()

        # Store original state for per-clip reset
        if reset_per_clip:
            self._original_ln_state: LayerNormState | None = self._get_layernorm_state()
        else:
            self._original_ln_state = None

        # Buffer to store the prediction from the previous timestep
        self.prev_prediction: Tensor | None = None
        self.prev_target_frame_idx: int | None = None

        # Tracking metrics
        self._adaptation_losses: list[float] = []
        self._num_adaptations = 0

    def _configure_trainable_params(self) -> None:
        """Freeze all parameters except LayerNorm weights and biases."""
        # First freeze everything
        self.model.requires_grad_(False)

        # Then selectively unfreeze LayerNorm parameters
        num_ln_params = 0
        for module in self.model.modules():
            if isinstance(module, nn.LayerNorm):
                for param in module.parameters():
                    param.requires_grad = True
                    num_ln_params += param.numel()

        # Log configuration
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        trainable_percentage = 100 * trainable_params / total_params if total_params > 0 else 0.0

        logger.info(
            "TTA model configured",
            extra={
                "trainable_params": trainable_params,
                "total_params": total_params,
                "trainable_percentage": trainable_percentage,
            },
        )

    def _get_layernorm_params(self) -> list[nn.Parameter]:
        """Get list of LayerNorm parameters for optimizer.

        Returns:
            List of LayerNorm parameters.
        """
        params = []
        for module in self.model.modules():
            if isinstance(module, nn.LayerNorm):
                params.extend(module.parameters())
        return params

    def _get_layernorm_state(self) -> LayerNormState:
        """Get current state of LayerNorm parameters.

        Returns:
            Dictionary mapping parameter names to their tensor values.
        """
        state: LayerNormState = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.LayerNorm):
                state[f"{name}.weight"] = module.weight.data.clone()
                state[f"{name}.bias"] = module.bias.data.clone()
        return state

    def _restore_layernorm_state(self, state: LayerNormState) -> None:
        """Restore LayerNorm parameters from saved state.

        Args:
            state: Dictionary of LayerNorm parameters to restore.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.LayerNorm):
                weight_key = f"{name}.weight"
                bias_key = f"{name}.bias"
                if weight_key in state:
                    module.weight.data.copy_(state[weight_key])
                if bias_key in state:
                    module.bias.data.copy_(state[bias_key])

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for LayerNorm parameters.

        Returns:
            Configured optimizer instance.
        """
        ln_params = self._get_layernorm_params()

        if self.optimizer_type.lower() == OptimizerType.ADAMW:
            return torch.optim.AdamW(
                ln_params,
                lr=self.tta_lr,
                betas=self.optimizer_betas,
            )
        return torch.optim.Adam(
            ln_params,
            lr=self.tta_lr,
            betas=self.optimizer_betas,
        )

    def reset_for_new_clip(self) -> None:
        """Reset model state for a new clip (per-clip TTA).

        This restores LayerNorm parameters to their original (checkpoint) values
        and clears the prediction buffer.
        """
        # Clear prediction buffer
        self.prev_prediction = None
        self.prev_target_frame_idx = None

        # Restore original LayerNorm state
        if self._original_ln_state is not None:
            self._restore_layernorm_state(self._original_ln_state)

        # Reset optimizer state (momentum etc.)
        self.optimizer = self._create_optimizer()

        # Clear tracking
        self._adaptation_losses = []
        self._num_adaptations = 0

    def adapt_step(
        self,
        prediction: Tensor,
        ground_truth: Tensor,
    ) -> float:
        """Perform a single adaptation step.

        Computes L1 loss between prediction and ground-truth, then updates
        LayerNorm parameters with gradient clipping.

        Args:
            prediction: Predicted features [B, N, D] (should have grad enabled)
            ground_truth: Ground-truth features [B, N, D]

        Returns:
            The adaptation loss value (for logging).
        """
        # Compute L1 loss with detached ground-truth (stop gradient into encoder)
        # Note: TTA wrapper always uses L1 loss for adaptation signal stability
        loss = F.l1_loss(prediction, ground_truth.detach())

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            self._get_layernorm_params(),
            max_norm=self.grad_clip_norm,
        )

        # Update parameters
        self.optimizer.step()

        # Track metrics
        loss_value = loss.item()
        self._adaptation_losses.append(loss_value)
        self._num_adaptations += 1

        return loss_value

    def step(
        self,
        z_context: Tensor,
        actions: Tensor,
        states: Tensor,
        ground_truth_next: Tensor | None = None,
        extrinsics: Tensor | None = None,
        target_timestep: int | None = None,
    ) -> Tensor:
        """Perform TTA step: adapt from previous prediction, then predict next.

        This implements the jump-based update scheme:
        1. If we have a previous prediction and ground-truth is available, adapt
        2. Make a new jump prediction with (potentially) updated weights
        3. Store the prediction for the next adaptation step

        Args:
            z_context: Current context features [B, N, D]
            actions: Actions [B, 1, action_dim]
            states: States [B, 1, action_dim]
            ground_truth_next: Ground-truth features for validation [B, N, D].
                If provided along with prev_prediction, triggers adaptation.
            extrinsics: Optional extrinsics [B, 1, action_dim-1]
            target_timestep: Target frame index τ for jump prediction (0-indexed).

        Returns:
            Predicted features [B, N, D] (detached for downstream use).
        """
        # --- Phase A: Look-Back Adaptation ---
        if self.prev_prediction is not None and ground_truth_next is not None:
            # We now have ground-truth for the previous prediction
            self.adapt_step(self.prev_prediction, ground_truth_next)

        # --- Phase B: Forward Inference ---
        # Enable gradients so we can backpropagate in the NEXT adaptation step
        with torch.set_grad_enabled(True):
            # Forward pass through predictor with optional target_timestep for jump
            z_pred_all = self.model(z_context, actions, states, extrinsics, target_timestep=target_timestep)

            # Extract only the last frame prediction (single-step jump prediction)
            N = z_context.shape[1]
            # For single-step, extract last N tokens
            z_pred_next = z_pred_all[:, -N:, :]

        # Store prediction for the next update cycle
        self.prev_prediction = z_pred_next

        # Return detached prediction for downstream tasks (avoid graph retention)
        return z_pred_next.detach()

    def get_adaptation_stats(self) -> AdaptationStats:
        """Get statistics about adaptation performed on this clip.

        Returns:
            Dictionary containing:
                - num_adaptations: Number of adaptation steps performed
                - mean_loss: Average loss across all adaptations
                - first_loss: Loss of the first adaptation step
                - last_loss: Loss of the last adaptation step
                - improvement: Difference between first and last loss
        """
        if not self._adaptation_losses:
            return {
                "num_adaptations": 0,
                "mean_loss": 0.0,
                "first_loss": 0.0,
                "last_loss": 0.0,
            }

        return {
            "num_adaptations": self._num_adaptations,
            "mean_loss": sum(self._adaptation_losses) / len(self._adaptation_losses),
            "first_loss": self._adaptation_losses[0],
            "last_loss": self._adaptation_losses[-1],
            "improvement": self._adaptation_losses[0] - self._adaptation_losses[-1]
            if len(self._adaptation_losses) > 1
            else 0.0,
        }

    def forward(
        self,
        features: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
    ) -> Tensor:
        """Standard forward pass (without TTA adaptation).

        Use this for evaluation after adaptation or when TTA is disabled.

        Args:
            features: Input features [B, C*N, D]
            actions: Actions [B, C, action_dim]
            states: States [B, C, action_dim]
            extrinsics: Optional extrinsics [B, C, action_dim-1]

        Returns:
            Model predictions.
        """
        with torch.no_grad():
            return self.model(features, actions, states, extrinsics)


class SequentialTTAProcessor:
    """Process a full sequence with per-target jump TTA adaptation.

    This processor handles a complete clip by iterating through jump targets,
    adapting at each step using the "look-back" scheme with jump predictions.

    For a sequence with T+1 frames (0 to T) and jump_k targets:
        - Frame 0: Initial context (z₀)
        - For each τ in {T+1-k, ..., T}: Jump-predict z_τ from z₀, then adapt
    """

    def __init__(
        self,
        tta_agent: JumpTTAAgent,
        jump_k: int = 3,
        patches_per_frame: int = DEFAULT_PATCHES_PER_FRAME,
        normalize_reps: bool = True,
    ) -> None:
        """Initialize the sequential processor.

        Args:
            tta_agent: The TTA agent wrapping the predictor
            jump_k: Number of candidate jump targets
            patches_per_frame: Number of patches (tokens) per frame
            normalize_reps: Whether to apply layer normalization to features
        """
        self.tta_agent = tta_agent
        self.jump_k = jump_k
        self.patches_per_frame = patches_per_frame
        self.normalize_reps = normalize_reps

    def process_clip(
        self,
        features: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float | int]]:
        """Process a complete clip with jump-based TTA adaptation.

        Iterates through jump targets, making predictions and adapting
        at each target using the look-back scheme.

        Args:
            features: [B, T+1, N, D] - Full sequence features
            actions: [B, T, action_dim] - Actions for T steps
            states: [B, T, action_dim] - States for T steps
            extrinsics: [B, T, action_dim-1] - Optional extrinsics

        Returns:
            Tuple of:
                - predictions: [B, k, N, D] - Jump predictions for all τ
                - stats: Dictionary with adaptation statistics
        """
        B, T_plus_1, N, D = features.shape
        T = T_plus_1 - 1
        tau_min = T + 1 - self.jump_k
        tau_max = T

        # Reset TTA agent for new clip
        if self.tta_agent.reset_per_clip:
            self.tta_agent.reset_for_new_clip()

        # Normalize features if enabled
        h = features
        if self.normalize_reps:
            h = F.layer_norm(
                features.reshape(B, -1, D), (D,), eps=LAYER_NORM_EPS
            ).reshape(B, T_plus_1, N, D)

        # Initial state: first frame
        z_input = h[:, :1, :, :].reshape(B, N, D)

        # Store all predictions
        all_predictions: list[Tensor] = []

        # Process each jump target
        for i, tau in enumerate(range(tau_min, tau_max + 1)):
            # Single action/state for initial step
            step_actions = actions[:, :1, :]
            step_states = states[:, :1, :]
            step_extrinsics = (
                extrinsics[:, :1, :]
                if extrinsics is not None
                else None
            )

            # Ground-truth for this target
            gt_tau = h[:, tau, :, :]  # [B, N, D]

            # Make prediction with TTA (adapts from previous if available)
            z_pred = self.tta_agent.step(
                z_context=z_input,
                actions=step_actions,
                states=step_states,
                ground_truth_next=gt_tau if i > 0 else None,
                extrinsics=step_extrinsics,
            )

            all_predictions.append(z_pred)

        # Final adaptation with last ground-truth
        if self.jump_k > 0:
            final_gt = h[:, tau_max, :, :]
            if self.tta_agent.prev_prediction is not None:
                self.tta_agent.adapt_step(
                    self.tta_agent.prev_prediction,
                    final_gt,
                )

        # Stack predictions
        predictions = torch.stack(all_predictions, dim=1)  # [B, k, N, D]

        # Get adaptation statistics
        stats = self.tta_agent.get_adaptation_stats()

        return predictions, stats
