"""Loss computation mixin for AC predictor models.

Provides reusable loss computation methods for both the ViT AC Predictor
and baseline models, ensuring scientific comparability.

Loss formulations:
- Teacher-Forcing: L_tf — single forward pass with full context, predicts next frame.
- Jump Prediction: L_jump — single forward pass from z₀ + a₀ directly to z_τ,
  where τ is sampled uniformly from the last k frames. RoPE encodes the target
  position so the model knows which frame to predict.
- Combined: L = (1 - λ) * L_tf + λ * L_jump  (λ ramps up via curriculum)
"""

import logging

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# Valid loss type names
VALID_LOSS_TYPES = ("l1", "l2", "huber")
DEFAULT_HUBER_DELTA = 1.0


class ACPredictorLossMixin:
    """Mixin providing loss computation methods for AC predictors.

    This mixin expects the following attributes to be set by the inheriting class:
        - normalize_reps: bool - Whether to apply layer normalization
        - loss_type: str - Loss function type: "l1", "l2", or "huber"
        - huber_delta: float - Delta parameter for Huber loss (only used when loss_type="huber")
        - T_teacher: int - Number of teacher-forcing steps
        - jump_k: int - Number of candidate target frames for jump prediction
        - patches_per_frame: int - Number of patches (tokens) per frame (N = H*W)
        - num_timesteps: int - Total encoded timesteps (T)

    The inheriting class must also implement:
        - _step_predictor(z, actions, states, extrinsics, target_timestep=None) -> Tensor
            Forward pass through the model with optional normalization.
            When target_timestep is provided, RoPE positions are overridden
            for jump prediction.
    """

    # Type hints for expected attributes (set by inheriting class)
    normalize_reps: bool
    loss_type: str
    huber_delta: float
    T_teacher: int
    jump_k: int
    patches_per_frame: int
    num_timesteps: int
    loss_weight_teacher: float
    loss_weight_jump: float

    def _compute_loss(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute loss with configurable loss type.

        Supports:
            - "l1": L1 / Mean Absolute Error
            - "l2": L2 / Mean Squared Error
            - "huber": Huber loss (smooth L1), controlled by self.huber_delta

        Args:
            pred: Predicted features
            target: Target features

        Returns:
            Scalar loss tensor
        """
        logger.debug(
            f"    [LOSS] pred shape: {pred.shape}, target shape: {target.shape}"
        )

        if self.loss_type == "l1":
            loss = F.l1_loss(pred, target)
        elif self.loss_type == "l2":
            loss = F.mse_loss(pred, target)
        elif self.loss_type == "huber":
            loss = F.huber_loss(pred, target, delta=self.huber_delta)
        else:
            raise ValueError(
                f"Unknown loss_type '{self.loss_type}'. "
                f"Must be one of {VALID_LOSS_TYPES}."
            )

        logger.debug(f"    [LOSS] {self.loss_type} loss: {loss.item():.8f}")
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
            loss = loss_fn(z_tf, h[:, tokens_per_frame:])

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

        # Apply layer norm to input if enabled (matching paper)
        if self.normalize_reps:
            context_features = F.layer_norm(
                context_features, (context_features.size(-1),), eps=1e-6
            )

        # Actions/states for T steps
        context_actions = actions[:, :T, :]
        context_states = states[:, :T, :]
        context_extrinsics = extrinsics[:, :T, :] if extrinsics is not None else None

        # Single forward pass - predicts all T frames (no target_timestep override)
        z_tf = self._step_predictor(
            context_features, context_actions, context_states, context_extrinsics
        )

        # Target: frames 1 to T (shifted by one from context)
        # Shape: [B, T*N, D]
        target = features[:, 1 : T + 1, :, :].reshape(B, T * N, D)
        if self.normalize_reps:
            target = F.layer_norm(target, (target.size(-1),), eps=1e-6)

        return self._compute_loss(z_tf, target)

    def _compute_jump_loss(
        self,
        features: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
    ) -> Tensor:
        """Compute stochastic jump prediction loss.

        Instead of jump prediction, the model directly predicts a
        randomly chosen future frame τ from the initial state z₀ and action a₀.
        The target frame position is communicated via 3D RoPE conditioning.

        Samples τ uniformly from the last ``jump_k`` frames of the sequence:
            τ ∈ {T+1-k, T+1-k+1, ..., T}  (indices into features tensor)

        For T=8, k=3: τ ∈ {6, 7, 8}

        The loss is:
            L_jump = E_τ [ || P(z₀, a₀, p_τ) - z_τ ||₁ ]

        Args:
            features: [B, T+1, N, D] - Features for T+1 frames
            actions: [B, T, action_dim] - Actions for T steps
            states: [B, T, action_dim] - States for T steps
            extrinsics: [B, T, action_dim-1] - Optional extrinsics

        Returns:
            Scalar loss tensor
        """
        B, T_plus_1, N, D = features.shape
        T = T_plus_1 - 1  # Number of encoded timesteps (e.g. 8)
        k = min(self.jump_k, T)  # Clamp k to available frames

        # Sample target timestep τ uniformly from last k frames
        # τ ranges from (T+1-k) to T inclusive (0-indexed into features tensor)
        tau_min = T + 1 - k  # e.g. 6 for T=8, k=3
        tau = torch.randint(tau_min, T + 1, (1,), device=features.device).item()
        logger.debug(
            f"  [JUMP] Sampled τ={tau} from [{tau_min}, {T}], k={k}"
        )

        # Normalize features if enabled
        h = features
        if self.normalize_reps:
            h = F.layer_norm(features.reshape(B, -1, D), (D,), eps=1e-6).reshape(
                B, T_plus_1, N, D
            )

        # Input: z₀ (frame 0) with a₀, s₀
        z_0 = h[:, 0:1, :, :].reshape(B, 1 * N, D)  # [B, N, D]
        a_0 = actions[:, 0:1, :]  # [B, 1, action_dim]
        s_0 = states[:, 0:1, :]  # [B, 1, action_dim]
        ext_0 = extrinsics[:, 0:1, :] if extrinsics is not None else None

        # Forward pass with target_timestep conditioning via RoPE
        z_pred = self._step_predictor(z_0, a_0, s_0, ext_0, target_timestep=tau)

        # Target: ground-truth features at frame τ
        target = h[:, tau, :, :].reshape(B, N, D)  # [B, N, D]

        # z_pred has shape [B, N, D] (single frame output)
        return self._compute_loss(z_pred, target)

    def _compute_jump_loss_per_timestep(
        self,
        features: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """Compute jump prediction loss with per-timestep breakdown for analysis.

        Evaluates the model at each possible target frame in the jump range
        {T+1-k, ..., T} and returns per-timestep and per-sample losses.

        Args:
            features: [B, T+1, N, D] - Features for T+1 frames
            actions: [B, T, action_dim] - Actions for T steps
            states: [B, T, action_dim] - States for T steps
            extrinsics: [B, T, action_dim-1] - Optional extrinsics

        Returns:
            Tuple of:
                - total_loss: Scalar loss tensor (average over all target frames)
                - per_timestep_losses: List of [B] tensors, one per target frame
                - per_sample_losses: List of [B] tensors with per-sample average loss
        """
        B, T_plus_1, N, D = features.shape
        T = T_plus_1 - 1
        k = min(self.jump_k, T)
        tau_min = T + 1 - k

        # Normalize features if enabled
        h = features
        if self.normalize_reps:
            h = F.layer_norm(features.reshape(B, -1, D), (D,), eps=1e-6).reshape(
                B, T_plus_1, N, D
            )

        # Input: z₀ with a₀, s₀
        z_0 = h[:, 0:1, :, :].reshape(B, 1 * N, D)
        a_0 = actions[:, 0:1, :]
        s_0 = states[:, 0:1, :]
        ext_0 = extrinsics[:, 0:1, :] if extrinsics is not None else None

        per_timestep_losses: list[Tensor] = []
        per_sample_total_losses = torch.zeros(B, device=features.device)

        for tau in range(tau_min, T + 1):
            # Forward pass with target_timestep conditioning
            z_pred = self._step_predictor(z_0, a_0, s_0, ext_0, target_timestep=tau)

            # Target: ground-truth at frame τ
            target_frame = h[:, tau, :, :]  # [B, N, D]

            # Per-sample loss for this target frame
            if self.loss_type == "l1":
                per_sample_loss = torch.abs(z_pred - target_frame).mean(dim=(1, 2))
            elif self.loss_type == "l2":
                per_sample_loss = ((z_pred - target_frame) ** 2).mean(dim=(1, 2))
            elif self.loss_type == "huber":
                per_sample_loss = F.huber_loss(
                    z_pred, target_frame, delta=self.huber_delta, reduction="none"
                ).mean(dim=(1, 2))
            else:
                raise ValueError(f"Unknown loss_type '{self.loss_type}'")

            per_timestep_losses.append(per_sample_loss)
            per_sample_total_losses += per_sample_loss

        # Average across target frames
        per_sample_total_losses = per_sample_total_losses / k
        total_loss = per_sample_total_losses.mean()

        return total_loss, per_timestep_losses, [per_sample_total_losses]

    def _shared_step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        """Shared step for training and validation.

        Computes combined teacher-forcing + jump prediction loss:
            L = (1 - λ) * L_teacher + λ * L_jump
        where λ = loss_weight_jump (controlled by curriculum schedule).

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

        # Reshape features if needed: [B, (T+1)*N, D] -> [B, T+1, N, D]
        if features.dim() == 3:
            B, total_tokens, D = features.shape
            T_plus_1 = total_tokens // self.patches_per_frame
            features = features.reshape(B, T_plus_1, self.patches_per_frame, D)
            logger.debug(f"[SHARED STEP] Reshaped features to: {features.shape}")

        logger.debug("\n--- COMPUTING TEACHER FORCING LOSS ---")
        loss_teacher = self._compute_teacher_forcing_loss(
            features, actions, states, extrinsics
        )
        logger.debug(f"[SHARED STEP] Teacher forcing loss: {loss_teacher.item():.8f}")

        logger.debug("\n--- COMPUTING JUMP PREDICTION LOSS ---")
        loss_jump = self._compute_jump_loss(
            features, actions, states, extrinsics
        )
        logger.debug(f"[SHARED STEP] Jump prediction loss: {loss_jump.item():.8f}")

        # Combined loss
        loss = (
            self.loss_weight_teacher * loss_teacher
            + self.loss_weight_jump * loss_jump
        )
        logger.debug(
            f"[SHARED STEP] Combined loss = {self.loss_weight_teacher} * "
            f"{loss_teacher.item():.8f} + {self.loss_weight_jump} * "
            f"{loss_jump.item():.8f} = {loss.item():.8f}"
        )

        # Log metrics
        self.log(f"{stage}/loss_teacher", loss_teacher, prog_bar=True, sync_dist=True)  # type: ignore[attr-defined]
        self.log(f"{stage}/loss_jump", loss_jump, prog_bar=True, sync_dist=True)  # type: ignore[attr-defined]
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)  # type: ignore[attr-defined]

        return loss
