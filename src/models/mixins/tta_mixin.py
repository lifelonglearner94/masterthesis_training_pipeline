"""Test Time Adaptation (TTA) mixin for Lightning modules.

Provides TTA capabilities that can be mixed into LightningModule subclasses.
Implements per-clip TTA with LayerNorm-only adaptation.
"""

import logging
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MPSFallbackResult(NamedTuple):
    """Result of MPS device fallback check for TTA."""

    use_fallback: bool
    original_device: torch.device | None
    features: Tensor
    actions: Tensor
    states: Tensor
    extrinsics: Tensor | None

logger = logging.getLogger(__name__)


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
    tta_mode: str
    tta_num_adaptation_steps: int
    tta_adapt_layers: str

    # Internal state
    _tta_optimizer: torch.optim.Optimizer | None
    _tta_original_ln_state: dict[str, Tensor] | None
    _tta_clip_stats: dict[str, Any]
    _tta_all_clip_stats: list[dict[str, Any]]

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
        tta_adapt_layers: str = "layernorm",
    ) -> None:
        """Initialize TTA configuration.

        Call this in the __init__ of the inheriting class.

        Args:
            tta_enabled: Whether TTA is enabled.
            tta_lr: Learning rate for TTA updates.
            tta_grad_clip: Maximum gradient norm for clipping.
            tta_reset_per_clip: Whether to reset model state per clip.
            tta_adaptation_horizon: Number of steps for adaptation (deprecated, use T_rollout).
            tta_optimizer_type: Type of optimizer ("adam" or "adamw").
            tta_optimizer_betas: Beta parameters for Adam.
            tta_mode: TTA mode - "full_rollout" (recommended) or "sequential".
            tta_num_adaptation_steps: Number of TTA update iterations per clip.
            tta_adapt_layers: Which layers to adapt during TTA. Options:
                - "layernorm": Only LayerNorm γ and β (default, ~0.1% params).
                - "layernorm+attn_proj": LayerNorm + attention output projections (~1-2% params).
                - "layernorm+bias": LayerNorm + all bias terms (~0.5% params).
                - "layernorm+mlp_out": LayerNorm + MLP output layers (~3-5% params).
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
        self.tta_adapt_layers = tta_adapt_layers

        self._tta_optimizer = None
        self._tta_original_ln_state = None
        self._tta_clip_stats = {}
        self._tta_all_clip_stats = []

    def _tta_configure_params(self) -> None:
        """Configure trainable parameters for TTA based on tta_adapt_layers setting."""
        if not self.tta_enabled:
            return

        self.model.requires_grad_(False)

        num_adapted_params = 0
        adapted_layer_types = []

        num_adapted_params += self._enable_layernorm_params(adapted_layer_types)

        if "attn_proj" in self.tta_adapt_layers:
            num_adapted_params += self._enable_attn_proj_params(adapted_layer_types)

        if "bias" in self.tta_adapt_layers and "attn" not in self.tta_adapt_layers:
            num_adapted_params += self._enable_bias_params(adapted_layer_types)

        if "mlp_out" in self.tta_adapt_layers:
            num_adapted_params += self._enable_mlp_out_params(adapted_layer_types)

        total_params = sum(p.numel() for p in self.model.parameters())

        logger.info(
            f"[TTA] Configured: {num_adapted_params:,} / {total_params:,} params trainable "
            f"({100 * num_adapted_params / total_params:.2f}%) - "
            f"Layers: {', '.join(adapted_layer_types)}"
        )

    def _enable_layernorm_params(
        self,
        adapted_layer_types: list[str],
    ) -> int:
        """Enable LayerNorm parameters for TTA."""
        count = 0
        for module in self.model.modules():
            if isinstance(module, nn.LayerNorm):
                for param in module.parameters():
                    param.requires_grad = True
                    count += param.numel()
        adapted_layer_types.append("LayerNorm")
        return count

    def _enable_attn_proj_params(
        self,
        adapted_layer_types: list[str],
    ) -> int:
        """Enable attention projection parameters for TTA."""
        count = 0
        for name, module in self.model.named_modules():
            proj = self._extract_attn_proj(name, module)
            if proj is not None and isinstance(proj, nn.Linear):
                for param in proj.parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                        count += param.numel()
        adapted_layer_types.append("Attention.proj")
        return count

    def _extract_attn_proj(self, name: str, module: nn.Module) -> nn.Module | None:
        """Extract attention projection module from name or module."""
        if name.endswith(".attn.proj") or (name.endswith(".attn") and hasattr(module, "proj")):
            return module.proj if hasattr(module, "proj") else module
        if ".proj" in name and ".attn" in name and isinstance(module, nn.Linear):
            return module
        return None

    def _enable_bias_params(
        self,
        adapted_layer_types: list[str],
    ) -> int:
        """Enable all bias parameters for TTA."""
        count = 0
        for name, param in self.model.named_parameters():
            if "bias" in name and not param.requires_grad:
                param.requires_grad = True
                count += param.numel()
        adapted_layer_types.append("All biases")
        return count

    def _enable_mlp_out_params(
        self,
        adapted_layer_types: list[str],
    ) -> int:
        """Enable MLP output layer parameters for TTA."""
        count = 0
        for name, module in self.model.named_modules():
            if name.endswith(".mlp.fc2") or name.endswith(".mlp.fc3"):
                if isinstance(module, nn.Linear):
                    for param in module.parameters():
                        if not param.requires_grad:
                            param.requires_grad = True
                            count += param.numel()
        adapted_layer_types.append("MLP output")
        return count

    def _tta_get_ln_params(self) -> list[nn.Parameter]:
        """Get list of all trainable TTA parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def _tta_save_ln_state(self) -> dict[str, Tensor]:
        """Save current state of all adapted parameters."""
        return {name: param.data.clone() for name, param in self.model.named_parameters() if param.requires_grad}

    def _tta_restore_ln_state(self, state: dict[str, Tensor]) -> None:
        """Restore adapted parameters from saved dict."""
        for name, param in self.model.named_parameters():
            if name in state:
                param.data.copy_(state[name])

    def _tta_set_ln_train_mode(self, training: bool) -> dict[str, bool]:
        """Set LayerNorm modules to train/eval mode.

        Args:
            training: Whether to set train mode (True) or eval mode (False).

        Returns:
            Dict mapping module names to their previous training state.
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
            states: Dict mapping module names to their training state.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.LayerNorm) and name in states:
                module.train(states[name])

    def _tta_is_mps_device(self) -> bool:
        """Check if the model is on an MPS (Apple Metal) device.

        MPS has known bugs with LayerNorm backward that produce NaN gradients,
        so we need to fall back to CPU during TTA adaptation.
        """
        try:
            for param in self.model.parameters():
                return param.device.type == "mps"
        except StopIteration:
            pass
        return False

    def _tta_make_model_params_non_inference(self) -> None:
        """Clone ALL model parameters to escape PyTorch Lightning's inference mode.

        When Lightning runs test_step in inference_mode, model parameters become
        "inference tensors" that cannot be used for backward passes. This method
        saves and reloads the model state dict to create fresh non-inference tensors.
        Only LayerNorm parameters are made trainable for TTA.
        """
        state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}

        inference_mode_was_enabled = torch.is_inference_mode_enabled()

        if inference_mode_was_enabled:
            with torch._C._DisableFuncTorch():
                torch._C._set_grad_enabled(True)

        self.model.load_state_dict(state_dict, strict=True)
        self._clone_non_buffer_tensors()
        self._set_layernorm_trainable()

    def _clone_non_buffer_tensors(self) -> None:
        """Clone non-parameter, non-buffer tensors to escape inference mode."""
        for name, module in self.model.named_modules():
            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue
                self._clone_tensor_if_needed(module, attr_name)

    def _clone_tensor_if_needed(self, module: nn.Module, attr_name: str) -> None:
        """Clone a tensor attribute if it's not a buffer or parameter."""
        try:
            attr = getattr(module, attr_name)
            if isinstance(attr, torch.Tensor) and not isinstance(attr, nn.Parameter):
                if attr_name not in module._buffers:
                    setattr(module, attr_name, attr.clone())
        except (AttributeError, RuntimeError):
            pass

    def _set_layernorm_trainable(self) -> None:
        """Set LayerNorm parameters as trainable."""
        self.model.requires_grad_(False)
        for module in self.model.modules():
            if isinstance(module, nn.LayerNorm):
                for param in module.parameters():
                    param.requires_grad = True

    def _tta_create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for TTA."""
        ln_params = self._tta_get_ln_params()

        if self.tta_optimizer_type.lower() == "adamw":
            return torch.optim.AdamW(
                ln_params,
                lr=self.tta_lr,
                betas=self.tta_optimizer_betas,
            )
        return torch.optim.Adam(
            ln_params,
            lr=self.tta_lr,
            betas=self.tta_optimizer_betas,
        )

    def _tta_reset_for_clip(self) -> None:
        """Reset TTA state for a new clip."""
        if not self.tta_enabled:
            return

        if self._tta_original_ln_state is not None:
            self._tta_restore_ln_state(self._tta_original_ln_state)

        self._tta_optimizer = self._tta_create_optimizer()
        self._tta_clip_stats = {
            "adaptation_losses": [],
            "num_adaptations": 0,
        }

    def _tta_adapt(self, pred: Tensor, target: Tensor) -> float:
        """Perform single TTA adaptation step.

        Args:
            pred: Predicted features (with gradients).
            target: Ground-truth features (will be detached).

        Returns:
            Loss value for logging.
        """
        if self._tta_optimizer is None:
            msg = "TTA optimizer not initialized. Call _tta_reset_for_clip first."
            raise RuntimeError(msg)

        ln_params = self._tta_get_ln_params()
        pre_update_params = [p.data.clone() for p in ln_params]

        loss = F.l1_loss(pred, target.detach())

        self._tta_optimizer.zero_grad()
        loss.backward()

        grad_norm_before = torch.nn.utils.clip_grad_norm_(
            ln_params,
            max_norm=float("inf"),
        ).item()

        grad_norm_after = torch.nn.utils.clip_grad_norm_(
            ln_params,
            max_norm=self.tta_grad_clip,
        ).item()

        self._tta_optimizer.step()

        param_delta_norm = self._compute_param_delta_norm(pre_update_params, ln_params)

        self._update_adaptation_stats(loss.item(), grad_norm_before, grad_norm_after, param_delta_norm)

        return loss.item()

    def _compute_param_delta_norm(self, pre_update_params: list[Tensor], ln_params: list[nn.Parameter]) -> float:
        """Compute the L2 norm of parameter changes."""
        param_delta_norm = 0.0
        for pre_param, param in zip(pre_update_params, ln_params):
            param_delta_norm += (param.data - pre_param).norm().item() ** 2
        return param_delta_norm**0.5

    def _update_adaptation_stats(
        self,
        loss_val: float,
        grad_norm_before: float,
        grad_norm_after: float,
        param_delta_norm: float,
    ) -> None:
        """Update adaptation statistics."""
        self._tta_clip_stats["adaptation_losses"].append(loss_val)
        self._tta_clip_stats["num_adaptations"] += 1
        self._tta_clip_stats["grad_norm_before_clip"] = grad_norm_before
        self._tta_clip_stats["grad_norm_after_clip"] = grad_norm_after
        self._tta_clip_stats["param_delta_norm"] = param_delta_norm

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
            features: [B, T+1, N, D] - Full sequence features.
            actions: [B, T, action_dim] - Actions.
            states: [B, T, action_dim] - States.
            extrinsics: [B, T, action_dim-1] - Optional extrinsics.
            detach: Whether to detach predictions (for evaluation).

        Returns:
            Tuple of (predictions, targets) where:
                - predictions: [B, T_pred*N, D] - Autoregressive predictions.
                - targets: [B, T_pred*N, D] - Ground-truth targets.
        """
        B, T_plus_1, N, D = features.shape
        C = self.context_frames
        T_pred = min(T_plus_1 - C, self.T_rollout)

        h = self._normalize_features(features, B, T_plus_1, N, D)

        z_ar = h[:, :C, :, :].reshape(B, C * N, D)
        predictions_list = []

        for step in range(T_pred):
            num_action_steps = C + step
            step_actions = actions[:, :num_action_steps, :]
            step_states = states[:, :num_action_steps, :]
            step_extrinsics = extrinsics[:, :num_action_steps, :] if extrinsics is not None else None

            z_pred_full = self._step_predictor(z_ar, step_actions, step_states, step_extrinsics)  # type: ignore[attr-defined]
            z_pred = z_pred_full[:, -N:, :]
            predictions_list.append(z_pred)
            z_ar = torch.cat([z_ar, z_pred.detach()], dim=1)

        predictions = torch.cat(predictions_list, dim=1)
        targets = h[:, C : C + T_pred, :, :].reshape(B, T_pred * N, D)

        if detach:
            predictions = predictions.detach()

        return predictions, targets

    def _normalize_features(
        self,
        features: Tensor,
        B: int,
        T_plus_1: int,
        N: int,
        D: int,
    ) -> Tensor:
        """Normalize features if enabled."""
        if self.normalize_reps:
            return F.layer_norm(features.reshape(B, -1, D), (D,), eps=1e-6).reshape(B, T_plus_1, N, D)
        return features

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
        1. Perform full autoregressive rollout (z₁→z₇).
        2. Compute rollout loss against ground-truth.
        3. Update LayerNorm parameters.
        4. (Optionally repeat for multiple adaptation steps).
        5. Evaluate with adapted model.

        Args:
            features: [B, T+1, N, D] - Full sequence features.
            actions: [B, T, action_dim] - Actions.
            states: [B, T, action_dim] - States.
            extrinsics: [B, T, action_dim-1] - Optional extrinsics.
            num_adaptation_steps: Number of TTA update iterations.

        Returns:
            Tuple of (final_predictions, targets, stats).
        """
        if self.tta_reset_per_clip:
            self._tta_reset_for_clip()

        self._tta_clip_stats = {
            "adaptation_losses": [],
            "num_adaptations": 0,
        }

        original_device = features.device
        use_cpu_for_tta = str(original_device).startswith("mps")

        with torch.no_grad():
            pre_adapt_pred, targets = self._tta_full_rollout(
                features, actions, states, extrinsics, detach=True
            )
            pre_adapt_loss = F.l1_loss(pre_adapt_pred, targets).item()

        self._tta_clip_stats["pre_adapt_loss"] = pre_adapt_loss

        with torch.inference_mode(False):
            target_device = "cpu" if use_cpu_for_tta else original_device

            features_for_tta = features.detach().clone().to(target_device)
            actions_for_tta = actions.detach().clone().to(target_device)
            states_for_tta = states.detach().clone().to(target_device)
            extrinsics_for_tta = (
                extrinsics.detach().clone().to(target_device) if extrinsics is not None else None
            )

            if use_cpu_for_tta:
                logger.debug("[TTA] MPS detected - moving to CPU for adaptation to avoid NaN gradients")
                self.model.to("cpu")
                self._tta_optimizer = self._tta_create_optimizer()

            try:
                for _ in range(num_adaptation_steps):
                    prev_ln_states = self._tta_set_ln_train_mode(True)
                    try:
                        predictions, targets_tta = self._tta_full_rollout(
                            features_for_tta,
                            actions_for_tta,
                            states_for_tta,
                            extrinsics_for_tta,
                            detach=False,
                        )
                        self._tta_adapt(predictions, targets_tta)
                    finally:
                        self._tta_restore_ln_train_mode(prev_ln_states)
            finally:
                if use_cpu_for_tta:
                    self.model.to(original_device)
                    self._tta_optimizer = self._tta_create_optimizer()

        with torch.no_grad():
            final_predictions, targets = self._tta_full_rollout(
                features, actions, states, extrinsics, detach=True
            )
            post_adapt_loss = F.l1_loss(final_predictions, targets).item()

        self._tta_clip_stats["post_adapt_loss"] = post_adapt_loss
        self._tta_clip_stats["improvement"] = pre_adapt_loss - post_adapt_loss

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
        1. Predict next frame.
        2. Wait for ground-truth.
        3. Adapt.
        4. Predict next frame with updated model.

        Args:
            features: [B, T+1, N, D] - Full sequence features.
            actions: [B, T, action_dim] - Actions.
            states: [B, T, action_dim] - States.
            extrinsics: [B, T, action_dim-1] - Optional extrinsics.

        Returns:
            Tuple of (adapted_predictions, targets, stats).
        """
        B, T_plus_1, N, D = features.shape
        C = self.context_frames
        T_pred = min(T_plus_1 - C, self.T_rollout)

        if self.tta_reset_per_clip:
            self._tta_reset_for_clip()

        self._tta_make_model_params_non_inference()
        self._tta_optimizer = self._tta_create_optimizer()

        (
            features_for_tta,
            actions_for_tta,
            states_for_tta,
            extrinsics_for_tta,
        ) = self._prepare_inputs_for_tta(features, actions, states, extrinsics)

        mps_result = self._handle_mps_fallback(
            features_for_tta,
            actions_for_tta,
            states_for_tta,
            extrinsics_for_tta,
        )

        try:
            predictions, targets = self._run_sequential_adaptation(
                mps_result.features,
                mps_result.actions,
                mps_result.states,
                mps_result.extrinsics,
                B,
                T_plus_1,
                N,
                D,
                C,
                T_pred,
            )
        finally:
            if mps_result.use_fallback and mps_result.original_device is not None:
                self.model.to(mps_result.original_device)
                self._tta_optimizer = self._tta_create_optimizer()

        stats = self._tta_get_clip_stats()
        self._tta_all_clip_stats.append(stats)

        return predictions, targets, stats

    def _prepare_inputs_for_tta(
        self,
        features: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Clone inputs for TTA to escape inference mode."""
        return (
            features.detach().clone().requires_grad_(False),
            actions.detach().clone().requires_grad_(False),
            states.detach().clone().requires_grad_(False),
            extrinsics.detach().clone().requires_grad_(False) if extrinsics is not None else None,
        )

    def _handle_mps_fallback(
        self,
        features_for_tta: Tensor,
        actions_for_tta: Tensor,
        states_for_tta: Tensor,
        extrinsics_for_tta: Tensor | None,
    ) -> MPSFallbackResult:
        """Handle MPS device fallback to CPU for TTA.

        Returns:
            MPSFallbackResult with fallback flag, original device, and tensors.
        """
        if not self._tta_is_mps_device():
            return MPSFallbackResult(
                use_fallback=False,
                original_device=None,
                features=features_for_tta,
                actions=actions_for_tta,
                states=states_for_tta,
                extrinsics=extrinsics_for_tta,
            )

        logger.debug("[TTA] MPS detected - moving to CPU for adaptation to avoid NaN gradients")
        original_device = next(self.model.parameters()).device
        self.model.to("cpu")
        self._tta_optimizer = self._tta_create_optimizer()

        return MPSFallbackResult(
            use_fallback=True,
            original_device=original_device,
            features=features_for_tta.to("cpu"),
            actions=actions_for_tta.to("cpu"),
            states=states_for_tta.to("cpu"),
            extrinsics=extrinsics_for_tta.to("cpu") if extrinsics_for_tta is not None else None,
        )

    def _run_sequential_adaptation(
        self,
        features: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None,
        B: int,
        T_plus_1: int,
        N: int,
        D: int,
        C: int,
        T_pred: int,
    ) -> tuple[Tensor, Tensor]:
        """Run sequential adaptation loop."""
        h = self._normalize_features(features, B, T_plus_1, N, D)
        z_ar = h[:, :C, :, :].reshape(B, C * N, D)
        predictions_list = []
        prev_pred = None

        for step in range(T_pred):
            target_idx = C + step
            num_action_steps = C + step

            step_actions = actions[:, :num_action_steps, :]
            step_states = states[:, :num_action_steps, :]
            step_extrinsics = extrinsics[:, :num_action_steps, :] if extrinsics is not None else None

            if prev_pred is not None and step > 0:
                prev_target = h[:, target_idx - 1, :, :]
                self._tta_adapt(prev_pred, prev_target)

            prev_ln_states = self._tta_set_ln_train_mode(True)
            try:
                z_pred_full = self._step_predictor(z_ar, step_actions, step_states, step_extrinsics)  # type: ignore[attr-defined]
                z_pred = z_pred_full[:, -N:, :]
            finally:
                self._tta_restore_ln_train_mode(prev_ln_states)

            prev_pred = z_pred
            predictions_list.append(z_pred.detach())
            z_ar = torch.cat([z_ar, z_pred.detach()], dim=1)

        if prev_pred is not None and T_pred > 0:
            final_target = h[:, C + T_pred - 1, :, :]
            self._tta_adapt(prev_pred, final_target)

        predictions = torch.cat(predictions_list, dim=1)
        targets = h[:, C : C + T_pred, :, :].reshape(B, T_pred * N, D)

        return predictions, targets

    def _tta_get_clip_stats(self) -> dict[str, Any]:
        """Get statistics for current clip."""
        losses = self._tta_clip_stats.get("adaptation_losses", [])
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
            "improvement": (
                improvement
                if improvement != 0.0
                else (losses[0] - losses[-1] if len(losses) > 1 else 0.0)
            ),
        }

    def _tta_get_epoch_stats(self) -> dict[str, Any]:
        """Get aggregated TTA statistics for the epoch."""
        if not self._tta_all_clip_stats:
            return {}

        total_clips = len(self._tta_all_clip_stats)
        total_adaptations = sum(s["num_adaptations"] for s in self._tta_all_clip_stats)
        mean_improvement = sum(s["improvement"] for s in self._tta_all_clip_stats) / total_clips

        if "pre_adapt_loss" in self._tta_all_clip_stats[0]:
            mean_pre_adapt = sum(s["pre_adapt_loss"] for s in self._tta_all_clip_stats) / total_clips
            mean_post_adapt = sum(s["post_adapt_loss"] for s in self._tta_all_clip_stats) / total_clips
        else:
            mean_pre_adapt = sum(s["first_loss"] for s in self._tta_all_clip_stats) / total_clips
            mean_post_adapt = sum(s["last_loss"] for s in self._tta_all_clip_stats) / total_clips

        return {
            "total_clips": total_clips,
            "total_adaptations": total_adaptations,
            "mean_improvement": mean_improvement,
            "mean_pre_adapt_loss": mean_pre_adapt,
            "mean_post_adapt_loss": mean_post_adapt,
        }
