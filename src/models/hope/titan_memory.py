"""Titan Memory — Self-Modifying Associative Memory (Behrouz 2025, HOPE).

Implements the Deep Self-Referential Titan memory from the Nested Learning paper.
The memory is a 2-layer MLP whose weights are updated *during the forward pass*
via Descending-with-Gradient Descent (DGD).

Key equations (from Behrouz 2025, Section 8.1):
    Retrieval:     o_t = M_{t-1}(q_t)
    Target gen:    v̂_□,t = M_{□,t-1}(v_t)
    DGD update:    M_{□,t} = M_{□,t-1}(α_t I − η_t k_t k_t^T)
                            − η_t (M_{□,t-1} k_t − v̂_□,t) k_t^T

Implementation approach — functional forward:
    During training, the memory uses **plain tensors** (not nn.Parameters) for
    the active forward computation. This avoids in-place modification of
    parameters that are part of the outer autograd graph.

    - nn.Parameters (w1.weight, w2.weight) are the meta-learned initial state,
      updated by the outer optimizer via standard backprop.
    - Active weights (_active_w1, _active_w2) are detached clones used for the
      actual forward computation and DGD self-modification within a sequence.
    - reset_active_weights() copies parameters → active weights (call per seq).
    - compute_and_apply_update() modifies active weights (not parameters).
    - forward() uses F.linear() with the active weights.

The memory supports:
    - Gradient-based self-modification via functional forward
    - Surprise-gated updates (only write when retrieval error is high)
    - Gradient norm clipping for inner-loop stability
    - Diagnostic logging of inner-loop gradient norms, surprise, param norms
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

log = logging.getLogger(__name__)


@dataclass
class TitanMemoryConfig:
    """Configuration for a single Titan memory module."""

    dim: int = 384
    hidden_multiplier: int = 4
    num_layers: int = 2
    activation: str = "gelu"
    grad_clip_inner: float = 1.0
    output_norm: bool = True
    # Detach memory graph every N tokens to bound memory cost (0 = never detach)
    detach_interval: int = 0


class TitanMemory(nn.Module):
    """MLP-based associative memory with DGD self-modification.

    Structure: x → W1 → σ → W2 → + x  (residual 2-layer MLP)

    Uses a **functional forward** pattern for self-modification:
    - nn.Parameters (w1.weight, w2.weight) are the meta-learned initial state.
    - Active weights (_active_w1, _active_w2) are plain tensors derived from
      the parameters and updated via DGD during the forward pass.
    - reset_active_weights() copies from parameters → active weights.
    - forward() uses F.linear() with active weights instead of self.w1/w2.
    - compute_and_apply_update() modifies active weights (not parameters).

    This design ensures that:
    1. During inference without reset: falls back to nn.Parameter modules.
    2. During training: active weights are detached clones, so DGD updates
       don't trigger in-place modification errors on the autograd graph.
    3. The outer optimizer updates the meta-learned initial state via standard
       backprop through the forward() calls that happen BEFORE reset.

    Args:
        config: TitanMemoryConfig with architecture settings.
    """

    def __init__(self, config: TitanMemoryConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.dim
        hidden = dim * config.hidden_multiplier

        # Meta-learned initial memory weights (2-layer MLP with residual)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.norm = nn.LayerNorm(dim) if config.output_norm else nn.Identity()

        # Activation function
        act_map = {"gelu": nn.GELU(), "silu": nn.SiLU(), "relu": nn.ReLU()}
        self.act = act_map.get(config.activation, nn.GELU())

        # Active weights for functional forward (plain tensors, NOT nn.Parameters).
        # When None, forward() falls back to using nn.Linear modules directly.
        self._active_w1: Tensor | None = None
        self._active_w2: Tensor | None = None

        # Running diagnostic counters (not saved in state_dict)
        self.register_buffer(
            "_step_counter", torch.tensor(0, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_total_surprise", torch.tensor(0.0), persistent=False
        )
        self.register_buffer(
            "_total_inner_grad_norm", torch.tensor(0.0), persistent=False
        )

    # ──────────────────────────────────────────────────────────────────────
    # Active-weight management
    # ──────────────────────────────────────────────────────────────────────

    def reset_active_weights(self) -> None:
        """Reset active weights to meta-learned initial state (from Parameters).

        Call this at the start of each sequence to ensure fresh memory state.
        Creates **detached clones** that can be freely modified by DGD without
        touching the original nn.Parameters.
        """
        self._active_w1 = self.w1.weight.detach().clone()
        self._active_w2 = self.w2.weight.detach().clone()

    def clear_active_weights(self) -> None:
        """Clear active weights, reverting to nn.Parameter-based forward."""
        self._active_w1 = None
        self._active_w2 = None

    # ──────────────────────────────────────────────────────────────────────
    # Forward (functional)
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, query: Tensor) -> Tensor:
        """Retrieve from memory: o_t = M_{t-1}(q_t).

        Uses **functional** F.linear() with active weights when available,
        falling back to the nn.Linear modules (w1, w2) otherwise.

        Args:
            query: [B, ..., D] input query tokens.

        Returns:
            [B, ..., D] retrieved memory output (with LayerNorm).
        """
        if self._active_w1 is not None and self._active_w2 is not None:
            # Functional path — uses detached active weight clones
            h = F.linear(query, self._active_w1)
            h = self.act(h)
            out = F.linear(h, self._active_w2) + query  # Residual
        else:
            # Standard nn.Module path (inference without reset, or eval)
            h = self.act(self.w1(query))
            out = self.w2(h) + query  # Residual
        return self.norm(out)

    # ──────────────────────────────────────────────────────────────────────
    # Surprise signal
    # ──────────────────────────────────────────────────────────────────────

    def surprise(self, residual: Tensor) -> Tensor:
        """Compute surprise signal ||residual||_2 per sample.

        Higher surprise → memory should update more aggressively.

        Args:
            residual: [B, ..., D] retrieval error (target - retrieved).

        Returns:
            [B] scalar surprise per sample.
        """
        norms = residual.detach().norm(dim=-1)  # [B, ...] or [B]
        if norms.dim() > 1:
            s = norms.mean(dim=list(range(1, norms.dim())))
        else:
            s = norms
        return s

    # ──────────────────────────────────────────────────────────────────────
    # DGD inner-loop update
    # ──────────────────────────────────────────────────────────────────────

    def compute_and_apply_update(
        self,
        key: Tensor,
        value: Tensor,
        error_signal: Tensor,
        lr: Tensor,
        alpha: Tensor | None = None,
    ) -> None:
        """Compute DGD parameter deltas and apply them to active weights.

        This is the core self-modification step. It:
        1. Creates temporary grad-enabled copies of active weights.
        2. Computes the inner loss ||M(key) - value||^2 with surprise gating.
        3. Computes first-order gradients (create_graph=False).
        4. Applies the DGD update rule to the active weight tensors.

        Because active weights are plain tensors (detached from the outer graph),
        this cannot cause in-place modification errors during outer backward.

        DGD update rule (Eq. 86-93 in Behrouz 2025):
            delta_W = -eta * grad_W L_inner
            W_new = alpha * W_old + delta_W   (weight decay via alpha < 1)

        Args:
            key:   [B, N, D]  keys to write.
            value: [B, N, D]  target values.
            error_signal: [B]  per-sample gating (surprise).
            lr:    [B, N, 1] or [B, 1, 1]  adaptive learning rate eta_t.
            alpha: [B, N, 1] or [B, 1, 1]  adaptive decay alpha_t (default 1.0).

        Raises:
            RuntimeError: If active weights have not been initialized via
                reset_active_weights().
        """
        if self._active_w1 is None or self._active_w2 is None:
            raise RuntimeError(
                "Active weights not initialized. Call reset_active_weights() "
                "before compute_and_apply_update()."
            )

        # Create temporary copies with grad enabled for inner-loop gradient
        w1_tmp = self._active_w1.detach().requires_grad_(True)
        w2_tmp = self._active_w2.detach().requires_grad_(True)

        # Forward through memory with grad-enabled temp weights
        h = F.linear(key, w1_tmp)
        h = self.act(h)
        retrieved = F.linear(h, w2_tmp) + key  # Residual

        # Inner loss: MSE between retrieved and target values
        inner_loss = F.mse_loss(
            retrieved, value.detach(), reduction="none"
        )  # [B, N, D]

        # Weight by error_signal (surprise gating)
        gate = error_signal.detach().unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        inner_loss = (inner_loss * gate).sum()

        # Compute inner-loop gradients (first-order, no Hessian)
        grads = torch.autograd.grad(
            inner_loss,
            [w1_tmp, w2_tmp],
            create_graph=False,
            allow_unused=True,
        )

        # Extract scalars for the update
        lr_scalar = lr.detach().mean().item()
        alpha_scalar = (
            alpha.detach().mean().item() if alpha is not None else 1.0
        )

        inner_grad_norm_sq = 0.0
        new_weights: list[Tensor] = []

        for w_old, grad in zip(
            [self._active_w1, self._active_w2], grads
        ):
            if grad is None:
                new_weights.append(w_old)
                continue

            grad = grad.detach()
            grad_norm = grad.norm().item()
            inner_grad_norm_sq += grad_norm ** 2

            # Clip inner-loop gradients for stability (Criticism S1)
            if (
                self.config.grad_clip_inner > 0
                and grad_norm > self.config.grad_clip_inner
            ):
                grad = grad * (
                    self.config.grad_clip_inner / (grad_norm + 1e-8)
                )

            # DGD update: W_new = alpha * W_old - eta * grad
            new_w = alpha_scalar * w_old - lr_scalar * grad
            new_weights.append(new_w)

        # Assign updated active weights (plain tensor reassignment, no in-place)
        self._active_w1 = new_weights[0]
        self._active_w2 = new_weights[1]

        # Update diagnostics
        self._step_counter += 1
        self._total_inner_grad_norm += inner_grad_norm_sq ** 0.5

    # ──────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────

    def reset_diagnostics(self) -> None:
        """Reset diagnostic counters (call at start of each forward sequence)."""
        self._step_counter.zero_()
        self._total_surprise.zero_()
        self._total_inner_grad_norm.zero_()

    def get_diagnostics(self) -> dict[str, float]:
        """Get averaged diagnostic metrics for logging.

        Returns dict with keys:
            - titan/mean_inner_grad_norm
            - titan/param_norm_w1
            - titan/param_norm_w2
            - titan/num_updates
        """
        n = max(self._step_counter.item(), 1)
        return {
            "titan/mean_inner_grad_norm": (
                self._total_inner_grad_norm.item() / n
            ),
            "titan/param_norm_w1": self.w1.weight.detach().norm().item(),
            "titan/param_norm_w2": self.w2.weight.detach().norm().item(),
            "titan/num_updates": self._step_counter.item(),
        }
