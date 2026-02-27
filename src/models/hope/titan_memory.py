"""Titan Memory — Self-Modifying Associative Memory (Behrouz 2025, HOPE).

Implements the Deep Self-Referential Titan memory from the Nested Learning paper.
The memory is a 2-layer MLP whose weights are updated *during the forward pass*
via Delta Gradient Descent (DGD).

Key equations (from Behrouz 2025, Section 8.1):
    Retrieval:     o_t = M_{t-1}(q_t)
    Self-targets:  v̂_□,t = M_{□,t-1}(v_t)          (Eq. 83)
    DGD update:    M_{□,t} = M_{□,t-1}(α_t I − η_t k_t k_t^T)
                            − η_t (M_{□,t-1} k_t − v̂_□,t) k_t^T  (Eq. 93)

Implementation approach — functional forward with meta-learning:
    - nn.Parameters (w1.weight, w2.weight) are the meta-learned initial state,
      updated by the outer optimizer via standard backprop.
    - Active weights (_active_w1, _active_w2) are derived from the parameters
      WITHOUT detach, so the outer optimizer can compute meta-gradients
      (FOMAML-style first-order) through the forward pass.
    - reset_active_weights() copies parameters → active weights (keeping grad).
    - compute_and_apply_update() replaces active weights with updated tensors.
    - forward() uses F.linear() with the active weights.
    - detach_interval bounds memory cost by periodically detaching the graph.

The memory supports:
    - Meta-learned initial states via preserved gradient flow (Pillar 3)
    - Self-generated target values v̂ = M(v) for self-referential learning
    - Per-token, per-feature η and α ∈ R^d for fine-grained adaptation (Eq. 88)
    - DGD preconditioner (diagonal approx of k_t k_t^T) for decorrelated updates
    - Surprise-gated updates (only write when retrieval error is high)
    - Gradient norm clipping for inner-loop stability
    - Periodic graph detachment to bound VRAM usage
    - Diagnostic logging of inner-loop gradient norms, surprise, param norms
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Custom autograd functions for stable DGD backward
# ──────────────────────────────────────────────────────────────────────────

class _StableGradPrecondScale(torch.autograd.Function):
    """w_old * precond in forward; normalised backward to prevent overflow.

    Forward:  out = w_old * precond                         (unchanged)
    Backward: grad_w_old   = grad_output * precond          (standard)
              grad_precond = grad_output * sign(w_old)       (normalised)

    Why sign(w_old) instead of w_old?
    The DGD preconditioner `precond = clamp(α − η·k², 0, 1)` depends on
    lr_feat (from M_eta) and alpha_feat (from M_alpha).  The raw backward
    `grad_precond = grad_output * w_old` scales every element by the full
    weight magnitude.  With ~30 memory-chain gradient paths per block
    (5 memories × 2 weights × live chunk pairs), each contributing
    O(||w_old||) amplification, the accumulated gradient compounds
    exponentially across 6 blocks → float32 overflow → NaN.

    Replacing w_old with sign(w_old) preserves the gradient *direction*
    while bounding each element's magnitude to |grad_output|.  Combined
    with _BackwardGradClip (Level 2), this prevents both per-element
    and per-tensor gradient explosion.

    See docs/20260227_phase2_nan_debugging_session_report.md §3 for the
    full numerical analysis.
    """

    @staticmethod
    def forward(ctx, w_old: Tensor, precond: Tensor) -> Tensor:
        ctx.save_for_backward(w_old, precond)
        return w_old * precond

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        w_old, precond = ctx.saved_tensors
        grad_w_old = grad_output * precond
        # Normalise: sign(w_old) instead of w_old
        grad_precond = grad_output * w_old.sign()
        return grad_w_old, grad_precond


class _BackwardGradClip(torch.autograd.Function):
    """Identity in forward; clips backward gradient norm to *max_norm*.

    This is the Level-2 defense: after each DGD update the new active
    weights are wrapped with this function so that the backward gradient
    flowing *through* them (from the next chunk's forward) is norm-clipped.

    Without this, the SUM of ~30 bounded gradient paths (from
    _StableGradPrecondScale * 5 memories * 2 weights) compounds across
    blocks.  With this clip each memory's backward contribution is bounded,
    ensuring linear (not exponential) gradient growth across blocks.

    The default max_norm = grad_clip_inner (typically 1.0), matching the
    inner-loop gradient clipping for symmetry.
    """

    @staticmethod
    def forward(ctx, x: Tensor, max_norm: float) -> Tensor:
        ctx.max_norm = max_norm
        return x

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        max_norm = ctx.max_norm
        grad_norm = grad_output.norm()
        if grad_norm > max_norm:
            grad_output = grad_output * (max_norm / (grad_norm + 1e-8))
        return grad_output, None


def _stable_precond_scale(w_old: Tensor, precond: Tensor) -> Tensor:
    """Convenience wrapper for _StableGradPrecondScale.apply."""
    return _StableGradPrecondScale.apply(w_old, precond)


def _backward_grad_clip(x: Tensor, max_norm: float) -> Tensor:
    """Convenience wrapper for _BackwardGradClip.apply."""
    return _BackwardGradClip.apply(x, max_norm)


class Activation(StrEnum):
    """Activation function types for Titan Memory."""
    GELU = "gelu"
    SILU = "silu"
    RELU = "relu"


@dataclass
class TitanMemoryConfig:
    """Configuration for a single Titan memory module.

    Attributes:
        dim: Feature dimension (default: 384).
        hidden_multiplier: Multiplier for hidden layer size (default: 4).
        num_layers: Number of layers (default: 2).
        activation: Activation function name (default: "gelu").
        grad_clip_inner: Gradient clipping threshold for inner loop (default: 1.0).
        grad_clip_backward: Max gradient norm for DGD backward path (default: 1.0).
            Level-2 defense: clips the backward gradient flowing through active
            weights after DGD updates.  Set to 0 to disable.
        output_norm: Apply LayerNorm to output (default: True).
        detach_interval: Detach memory graph every N update steps, 0 = never (default: 0).
    """

    dim: int = 384
    hidden_multiplier: int = 4
    num_layers: int = 2
    activation: str = "gelu"
    grad_clip_inner: float = 1.0
    grad_clip_backward: float = 1.0
    output_norm: bool = True
    detach_interval: int = 0


class TitanMemory(nn.Module):
    """MLP-based associative memory with DGD self-modification.

    Structure: x → W1 → σ → W2 → + x  (residual 2-layer MLP, Eq. 89)

    Uses a **functional forward** pattern for self-modification:
    - nn.Parameters (w1.weight, w2.weight) are the meta-learned initial state.
    - Active weights (_active_w1, _active_w2) are derived from the parameters
      and updated via DGD during the forward pass.
    - reset_active_weights() copies from parameters → active weights, keeping
      the gradient connection for meta-learning (no detach).
    - forward() uses F.linear() with active weights instead of self.w1/w2.
    - compute_and_apply_update() creates new active weight tensors via DGD.

    Meta-learning design (FOMAML-style, first-order):
    1. reset_active_weights() clones parameters WITHOUT detach.
    2. forward() uses F.linear(query, active_w) — outer loss depends on
       active weights which depend on nn.Parameters.
    3. compute_and_apply_update() computes inner-loop gradients with
       create_graph=False (first-order), then creates NEW tensors for active
       weights. The new tensors are functions of the OLD active weights
       (via alpha * w_old - eta * grad), preserving the meta-gradient chain.
    4. detach_interval periodically detaches to bound VRAM.

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
        self.norm = (
            nn.LayerNorm(dim) if config.output_norm else nn.Identity()
        )

        # Activation function
        act_map: dict[str, nn.Module] = {
            Activation.GELU: nn.GELU(),
            Activation.SILU: nn.SiLU(),
            Activation.RELU: nn.ReLU(),
        }
        self.act = act_map.get(config.activation, nn.GELU())

        # Active weights for functional forward (plain tensors, NOT nn.Parameters)
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

        CRITICAL: Does NOT detach — the active weights remain connected to the
        nn.Parameters so the outer optimizer can compute meta-gradients through
        the forward pass (FOMAML-style first-order meta-learning).

        The gradient chain is:
            nn.Parameter → active_weight → F.linear(query, active_weight) → loss
        so backprop through the loss updates the meta-learned initial state.
        """
        self._active_w1 = self.w1.weight.clone()
        self._active_w2 = self.w2.weight.clone()

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
            h = F.linear(query, self._active_w1)
            h = self.act(h)
            out = F.linear(h, self._active_w2) + query
        else:
            h = self.act(self.w1(query))
            out = self.w2(h) + query
        return self.norm(out)

    # ──────────────────────────────────────────────────────────────────────
    # Self-generated target values (Eq. 83)
    # ──────────────────────────────────────────────────────────────────────

    def generate_self_target(self, v: Tensor) -> Tensor:
        """Generate self-referential target values: v̂_□,t = M_{□,t-1}(v_t).

        This is the "self-modifying" property (Eq. 83, Behrouz 2025):
        each memory generates its own target using its own current weights.

        Args:
            v: [B, N, D] value tokens.

        Returns:
            [B, N, D] self-generated target values.
        """
        return self.forward(v)

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
        norms = residual.detach().norm(dim=-1)
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

        This is the core self-modification step (Eq. 88-93, Behrouz 2025).

        The update preserves the meta-gradient chain:
            new_w = alpha * w_old - eta * grad
        where w_old is connected to nn.Parameters, so backprop through any
        subsequent forward(query) that uses new_w will flow gradients back
        to the meta-learned initial state.

        Inner-loop gradients are computed with create_graph=False (first-order,
        FOMAML-style). The meta-gradient flows through w_old, NOT through
        the inner-loop gradient computation itself.

        Per-feature η and α are averaged over batch/tokens but kept as D-dim
        vectors for per-feature adaptation granularity (Eq. 88, Behrouz 2025).

        Full DGD update rule (Eq. 93 in Behrouz 2025):
            M_t = M_{t-1} (α_t I − η_t k_t k_t^T) − η_t (M_{t-1} k_t − v̂_t) k_t^T

        For nonlinear MLPs, the k_t k_t^T preconditioner is applied as a
        diagonal approximation: each weight row/column is scaled by the
        mean squared key magnitude along its corresponding dimension.
        This decorrelates correlated token updates (the core DGD innovation).

        Args:
            key: [B, N, D] keys to write.
            value: [B, N, D] self-generated target values v̂_□.
            error_signal: [B] per-sample gating (surprise).
            lr: [B, N, D] per-token, per-feature adaptive learning rate η_t.
            alpha: [B, N, D] per-token, per-feature adaptive decay α_t (default 1.0).

        Raises:
            RuntimeError: If active weights have not been initialized via
                reset_active_weights().
        """
        if self._active_w1 is None or self._active_w2 is None:
            raise RuntimeError(
                "Active weights not initialized. Call reset_active_weights() "
                "before compute_and_apply_update()."
            )

        # ─── Compute inner-loop gradient ───
        # torch.enable_grad() needed because Lightning disables gradients during
        # validation/test, but the inner-loop still needs gradients for DGD.
        # create_graph=False ensures no second-order gradients (FOMAML).
        with torch.enable_grad():
            # Leaf copies for autograd.grad() — detached from meta-gradient chain
            # (first-order: we don't differentiate through the gradient computation)
            w1_leaf = self._active_w1.detach().requires_grad_(True)
            w2_leaf = self._active_w2.detach().requires_grad_(True)

            h = F.linear(key, w1_leaf)
            h = self.act(h)
            retrieved = F.linear(h, w2_leaf) + key

            inner_loss = F.mse_loss(retrieved, value.detach(), reduction="none")

            gate = error_signal.detach().unsqueeze(-1).unsqueeze(-1)
            inner_loss = (inner_loss * gate).mean()

            grads = torch.autograd.grad(
                inner_loss,
                [w1_leaf, w2_leaf],
                create_graph=False,
                allow_unused=True,
            )

        # ─── Compute per-feature η and α for weight update ───
        # LIVE (not detached): lr_feat and alpha_feat remain in the computation
        # graph so that M_eta/M_alpha receive outer-loss meta-gradients.
        # This preserves the paper's core requirement (Behrouz 2025, §8.1):
        # "the initial states of ALL memories are meta-learned".
        #
        # Gradient stability is ensured by a two-level defense:
        #   Level 1 — _StableGradPrecondScale: element-wise sign normalisation
        #             on w_old * precond prevents individual weight magnitudes
        #             from amplifying backward gradients.
        #   Level 2 — _BackwardGradClip: norm clipping on the DGD output active
        #             weights bounds each memory's total backward contribution,
        #             preventing the sum of ~30 gradient paths per block from
        #             compounding exponentially across 6 blocks.
        #
        # See docs/20260227_phase2_nan_debugging_session_report.md for analysis.
        lr_feat = lr.mean(dim=(0, 1))
        alpha_feat = (
            alpha.mean(dim=(0, 1)) if alpha is not None
            else torch.ones(key.shape[-1], device=key.device)
        )

        # ─── DGD preconditioner: diagonal approximation of k_t k_t^T ───
        k_sq = (key.detach() ** 2).mean(dim=(0, 1))

        inner_grad_norm_sq = 0.0
        new_weights: list[Tensor] = []

        for idx, (w_old, grad) in enumerate(zip(
            [self._active_w1, self._active_w2], grads
        )):
            if grad is None:
                new_weights.append(w_old)
                continue

            grad = grad.detach()
            grad_norm = grad.norm().item()
            inner_grad_norm_sq += grad_norm ** 2

            if (
                self.config.grad_clip_inner > 0
                and grad_norm > self.config.grad_clip_inner
            ):
                grad = grad * (
                    self.config.grad_clip_inner / (grad_norm + 1e-8)
                )

            # DGD preconditioned update (Eq. 93): W_new = W_old * (α − η·k²) − η·grad
            # STABILITY: Clamp preconditioner to [0, 1] — without this, precond^7
            # across sequential chunk updates causes exponential weight growth / NaN.
            # Level 1: _stable_precond_scale uses sign(w_old) in backward to
            # prevent element-wise gradient amplification by weight magnitudes.
            if idx == 0:
                # w1 [hidden, D]: preconditioner scales columns (input dim = D)
                precond = alpha_feat.unsqueeze(0) - lr_feat.unsqueeze(0) * k_sq.unsqueeze(0)
                precond = precond.clamp(0.0, 1.0)
                new_w = _stable_precond_scale(w_old, precond) - lr_feat.unsqueeze(0) * grad
            else:
                # w2 [D, hidden]: preconditioner scales rows (output dim = D)
                precond = alpha_feat.unsqueeze(1) - lr_feat.unsqueeze(1) * k_sq.unsqueeze(1)
                precond = precond.clamp(0.0, 1.0)
                new_w = _stable_precond_scale(w_old, precond) - lr_feat.unsqueeze(1) * grad
            new_weights.append(new_w)

        # Level 2: Clip backward gradient norm on the new active weights.
        # This bounds each memory's total backward contribution, preventing
        # accumulation of ~30 gradient paths from compounding across blocks.
        clip_bw = self.config.grad_clip_backward
        if clip_bw > 0:
            self._active_w1 = _backward_grad_clip(new_weights[0], clip_bw)
            self._active_w2 = _backward_grad_clip(new_weights[1], clip_bw)
        else:
            self._active_w1 = new_weights[0]
            self._active_w2 = new_weights[1]

        # ─── Periodic detach to bound VRAM ───
        step = self._step_counter.item() + 1
        if (
            self.config.detach_interval > 0
            and step % self.config.detach_interval == 0
        ):
            self._active_w1 = self._active_w1.detach().clone()
            self._active_w2 = self._active_w2.detach().clone()

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

        Returns:
            Dict with keys:
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
