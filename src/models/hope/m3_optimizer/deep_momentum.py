"""
Deep Momentum optimizer variants from the Nested Learning paper.

Implements momentum as an associative memory with several variants:
- preconditioned: Adam-style diagonal preconditioning
- dmgd: Deep Momentum Gradient Descent (tanh nonlinearity)
- muon: Preconditioning + nonlinearity (Muon-equivalent, Eq. 24)
- l2_objective: L2 regression for gradient aggregation
- nl_l2_precond: Context-aware orthogonal projection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DeepMomentumState:
    """Internal running-average state for :class:`DeepMomentum`."""

    grad_avg: Optional[torch.Tensor] = None
    sq_avg: Optional[torch.Tensor] = None


class DeepMomentum(nn.Module):
    """Implements momentum variants described in the Nested Learning paper.

    This module is used as an *inner* optimizer (called like a function on each
    gradient tensor) and maintains exponential-moving-average state internally.

    Parameters
    ----------
    beta : float
        EMA coefficient for the first moment (gradient average).
    beta2 : float
        EMA coefficient for the second moment (used by ``preconditioned`` and
        ``muon`` variants).
    eps : float
        Numerical stability term for preconditioning denominator.
    variant : str
        One of ``"preconditioned"``, ``"dmgd"``, ``"muon"``,
        ``"l2_objective"``, ``"nl_l2_precond"``.

    Example
    -------
    >>> opt = DeepMomentum(beta=0.9, beta2=0.999, variant="muon")
    >>> update = opt(grad_tensor)
    >>> param.add_(update, alpha=-lr)
    """

    SUPPORTED_VARIANTS = {"preconditioned", "dmgd", "muon", "l2_objective", "nl_l2_precond"}

    def __init__(
        self,
        *,
        beta: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        variant: str = "preconditioned",
    ) -> None:
        super().__init__()
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}. "
                f"Supported: {sorted(self.SUPPORTED_VARIANTS)}"
            )
        self.beta = beta
        self.beta2 = beta2
        self.eps = eps
        self.variant = variant
        self.state = DeepMomentumState()
        self.nonlinearity = nn.Tanh() if variant in {"dmgd", "muon"} else nn.Identity()
        self.last_metrics: dict[str, float] = {}

    def reset_state(self) -> None:
        """Clear running averages (e.g. between epochs or tasks)."""
        self.state = DeepMomentumState()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _precondition(self, grad: torch.Tensor) -> torch.Tensor:
        """Adam-style diagonal preconditioning."""
        if self.state.sq_avg is None or self.state.sq_avg.shape != grad.shape:
            self.state.sq_avg = torch.zeros_like(grad)
        self.state.sq_avg.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        denom = self.state.sq_avg.sqrt().add_(self.eps)
        return grad / denom

    def _nl_precondition(
        self,
        grad: torch.Tensor,
        context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Context-aware orthogonal projection.

        Projects the gradient orthogonal to the context direction, effectively
        removing the component already explained by the context signal.
        """
        metrics: dict[str, float] = {"ctx_norm": 0.0, "proj_norm": 0.0}
        if context is None:
            return grad, metrics
        ctx = context
        if ctx.ndim > 1:
            ctx = ctx.reshape(-1, ctx.shape[-1]).mean(dim=0)
        ctx_norm = torch.norm(ctx)
        metrics["ctx_norm"] = ctx_norm.item()

        if ctx_norm > 0:
            unit = ctx / (ctx_norm + self.eps)
            # Project grad orthogonal to context (rank-1 projector).
            projection = (grad * unit).sum(dim=-1, keepdim=True) * unit
            update = grad - projection
            metrics["proj_norm"] = torch.norm(update).item()
            return update, metrics
        return grad, metrics

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(  # type: ignore[override]
        self,
        grad: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the momentum update for a single gradient tensor.

        Parameters
        ----------
        grad : Tensor
            Raw gradient for one parameter.
        context : Tensor, optional
            Context vector used only by the ``nl_l2_precond`` variant.

        Returns
        -------
        Tensor
            The smoothed/transformed update (same shape as *grad*).
        """
        if self.state.grad_avg is None or self.state.grad_avg.shape != grad.shape:
            self.state.grad_avg = torch.zeros_like(grad)
        self.last_metrics = {}

        update = grad

        if self.variant in {"preconditioned", "muon"}:
            update = self._precondition(grad)

        if self.variant == "l2_objective":
            update = grad + 0.1 * torch.mean(grad, dim=-1, keepdim=True)

        if self.variant == "nl_l2_precond":
            update, metrics = self._nl_precondition(grad, context)
            self.last_metrics.update(metrics)

        if self.variant in {"dmgd", "muon"}:
            update = self.nonlinearity(update)

        self.state.grad_avg.mul_(self.beta).add_(update, alpha=1 - self.beta)
        return self.state.grad_avg
