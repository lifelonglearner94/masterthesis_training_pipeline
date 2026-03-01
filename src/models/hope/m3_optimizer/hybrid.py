"""
Hybrid Muon + AdamW outer optimizer.

Routes model parameters to the appropriate optimizer backend:
- ≥2-D weight matrices (excluding embeddings and norms) → ``torch.optim.Muon``
- Everything else (biases, embeddings, LayerNorm) → ``torch.optim.AdamW``

Requires **PyTorch ≥ 2.9** for ``torch.optim.Muon``.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Parameter classification
# ---------------------------------------------------------------------------

def is_muon_candidate(name: str, param: torch.nn.Parameter) -> bool:
    """Decide whether a named parameter should be routed to Muon.

    Rules:
    - Must be ≥ 2-D (i.e. a matrix, not a bias or scalar).
    - Must *not* be an embedding or normalisation parameter.
    """
    if param.ndim < 2:
        return False
    lowered = name.lower()
    if "norm" in lowered or "embed" in lowered:
        return False
    return True


# ---------------------------------------------------------------------------
# HybridMuonAdamW
# ---------------------------------------------------------------------------

class HybridMuonAdamW:
    """Wraps a ``torch.optim.Muon`` and a ``torch.optim.AdamW`` instance,
    dispatching :meth:`zero_grad`, :meth:`step`, and state-dict operations to
    both.

    Parameters
    ----------
    muon_opt : Optimizer or None
        Muon optimizer handling matrix parameters.
    adamw_opt : Optimizer or None
        AdamW optimizer handling the remaining parameters.
    muon_param_elems : int
        Total number of scalar elements managed by Muon.
    adamw_param_elems : int
        Total number of scalar elements managed by AdamW.
    """

    def __init__(
        self,
        muon_opt: torch.optim.Optimizer | None,
        adamw_opt: torch.optim.Optimizer | None,
        muon_param_elems: int,
        adamw_param_elems: int,
    ) -> None:
        self.muon_opt = muon_opt
        self.adamw_opt = adamw_opt
        self.muon_param_elems = muon_param_elems
        self.adamw_param_elems = adamw_param_elems

    # -- Core optimizer API ------------------------------------------------

    def zero_grad(self) -> None:
        if self.muon_opt:
            self.muon_opt.zero_grad()
        if self.adamw_opt:
            self.adamw_opt.zero_grad()

    def step(self) -> None:
        if self.muon_opt:
            self.muon_opt.step()
        if self.adamw_opt:
            self.adamw_opt.step()

    # -- State dict --------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "muon": self.muon_opt.state_dict() if self.muon_opt else None,
            "adamw": self.adamw_opt.state_dict() if self.adamw_opt else None,
        }

    def load_state_dict(self, state: dict) -> None:
        if self.muon_opt and state.get("muon") is not None:
            self.muon_opt.load_state_dict(state["muon"])
        if self.adamw_opt and state.get("adamw") is not None:
            self.adamw_opt.load_state_dict(state["adamw"])

    # -- Convenience -------------------------------------------------------

    @property
    def state(self) -> dict:
        """Combined optimizer state from both sub-optimizers.

        Lightning accesses ``optimizer.state`` to move tensors between
        devices (CPU ↔ GPU). We merge both sub-optimizer state dicts
        into a single view.
        """
        combined: dict = {}
        if self.muon_opt:
            combined.update(self.muon_opt.state)
        if self.adamw_opt:
            combined.update(self.adamw_opt.state)
        return combined

    @state.setter
    def state(self, value: dict) -> None:
        """Allow Lightning to set state (no-op, state lives in sub-optimizers)."""
        pass

    @property
    def param_groups(self) -> list:
        groups: list = []
        if self.muon_opt:
            groups.extend(self.muon_opt.param_groups)
        if self.adamw_opt:
            groups.extend(self.adamw_opt.param_groups)
        return groups

    def get_param_split(self) -> dict[str, int]:
        """Return the number of scalar elements routed to each sub-optimizer."""
        return {
            "muon": self.muon_param_elems,
            "adamw": self.adamw_param_elems,
        }

    def __repr__(self) -> str:
        return (
            f"HybridMuonAdamW(muon_elems={self.muon_param_elems:,}, "
            f"adamw_elems={self.adamw_param_elems:,})"
        )


# ---------------------------------------------------------------------------
# Builder function
# ---------------------------------------------------------------------------

def build_hybrid_muon_adamw(
    model: nn.Module,
    *,
    lr: float = 2.5e-4,
    weight_decay: float = 0.02,
    momentum: float = 0.95,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-7,
    ns_coefficients: Sequence[float] | None = None,
    ns_steps: int | None = None,
    fused: str | bool = "auto",
    muon_filter_fn: Any | None = None,
) -> HybridMuonAdamW:
    """Build a hybrid Muon + AdamW optimizer for *model*.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters will be optimized.
    lr : float
        Learning rate for both sub-optimizers.
    weight_decay : float
        Weight decay for both sub-optimizers.
    momentum : float
        Muon momentum coefficient.
    betas : tuple
        AdamW beta coefficients.
    eps : float
        Muon epsilon.
    ns_coefficients : sequence of float, optional
        Newton-Schulz iteration coefficients for Muon.
    ns_steps : int, optional
        Number of Newton-Schulz iterations for Muon.
    fused : str or bool
        ``"auto"`` enables fused AdamW on CUDA. Pass ``True``/``False`` to
        force.
    muon_filter_fn : callable, optional
        Custom ``(name, param) -> bool`` function to decide Muon eligibility.
        Defaults to :func:`is_muon_candidate`.

    Returns
    -------
    HybridMuonAdamW

    Raises
    ------
    RuntimeError
        If ``torch.optim.Muon`` is not available in the current PyTorch build
        and there are Muon-eligible parameters.
    """
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError(
            "torch.optim.Muon is not available in this PyTorch build. "
            "Requires PyTorch >= 2.9."
        )

    filter_fn = muon_filter_fn or is_muon_candidate

    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if filter_fn(name, param):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    # --- Muon sub-optimizer ---
    muon_kwargs: dict[str, Any] = {
        "lr": lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "eps": eps,
    }
    if ns_coefficients is not None:
        muon_kwargs["ns_coefficients"] = tuple(ns_coefficients)
    if ns_steps is not None:
        muon_kwargs["ns_steps"] = int(ns_steps)

    muon_opt = (
        torch.optim.Muon(muon_params, **muon_kwargs)  # type: ignore[attr-defined]
        if muon_params
        else None
    )

    # --- AdamW sub-optimizer ---
    use_fused: bool
    if fused == "auto":
        use_fused = torch.cuda.is_available()
    else:
        use_fused = bool(fused)

    adamw_kwargs: dict[str, Any] = {
        "lr": lr,
        "betas": betas,
        "weight_decay": weight_decay,
    }
    if use_fused:
        adamw_kwargs["fused"] = True

    adamw_opt = (
        torch.optim.AdamW(adamw_params, **adamw_kwargs) if adamw_params else None
    )

    muon_elems = int(sum(p.numel() for p in muon_params))
    adamw_elems = int(sum(p.numel() for p in adamw_params))

    return HybridMuonAdamW(muon_opt, adamw_opt, muon_elems, adamw_elems)
