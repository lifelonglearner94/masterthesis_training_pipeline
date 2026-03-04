"""Dynamic Nested Hierarchy — variable-depth chain of Self-Modifying Memories.

Manages L_t ∈ [L_min, L_max] levels of SelfModifyingMemory modules, composing
them as a nested chain (Jafari 2025):

    y_t = M_t^(1) ∘ M_t^(2) ∘ ... ∘ M_t^(L_t)(x_t)

Each level:
    - Has its own SelfModifyingMemory (Titan + meta-network g_ψ)
    - Has a learned update frequency f^(l) (continuous, mapped to discrete period)
    - Has per-level η and α parameters inside its SMM

The hierarchy starts with L_0 = L_init levels and can grow up to L_max via
the StructuralEvolutionController (separate module).

Shared Q/K/V projections feed all levels — levels specialise via their
internal weights, not input representations.

CL lifecycle:
    - reset_all_levels(): Reset clip-level memory states
    - clear_all_levels(): Clear active weights (pre-device-move)
    - detach_all_levels(): Detach gradient history (task boundaries)
    - freeze_inner_loops() / unfreeze_inner_loops(): DGD control
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.hope.dnh.smm import SMMConfig, SelfModifyingMemory

log = logging.getLogger(__name__)


@dataclass
class DNHConfig:
    """Configuration for the Dynamic Nested Hierarchy.

    Attributes:
        dim: Feature dimension.
        L_init: Initial number of levels (default: 2).
        L_max: Maximum number of levels (default: 5).
        L_min: Minimum number of levels (default: 2).
        titan_hidden_multiplier: Titan MLP hidden dim multiplier.
        titan_layers: Number of layers in Titan MLP.
        titan_activation: Activation function for Titan memory.
        titan_grad_clip_inner: Gradient clip for inner-loop DGD.
        titan_grad_clip_backward: Max gradient norm for DGD backward.
        titan_detach_interval: Detach memory graph every N steps.
        meta_hidden_dim: Hidden dim for meta-networks (None = dim).
        surprise_threshold: Min surprise to trigger DGD update.
        use_longterm_memory: Whether to use persistent longterm memory.
        longterm_hidden_multiplier: Longterm memory hidden dim multiplier.
        longterm_lr_scale: DGD learning rate scale for longterm memory.
    """

    dim: int = 384
    L_init: int = 2
    L_max: int = 5
    L_min: int = 2
    titan_hidden_multiplier: int = 2
    titan_layers: int = 2
    titan_activation: str = "gelu"
    titan_grad_clip_inner: float = 1.0
    titan_grad_clip_backward: float = 1.0
    titan_detach_interval: int = 0
    meta_hidden_dim: int | None = None
    surprise_threshold: float = 0.0
    use_longterm_memory: bool = False
    longterm_hidden_multiplier: int = 2
    longterm_lr_scale: float = 0.1


class DynamicNestedHierarchy(nn.Module):
    """Dynamic Nested Hierarchy of Self-Modifying Memories.

    Manages a variable-depth chain of SMM modules with:
        - Chain composition forward pass
        - Per-level frequency control (continuous → discrete period)
        - Level add/remove interface for structural evolution
        - Longterm memory support (optional, for CL persistence)

    The hierarchy is the core Phase B replacement in DNHHybridBlock,
    taking over from the single TitanMemory in Phase 8's HybridBlock.

    Args:
        config: DNHConfig with all hierarchy settings.
    """

    def __init__(self, config: DNHConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.dim

        # ─── Q/K/V projections (shared across all levels) ───
        self.mem_q_proj = nn.Linear(dim, dim, bias=False)
        self.mem_k_proj = nn.Linear(dim, dim, bias=False)
        self.mem_v_proj = nn.Linear(dim, dim, bias=False)

        # ─── Output projection (single, after full chain) ───
        self.mem_out_proj = nn.Linear(dim, dim, bias=True)

        # ─── SMM levels ───
        self.levels = nn.ModuleList()
        for i in range(config.L_init):
            self.levels.append(self._create_level(level_index=i))

        # ─── Per-level frequency parameters ───
        # Stored as raw parameters; softplus maps to R+.
        # Initial frequencies: level 0 fastest, higher levels progressively slower.
        # f_0 ≈ 1.0 (every step), f_1 ≈ 3.0, f_2 ≈ 5.0, etc.
        init_freqs = [1.0 + 2.0 * i for i in range(config.L_init)]
        self.freq_raw = nn.ParameterList([
            nn.Parameter(torch.tensor(self._inv_softplus(f)))
            for f in init_freqs
        ])

        # ─── Longterm memory (optional, single shared across all levels) ───
        self.use_longterm_memory = config.use_longterm_memory
        if config.use_longterm_memory:
            from src.models.hope.titan_memory import TitanMemory, TitanMemoryConfig
            longterm_cfg = TitanMemoryConfig(
                dim=dim,
                hidden_multiplier=config.longterm_hidden_multiplier,
                num_layers=config.titan_layers,
                activation=config.titan_activation,
                grad_clip_inner=config.titan_grad_clip_inner,
                grad_clip_backward=config.titan_grad_clip_backward,
                detach_interval=config.titan_detach_interval,
            )
            self.M_longterm = TitanMemory(longterm_cfg)
            self.longterm_gate = nn.Linear(dim, 1, bias=True)
            nn.init.constant_(self.longterm_gate.bias, 1.0)
            nn.init.zeros_(self.longterm_gate.weight)
            self.longterm_lr_scale = config.longterm_lr_scale

        # ─── State ───
        self.freeze_inner_loop: bool = False

        # Step counter for frequency-based scheduling
        self.register_buffer(
            "_step", torch.tensor(0, dtype=torch.long), persistent=False
        )

    # ─── Level creation ───

    def _create_level(self, level_index: int) -> SelfModifyingMemory:
        """Create a new SMM level with the given index."""
        cfg = self.config
        smm_cfg = SMMConfig(
            dim=cfg.dim,
            titan_hidden_multiplier=cfg.titan_hidden_multiplier,
            titan_layers=cfg.titan_layers,
            titan_activation=cfg.titan_activation,
            titan_grad_clip_inner=cfg.titan_grad_clip_inner,
            titan_grad_clip_backward=cfg.titan_grad_clip_backward,
            titan_detach_interval=cfg.titan_detach_interval,
            meta_hidden_dim=cfg.meta_hidden_dim,
            level_index=level_index,
        )
        return SelfModifyingMemory(smm_cfg)

    @staticmethod
    def _inv_softplus(y: float) -> float:
        """Inverse of softplus: x such that softplus(x) = y."""
        if y > 20.0:
            return y
        return math.log(math.exp(y) - 1.0)

    # ─── Properties ───

    @property
    def num_levels(self) -> int:
        """Current number of active levels."""
        return len(self.levels)

    @property
    def frequencies(self) -> list[float]:
        """Current update frequencies (softplus-mapped)."""
        return [F.softplus(f).item() for f in self.freq_raw]

    @property
    def periods(self) -> list[int]:
        """Integer update periods derived from frequencies."""
        return [max(1, round(F.softplus(f).item())) for f in self.freq_raw]

    # ─── Forward ───

    def forward(self, x: Tensor) -> Tensor:
        """Forward: project → chain composition → output projection.

        Args:
            x: [B, N, D] pre-normed input (contextualized by attention).

        Returns:
            [B, N, D] hierarchy output.
        """
        # Shared projections
        q = self.mem_q_proj(x)
        k = self.mem_k_proj(x)
        v = self.mem_v_proj(x)

        current_step = self._step.item()
        periods = self.periods

        # Chain composition: y = M^(1) ∘ M^(2) ∘ ... ∘ M^(L)(q)
        # Only levels whose period divides the current step participate.
        y = q
        for i, (level, period) in enumerate(zip(self.levels, periods)):
            # Frequency-based scheduling: level active if step % period == 0
            # Level 0 (fastest) is always active.
            if period <= 1 or current_step % period == 0:
                y = level(y, k, v)

                # DGD update for this level
                level.update_memory(
                    k=k, v=v, output=y,
                    surprise_threshold=self.config.surprise_threshold,
                    freeze_inner=self.freeze_inner_loop,
                )

        # Gated longterm memory (optional)
        if self.use_longterm_memory:
            output_longterm = self.M_longterm(q)
            gate = torch.sigmoid(self.longterm_gate(q))  # [B, N, 1]
            y = gate * y + (1.0 - gate) * output_longterm

            # Update longterm memory (slower DGD)
            if not self.freeze_inner_loop:
                retrieval_error = v - y
                surprise = self.M_longterm.surprise(retrieval_error)
                # Average η from the first level for longterm
                first_level = self.levels[0]
                eta = F.softplus(first_level.eta_base)
                alpha = torch.sigmoid(first_level.alpha_base)
                B, N, D = k.shape
                eta_lt = eta.unsqueeze(0).unsqueeze(0).expand(B, N, D) * self.longterm_lr_scale
                alpha_expanded = alpha.unsqueeze(0).unsqueeze(0).expand(B, N, D)
                self_target_lt = self.M_longterm.generate_self_target(v)
                self.M_longterm.compute_and_apply_update(
                    key=k,
                    value=self_target_lt,
                    error_signal=surprise,
                    lr=eta_lt,
                    alpha=alpha_expanded,
                )

        # Output projection
        output = self.mem_out_proj(y)

        # Advance step counter
        self._step += 1

        return output

    # ─── Structural evolution interface ───

    def add_level(self, init_from_level: int | None = None) -> bool:
        """Add a new level to the hierarchy.

        Args:
            init_from_level: Index of existing level to initialise from.
                If None, uses the outermost (last) level.

        Returns:
            True if level was added, False if at L_max.
        """
        if self.num_levels >= self.config.L_max:
            log.info(
                f"DNH: Cannot add level — already at L_max={self.config.L_max}"
            )
            return False

        new_idx = self.num_levels
        new_level = self._create_level(level_index=new_idx)

        # Move new level to same device as existing levels
        device = next(self.parameters()).device
        new_level = new_level.to(device)

        # Hebbian-like initialisation from existing level (paper Eq.)
        if init_from_level is None:
            init_from_level = self.num_levels - 1

        src_level = self.levels[init_from_level]
        with torch.no_grad():
            # Copy TitanMemory initial weights
            new_level.memory.w1.weight.copy_(src_level.memory.w1.weight)
            new_level.memory.w2.weight.copy_(src_level.memory.w2.weight)
            # Add Hebbian perturbation scaled by context direction
            # w1 is [hidden, dim], context is [dim] — use directional noise
            ctx = src_level.level_context.data
            ctx_norm = ctx / (ctx.norm() + 1e-8)
            w1_shape = new_level.memory.w1.weight.shape  # [hidden, dim]
            # Perturbation: rows are scaled by random projection of context
            random_proj = torch.randn(w1_shape[0], device=ctx.device)
            perturbation = 0.01 * random_proj.unsqueeze(1) * ctx_norm.unsqueeze(0)
            new_level.memory.w1.weight.add_(perturbation)
            # Copy η and α
            new_level.eta_base.copy_(src_level.eta_base)
            new_level.alpha_base.copy_(src_level.alpha_base)
            # Slightly different context to encourage specialisation
            new_level.level_context.copy_(src_level.level_context + torch.randn_like(ctx) * 0.02)

        self.levels.append(new_level)

        # New frequency: mean of existing frequencies
        mean_freq = sum(self.frequencies[:-1]) / max(len(self.frequencies) - 1, 1)
        new_freq_raw = nn.Parameter(
            torch.tensor(self._inv_softplus(mean_freq), device=device)
        )
        self.freq_raw.append(new_freq_raw)

        log.info(
            f"DNH: Added level {new_idx} (init from level {init_from_level}), "
            f"freq={mean_freq:.2f}, total levels={self.num_levels}"
        )
        return True

    def remove_level(self, level_index: int) -> bool:
        """Remove a level from the hierarchy.

        Args:
            level_index: Index of the level to remove.

        Returns:
            True if level was removed, False if at L_min or invalid index.
        """
        if self.num_levels <= self.config.L_min:
            log.info(
                f"DNH: Cannot remove level — already at L_min={self.config.L_min}"
            )
            return False
        if level_index < 0 or level_index >= self.num_levels:
            log.warning(f"DNH: Invalid level index {level_index}")
            return False

        # Remove from both ModuleList and ParameterList
        del self.levels[level_index]
        # ParameterList doesn't support __delitem__, so rebuild
        remaining = [p for i, p in enumerate(self.freq_raw) if i != level_index]
        self.freq_raw = nn.ParameterList(remaining)

        log.info(
            f"DNH: Removed level {level_index}, remaining levels={self.num_levels}"
        )
        return True

    # ─── CL lifecycle ───

    def reset_all_levels(self) -> None:
        """Reset all clip-level memory states."""
        for level in self.levels:
            level.reset_active_weights()
            level.reset_diagnostics()
        self._step.zero_()

        if self.use_longterm_memory:
            if self.M_longterm._active_w1 is None:
                self.M_longterm.reset_active_weights()
            else:
                self.M_longterm._active_w1 = self.M_longterm._active_w1.detach()
                self.M_longterm._active_w2 = self.M_longterm._active_w2.detach()
            self.M_longterm.reset_diagnostics()

    def clear_all_levels(self) -> None:
        """Clear all active weights (pre-device-move safety)."""
        for level in self.levels:
            level.clear_active_weights()
        if self.use_longterm_memory:
            self.M_longterm.clear_active_weights()

    def detach_all_levels(self) -> None:
        """Detach gradient history from all levels (CL task boundaries)."""
        for level in self.levels:
            level.detach_active_weights()
        if self.use_longterm_memory:
            self.M_longterm.detach_active_weights()

    def freeze_inner_loops(self) -> None:
        """Freeze DGD memory updates (evaluation mode)."""
        self.freeze_inner_loop = True

    def unfreeze_inner_loops(self) -> None:
        """Unfreeze DGD memory updates (training mode)."""
        self.freeze_inner_loop = False

    def reset_longterm_memory(self) -> None:
        """Explicitly reset longterm memory."""
        if self.use_longterm_memory:
            self.M_longterm.reset_active_weights()
            self.M_longterm.reset_diagnostics()

    # ─── Diagnostics ───

    def get_diagnostics(self) -> dict[str, float]:
        """Aggregate diagnostics from all levels."""
        diag: dict[str, float] = {}

        diag["dnh/num_levels"] = float(self.num_levels)

        for i, level in enumerate(self.levels):
            for key, val in level.get_diagnostics().items():
                diag[f"dnh/level_{i}/{key}"] = val

        # Log frequencies
        for i, f in enumerate(self.frequencies):
            diag[f"dnh/level_{i}/frequency"] = f

        if self.use_longterm_memory:
            for key, val in self.M_longterm.get_diagnostics().items():
                diag[f"dnh/longterm/{key}"] = val

        return diag

    def get_level_gradient_norms(self) -> list[float]:
        """Get gradient norms for each level (for structural evolution)."""
        return [level.get_gradient_norm() for level in self.levels]
