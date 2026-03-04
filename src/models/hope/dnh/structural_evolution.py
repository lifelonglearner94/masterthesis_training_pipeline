"""Structural Evolution Controller — manages level add/prune/frequency modulation.

Implements the three structural evolution mechanisms from Jafari 2025:

1. **Level Addition**: Insert new SMM level when meta-loss > τ_add AND L < L_max.
   New level is initialised from the outermost level via Hebbian-like rule.

2. **Level Pruning**: Remove SMM level when its gradient norm contribution < ε_prune
   for a sustained period (EMA smoothing prevents premature removal) AND L > L_min.

3. **Frequency Modulation**: Update frequencies via LSS-based momentum rule:
       Δf^(l) = γ · LSS^(l)
       m_{t+1}^(l) = β · m_t^(l) + (1 - β) · LSS^(l)
       f_{t+1}^(l) = f^(l) + η_f · m_{t+1}^(l)

The controller operates on DynamicNestedHierarchy instances and is called from
the Lightning module's training_step at a configurable interval.

This is a coordination module only — it does NOT own any weights. All weights
live in the DynamicNestedHierarchy and its SMM levels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from src.models.hope.dnh.dynamic_hierarchy import DynamicNestedHierarchy

log = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """Configuration for structural evolution.

    Attributes:
        tau_add: Meta-loss threshold for level addition.
        epsilon_prune: Gradient norm threshold for level pruning.
        gamma_freq: Frequency modulation coefficient.
        eta_freq: Learning rate for frequency updates.
        beta_momentum: Momentum coefficient for frequency modulation.
        f_min: Minimum allowed frequency.
        f_max: Maximum allowed frequency.
        prune_patience: Steps below ε before pruning (EMA smoothing).
        evolution_interval: Check evolution every N training steps.
        enable_addition: Whether level addition is enabled.
        enable_pruning: Whether level pruning is enabled.
        enable_freq_modulation: Whether frequency modulation is enabled.
        warmup_steps: Skip evolution for the first N steps.
    """

    tau_add: float = 0.5
    epsilon_prune: float = 0.01
    gamma_freq: float = 0.1
    eta_freq: float = 0.01
    beta_momentum: float = 0.9
    f_min: float = 0.5
    f_max: float = 20.0
    prune_patience: int = 100
    evolution_interval: int = 50
    enable_addition: bool = True
    enable_pruning: bool = True
    enable_freq_modulation: bool = True
    warmup_steps: int = 200


class StructuralEvolutionController:
    """Manages structural evolution of DynamicNestedHierarchy instances.

    Stateful controller that tracks:
        - Per-level gradient norm EMAs (for pruning decisions)
        - Per-level frequency momentum (for smooth frequency modulation)
        - Evolution event counts (for logging/diagnostics)
        - Steps-below-threshold counters (for pruning patience)

    Called from the Lightning module's training_step to:
        1. Check if structural changes are needed
        2. Execute add/prune/modulate operations
        3. Log evolution events

    The controller does NOT own any trainable parameters — it operates on
    the DynamicNestedHierarchy's levels and frequency parameters.

    Args:
        config: EvolutionConfig with all thresholds and hyperparameters.
        max_levels: Maximum number of levels (from DNHConfig.L_max).
    """

    def __init__(self, config: EvolutionConfig, max_levels: int = 5) -> None:
        self.config = config
        self.max_levels = max_levels

        # ─── Per-level tracking state ───
        # These are lists that grow/shrink with the hierarchy.
        self._grad_norm_ema: list[float] = []
        self._freq_momentum: list[float] = []
        self._steps_below_threshold: list[int] = []

        # ─── Global counters ───
        self._global_step: int = 0
        self._additions: int = 0
        self._prunings: int = 0
        self._last_meta_loss: float = 0.0

    def initialize_tracking(self, num_levels: int) -> None:
        """Initialize tracking state for the given number of levels.

        Should be called once at the start of training (or when the
        hierarchy is first created).
        """
        self._grad_norm_ema = [0.0] * num_levels
        self._freq_momentum = [0.0] * num_levels
        self._steps_below_threshold = [0] * num_levels

    def step(
        self,
        hierarchy: "DynamicNestedHierarchy",  # noqa: F821 — forward ref
        meta_loss: float,
    ) -> dict[str, int | float]:
        """Execute one evolution step.

        Should be called every training step. The controller internally
        tracks the interval and only performs evolution checks at the
        configured `evolution_interval`.

        Args:
            hierarchy: The DynamicNestedHierarchy to evolve.
            meta_loss: Current meta-loss value (scalar).

        Returns:
            Dict with evolution event info for logging.
        """
        self._global_step += 1
        self._last_meta_loss = meta_loss

        events: dict[str, int | float] = {
            "evolution/step": self._global_step,
            "evolution/meta_loss": meta_loss,
            "evolution/num_levels": hierarchy.num_levels,
        }

        # Skip during warmup
        if self._global_step < self.config.warmup_steps:
            return events

        # Only check at interval
        if self._global_step % self.config.evolution_interval != 0:
            return events

        # Ensure tracking arrays match current hierarchy size
        self._sync_tracking(hierarchy.num_levels)

        # 1. Get gradient norms per level
        grad_norms = hierarchy.get_level_gradient_norms()
        events["evolution/mean_grad_norm"] = (
            sum(grad_norms) / max(len(grad_norms), 1)
        )

        # 2. Update EMA of gradient norms
        ema_decay = 0.95
        for i, gn in enumerate(grad_norms):
            if i < len(self._grad_norm_ema):
                self._grad_norm_ema[i] = (
                    ema_decay * self._grad_norm_ema[i] + (1 - ema_decay) * gn
                )

        # 3. Frequency modulation (LSS-based)
        if self.config.enable_freq_modulation:
            self._modulate_frequencies(hierarchy, grad_norms)

        # 4. Level addition
        added = False
        if self.config.enable_addition:
            added = self._maybe_add_level(hierarchy, meta_loss)
            if added:
                events["evolution/added_level"] = 1
                self._additions += 1

        # 5. Level pruning (skip if we just added)
        pruned = False
        if self.config.enable_pruning and not added:
            pruned = self._maybe_prune_level(hierarchy)
            if pruned:
                events["evolution/pruned_level"] = 1
                self._prunings += 1

        events["evolution/total_additions"] = self._additions
        events["evolution/total_prunings"] = self._prunings

        return events

    def _sync_tracking(self, num_levels: int) -> None:
        """Ensure tracking arrays match the current number of levels."""
        while len(self._grad_norm_ema) < num_levels:
            self._grad_norm_ema.append(0.0)
            self._freq_momentum.append(0.0)
            self._steps_below_threshold.append(0)
        # Trim if levels were removed externally
        self._grad_norm_ema = self._grad_norm_ema[:num_levels]
        self._freq_momentum = self._freq_momentum[:num_levels]
        self._steps_below_threshold = self._steps_below_threshold[:num_levels]

    def _maybe_add_level(
        self,
        hierarchy: "DynamicNestedHierarchy",  # noqa: F821
        meta_loss: float,
    ) -> bool:
        """Add a level if meta-loss exceeds threshold."""
        if meta_loss > self.config.tau_add:
            return hierarchy.add_level()
        return False

    def _maybe_prune_level(
        self,
        hierarchy: "DynamicNestedHierarchy",  # noqa: F821
    ) -> bool:
        """Prune a level if its gradient norm is consistently below threshold.

        Uses EMA-smoothed gradient norms with a patience counter to avoid
        premature pruning from momentary fluctuations.

        Prunes the level with the lowest gradient norm contribution,
        NOT level 0 (the fastest / innermost level is protected).
        """
        # Find candidate: lowest EMA gradient norm, excluding level 0
        if hierarchy.num_levels <= hierarchy.config.L_min:
            return False

        candidates = list(range(1, len(self._grad_norm_ema)))
        if not candidates:
            return False

        # Update patience counters
        for i in candidates:
            if self._grad_norm_ema[i] < self.config.epsilon_prune:
                self._steps_below_threshold[i] += 1
            else:
                self._steps_below_threshold[i] = 0

        # Find level that has been below threshold longest
        best_candidate = max(candidates, key=lambda i: self._steps_below_threshold[i])

        if self._steps_below_threshold[best_candidate] >= self.config.prune_patience:
            log.info(
                f"Evolution: Pruning level {best_candidate} "
                f"(EMA grad norm={self._grad_norm_ema[best_candidate]:.6f}, "
                f"below ε={self.config.epsilon_prune} for "
                f"{self._steps_below_threshold[best_candidate]} steps)"
            )
            success = hierarchy.remove_level(best_candidate)
            if success:
                # Remove tracking state for pruned level
                self._grad_norm_ema.pop(best_candidate)
                self._freq_momentum.pop(best_candidate)
                self._steps_below_threshold.pop(best_candidate)
            return success
        return False

    def _modulate_frequencies(
        self,
        hierarchy: "DynamicNestedHierarchy",  # noqa: F821
        grad_norms: list[float],
    ) -> None:
        """Modulate per-level frequencies via LSS-based momentum.

        Δf^(l) = γ · LSS^(l)
        m_{t+1}^(l) = β · m_t^(l) + (1 - β) · LSS^(l)
        f_{t+1}^(l) = f^(l) + η_f · m_{t+1}^(l)

        LSS^(l) is approximated by the gradient norm of level l.
        """
        cfg = self.config
        for i in range(min(len(grad_norms), len(self._freq_momentum))):
            if i >= len(hierarchy.freq_raw):
                break

            lss = grad_norms[i]

            # Update momentum
            self._freq_momentum[i] = (
                cfg.beta_momentum * self._freq_momentum[i]
                + (1 - cfg.beta_momentum) * lss
            )

            # Update frequency parameter (in raw space, before softplus)
            # We modify the raw parameter directly since softplus is monotone.
            delta = cfg.eta_freq * self._freq_momentum[i]
            with torch.no_grad():
                hierarchy.freq_raw[i].add_(delta)

                # Clamp in mapped space: ensure softplus(raw) ∈ [f_min, f_max]
                # softplus^-1(f_min) ≤ raw ≤ softplus^-1(f_max)
                import math
                raw_min = math.log(math.exp(cfg.f_min) - 1) if cfg.f_min < 20 else cfg.f_min
                raw_max = math.log(math.exp(cfg.f_max) - 1) if cfg.f_max < 20 else cfg.f_max
                hierarchy.freq_raw[i].clamp_(raw_min, raw_max)

    def get_diagnostics(self) -> dict[str, float]:
        """Get evolution diagnostics for logging."""
        return {
            "evolution/global_step": float(self._global_step),
            "evolution/total_additions": float(self._additions),
            "evolution/total_prunings": float(self._prunings),
            "evolution/last_meta_loss": self._last_meta_loss,
        }
