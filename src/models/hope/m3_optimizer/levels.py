"""
Level scheduling and multi-level optimizer management.

Provides a deterministic clock for nested-learning level updates and a manager
that pairs each level with its own :class:`DeepMomentum` inner optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, MutableMapping, Sequence

import torch
from torch import nn

from src.models.hope.m3_optimizer.deep_momentum import DeepMomentum
from src.models.hope.m3_optimizer.factory import build_deep_momentum


# ============================================================================
# LevelSpec / LevelState / LevelClock
# ============================================================================

@dataclass(frozen=True)
class LevelSpec:
    """Configuration for a single nested-learning level.

    Parameters
    ----------
    name : str
        Unique identifier (e.g. ``"fast"``, ``"slow"``).
    update_period : int
        How many clock ticks between updates.
    warmup_steps : int
        Skip updates before this many ticks.
    jitter : int
        Deterministic period variation: ``period + (step % (jitter + 1))``.
    optimizer_key : str or None
        Key into the optimizer config dict; ``None`` → ``"default"``.
    """

    name: str
    update_period: int
    warmup_steps: int = 0
    jitter: int = 0
    optimizer_key: str | None = None

    def __post_init__(self) -> None:
        if self.update_period <= 0:
            raise ValueError(f"update_period for level {self.name} must be positive")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps for level {self.name} must be non-negative")
        if self.jitter < 0:
            raise ValueError(f"jitter for level {self.name} must be non-negative")


@dataclass
class LevelState:
    last_step: int = -1
    updates: int = 0


class LevelClock:
    """Deterministic scheduler for nested-learning level updates."""

    def __init__(self, specs: Sequence[LevelSpec]) -> None:
        if not specs:
            raise ValueError("LevelClock requires at least one LevelSpec")
        self._specs: Dict[str, LevelSpec] = {spec.name: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("Duplicate level names provided to LevelClock")
        self._state: MutableMapping[str, LevelState] = {
            name: LevelState() for name in self._specs
        }
        self._step: int = 0
        self._timeline: List[dict] = []

    @property
    def step(self) -> int:
        return self._step

    def tick(self) -> None:
        """Advance the clock by one step."""
        self._step += 1

    def should_update(self, name: str) -> bool:
        """Check if level *name* should fire at the current step."""
        spec = self._specs[name]
        state = self._state[name]
        if self._step < spec.warmup_steps:
            return False
        delta = self._step - state.last_step
        period = spec.update_period
        if spec.jitter:
            period = period + (self._step % (spec.jitter + 1))
        return state.last_step < 0 or delta >= period

    def record_update(self, name: str) -> None:
        """Record that level *name* was updated at the current step."""
        state = self._state[name]
        state.last_step = self._step
        state.updates += 1
        self._timeline.append({"step": self._step, "level": name})

    def levels_in_frequency_order(self) -> List[LevelSpec]:
        """Return specs sorted by ``update_period`` (fastest first)."""
        return sorted(self._specs.values(), key=lambda spec: spec.update_period)

    def stats(self) -> Dict[str, LevelState]:
        return {
            name: LevelState(state.last_step, state.updates)
            for name, state in self._state.items()
        }

    def timeline(self) -> List[dict]:
        return list(self._timeline)


def ensure_level_specs(entries: Iterable[LevelSpec]) -> List[LevelSpec]:
    """Validate and deduplicate an iterable of :class:`LevelSpec`."""
    specs = list(entries)
    seen: set[str] = set()
    ordered: List[LevelSpec] = []
    for spec in specs:
        if spec.name in seen:
            raise ValueError(f"Duplicate level spec {spec.name}")
        seen.add(spec.name)
        ordered.append(spec)
    return ordered


# ============================================================================
# LevelConfig / LevelOptimizerManager
# ============================================================================

@dataclass
class LevelConfig:
    """Configuration bundle for :class:`LevelOptimizerManager`.

    Parameters
    ----------
    specs : iterable of LevelSpec
        One spec per level.
    optimizer_configs : dict
        Maps optimizer keys to config dicts like
        ``{"type": "deep_momentum", "params": {"variant": "muon"}}``.
    default_lr : float
        Fallback learning rate if not specified per-optimizer.
    """

    specs: Iterable[LevelSpec]
    optimizer_configs: Dict[str, dict]
    default_lr: float


class LevelOptimizerManager:
    """Manages per-level :class:`DeepMomentum` optimizers with scheduling.

    Each level gets its own ``DeepMomentum`` instance and learning rate.
    The manager's :meth:`tick` / :meth:`should_update` / :meth:`optimize`
    cycle drives the training loop.
    """

    def __init__(self, config: LevelConfig) -> None:
        self.clock = LevelClock(list(config.specs))
        self.learning_rates: Dict[str, float] = {}
        self.optimizers: Dict[str, DeepMomentum] = {}
        self._last_metrics: Dict[str, Dict[str, float]] = {}

        for spec in config.specs:
            key = spec.optimizer_key or "default"
            optim_cfg = config.optimizer_configs.get(
                key, {"type": "deep_momentum", "params": {}}
            )
            lr = optim_cfg.get("lr", config.default_lr)
            params_cfg = optim_cfg.get("params", {})
            optimizer = build_deep_momentum(
                {"type": optim_cfg.get("type", "deep_momentum"), "params": params_cfg}
            )
            self.optimizers[spec.name] = optimizer
            self.learning_rates[spec.name] = lr

    def should_update(self, level: str) -> bool:
        return self.clock.should_update(level)

    def optimize(
        self,
        level: str,
        module: nn.Module,
        loss: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
    ) -> float:
        """Run one inner-optimizer step for *level*.

        Returns the total gradient norm (useful for logging).
        """
        if not self.should_update(level):
            return 0.0
        params = tuple(module.parameters())
        if not params:
            return 0.0
        grads = torch.autograd.grad(loss, params, retain_graph=False)
        optimizer = self.optimizers[level]
        lr = self.learning_rates[level]
        total_norm = 0.0
        with torch.no_grad():
            for param, grad in zip(params, grads, strict=True):
                if grad is None:
                    continue
                update = optimizer(grad, context=context)
                param.add_(update, alpha=-lr)
                total_norm += grad.norm().item()
        self.clock.record_update(level)
        metrics = getattr(optimizer, "last_metrics", None)
        if metrics:
            self._last_metrics[level] = dict(metrics)
        else:
            self._last_metrics[level] = {}
        return total_norm

    def tick(self) -> None:
        """Advance the level clock by one step."""
        self.clock.tick()

    def pop_last_metrics(self, level: str) -> Dict[str, float]:
        """Pop and return the last optimizer metrics for *level*."""
        return self._last_metrics.pop(level, {})
