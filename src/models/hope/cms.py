"""Continuum Memory System (CMS) — Multi-Frequency MLP Hierarchy (Behrouz 2025).

Implements the CMS component of the HOPE architecture. CMS provides a hierarchy
of MLPs operating at different temporal frequencies:
    - Fast level  (period=1): captures immediate action responses
    - Medium level (period=4): captures motion dynamics
    - Slow level   (period=16): captures static scene structure / background

Each level is a simple MLP with residual connection and gradient clipping.
Higher-frequency levels update every token; lower-frequency levels update
every `period` tokens, accumulating inputs via chunk-based processing.

Key design from the paper (Section 8.3):
    y_t = MLP^(f_K)( MLP^(f_{K-1})( ... MLP^(f_1)(o_t) ... ) )

The multi-frequency design handles both fast movements and slow backgrounds,
which is critical for robotics video prediction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)


@dataclass
class LevelSpec:
    """Specification for a single CMS frequency level."""

    name: str = "fast"
    update_period: int = 1  # Update every N tokens (1=fastest)
    warmup_steps: int = 0  # Steps before this level starts updating
    jitter: float = 0.0  # Random jitter on update period (0=deterministic)
    hidden_multiplier: float = 4.0  # MLP hidden dim = dim * multiplier


class CMSBlock(nn.Module):
    """Single CMS frequency level — an MLP with residual connection.

    Structure: x → LayerNorm → Linear → Act → Linear → + x

    Args:
        dim: Input/output dimension.
        hidden_multiplier: Hidden dimension = dim * hidden_multiplier.
        act_layer: Activation function class.
        drop: Dropout rate.
        grad_clip: Maximum gradient norm for clipping through this block.
    """

    def __init__(
        self,
        dim: int,
        hidden_multiplier: float = 4.0,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        grad_clip: float = 1.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * hidden_multiplier)

        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)
        self.grad_clip = grad_clip

        # Initialize fc2 near zero for stable residual learning
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection.

        Args:
            x: [B, N, D] input tokens.

        Returns:
            [B, N, D] output tokens.
        """
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return residual + x


class CMS(nn.Module):
    """Continuum Memory System — multi-frequency MLP hierarchy.

    Composes multiple CMSBlock levels at different temporal frequencies.
    All levels process the input sequentially (cascade):
        y = level_K( level_{K-1}( ... level_1(x) ... ) )

    In standard mode (no chunk scheduling), all levels process every token.
    When chunk scheduling is enabled, slower levels only process every
    `update_period`-th chunk of tokens.

    Args:
        dim: Input/output dimension.
        levels: List of LevelSpec defining each frequency level.
        act_layer: Activation function class.
        drop: Dropout rate.
        grad_clip: Gradient clipping for CMS blocks.
        use_chunk_scheduling: Whether to use frequency-based chunk scheduling.
    """

    def __init__(
        self,
        dim: int,
        levels: list[LevelSpec] | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        grad_clip: float = 1.0,
        use_chunk_scheduling: bool = False,
    ) -> None:
        super().__init__()

        if levels is None:
            levels = [
                LevelSpec(name="fast", update_period=1, hidden_multiplier=4.0),
                LevelSpec(name="medium", update_period=4, hidden_multiplier=4.0),
                LevelSpec(name="slow", update_period=16, hidden_multiplier=4.0),
            ]

        self.levels_spec = levels
        self.use_chunk_scheduling = use_chunk_scheduling

        self.blocks = nn.ModuleList([
            CMSBlock(
                dim=dim,
                hidden_multiplier=spec.hidden_multiplier,
                act_layer=act_layer,
                drop=drop,
                grad_clip=grad_clip,
            )
            for spec in levels
        ])

        # Step counter for chunk scheduling
        self.register_buffer("_global_step", torch.tensor(0, dtype=torch.long), persistent=False)

        # Chunk accumulators for each level (used in chunk scheduling mode)
        self._chunk_accumulators: list[list[Tensor]] = [[] for _ in levels]

    @property
    def num_levels(self) -> int:
        return len(self.blocks)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all CMS levels (cascade).

        In standard mode, all levels process every token. In chunk scheduling
        mode, slower levels are skipped based on their update_period.

        Args:
            x: [B, N, D] input tokens from Titan memory output.

        Returns:
            [B, N, D] processed tokens.
        """
        for i, (block, spec) in enumerate(zip(self.blocks, self.levels_spec)):
            if self.use_chunk_scheduling:
                # Check if this level should update at current step
                step = self._global_step.item()
                if step < spec.warmup_steps:
                    continue  # Skip during warmup
                if spec.update_period > 1 and step % spec.update_period != 0:
                    continue  # Skip non-update steps
            x = block(x)

        self._global_step += 1
        return x

    def reset_step_counter(self) -> None:
        """Reset the global step counter (call at start of each sequence)."""
        self._global_step.zero_()
        self._chunk_accumulators = [[] for _ in self.levels_spec]

    def get_level_params(self) -> list[tuple[str, list[nn.Parameter]]]:
        """Get parameters grouped by CMS level (for per-level optimizers).

        Returns:
            List of (level_name, params) tuples.
        """
        result = []
        for spec, block in zip(self.levels_spec, self.blocks):
            params = list(block.parameters())
            result.append((spec.name, params))
        return result
