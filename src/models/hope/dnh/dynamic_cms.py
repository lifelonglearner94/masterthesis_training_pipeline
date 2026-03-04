"""Dynamic CMS — CMS with structural evolution capability (Phase 11).

Extends the Phase 8 CMS (Continuum Memory System) with DNH-style dynamic
level management:
    - Starts with the standard 3 levels (fast/medium/slow)
    - Can add new CMS frequency levels when per-level meta-loss is high
    - Can prune CMS levels when gradient norm contribution drops
    - Maximum CMS levels capped at cms_L_max (default: 5)

Uses the same CMSBlock class from cms.py for all levels — new levels are
zero-initialized for stable residual learning.

Frame-aware scheduling is preserved from the original CMS.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
from torch import Tensor

from src.models.hope.cms import CMS, CMSBlock, LevelSpec

log = logging.getLogger(__name__)


class DynamicCMS(nn.Module):
    """CMS with dynamic level add/prune capability.

    Wraps an nn.ModuleList of CMSBlock instances and a parallel list of
    LevelSpec metadata. Supports adding/removing levels at runtime for
    structural evolution.

    Args:
        dim: Feature dimension.
        levels: Initial list of LevelSpec for frequency levels.
        act_layer: Activation function class for all levels.
        drop: Dropout rate for all CMS blocks.
        grad_clip: Gradient clipping norm for CMS blocks.
        use_chunk_scheduling: Enable frame-aware temporal scheduling.
        cms_L_max: Maximum number of CMS levels (default: 5).
        cms_L_min: Minimum number of CMS levels (default: 2).
    """

    def __init__(
        self,
        dim: int,
        levels: list[LevelSpec] | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        grad_clip: float = 1.0,
        use_chunk_scheduling: bool = False,
        cms_L_max: int = 5,
        cms_L_min: int = 2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.act_layer = act_layer
        self.drop = drop
        self.grad_clip = grad_clip
        self.use_chunk_scheduling = use_chunk_scheduling
        self.cms_L_max = cms_L_max
        self.cms_L_min = cms_L_min

        if levels is None:
            levels = [
                LevelSpec(name="fast", update_period=1, hidden_multiplier=2.0),
                LevelSpec(name="medium", update_period=3, hidden_multiplier=2.5),
                LevelSpec(name="slow", update_period=7, hidden_multiplier=3.0),
            ]

        self.levels_spec: list[LevelSpec] = list(levels)

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
        self.register_buffer(
            "_global_step", torch.tensor(0, dtype=torch.long), persistent=False
        )

    @property
    def num_levels(self) -> int:
        return len(self.blocks)

    # ─── Forward ───

    def forward(
        self,
        x: Tensor,
        T: int | None = None,
        tokens_per_frame: int | None = None,
    ) -> Tensor:
        """Forward pass through all CMS levels (cascade).

        Identical to the original CMS.forward() — supports both standard
        mode and frame-aware chunk scheduling.

        Args:
            x: [B, N, D] input tokens from Titan memory output.
            T: Number of temporal frames.
            tokens_per_frame: Tokens per frame (H*W + action_tokens).

        Returns:
            [B, N, D] processed tokens.
        """
        if not self.use_chunk_scheduling or T is None or tokens_per_frame is None:
            for block in self.blocks:
                x = block(x)
            return x

        B, N, D = x.shape

        for block, spec in zip(self.blocks, self.levels_spec):
            if spec.update_period <= 1:
                x = block(x)
                continue

            frame_indices = torch.arange(T, device=x.device)
            active_frames = (frame_indices % spec.update_period == 0)

            if not active_frames.any():
                continue

            if active_frames.all():
                x = block(x)
            else:
                token_mask = active_frames.repeat_interleave(tokens_per_frame)
                token_mask = token_mask[:N]
                indices = token_mask.nonzero(as_tuple=True)[0]
                x_sub = x[:, indices, :]
                x_sub = block(x_sub)
                x = x.clone()
                x[:, indices, :] = x_sub

        return x

    # ─── Dynamic level management ───

    def add_level(
        self,
        name: str | None = None,
        update_period: int | None = None,
        hidden_multiplier: float | None = None,
    ) -> bool:
        """Add a new CMS frequency level.

        If parameters are not specified, they are interpolated from existing
        levels: period = max existing period + 2, hidden = mean hidden multiplier.

        Args:
            name: Level name (auto-generated if None).
            update_period: Update period for the new level.
            hidden_multiplier: Hidden dimension multiplier.

        Returns:
            True if level was added, False if at maximum.
        """
        if self.num_levels >= self.cms_L_max:
            log.info(f"DynamicCMS: Cannot add level — already at max {self.cms_L_max}")
            return False

        # Auto-determine parameters if not given
        if update_period is None:
            max_period = max(s.update_period for s in self.levels_spec)
            update_period = max_period + 2
        if hidden_multiplier is None:
            hidden_multiplier = sum(s.hidden_multiplier for s in self.levels_spec) / len(self.levels_spec)
        if name is None:
            name = f"dynamic_{self.num_levels}"

        spec = LevelSpec(
            name=name,
            update_period=update_period,
            hidden_multiplier=hidden_multiplier,
        )
        block = CMSBlock(
            dim=self.dim,
            hidden_multiplier=hidden_multiplier,
            act_layer=self.act_layer,
            drop=self.drop,
            grad_clip=self.grad_clip,
        )

        # Move to same device
        device = next(self.parameters()).device
        block = block.to(device)

        self.levels_spec.append(spec)
        self.blocks.append(block)

        log.info(
            f"DynamicCMS: Added level '{name}' "
            f"(period={update_period}, hidden_mult={hidden_multiplier:.1f}), "
            f"total={self.num_levels}"
        )
        return True

    def remove_level(self, level_index: int) -> bool:
        """Remove a CMS level.

        Args:
            level_index: Index of the level to remove.

        Returns:
            True if level was removed, False if at minimum or invalid.
        """
        if self.num_levels <= self.cms_L_min:
            log.info(f"DynamicCMS: Cannot remove level — at minimum {self.cms_L_min}")
            return False
        if level_index < 0 or level_index >= self.num_levels:
            log.warning(f"DynamicCMS: Invalid index {level_index}")
            return False

        name = self.levels_spec[level_index].name
        del self.levels_spec[level_index]
        del self.blocks[level_index]

        log.info(
            f"DynamicCMS: Removed level '{name}' (index {level_index}), "
            f"remaining={self.num_levels}"
        )
        return True

    def get_level_gradient_norms(self) -> list[float]:
        """Get gradient norms per CMS level (for pruning decisions)."""
        norms = []
        for block in self.blocks:
            total = 0.0
            for p in block.parameters():
                if p.grad is not None:
                    total += p.grad.data.norm(2).item() ** 2
            norms.append(total ** 0.5)
        return norms

    # ─── Lifecycle ───

    def reset_step_counter(self) -> None:
        """Reset the global step counter."""
        self._global_step.zero_()

    def get_diagnostics(self) -> dict[str, float]:
        """Get diagnostic metrics for all CMS levels."""
        diag: dict[str, float] = {"dynamic_cms/num_levels": float(self.num_levels)}
        for i, spec in enumerate(self.levels_spec):
            diag[f"dynamic_cms/level_{i}/period"] = float(spec.update_period)
            diag[f"dynamic_cms/level_{i}/hidden_mult"] = spec.hidden_multiplier
        return diag

    def get_level_params(self) -> list[tuple[str, list[nn.Parameter]]]:
        """Get parameters grouped by CMS level (for per-level optimizers)."""
        result = []
        for spec, block in zip(self.levels_spec, self.blocks):
            params = list(block.parameters())
            result.append((spec.name, params))
        return result
