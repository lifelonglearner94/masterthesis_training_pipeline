"""DNH Hybrid Block — Self-Attention + Dynamic Nested Hierarchy + Dynamic CMS (Phase 11).

Structurally identical to Phase 8's HybridBlock but with DNH components:
    Phase A: ACRoPEAttention — full multi-head self-attention with 3D RoPE
             (UNCHANGED from Phase 8)
    Phase B: DynamicNestedHierarchy — variable-depth chain of SMMs
             (replaces single TitanMemory)
    Phase C: DynamicCMS — multi-frequency MLP cascade with add/prune
             (replaces static CMS)

Key differences from Phase 8 HybridBlock:
    1. Phase B: Single TitanMemory → DynamicNestedHierarchy (2–5 SMM levels)
    2. Phase C: Static CMS → DynamicCMS (3–5 CMSBlock levels)
    3. Per-level η/α moved inside each SelfModifyingMemory
    4. Shared Q/K/V projections now live in DynamicNestedHierarchy
    5. Structural evolution interface exposed for the controller

The forward() signature is identical to HybridBlock for drop-in compatibility
in the backbone processing loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from src.models.ac_predictor.utils.modules import ACRoPEAttention, DropPath
from src.models.hope.cms import LevelSpec
from src.models.hope.dnh.dynamic_cms import DynamicCMS
from src.models.hope.dnh.dynamic_hierarchy import DNHConfig, DynamicNestedHierarchy
from src.models.hope.titan_memory import _backward_grad_clip

log = logging.getLogger(__name__)


@dataclass
class DNHHybridBlockConfig:
    """Configuration for a single DNH Hybrid block.

    Attributes shared with HybridBlockConfig are named identically for
    config compatibility.
    """

    dim: int = 384
    num_heads: int = 16

    # ─── Attention settings (Phase A, unchanged) ───
    qkv_bias: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    grid_size: int = 16

    # ─── DNH settings (Phase B) ───
    dnh_L_init: int = 2
    dnh_L_max: int = 5
    dnh_L_min: int = 2
    titan_hidden_multiplier: int = 2
    titan_layers: int = 2
    titan_activation: str = "gelu"
    titan_grad_clip_inner: float = 1.0
    titan_grad_clip_backward: float = 1.0
    titan_detach_interval: int = 0
    surprise_threshold: float = 0.0
    meta_hidden_dim: int | None = None

    # ─── CMS settings (Phase C) ───
    cms_levels: list[LevelSpec] = field(
        default_factory=lambda: [
            LevelSpec(name="fast", update_period=1, hidden_multiplier=2.0),
            LevelSpec(name="medium", update_period=3, hidden_multiplier=2.5),
            LevelSpec(name="slow", update_period=7, hidden_multiplier=3.0),
        ]
    )
    cms_use_chunk_scheduling: bool = False
    cms_L_max: int = 5
    cms_L_min: int = 2

    # ─── Longterm memory ───
    use_longterm_memory: bool = False
    longterm_hidden_multiplier: int = 2
    longterm_lr_scale: float = 0.1

    # ─── Regularization ───
    drop_path: float = 0.0
    drop: float = 0.0


class DNHHybridBlock(nn.Module):
    """DNH Hybrid block: Attention + Dynamic Nested Hierarchy + Dynamic CMS.

    Architecture per forward pass:
        1. Phase A: x = x + Attention(norm(x))      [token interaction + RoPE]
        2. Phase B: x = x + DNH(norm(x))            [dynamic nested memory chain]
        3. Phase C: x = x + DynamicCMS(norm(x))     [dynamic multi-frequency MLPs]

    Phase A is identical to Phase 8. Phase B replaces the single TitanMemory
    with a DynamicNestedHierarchy containing 2–5 SelfModifyingMemory levels.
    Phase C replaces the static CMS with DynamicCMS supporting add/prune.

    Args:
        config: DNHHybridBlockConfig with all settings.
    """

    def __init__(self, config: DNHHybridBlockConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.dim

        # ─── Phase A: Self-Attention with 3D RoPE (UNCHANGED) ───
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ACRoPEAttention(
            dim=dim,
            num_heads=config.num_heads,
            qkv_bias=config.qkv_bias,
            attn_drop=config.attn_drop,
            proj_drop=config.proj_drop,
            use_sdpa=True,
            grid_size=config.grid_size,
        )

        # ─── Phase B: Dynamic Nested Hierarchy ───
        self.norm2 = nn.LayerNorm(dim)
        dnh_cfg = DNHConfig(
            dim=dim,
            L_init=config.dnh_L_init,
            L_max=config.dnh_L_max,
            L_min=config.dnh_L_min,
            titan_hidden_multiplier=config.titan_hidden_multiplier,
            titan_layers=config.titan_layers,
            titan_activation=config.titan_activation,
            titan_grad_clip_inner=config.titan_grad_clip_inner,
            titan_grad_clip_backward=config.titan_grad_clip_backward,
            titan_detach_interval=config.titan_detach_interval,
            meta_hidden_dim=config.meta_hidden_dim,
            surprise_threshold=config.surprise_threshold,
            use_longterm_memory=config.use_longterm_memory,
            longterm_hidden_multiplier=config.longterm_hidden_multiplier,
            longterm_lr_scale=config.longterm_lr_scale,
        )
        self.dnh = DynamicNestedHierarchy(dnh_cfg)

        # ─── Phase C: Dynamic CMS ───
        self.norm3 = nn.LayerNorm(dim)
        self.dynamic_cms = DynamicCMS(
            dim=dim,
            levels=config.cms_levels,
            drop=config.drop,
            use_chunk_scheduling=config.cms_use_chunk_scheduling,
            cms_L_max=config.cms_L_max,
            cms_L_min=config.cms_L_min,
        )

        # ─── Drop path (stochastic depth) ───
        self.drop_path = (
            DropPath(config.drop_path) if config.drop_path > 0.0 else nn.Identity()
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        T: int | None = None,
        H: int | None = None,
        W: int | None = None,
        action_tokens: int = 0,
        target_timestep: int | None = None,
    ) -> Tensor:
        """Forward pass: Attention → DNH → Dynamic CMS.

        Signature compatible with Phase 8 HybridBlock.forward() for
        drop-in substitution in the backbone processing loop.

        Args:
            x: [B, N_total, D] input tokens.
            mask: Optional position mask.
            attn_mask: Block causal attention mask [N_total, N_total].
            T: Number of temporal frames.
            H: Grid height (patches).
            W: Grid width (patches).
            action_tokens: Number of conditioning tokens per frame.
            target_timestep: Target frame index for RoPE jump prediction.

        Returns:
            [B, N_total, D] output tokens.
        """
        # ─── Phase A: Self-Attention (with 3D RoPE) ───
        x = self._phase_a(x, mask, attn_mask, T, H, W, action_tokens, target_timestep)

        # ─── Phase B: Dynamic Nested Hierarchy ───
        # NOT safe for activation checkpointing (DGD mutates memory state)
        y = self.norm2(x)
        y = self.dnh(y)
        x = x + self.drop_path(y)

        # ─── Phase C: Dynamic CMS ───
        x = self._phase_c(x, T, H, W, action_tokens)

        return x

    def _phase_a(
        self,
        x: Tensor,
        mask: Tensor | None,
        attn_mask: Tensor | None,
        T: int | None,
        H: int | None,
        W: int | None,
        action_tokens: int,
        target_timestep: int | None,
    ) -> Tensor:
        """Phase A: Self-Attention with 3D RoPE (pure, checkpoint-safe)."""
        y = self.norm1(x)
        y = self.attn(
            y, mask=mask, attn_mask=attn_mask,
            T=T, H=H, W=W,
            action_tokens=action_tokens,
            target_timestep=target_timestep,
        )
        return x + self.drop_path(y)

    def _phase_c(
        self,
        x: Tensor,
        T: int | None,
        H: int | None,
        W: int | None,
        action_tokens: int,
    ) -> Tensor:
        """Phase C: Dynamic CMS (pure, checkpoint-safe)."""
        y = self.norm3(x)
        tokens_per_frame = (
            (action_tokens + H * W) if H is not None and W is not None else None
        )
        y = self.dynamic_cms(y, T=T, tokens_per_frame=tokens_per_frame)
        return x + self.drop_path(y)

    # ─── CL lifecycle hooks ───

    def reset_memory_state(self) -> None:
        """Reset clip-level memory states (call between clips)."""
        self.dnh.reset_all_levels()
        self.dynamic_cms.reset_step_counter()

    def reset_longterm_memory(self) -> None:
        """Explicitly reset longterm memory."""
        self.dnh.reset_longterm_memory()

    # ─── Diagnostics ───

    def get_diagnostics(self) -> dict[str, float]:
        """Get diagnostic metrics from this block."""
        diag: dict[str, float] = {}

        # DNH diagnostics
        for key, val in self.dnh.get_diagnostics().items():
            diag[key] = val

        # Dynamic CMS diagnostics
        for key, val in self.dynamic_cms.get_diagnostics().items():
            diag[key] = val

        return diag
