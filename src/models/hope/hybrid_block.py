"""Hybrid Block — Self-Attention + Titan Memory + CMS (Phase 8).

Combines the best of both architectures:
    Phase A: ACRoPEAttention — full multi-head self-attention with 3D RoPE
             Provides content-based token interaction (what the ViT baseline has)
    Phase B: Titan Memory — DGD-based in-context adaptation (what HOPE adds)
             Single M_memory with simplified η/α (learned per-block scalars)
    Phase C: CMS — multi-frequency MLP cascade
             Replaces the standard MLP/FFN with frequency-aware processing

Key design decisions vs original HOPE:
    1. Attention is RETAINED, not replaced → tokens can see each other
    2. Only 1 Titan memory (M_memory) per block instead of 5
       K,V come from simple projections, η/α are learned parameters
    3. Deeper architecture (12 blocks vs 5) enabled by lower per-block cost
    4. RoPE works in attention (Q·K dot product); temporal embeddings
       are additive at the model level (both coexist)

Comparison of what each phase provides:
    | Capability          | ViT Baseline (24×) | HOPE (5×)  | Hybrid (12×) |
    |---------------------|--------------------|------------|--------------|
    | Token interaction   | ✓ Attention        | ✗ (mixer)  | ✓ Attention  |
    | Positional encoding | ✓ 3D RoPE          | ✗ (embeds) | ✓ RoPE+Embed |
    | In-context adapt.   | ✗                  | ✓ Titan    | ✓ Titan      |
    | Multi-freq features | ✗ (single MLP)     | ✓ CMS      | ✓ CMS        |
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.ac_predictor.utils.modules import ACRoPEAttention, DropPath
from src.models.hope.cms import CMS, LevelSpec
from src.models.hope.titan_memory import TitanMemory, TitanMemoryConfig, _backward_grad_clip

log = logging.getLogger(__name__)

# Constants
DEFAULT_ETA_INIT = -3.0    # softplus(-3) ≈ 0.049 — conservative DGD lr
DEFAULT_ALPHA_INIT = 2.0   # sigmoid(2) ≈ 0.88 — moderate decay


@dataclass
class HybridBlockConfig:
    """Configuration for a single Hybrid block."""

    dim: int = 384
    num_heads: int = 16

    # ─── Attention settings ───
    qkv_bias: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    grid_size: int = 16

    # ─── Titan memory settings (simplified: single M_memory) ───
    titan_hidden_multiplier: int = 2
    titan_layers: int = 2
    titan_activation: str = "gelu"
    titan_grad_clip_inner: float = 1.0
    titan_grad_clip_backward: float = 1.0
    titan_detach_interval: int = 0
    surprise_threshold: float = 0.0

    # ─── CMS settings ───
    cms_levels: list[LevelSpec] = field(
        default_factory=lambda: [
            LevelSpec(name="fast", update_period=1, hidden_multiplier=2.0),
            LevelSpec(name="medium", update_period=3, hidden_multiplier=2.5),
            LevelSpec(name="slow", update_period=7, hidden_multiplier=3.0),
        ]
    )
    cms_use_chunk_scheduling: bool = False

    # ─── Longterm memory (optional, for CL) ───
    use_longterm_memory: bool = False
    longterm_hidden_multiplier: int = 2
    longterm_lr_scale: float = 0.1

    # ─── Regularization ───
    drop_path: float = 0.0
    drop: float = 0.0


class HybridBlock(nn.Module):
    """Hybrid block: Attention + Titan Memory + CMS.

    Replaces both the original ViT block (Attention + MLP) and the HOPE block
    (Titan-only + CMS) with a unified architecture that combines the strengths
    of both, see module docstring for the motivation.

    Architecture per forward pass:
        1. Phase A: x = x + Attention(norm(x))      [token interaction + RoPE]
        2. Phase B: x = x + TitanMemory(norm(x))    [in-context DGD adaptation]
        3. Phase C: x = x + CMS(norm(x))            [multi-frequency features]

    The Titan memory in Phase B uses a simplified update rule:
        - Single M_memory (instead of 5 separate memories)
        - η (learning rate) = learned per-feature parameter, not adaptive memory
        - α (decay) = learned per-feature parameter, not adaptive memory
        - Keys and values for DGD come from simple linear projections

    Args:
        config: HybridBlockConfig with all settings.
        titan_detach_interval: Detach memory graph every N steps (0 = never).
    """

    def __init__(
        self, config: HybridBlockConfig, titan_detach_interval: int = 0
    ) -> None:
        super().__init__()
        self.config = config
        dim = config.dim

        # ─── Phase A: Self-Attention with 3D RoPE ───
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

        # ─── Phase B: Titan Memory (simplified) ───
        self.norm2 = nn.LayerNorm(dim)

        # Query projection for memory retrieval
        self.mem_q_proj = nn.Linear(dim, dim, bias=False)

        # Key/Value projections for DGD write
        self.mem_k_proj = nn.Linear(dim, dim, bias=False)
        self.mem_v_proj = nn.Linear(dim, dim, bias=False)

        # The main retrieval/adaptation memory
        mem_cfg = TitanMemoryConfig(
            dim=dim,
            hidden_multiplier=config.titan_hidden_multiplier,
            num_layers=config.titan_layers,
            activation=config.titan_activation,
            grad_clip_inner=config.titan_grad_clip_inner,
            grad_clip_backward=config.titan_grad_clip_backward,
            detach_interval=titan_detach_interval,
        )
        self.M_memory = TitanMemory(mem_cfg)

        # Output projection for Phase B
        self.mem_out_proj = nn.Linear(dim, dim, bias=True)

        # Simplified η and α: learned per-feature parameters.
        # Instead of full TitanMemory networks (M_eta, M_alpha), we use
        # learnable vectors that the outer optimizer tunes. This drastically
        # reduces parameters while retaining per-feature DGD adaptation.
        # η: softplus(eta_base) → positive learning rate per feature
        # α: sigmoid(alpha_base) → decay factor in [0, 1] per feature
        self.eta_base = nn.Parameter(torch.full((dim,), DEFAULT_ETA_INIT))
        self.alpha_base = nn.Parameter(torch.full((dim,), DEFAULT_ALPHA_INIT))

        # ─── Longterm memory (optional, for CL persistence) ───
        self.use_longterm_memory = config.use_longterm_memory
        if config.use_longterm_memory:
            longterm_cfg = TitanMemoryConfig(
                dim=dim,
                hidden_multiplier=config.longterm_hidden_multiplier,
                num_layers=config.titan_layers,
                activation=config.titan_activation,
                grad_clip_inner=config.titan_grad_clip_inner,
                grad_clip_backward=config.titan_grad_clip_backward,
                detach_interval=titan_detach_interval,
            )
            self.M_longterm = TitanMemory(longterm_cfg)

            # Gating: learned per-token interpolation between clip and longterm
            self.longterm_gate = nn.Linear(dim, 1, bias=True)
            nn.init.constant_(self.longterm_gate.bias, 1.0)   # favour clip-level
            nn.init.zeros_(self.longterm_gate.weight)

            self.longterm_lr_scale = config.longterm_lr_scale

        # ─── Phase C: CMS ───
        self.norm3 = nn.LayerNorm(dim)
        self.cms = CMS(
            dim=dim,
            levels=config.cms_levels,
            drop=config.drop,
            use_chunk_scheduling=config.cms_use_chunk_scheduling,
        )

        # ─── Drop path (stochastic depth) ───
        self.drop_path = (
            DropPath(config.drop_path) if config.drop_path > 0.0 else nn.Identity()
        )

        # ─── State ───
        self._last_surprise: Tensor | None = None
        self.freeze_inner_loop: bool = False

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
        """Forward pass: Attention → Titan Memory → CMS.

        Compatible with ACBlock interface for drop-in substitution.

        Args:
            x: [B, N_total, D] input tokens (including action/state tokens).
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
        y = self.norm1(x)
        y = self.attn(
            y, mask=mask, attn_mask=attn_mask,
            T=T, H=H, W=W,
            action_tokens=action_tokens,
            target_timestep=target_timestep,
        )
        x = x + self.drop_path(y)

        # ─── Phase B: Titan Memory Read + DGD Write ───
        y = self.norm2(x)
        y = self._titan_forward(y)
        x = x + self.drop_path(y)

        # ─── Phase C: CMS ───
        y = self.norm3(x)
        tokens_per_frame = (
            (action_tokens + H * W) if H is not None and W is not None else None
        )
        y = self.cms(y, T=T, tokens_per_frame=tokens_per_frame)
        x = x + self.drop_path(y)

        return x

    def _titan_forward(self, x: Tensor) -> Tensor:
        """Phase B: Simplified Titan Memory read + DGD write.

        1. Project to query, key, value
        2. Read from M_memory using query
        3. Optionally read from M_longterm and gate
        4. Compute surprise and update M_memory via DGD

        Args:
            x: [B, N, D] pre-normed input (contextualized by attention).

        Returns:
            [B, N, D] Titan memory output.
        """
        # Project inputs for memory operations
        q = self.mem_q_proj(x)
        k = self.mem_k_proj(x)
        v = self.mem_v_proj(x)

        # Read from clip-level memory
        output = self.M_memory(q)

        # Gated longterm memory (optional)
        if self.use_longterm_memory:
            output_longterm = self.M_longterm(q)
            gate = torch.sigmoid(self.longterm_gate(q))  # [B, N, 1]
            output = gate * output + (1.0 - gate) * output_longterm

        # DGD update: write current experience into memory
        if not self.freeze_inner_loop and (
            torch.is_grad_enabled() or not self.training
        ):
            retrieval_error = v - output
            surprise = self.M_memory.surprise(retrieval_error)
            self._last_surprise = surprise.detach()

            # Compute η and α from learned parameters
            eta = F.softplus(self.eta_base)   # [D], positive
            alpha = torch.sigmoid(self.alpha_base)  # [D], in [0,1]

            # Gate update by surprise threshold
            if self.config.surprise_threshold > 0:
                update_mask = surprise > self.config.surprise_threshold
                if update_mask.any():
                    self._update_memory(k, v, surprise, eta, alpha)
            else:
                self._update_memory(k, v, surprise, eta, alpha)

        # Project output
        output = self.mem_out_proj(output)
        return output

    def _update_memory(
        self,
        k: Tensor,
        v: Tensor,
        surprise: Tensor,
        eta: Tensor,
        alpha: Tensor,
    ) -> None:
        """Update M_memory (and optionally M_longterm) via DGD.

        Uses self-generated targets (Eq. 83, Behrouz 2025): the memory
        generates its own target from the value, ensuring self-referential
        learning.

        Args:
            k: [B, N, D] keys for writing.
            v: [B, N, D] values for writing.
            surprise: [B] surprise signal per sample.
            eta: [D] per-feature learning rate.
            alpha: [D] per-feature decay factor.
        """
        # Expand η and α to match expected shape [B, N, D]
        B, N, D = k.shape
        eta_expanded = eta.unsqueeze(0).unsqueeze(0).expand(B, N, D)
        alpha_expanded = alpha.unsqueeze(0).unsqueeze(0).expand(B, N, D)

        # Self-generated target for M_memory
        self_target = self.M_memory.generate_self_target(v)
        self.M_memory.compute_and_apply_update(
            key=k,
            value=self_target,
            error_signal=surprise,
            lr=eta_expanded,
            alpha=alpha_expanded,
        )

        # Longterm memory update (slower DGD)
        if self.use_longterm_memory:
            self_target_lt = self.M_longterm.generate_self_target(v)
            eta_lt = eta_expanded * self.longterm_lr_scale
            self.M_longterm.compute_and_apply_update(
                key=k,
                value=self_target_lt,
                error_signal=surprise,
                lr=eta_lt,
                alpha=alpha_expanded,
            )

    def reset_memory_state(self) -> None:
        """Reset clip-level memory state (call between clips).

        M_longterm is NOT reset — it persists across clips.
        """
        self.M_memory.reset_active_weights()
        self.M_memory.reset_diagnostics()
        self.cms.reset_step_counter()

        # M_longterm: initialize on first call, detach thereafter
        if self.use_longterm_memory:
            if self.M_longterm._active_w1 is None:
                self.M_longterm.reset_active_weights()
            else:
                self.M_longterm._active_w1 = self.M_longterm._active_w1.detach()
                self.M_longterm._active_w2 = self.M_longterm._active_w2.detach()
            self.M_longterm.reset_diagnostics()

    def reset_longterm_memory(self) -> None:
        """Explicitly reset longterm memory to initial state."""
        if self.use_longterm_memory:
            self.M_longterm.reset_active_weights()
            self.M_longterm.reset_diagnostics()

    def get_diagnostics(self) -> dict[str, float]:
        """Get diagnostic metrics from this block."""
        diag: dict[str, float] = {}

        for key, val in self.M_memory.get_diagnostics().items():
            diag[f"M_memory/{key}"] = val

        if self.use_longterm_memory:
            for key, val in self.M_longterm.get_diagnostics().items():
                diag[f"M_longterm/{key}"] = val

        if self._last_surprise is not None:
            diag["hope/mean_surprise"] = self._last_surprise.mean().item()
            diag["hope/max_surprise"] = self._last_surprise.max().item()

        # Log learned η and α
        diag["titan/mean_eta"] = F.softplus(self.eta_base).mean().item()
        diag["titan/mean_alpha"] = torch.sigmoid(self.alpha_base).mean().item()

        return diag
