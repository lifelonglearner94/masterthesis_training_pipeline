"""HOPE Block — Self-Modifying Titan + CMS with optional 3D RoPE (Behrouz 2025).

Implements a single layer of the AC-HOPE-ViT architecture, combining:

Phase A — Self-Modifying Titan Layer (replaces standard attention):
    1. Adaptive projection: generate K, V, η, α from adaptive memories
       (only Q remains a static projection — Eq. 76)
    2. Optional 3D AC-RoPE injection on Q/K for spatiotemporal anchoring
    3. Memory read:  o_t = M_{memory,t-1}(q_t)
    4. Memory write: DGD update with surprise gating

Phase B — Continuum Memory System (replaces standard MLP):
    Multi-frequency MLP cascade (fast → medium → slow)

The block maintains per-sequence memory state that is reset between sequences.

Design decisions addressing criticism:
    - 3D RoPE is configurable via `use_rope` flag (Criticism §2)
    - All inner-loop diagnostics are logged (Criticism §1)
    - Gradient clipping on inner-loop gradients prevents instability
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.hope.cms import CMS, LevelSpec
from src.models.hope.titan_memory import TitanMemory, TitanMemoryConfig

log = logging.getLogger(__name__)


@dataclass
class HOPEBlockConfig:
    """Configuration for a single HOPE block."""

    dim: int = 384
    num_heads: int = 16

    # Titan memory settings
    titan_hidden_multiplier: int = 4
    titan_layers: int = 2
    titan_activation: str = "gelu"
    titan_grad_clip_inner: float = 1.0

    # CMS settings
    cms_levels: list[LevelSpec] = field(default_factory=lambda: [
        LevelSpec(name="fast", update_period=1, hidden_multiplier=4.0),
        LevelSpec(name="medium", update_period=4, hidden_multiplier=4.0),
        LevelSpec(name="slow", update_period=16, hidden_multiplier=4.0),
    ])
    cms_use_chunk_scheduling: bool = False

    # Self-modifier settings
    self_mod_dim: int = 64  # Hidden dim for the SelfModifier MLP

    # 3D RoPE toggle (Criticism §2: can be disabled for ablation)
    use_rope: bool = True
    grid_size: int = 16  # For RoPE position snapping

    # Surprise gating
    surprise_threshold: float = 0.0  # Min surprise to trigger memory update (0=always)

    # Drop path (stochastic depth)
    drop_path: float = 0.0

    # Dropout
    drop: float = 0.0


class SelfModifier(nn.Module):
    """Generates modulated target values for Titan memory self-modification.

    Takes key + value + error signal and produces modulated targets that
    the memory should learn to retrieve. This is the "self-referential"
    part — the model decides *what* to remember.

    Architecture: concat(key, value, error) → 3-layer MLP → target_delta
    Output: value + target_delta  (residual modulation)

    Args:
        dim: Token dimension.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        # Initialize output near zero for stable initial training
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, key: Tensor, value: Tensor, error_signal: Tensor) -> Tensor:
        """Compute modulated target for memory writing.

        Args:
            key:   [B, N, D] keys.
            value: [B, N, D] values.
            error_signal: [B] per-sample surprise scalar.

        Returns:
            [B, N, D] modulated target values.
        """
        B, N, D = key.shape
        # Expand error_signal: [B] → [B, N, 1]
        err = error_signal.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1)
        # Concatenate: [B, N, 2*D+1]
        inp = torch.cat([key, value, err], dim=-1)
        delta = self.net(inp)  # [B, N, D]
        return value + delta  # Residual modulation


class HOPEBlock(nn.Module):
    """Single HOPE block: Phase A (Titan) → Phase B (CMS).

    Replaces a standard Transformer block (Attention + MLP) with:
        - Phase A: Self-Modifying Titan memory (adaptive attention replacement)
        - Phase B: CMS multi-frequency MLP cascade

    The block maintains 5 adaptive memories:
        - M_k: generates keys
        - M_v: generates values
        - M_eta: generates learning rates η
        - M_alpha: generates decay factors α
        - M_memory: the main retrieval memory

    Only q_t = x_t W_q uses a static projection (Eq. 76, Behrouz 2025).

    Args:
        config: HOPEBlockConfig with all settings.
    """

    def __init__(self, config: HOPEBlockConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.dim
        head_dim = dim // config.num_heads

        # ─── Pre-norm ───
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # ─── Phase A: Self-Modifying Titan ───

        # Static query projection (only Q is non-adaptive, Eq. 76)
        self.q_proj = nn.Linear(dim, dim, bias=False)

        # Adaptive memories for K, V, η, α (Eq. 77-80)
        mem_cfg = TitanMemoryConfig(
            dim=dim,
            hidden_multiplier=config.titan_hidden_multiplier,
            num_layers=config.titan_layers,
            activation=config.titan_activation,
            grad_clip_inner=config.titan_grad_clip_inner,
        )
        self.M_k = TitanMemory(mem_cfg)      # Key generation memory
        self.M_v = TitanMemory(mem_cfg)      # Value generation memory
        self.M_eta = TitanMemory(mem_cfg)    # Learning rate generation memory
        self.M_alpha = TitanMemory(mem_cfg)  # Decay factor generation memory
        self.M_memory = TitanMemory(mem_cfg)  # Main retrieval memory

        # Self-modifier for target generation
        self.self_modifier = SelfModifier(dim, config.self_mod_dim)

        # Output projection for Phase A
        self.out_proj = nn.Linear(dim, dim, bias=True)

        # ─── RoPE support (optional, Criticism §2) ───
        if config.use_rope:
            self.d_dim = int(2 * ((head_dim // 3) // 2))
            self.h_dim = int(2 * ((head_dim // 3) // 2))
            self.w_dim = int(2 * ((head_dim // 3) // 2))
        else:
            self.d_dim = self.h_dim = self.w_dim = 0

        # ─── Phase B: CMS ───
        self.cms = CMS(
            dim=dim,
            levels=config.cms_levels,
            drop=config.drop,
            use_chunk_scheduling=config.cms_use_chunk_scheduling,
        )

        # ─── Drop path (stochastic depth) ───
        from src.models.ac_predictor.utils.modules import DropPath
        self.drop_path = DropPath(config.drop_path) if config.drop_path > 0.0 else nn.Identity()

        # Diagnostics
        self._last_surprise: Tensor | None = None
        self._last_inner_grad_norm: float = 0.0

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        T: int | None = None,
        H: int | None = None,
        W: int | None = None,
        action_tokens: int = 0,
    ) -> Tensor:
        """Forward pass through the HOPE block.

        Compatible with ACBlock interface from the original AC_ViT predictor.

        Args:
            x: [B, N_total, D] input tokens (including action/state tokens).
            mask: Optional position mask.
            attn_mask: Block causal attention mask [N_total, N_total] (used for
                       causal masking; the Titan memory inherently processes
                       sequentially so this mainly affects the RoPE positions).
            T: Number of temporal frames.
            H: Grid height (patches).
            W: Grid width (patches).
            action_tokens: Number of conditioning tokens per frame (usually 2).

        Returns:
            [B, N_total, D] output tokens.
        """
        # ─── Phase A: Self-Modifying Titan ───
        y = self.norm1(x)
        y = self._titan_forward(y, T=T, H=H, W=W, action_tokens=action_tokens)
        x = x + self.drop_path(y)

        # ─── Phase B: CMS ───
        y = self.norm2(x)
        y = self.cms(y)
        x = x + self.drop_path(y)

        return x

    def _titan_forward(
        self,
        x: Tensor,
        T: int | None = None,
        H: int | None = None,
        W: int | None = None,
        action_tokens: int = 0,
    ) -> Tensor:
        """Phase A: Self-Modifying Titan with optional 3D RoPE.

        Steps:
            1. Generate Q (static), K, V, η, α (adaptive from memories)
            2. Optionally apply 3D RoPE rotation to Q and K
            3. Retrieve from main memory: o_t = M_memory(q_t)
            4. Compute surprise and update all memories via DGD

        Args:
            x: [B, N, D] pre-normed input.
            T, H, W: Spatiotemporal grid dimensions.
            action_tokens: Conditioning tokens per frame.

        Returns:
            [B, N, D] Titan output.
        """
        B, N, D = x.shape

        # Step 1: Generate adaptive projections
        q = self.q_proj(x)          # Static: q_t = x_t W_q  (Eq. 76)
        k = self.M_k(x)             # Adaptive: k_t = M_{k,t-1}(x_t)  (Eq. 77)
        v = self.M_v(x)             # Adaptive: v_t = M_{v,t-1}(x_t)  (Eq. 78)
        eta_raw = self.M_eta(x)     # Adaptive: η_t raw  (Eq. 79)
        alpha_raw = self.M_alpha(x)  # Adaptive: α_t raw  (Eq. 80)

        # Normalize η and α to reasonable ranges
        # η (learning rate): softplus to ensure positive, then scale down
        eta = F.softplus(eta_raw.mean(dim=-1, keepdim=True)) * 0.01  # [B, N, 1]
        # α (decay): sigmoid to [0, 1] range
        alpha = torch.sigmoid(alpha_raw.mean(dim=-1, keepdim=True))  # [B, N, 1]

        # Step 2: Optional 3D RoPE on Q and K (Criticism §2: configurable)
        if self.config.use_rope and T is not None and H is not None and W is not None:
            q, k = self._apply_rope(q, k, T, H, W, action_tokens)

        # Step 3: Retrieve from main memory
        output = self.M_memory(q)   # o_t = M_{memory,t-1}(q_t)  (Eq. 83)

        # Step 4: Compute surprise and update memories
        if self.training and torch.is_grad_enabled():
            # Retrieval error as surprise signal
            retrieval_error = v - output  # [B, N, D]
            surprise = self.M_memory.surprise(retrieval_error)  # [B]
            self._last_surprise = surprise.detach()

            # Generate modulated targets via SelfModifier
            target_v = self.self_modifier(k, v, surprise)  # [B, N, D]

            # Only update if surprise exceeds threshold
            if self.config.surprise_threshold > 0:
                update_mask = surprise > self.config.surprise_threshold
                if update_mask.any():
                    self._update_memories(k, v, target_v, surprise, eta, alpha)
            else:
                self._update_memories(k, v, target_v, surprise, eta, alpha)

        # Project output
        output = self.out_proj(output)

        return output

    def _apply_rope(
        self,
        q: Tensor,
        k: Tensor,
        T: int,
        H: int,
        W: int,
        action_tokens: int,
    ) -> tuple[Tensor, Tensor]:
        """Apply 3D RoPE rotation to Q and K tensors.

        Reuses the exact same rotation logic as ACRoPEAttention from the
        existing AC predictor to ensure spatiotemporal position encoding
        is consistent with the V-JEPA2 architecture.

        The rotation is applied per-head (Q and K are reshaped to multi-head
        format, rotated, then flattened back).

        Args:
            q: [B, N_total, D] query tensor.
            k: [B, N_total, D] key tensor.
            T: Number of temporal frames.
            H: Grid height.
            W: Grid width.
            action_tokens: Number of conditioning tokens per frame.

        Returns:
            Tuple of rotated (q, k), each [B, N_total, D].
        """
        from src.models.ac_predictor.utils.modules import rotate_queries_or_keys

        B, N_total, D = q.shape
        num_heads = self.config.num_heads
        head_dim = D // num_heads

        # Separate action tokens from frame tokens
        if action_tokens > 0:
            # Reshape to [B, T, action_tokens + H*W, D]
            q_reshaped = q.view(B, T, action_tokens + H * W, D)
            k_reshaped = k.view(B, T, action_tokens + H * W, D)

            # Process action tokens: only apply temporal RoPE
            q_act = q_reshaped[:, :, :action_tokens, :].reshape(B, T * action_tokens, D)
            k_act = k_reshaped[:, :, :action_tokens, :].reshape(B, T * action_tokens, D)

            # Reshape to multi-head for rotation
            q_act = q_act.view(B, T * action_tokens, num_heads, head_dim).transpose(1, 2)
            k_act = k_act.view(B, T * action_tokens, num_heads, head_dim).transpose(1, 2)

            # Temporal positions for action tokens
            act_time_pos = torch.arange(T, device=q.device).unsqueeze(1).expand(T, action_tokens).reshape(-1).float()

            # Rotate only depth dimension for action tokens
            qd_act = rotate_queries_or_keys(q_act[..., :self.d_dim], pos=act_time_pos)
            kd_act = rotate_queries_or_keys(k_act[..., :self.d_dim], pos=act_time_pos)
            qr_act = q_act[..., self.d_dim:]
            kr_act = k_act[..., self.d_dim:]
            q_act = torch.cat([qd_act, qr_act], dim=-1)
            k_act = torch.cat([kd_act, kr_act], dim=-1)

            # Reshape back: [B, num_heads, T*action_tokens, head_dim] → [B, T, action_tokens, D]
            q_act = q_act.transpose(1, 2).reshape(B, T, action_tokens, D)
            k_act = k_act.transpose(1, 2).reshape(B, T, action_tokens, D)

            # Process frame tokens: apply full 3D RoPE
            q_frame = q_reshaped[:, :, action_tokens:, :].reshape(B, T * H * W, D)
            k_frame = k_reshaped[:, :, action_tokens:, :].reshape(B, T * H * W, D)
        else:
            q_frame = q
            k_frame = k

        # Compute 3D positions for frame tokens
        frame_ids = torch.arange(T * H * W, device=q.device)
        tokens_per_frame = H * W
        d_pos = (frame_ids // tokens_per_frame).float()
        h_pos = ((frame_ids % tokens_per_frame) // W).float()
        w_pos = ((frame_ids % tokens_per_frame) % W).float()

        # Snap to grid (matching ACRoPEAttention)
        h_pos = h_pos * (self.config.grid_size / H)
        w_pos = w_pos * (self.config.grid_size / W)

        # Reshape to multi-head
        q_frame = q_frame.view(B, T * H * W, num_heads, head_dim).transpose(1, 2)
        k_frame = k_frame.view(B, T * H * W, num_heads, head_dim).transpose(1, 2)

        # Rotate each dimension
        s = 0
        qd = rotate_queries_or_keys(q_frame[..., s:s + self.d_dim], pos=d_pos)
        kd = rotate_queries_or_keys(k_frame[..., s:s + self.d_dim], pos=d_pos)
        s += self.d_dim

        qh = rotate_queries_or_keys(q_frame[..., s:s + self.h_dim], pos=h_pos)
        kh = rotate_queries_or_keys(k_frame[..., s:s + self.h_dim], pos=h_pos)
        s += self.h_dim

        qw = rotate_queries_or_keys(q_frame[..., s:s + self.w_dim], pos=w_pos)
        kw = rotate_queries_or_keys(k_frame[..., s:s + self.w_dim], pos=w_pos)
        s += self.w_dim

        # Combine rotated and unrotated dimensions
        if s < head_dim:
            qr = q_frame[..., s:]
            kr = k_frame[..., s:]
            q_frame = torch.cat([qd, qh, qw, qr], dim=-1)
            k_frame = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q_frame = torch.cat([qd, qh, qw], dim=-1)
            k_frame = torch.cat([kd, kh, kw], dim=-1)

        # Reshape back: [B, num_heads, T*H*W, head_dim] → [B, T*H*W, D]
        q_frame = q_frame.transpose(1, 2).reshape(B, T * H * W, D)
        k_frame = k_frame.transpose(1, 2).reshape(B, T * H * W, D)

        # Merge action and frame tokens back
        if action_tokens > 0:
            # Interleave: [B, T, action_tokens, D] ++ [B, T, H*W, D] → [B, T*(action_tokens+H*W), D]
            q_frame_3d = q_frame.view(B, T, H * W, D)
            k_frame_3d = k_frame.view(B, T, H * W, D)
            q_out = torch.cat([q_act, q_frame_3d], dim=2).flatten(1, 2)
            k_out = torch.cat([k_act, k_frame_3d], dim=2).flatten(1, 2)
        else:
            q_out = q_frame
            k_out = k_frame

        return q_out, k_out

    def _update_memories(
        self,
        k: Tensor,
        v: Tensor,
        target_v: Tensor,
        surprise: Tensor,
        eta: Tensor,
        alpha: Tensor,
    ) -> None:
        """Update all adaptive memories via DGD (Eq. 86-93).

        Each memory M_□ is updated independently:
            M_□,t = M_□,t-1 (α_t I − η_t k_t k_t^T) − η_t ∇L(M_□; k, v̂_□)

        Args:
            k: [B, N, D] keys.
            v: [B, N, D] values.
            target_v: [B, N, D] modulated target values.
            surprise: [B] surprise signal per sample.
            eta: [B, N, 1] adaptive learning rate.
            alpha: [B, N, 1] adaptive decay factor.
        """
        memories = [self.M_k, self.M_v, self.M_eta, self.M_alpha, self.M_memory]

        for mem in memories:
            mem.compute_and_apply_update(
                key=k,
                value=target_v,
                error_signal=surprise,
                lr=eta,
                alpha=alpha,
            )

    def reset_memory_state(self) -> None:
        """Reset all memory states to meta-learned initial parameters.

        Call this at the start of each new sequence to ensure memories
        don't carry state between independent sequences.
        Initializes active weights (detached clones) for functional forward.
        """
        for mem in [self.M_k, self.M_v, self.M_eta, self.M_alpha, self.M_memory]:
            mem.reset_active_weights()
            mem.reset_diagnostics()
        self.cms.reset_step_counter()

    def get_diagnostics(self) -> dict[str, float]:
        """Aggregate diagnostics from all sub-components.

        Returns:
            Dict with prefixed diagnostic metrics.
        """
        diag = {}
        for prefix, mem in [
            ("M_memory", self.M_memory),
            ("M_k", self.M_k),
            ("M_v", self.M_v),
        ]:
            for key, val in mem.get_diagnostics().items():
                diag[f"{prefix}/{key}"] = val

        if self._last_surprise is not None:
            diag["hope/mean_surprise"] = self._last_surprise.mean().item()
            diag["hope/max_surprise"] = self._last_surprise.max().item()

        return diag
