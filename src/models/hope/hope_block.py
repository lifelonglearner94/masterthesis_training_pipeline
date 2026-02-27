"""HOPE Block — Self-Modifying Titan + CMS with optional 3D RoPE (Behrouz 2025).

Implements a single layer of the AC-HOPE-ViT architecture, combining:

Phase A — Self-Modifying Titan Layer (replaces standard attention):
    1. Adaptive projection: generate K, V, η, α from adaptive memories
       (only Q remains a static projection — Eq. 76)
    2. Optional 3D AC-RoPE injection on Q/K for spatiotemporal anchoring
    3. Memory read:  o_t = M_{memory,t-1}(q_t)
    4. Self-generated targets: v̂_□ = M_{□,t-1}(v_t)  (Eq. 83)
    5. Memory write: DGD update with surprise gating and per-memory targets

Phase B — Continuum Memory System (replaces standard MLP):
    Multi-frequency MLP cascade (fast → medium → slow)

The block maintains per-sequence memory state that is reset between sequences.
Chunk-wise processing (Section 8.2) splits tokens into chunks and updates
memories between chunks, enabling intra-sequence memory evolution.

Design decisions addressing criticism:
    - 3D RoPE is configurable via `use_rope` flag (Criticism §2)
    - All inner-loop diagnostics are logged (Criticism §1)
    - Gradient clipping on inner-loop gradients prevents instability
    - Per-memory self-generated targets (Eq. 83, Review §4.2)
    - Chunk-wise sequential updates (Section 8.2, Review §4.5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.hope.cms import CMS, LevelSpec
from src.models.hope.titan_memory import TitanMemory, TitanMemoryConfig

log = logging.getLogger(__name__)


# Constants
DEFAULT_ETA_SCALE = 0.01  # Scaling factor for learning rate normalization


class TitanActivation(StrEnum):
    """Supported activation functions for Titan memory layers."""

    GELU = "gelu"
    RELU = "relu"
    SILU = "silu"
    TANH = "tanh"


@dataclass
class HOPEBlockConfig:
    """Configuration for a single HOPE block."""

    dim: int = 384
    num_heads: int = 16

    # Titan memory settings
    titan_hidden_multiplier: int = 4
    titan_layers: int = 2
    titan_activation: str = TitanActivation.GELU
    titan_grad_clip_inner: float = 1.0
    titan_grad_clip_backward: float = 1.0  # Level-2 backward clip for DGD chain

    # CMS settings
    cms_levels: list[LevelSpec] = field(
        default_factory=lambda: [
            LevelSpec(name="fast", update_period=1, hidden_multiplier=4.0),
            LevelSpec(name="medium", update_period=4, hidden_multiplier=4.0),
            LevelSpec(name="slow", update_period=16, hidden_multiplier=4.0),
        ]
    )
    cms_use_chunk_scheduling: bool = False

    # Chunk-wise processing (Section 8.2)
    # Chunk size for intra-sequence memory updates. 0 = no chunking (all tokens at once).
    # Recommended: tokens_per_frame (e.g. 258 = 256 patches + 2 action tokens).
    chunk_size: int = 0

    # 3D RoPE toggle (Criticism §2: can be disabled for ablation)
    use_rope: bool = True
    grid_size: int = 16  # For RoPE position snapping

    # Surprise gating
    surprise_threshold: float = 0.0  # Min surprise to trigger memory update (0=always)

    # Drop path (stochastic depth)
    drop_path: float = 0.0

    # Dropout
    drop: float = 0.0


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
        titan_detach_interval: Detach memory graph every N update steps
            to bound VRAM. 0 = never detach (full meta-gradient chain).
    """

    def __init__(self, config: HOPEBlockConfig, titan_detach_interval: int = 0) -> None:
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
            grad_clip_backward=config.titan_grad_clip_backward,
            detach_interval=titan_detach_interval,
        )
        self.M_k = TitanMemory(mem_cfg)  # Key generation memory
        self.M_v = TitanMemory(mem_cfg)  # Value generation memory
        self.M_eta = TitanMemory(mem_cfg)  # Learning rate generation memory
        self.M_alpha = TitanMemory(mem_cfg)  # Decay factor generation memory
        self.M_memory = TitanMemory(mem_cfg)  # Main retrieval memory

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

        self.drop_path = (
            DropPath(config.drop_path) if config.drop_path > 0.0 else nn.Identity()
        )

        # Diagnostics
        self._last_surprise: Tensor | None = None
        self._last_inner_grad_norm: float = 0.0

        # Auxiliary loss for M_k/M_v gradient flow (accumulated per forward)
        self._aux_loss: Tensor = torch.tensor(0.0)

        # Freeze flag: when True, skip all inner-loop DGD memory updates.
        # Memories are still READ (for retrieval) but never WRITTEN.
        # This enables pure inference for CL evaluation phases.
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
            target_timestep: Target frame index for jump prediction.

        Returns:
            [B, N_total, D] output tokens.
        """
        # ─── Phase A: Self-Modifying Titan ───
        y = self.norm1(x)
        y = self._titan_forward(y, T=T, H=H, W=W, action_tokens=action_tokens, target_timestep=target_timestep)
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
        target_timestep: int | None = None,
    ) -> Tensor:
        """Phase A: Self-Modifying Titan with optional 3D RoPE and chunking.

        Implements chunk-wise sequential processing (Section 8.2):
        tokens are split into chunks, memories are updated between chunks,
        enabling intra-sequence memory evolution.

        Steps per chunk:
            1. Generate Q (static), K, V, η, α (adaptive from current memory)
            2. Optionally apply 3D RoPE rotation to Q and K
            3. Retrieve from main memory: o_t = M_memory(q_t)
            4. Compute surprise signal
            5. Generate per-memory self-targets: v̂_□ = M_□(v_t)  (Eq. 83)
            6. Update all memories via DGD

        Args:
            x: [B, N, D] pre-normed input.
            T, H, W: Spatiotemporal grid dimensions.
            action_tokens: Conditioning tokens per frame.

        Returns:
            [B, N, D] Titan output.
        """
        B, N, D = x.shape
        chunk_size = self.config.chunk_size
        tokens_per_frame = (action_tokens + H * W) if H is not None else 0

        # If no chunking or T is unknown, process all tokens at once
        if chunk_size <= 0 or T is None or chunk_size >= T:
            return self._titan_forward_chunk(
                x, T=T, H=H, W=W, action_tokens=action_tokens, target_timestep=target_timestep
            )

        # ─── Chunk-wise processing along temporal dimension ───
        # Split into groups of `chunk_size` timesteps. Each chunk has
        # chunk_T * tokens_per_frame tokens. Memories update between chunks
        # so chunk 2 reads state modified by chunk 1 (meta-gradient chain).
        output_chunks = []

        for t_start in range(0, T, chunk_size):
            t_end = min(t_start + chunk_size, T)
            chunk_T = t_end - t_start

            # Extract tokens for these timesteps
            tok_start = t_start * tokens_per_frame
            tok_end = t_end * tokens_per_frame
            x_chunk = x[:, tok_start:tok_end, :]  # [B, chunk_T * tpf, D]

            # Process chunk with current memory state
            out_chunk = self._titan_forward_chunk(
                x_chunk, T=chunk_T, H=H, W=W, action_tokens=action_tokens, target_timestep=target_timestep
            )
            output_chunks.append(out_chunk)

        # Concatenate all chunk outputs
        return torch.cat(output_chunks, dim=1)  # [B, N, D]

    def _titan_forward_chunk(
        self,
        x: Tensor,
        T: int | None = None,
        H: int | None = None,
        W: int | None = None,
        action_tokens: int = 0,
        target_timestep: int | None = None,
    ) -> Tensor:
        """Process a single chunk through the Titan memory.

        This is the core computation unit. When chunk_size=0, the entire
        sequence is processed as one chunk (original behavior). With chunking,
        this is called per-chunk and memories accumulate between calls.

        Args:
            x: [B, C, D] chunk of pre-normed input tokens.
            T, H, W: Full spatiotemporal grid dimensions (for RoPE).
            action_tokens: Conditioning tokens per frame.

        Returns:
            [B, C, D] Titan output for this chunk.
        """
        B, C, D = x.shape

        # Step 1: Generate adaptive projections using current memory state
        q = self.q_proj(x)  # Static: q_t = x_t W_q  (Eq. 76)
        k = self.M_k(x)  # Adaptive: k_t = M_{k,t-1}(x_t)  (Eq. 79)
        v = self.M_v(x)  # Adaptive: v_t = M_{v,t-1}(x_t)  (Eq. 79)
        eta_raw = self.M_eta(x)  # Adaptive: η_t raw  (Eq. 80)
        alpha_raw = self.M_alpha(x)  # Adaptive: α_t raw  (Eq. 80)

        # Normalize η and α to reasonable ranges — per-token, per-feature [B, C, D]
        # Paper Eq. 88: η_t, α_t ∈ R^d (full-dimensional, per-token)
        # η (learning rate): softplus to ensure positive, then scale down
        eta = F.softplus(eta_raw) * DEFAULT_ETA_SCALE  # [B, C, D]
        # α (decay): sigmoid to [0, 1] range
        alpha = torch.sigmoid(alpha_raw)  # [B, C, D]

        # Step 2: Optional 3D RoPE on Q and K (Criticism §2: configurable)
        if self.config.use_rope and T is not None and H is not None and W is not None:
            q, k = self._apply_rope(q, k, T, H, W, action_tokens, target_timestep=target_timestep)

        # Step 3: Retrieve from main memory
        output = self.M_memory(q)  # o_t = M_{memory,t-1}(q_t)  (Eq. 86)

        # Step 4-6: Compute surprise + per-memory self-targets + DGD update
        # Paper: "There is no distinction between training and test time."
        # The inner loop runs during BOTH training and inference so the model
        # adapts in-context to each sequence (core HOPE advantage).
        # When freeze_inner_loop=True, skip all memory writes for CL evaluation.
        if not self.freeze_inner_loop and (torch.is_grad_enabled() or not self.training):
            # Retrieval error as surprise signal
            retrieval_error = v - output  # [B, C, D]
            surprise = self.M_memory.surprise(retrieval_error)  # [B]
            self._last_surprise = surprise.detach()

            # Only update if surprise exceeds threshold
            if self.config.surprise_threshold > 0:
                update_mask = surprise > self.config.surprise_threshold
                if update_mask.any():
                    self._update_memories(k, v, surprise, eta, alpha)
            else:
                self._update_memories(k, v, surprise, eta, alpha)

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
        target_timestep: int | None = None,
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
            target_timestep: Target frame index for jump prediction.

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
            q_act = q_reshaped[:, :, :action_tokens, :].reshape(
                B, T * action_tokens, D
            )
            k_act = k_reshaped[:, :, :action_tokens, :].reshape(
                B, T * action_tokens, D
            )

            # Reshape to multi-head for rotation
            q_act = q_act.view(B, T * action_tokens, num_heads, head_dim).transpose(1, 2)
            k_act = k_act.view(B, T * action_tokens, num_heads, head_dim).transpose(1, 2)

            # Temporal positions for action tokens
            if target_timestep is not None:
                act_time_pos = (
                    torch.full((T, action_tokens), target_timestep - 1, device=q.device)
                    .reshape(-1)
                    .float()
                )
            else:
                act_time_pos = (
                    torch.arange(T, device=q.device)
                    .unsqueeze(1)
                    .expand(T, action_tokens)
                    .reshape(-1)
                    .float()
                )

            # Rotate only depth dimension for action tokens
            qd_act = rotate_queries_or_keys(q_act[..., : self.d_dim], pos=act_time_pos)
            kd_act = rotate_queries_or_keys(k_act[..., : self.d_dim], pos=act_time_pos)
            qr_act = q_act[..., self.d_dim :]
            kr_act = k_act[..., self.d_dim :]
            q_act = torch.cat([qd_act, qr_act], dim=-1)
            k_act = torch.cat([kd_act, kr_act], dim=-1)

            # Reshape back: [B, num_heads, T*action_tokens, head_dim] → [B, T, action_tokens, D]
            q_act = q_act.transpose(1, 2).reshape(B, T, action_tokens, D)
            k_act = k_act.transpose(1, 2).reshape(B, T, action_tokens, D)

            # Process frame tokens: apply full 3D RoPE
            q_frame = q_reshaped[:, :, action_tokens :, :].reshape(B, T * H * W, D)
            k_frame = k_reshaped[:, :, action_tokens :, :].reshape(B, T * H * W, D)
        else:
            q_frame = q
            k_frame = k

        # Compute 3D positions for frame tokens
        if target_timestep is not None:
            # Jump prediction: all tokens at temporal position (target_timestep - 1)
            base_spatial = torch.arange(H * W, device=q.device)
            frame_ids = (target_timestep - 1) * H * W + base_spatial
            # Repeat for T timesteps (T should be 1 in jump mode but handle gracefully)
            if T > 1:
                frame_ids = frame_ids.repeat(T)
        else:
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
        qd = rotate_queries_or_keys(q_frame[..., s : s + self.d_dim], pos=d_pos)
        kd = rotate_queries_or_keys(k_frame[..., s : s + self.d_dim], pos=d_pos)
        s += self.d_dim

        qh = rotate_queries_or_keys(q_frame[..., s : s + self.h_dim], pos=h_pos)
        kh = rotate_queries_or_keys(k_frame[..., s : s + self.h_dim], pos=h_pos)
        s += self.h_dim

        qw = rotate_queries_or_keys(q_frame[..., s : s + self.w_dim], pos=w_pos)
        kw = rotate_queries_or_keys(k_frame[..., s : s + self.w_dim], pos=w_pos)
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
        surprise: Tensor,
        eta: Tensor,
        alpha: Tensor,
    ) -> None:
        """Update all adaptive memories via DGD with self-generated targets.

        Each memory M_□ generates its own target (Eq. 83):
            v̂_□,t = M_{□,t-1}(v_t)

        Then is updated independently (Eq. 88-93):
            M_□,t = M_□,t-1 (α_t I − η_t k_t k_t^T) − η_t ∇L(M_□; k, v̂_□)

        This is the "self-referential" property: each memory uses its own
        weights to decide what it should learn.

        Auxiliary loss for M_k/M_v gradient flow:
            Under first-order meta-learning (FOMAML), M_k and M_v outputs
            are only consumed inside the inner-loop where weights are detached,
            so no outer-loss gradient reaches M_k/M_v parameters. To fix this,
            we compute a small auxiliary loss measuring how well the adaptive
            k,v projections serve the main memory retrieval. This loss is
            accumulated in self._aux_loss and added to the outer loss by the
            Lightning module (with a small weight to avoid dominating training).

        Args:
            k: [B, N, D] keys (output of M_k, in computation graph).
            v: [B, N, D] values (output of M_v, in computation graph).
            surprise: [B] surprise signal per sample.
            eta: [B, N, D] adaptive learning rate.
            alpha: [B, N, D] adaptive decay factor.
        """
        memories = [self.M_k, self.M_v, self.M_eta, self.M_alpha, self.M_memory]

        for mem in memories:
            # Each memory generates its OWN target (Eq. 83: v̂_□ = M_□(v))
            self_target = mem.generate_self_target(v)  # [B, N, D]
            mem.compute_and_apply_update(
                key=k,
                value=self_target,
                error_signal=surprise,
                lr=eta,
                alpha=alpha,
            )

        # ─── Auxiliary loss for M_k / M_v gradient flow ───
        # Compute how well the adaptive k from M_k retrieves from M_memory.
        # This creates a differentiable path: M_k params → k → retrieval → aux_loss
        # and similarly M_v params → v → target matching → aux_loss.
        # Both k and v are live tensors (in the computation graph from M_k/M_v
        # forward passes above), so gradients flow back to M_k/M_v parameters.
        #
        # CRITICAL: We use DETACHED M_memory weights for this retrieval so that
        # the aux gradient only flows through k (→ M_k) and v (→ M_v), NOT
        # through M_memory's DGD update chain. With detach_interval > 1, the
        # post-update weights are still live (connected to nn.Parameters via the
        # preconditioner), and backprop through [aux_loss → M_memory(k) →
        # post-update weights → precond → w_old] includes a factor of ||w_old||
        # (the full weight matrix). Across 5 memories × 6 blocks × 7 chunks,
        # these terms overflow intermediate backward values to inf/NaN before
        # gradient_clip_val can be applied. With detach_interval=1 this was
        # harmless (weights are already detached), but with >1 it's fatal.
        if self.training and torch.is_grad_enabled():
            # Retrieve using adaptive keys through M_memory with DETACHED weights
            # so gradient flows through k only — not through the DGD update chain.
            _w1 = self.M_memory._active_w1.detach()
            _w2 = self.M_memory._active_w2.detach()
            h = F.linear(k, _w1)
            h = self.M_memory.act(h)
            retrieved_via_k = self.M_memory.norm(F.linear(h, _w2) + k)
            # Measure how close retrieved values are to the adaptive v
            aux = F.mse_loss(retrieved_via_k, v.detach())  # M_k path
            aux = aux + F.mse_loss(retrieved_via_k.detach(), v)  # M_v path
            self._aux_loss = self._aux_loss + aux

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
        # Reset auxiliary loss accumulator for this sequence
        self._aux_loss = torch.tensor(0.0, device=next(self.parameters()).device)

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
            ("M_eta", self.M_eta),
            ("M_alpha", self.M_alpha),
        ]:
            for key, val in mem.get_diagnostics().items():
                diag[f"{prefix}/{key}"] = val

        if self._last_surprise is not None:
            diag["hope/mean_surprise"] = self._last_surprise.mean().item()
            diag["hope/max_surprise"] = self._last_surprise.max().item()

        return diag
