# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import drop_path

logger = logging.getLogger(__name__)

_DEFAULT_ALIGNMENT: Final = 8
_THREE_WAY_DIVISOR: Final = 3
_PAIR_DIVISOR: Final = 2
_SWIGLU_WIDTH_FACTOR: Final = 2  # From SwiGLU paper: hidden = 2h/3


def build_action_block_causal_attention_mask(T: int, H: int, W: int, add_tokens: int = 1) -> torch.Tensor:
    """Build a causal attention mask for action blocks.

    Args:
        T: Number of temporal frames.
        H: Height of spatial grid.
        W: Width of spatial grid.
        add_tokens: Number of additional tokens to prepend (e.g., action tokens).

    Returns:
        Boolean attention mask of shape (N, N) where N = T * (add_tokens + H * W).
    """
    n_t = add_tokens + (H * W)
    n = T * n_t
    mask = torch.zeros(n, n, dtype=torch.bool)
    mask_block = torch.ones(n_t, n_t, dtype=torch.bool)
    local_window_time = T

    for t1 in range(T):
        for t2 in range(max(0, t1 - local_window_time + 1), t1 + 1):
            mask[t1 * n_t : (t1 + 1) * n_t, t2 * n_t : (t2 + 1) * n_t] = mask_block

    return mask


def rotate_queries_or_keys(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to queries or keys.

    Args:
        x: Input tensor of shape (B, num_heads, N, D).
        pos: Position tensor of shape (..., N).

    Returns:
        Rotated tensor of the same shape as input.
    """
    B, num_heads, N, D = x.size()
    if D % _PAIR_DIVISOR != 0:
        raise ValueError(f"Embedding dimension must be a multiple of {_PAIR_DIVISOR} for block matrix rotation, got {D}")

    # Compute angle for each position
    omega = torch.arange(D // _PAIR_DIVISOR, dtype=x.dtype, device=x.device)
    omega /= D / _PAIR_DIVISOR
    omega = 1.0 / 10000**omega  # (D/2,)
    freq = torch.einsum("..., f -> ... f", pos, omega)  # (..., N, D/2), outer product

    # Build rotation matrix and apply
    emb_sin = freq.sin()  # (..., N, D/2)
    emb_cos = freq.cos()  # (..., N, D/2)

    # NOTE: This expansion has a subtle bug where frequencies are duplicated across the vector pair.
    # Fixing the bug would break compatibility with the pretrained model, but the fix can be applied
    # by commenting out the two lines below, and uncommenting the following two lines.
    # Thanks to @echosprint, original PR: https://github.com/facebookresearch/vjepa2/pull/15
    emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, _PAIR_DIVISOR)
    emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, _PAIR_DIVISOR)
    # emb_sin = emb_sin.repeat_interleave(2, dim=-1)  # (..., N, D)
    # emb_cos = emb_cos.repeat_interleave(2, dim=-1)  # (..., N, D)

    y = x.unflatten(-1, (-1, _PAIR_DIVISOR))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)

    return (x * emb_cos) + (y * emb_sin)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float | None = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.SiLU,
        drop: float = 0.0,
        wide_silu: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        swiglu_hidden_features = hidden_features = hidden_features or in_features

        if wide_silu:
            swiglu_hidden_features = int(_SWIGLU_WIDTH_FACTOR * hidden_features / 3)
            swiglu_hidden_features = (swiglu_hidden_features + _DEFAULT_ALIGNMENT - 1) // _DEFAULT_ALIGNMENT * _DEFAULT_ALIGNMENT

        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)


class ACRoPEAttention(nn.Module):
    """Attention module with Action-Caused Rotary Position Embeddings (ACRoPE)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_sdpa: bool = True,
        is_causal: bool = False,
        grid_size: int = 16,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

        # Dimension allocation for 3D rotary embeddings
        self.d_dim = int(_PAIR_DIVISOR * ((head_dim // _THREE_WAY_DIVISOR) // _PAIR_DIVISOR))
        self.h_dim = int(_PAIR_DIVISOR * ((head_dim // _THREE_WAY_DIVISOR) // _PAIR_DIVISOR))
        self.w_dim = int(_PAIR_DIVISOR * ((head_dim // _THREE_WAY_DIVISOR) // _PAIR_DIVISOR))
        self.grid_size = grid_size

    def _get_frame_pos(self, ids: torch.Tensor, H_patches: int, W_patches: int) -> torch.Tensor:
        """Get frame position indices from token IDs."""
        tokens_per_frame = H_patches * W_patches
        return ids // tokens_per_frame

    def _get_height_pos(self, ids: torch.Tensor, H_patches: int, W_patches: int) -> torch.Tensor:
        """Get height position indices from token IDs."""
        tokens_per_frame = H_patches * W_patches
        tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        return ids // tokens_per_row

    def separate_positions(
        self, ids: torch.Tensor, H_patches: int, W_patches: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Separate token IDs into frame, height, and width position components."""
        tokens_per_frame = H_patches * W_patches
        tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        # Remove frame component and height component to get width
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return frame_ids.float(), height_ids.float(), width_ids.float()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        T: int | None = None,
        H: int | None = None,
        W: int | None = None,
        action_tokens: int = 0,
        target_timestep: int | None = None,
    ) -> torch.Tensor:
        """Forward pass of ACRoPE attention.

        Args:
            x: Input tensor of shape (B, N, C).
            mask: Optional position mask.
            attn_mask: Optional attention mask for SDPA.
            T: Number of temporal frames.
            H: Height of spatial grid.
            W: Width of spatial grid.
            action_tokens: Number of action tokens to process separately.
            target_timestep: Target frame index for jump prediction (0-indexed
                into the features tensor). When set, RoPE temporal positions are
                overridden to ``target_timestep - 1`` so the model's output
                corresponds to the prediction for frame ``target_timestep``.

        Returns:
            Output tensor of shape (B, N, C).
        """
        B, N, C = x.size()

        # Compute position of each frame token
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H, W)
        elif target_timestep is not None:
            # Jump prediction: place all tokens at temporal position (target_timestep - 1)
            # so the model output corresponds to frame target_timestep.
            base_spatial = torch.arange(H * W, device=x.device)
            mask = (target_timestep - 1) * H * W + base_spatial
            d_mask, h_mask, w_mask = self.separate_positions(mask, H, W)
        else:
            mask = torch.arange(int(T * H * W), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H, W)

        # Snap spatial positions to grid size
        h_mask *= self.grid_size / H
        w_mask *= self.grid_size / W

        # Split out action tokens from sequence
        if action_tokens > 0:
            x = x.view(B, -1, action_tokens + H * W, C)  # [B, T, 1+H*W, D]

            action_q = []
            action_k = []
            action_v = []
            # Temporal positions for action tokens: override for jump prediction
            if target_timestep is not None:
                act_time_pos = torch.full((T,), target_timestep - 1, dtype=torch.long, device=x.device)
            else:
                act_time_pos = torch.arange(T, device=x.device)
            for i in range(action_tokens):
                a = x[:, :, i : i + 1, :].flatten(1, 2)
                # Compute qkv for action tokens and rotate
                qkv = self.qkv(a).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

                # Rotate temporal dimension for action tokens
                qd = rotate_queries_or_keys(q[..., : self.d_dim], pos=act_time_pos)
                kd = rotate_queries_or_keys(k[..., : self.d_dim], pos=act_time_pos)
                qr = q[..., self.d_dim :]
                kr = k[..., self.d_dim :]

                action_q.append(torch.cat([qd, qr], dim=-1).view(B, self.num_heads, T, 1, -1))
                action_k.append(torch.cat([kd, kr], dim=-1).view(B, self.num_heads, T, 1, -1))
                action_v.append(v.view(B, self.num_heads, T, 1, -1))

            action_q = torch.cat(action_q, dim=3).flatten(2, 3)
            action_k = torch.cat(action_k, dim=3).flatten(2, 3)
            action_v = torch.cat(action_v, dim=3).flatten(2, 3)
            x = x[:, :, action_tokens:, :].flatten(1, 2)

        # Compute qkv for frame tokens and rotate
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        s = 0
        # Rotate depth (temporal) dimension
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_mask)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_mask)
        s += self.d_dim

        # Rotate height dimension
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_mask)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim

        # Rotate width dimension
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_mask)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim

        # Combine rotated dimensions
        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        # Merge action tokens back with frame tokens
        if action_tokens > 0:

            def merge_(tx: torch.Tensor, ta: torch.Tensor) -> torch.Tensor:
                """Merge frame and action tokens.

                Args:
                    tx: Frame tokens in [B, num_heads, N, D].
                    ta: Action tokens in [B, num_heads, N, D].

                Returns:
                    Merged tokens.
                """
                tx = tx.view(B, self.num_heads, T, H * W, -1)  # [B, T, H*W, D]
                ta = ta.view(B, self.num_heads, T, action_tokens, -1)  # [B, T, A, D]
                return torch.cat([ta, tx], dim=3).flatten(2, 3)

            q = merge_(q, action_q)
            k = merge_(k, action_k)
            v = merge_(v, action_v)

        # Compute attention
        if attn_mask is not None or self.use_sdpa:
            logger.debug(f"      [ACRoPEAttn] scaled_dot_product_attention: Q={q.shape}, K={k.shape}, V={v.shape}")
            logger.debug(
                f"      [ACRoPEAttn] Q stats: min={q.min().item():.4f}, max={q.max().item():.4f}, "
                f"mean={q.mean().item():.4f}"
            )
            logger.debug(
                f"      [ACRoPEAttn] K stats: min={k.min().item():.4f}, max={k.max().item():.4f}, "
                f"mean={k.mean().item():.4f}"
            )
            logger.debug(
                f"      [ACRoPEAttn] V stats: min={v.min().item():.4f}, max={v.max().item():.4f}, "
                f"mean={v.mean().item():.4f}"
            )

            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p, is_causal=self.is_causal, attn_mask=attn_mask
            )

            logger.debug(
                f"      [ACRoPEAttn] Output: shape={x.shape}, min={x.min().item():.4f}, max={x.max().item():.4f}"
            )
            attn = None
        else:
            logger.debug(f"      [ACRoPEAttn] Manual attention: Q@K.T matmul: [{q.shape}] @ [{k.shape}]")
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
            logger.debug(
                f"      [ACRoPEAttn] Attention scores: shape={attn.shape}, min={attn.min().item():.4f}, "
                f"max={attn.max().item():.4f}"
            )
            attn = attn.softmax(dim=-1)
            logger.debug(f"      [ACRoPEAttn] After softmax: min={attn.min().item():.4f}, max={attn.max().item():.4f}")
            attn = self.attn_drop(attn)
            logger.debug(f"      [ACRoPEAttn] Attn@V matmul: [{attn.shape}] @ [{v.shape}]")
            x = attn @ v
            logger.debug(f"      [ACRoPEAttn] Output: shape={x.shape}")

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        logger.debug(
            f"      [ACRoPEAttn] After projection: shape={x.shape}, min={x.min().item():.4f}, "
            f"max={x.max().item():.4f}"
        )
        x = self.proj_drop(x)
        return x


class RoPEAttention(nn.Module):
    """Attention module with Rotary Position Embeddings (RoPE)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_sdpa: bool = True,
        grid_size: int = 14,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

        # Dimension allocation for 3D rotary embeddings
        self.d_dim = int(_PAIR_DIVISOR * ((head_dim // _THREE_WAY_DIVISOR) // _PAIR_DIVISOR))
        self.h_dim = int(_PAIR_DIVISOR * ((head_dim // _THREE_WAY_DIVISOR) // _PAIR_DIVISOR))
        self.w_dim = int(_PAIR_DIVISOR * ((head_dim // _THREE_WAY_DIVISOR) // _PAIR_DIVISOR))
        self.grid_size = grid_size

    def _get_frame_pos(self, ids: torch.Tensor, H_patches: int | None = None, W_patches: int | None = None) -> torch.Tensor:
        """Get frame position indices from token IDs."""
        if H_patches is None or W_patches is None:
            tokens_per_frame = self.grid_size * self.grid_size
        else:
            tokens_per_frame = H_patches * W_patches
        return ids // tokens_per_frame

    def _get_height_pos(
        self, ids: torch.Tensor, H_patches: int | None = None, W_patches: int | None = None
    ) -> torch.Tensor:
        """Get height position indices from token IDs."""
        if H_patches is None or W_patches is None:
            tokens_per_frame = self.grid_size * self.grid_size
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = H_patches * W_patches
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        return ids // tokens_per_row

    def separate_positions(
        self, ids: torch.Tensor, H_patches: int | None = None, W_patches: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Separate token IDs into frame, height, and width position components."""
        if H_patches is None or W_patches is None:
            tokens_per_frame = self.grid_size * self.grid_size
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = H_patches * W_patches
            tokens_per_row = W_patches

        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        # Remove frame component and height component to get width
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return frame_ids, height_ids, width_ids

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        T: int | None = None,
        H_patches: int | None = None,
        W_patches: int | None = None,
    ) -> torch.Tensor:
        """Forward pass of RoPE attention.

        Args:
            x: Input tensor of shape (B, N, C).
            mask: Optional position mask.
            attn_mask: Optional attention mask for SDPA.
            T: Number of temporal frames.
            H_patches: Height of spatial grid in patches.
            W_patches: Width of spatial grid in patches.

        Returns:
            Output tensor of shape (B, N, C).
        """
        B, N, C = x.size()
        grid_depth = int(N // (self.grid_size * self.grid_size))

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)
        else:
            if T is None or H_patches is None or W_patches is None:
                mask = torch.arange(grid_depth * self.grid_size * self.grid_size, device=x.device, dtype=torch.long)
            else:
                mask = torch.arange(T * H_patches * W_patches, device=x.device, dtype=torch.long)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)

        s = 0
        # Rotate depth (temporal) dimension
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_mask)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_mask)
        s += self.d_dim

        # Rotate height dimension
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_mask)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim

        # Rotate width dimension
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_mask)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim

        # Combine rotated dimensions
        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        # Compute attention
        if attn_mask is not None or self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p, is_causal=self.is_causal, attn_mask=attn_mask
            )
            attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    """Standard multi-head attention module."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_sdpa: bool = True,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass of standard attention.

        Args:
            x: Input tensor of shape (B, N, C).
            mask: Optional position mask (unused, kept for API compatibility).
            attn_mask: Optional attention mask for SDPA.

        Returns:
            Output tensor of shape (B, N, C).
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if attn_mask is not None or self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p, is_causal=self.is_causal, attn_mask=attn_mask
            )
            attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ACBlock(nn.Module):
    """Transformer block with optional Action-Caused RoPE attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        wide_silu: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        use_sdpa: bool = True,
        is_causal: bool = False,
        grid_size: int = 16,
        use_rope: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)

        if use_rope:
            self.attn = ACRoPEAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                grid_size=grid_size,
                proj_drop=drop,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                wide_silu=wide_silu,
                drop=drop,
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        T: int | None = None,
        H: int | None = None,
        W: int | None = None,
        action_tokens: int = 0,
        target_timestep: int | None = None,
    ) -> torch.Tensor:
        """Forward pass of ACBlock.

        Args:
            x: Input tensor of shape (B, N, C).
            mask: Optional position mask.
            attn_mask: Optional attention mask for SDPA.
            T: Number of temporal frames.
            H: Height of spatial grid.
            W: Width of spatial grid.
            action_tokens: Number of action tokens to process separately.
            target_timestep: Target frame index for jump prediction.

        Returns:
            Output tensor of shape (B, N, C).
        """
        y = self.norm1(x)
        if isinstance(self.attn, ACRoPEAttention):
            y = self.attn(y, mask=mask, attn_mask=attn_mask, T=T, H=H, W=W, action_tokens=action_tokens, target_timestep=target_timestep)
        else:
            y = self.attn(y, mask=mask, attn_mask=attn_mask)
        x = x + self.drop_path(y)
        y = self.norm2(x)
        x = x + self.drop_path(self.mlp(y))
        return x


class Block(nn.Module):
    """Transformer block with optional RoPE attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        wide_silu: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        use_sdpa: bool = True,
        is_causal: bool = False,
        grid_size: int = 16,
        use_rope: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)

        if use_rope:
            self.attn = RoPEAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                grid_size=grid_size,
                proj_drop=drop,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                wide_silu=wide_silu,
                drop=drop,
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        T: int | None = None,
        H_patches: int | None = None,
        W_patches: int | None = None,
    ) -> torch.Tensor:
        """Forward pass of Block.

        Args:
            x: Input tensor of shape (B, N, C).
            mask: Optional position mask.
            attn_mask: Optional attention mask for SDPA.
            T: Number of temporal frames.
            H_patches: Height of spatial grid in patches.
            W_patches: Width of spatial grid in patches.

        Returns:
            Output tensor of shape (B, N, C).
        """
        if isinstance(self.attn, RoPEAttention):
            y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)
        else:
            y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    """Cross-attention module for attending from query to context."""

    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = False, use_sdpa: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim * 2), bias=qkv_bias)
        self.use_sdpa = use_sdpa

    def forward(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of cross-attention.

        Args:
            q: Query tensor of shape (B, n, C).
            x: Context tensor of shape (B, N, C).

        Returns:
            Output tensor of shape (B, n, C).
        """
        B, n, C = q.shape
        q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)

        if self.use_sdpa:
            q = F.scaled_dot_product_attention(q, k, v)
        else:
            xattn = (q @ k.transpose(-2, -1)) * self.scale
            xattn = xattn.softmax(dim=-1)  # (batch_size, num_heads, query_len, seq_len)
            q = xattn @ v

        q = q.transpose(1, 2).reshape(B, n, C)
        return q


class CrossAttentionBlock(nn.Module):
    """Transformer block with cross-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of cross-attention block.

        Args:
            q: Query tensor of shape (B, n, C).
            x: Context tensor of shape (B, N, C).

        Returns:
            Output tensor of shape (B, n, C).
        """
        y = self.xattn(q, self.norm1(x))
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q
