"""GatedDeltaNet action-conditioned backbone for spatiotemporal prediction.

Wraps the GatedDeltaNet recurrent-linear-attention layer (Yang et al., 2024)
into the same encoder → action-tile → process → residual-decode pattern
used by all CL models in this repository.

Architecture per timestep:
    1. 1×1 Conv Encoder: 1024 → hidden_dim
    2. Spatial action tiling + linear projection
    3. Temporal position embedding
    4. N stacked GatedDeltaNet blocks (RMSNorm → GatedDeltaNet → RMSNorm → SwiGLU)
    5. 1×1 Conv Decoder: hidden_dim → 1024, output = z_t + delta

Reference: Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule", 2024
"""

from __future__ import annotations

import logging
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fla.modules import RMSNorm as _FlaRMSNorm  # noqa: F401 (available but not used directly)

from .gated_delta_net import GatedDeltaNet


class RMSNorm(nn.Module):
    """Pure-PyTorch RMSNorm (works on CPU and CUDA)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLU(nn.Module):
    """Pure-PyTorch SwiGLU matching xformers.ops.SwiGLU API.

    Computes: output = W2 · (SiLU(W1 · x) * W3 · x)
    """

    def __init__(self, in_features: int, hidden_features: int, bias: bool = False, **kwargs) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.w3 = nn.Linear(in_features, hidden_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

log = logging.getLogger(__name__)

# ───────────────────────── Constants ─────────────────────────
DEFAULT_INPUT_DIM: Final = 1024
DEFAULT_HIDDEN_DIM: Final = 768
DEFAULT_ACTION_DIM: Final = 2
DEFAULT_SPATIAL_SIZE: Final = 16
DEFAULT_N_LAYERS: Final = 5
DEFAULT_NUM_HEADS: Final = 12
DEFAULT_EXPAND_K: Final = 0.75
DEFAULT_EXPAND_V: Final = 1.5
DEFAULT_INTERMEDIATE_SIZE: Final = 2048
DEFAULT_CONV_SIZE: Final = 4
DEFAULT_NUM_TIMESTEPS: Final = 8


class GatedDeltaNetBlock(nn.Module):
    """Single GatedDeltaNet block: RMSNorm → GatedDeltaNet → residual → RMSNorm → SwiGLU → residual."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        expand_k: float,
        expand_v: float,
        intermediate_size: int,
        conv_size: int = 4,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size=hidden_dim)
        self.attn = GatedDeltaNet(
            mode='chunk',
            hidden_size=hidden_dim,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            conv_size=conv_size,
            layer_idx=layer_idx,
            use_mamba_gate=True,
        )
        self.norm2 = RMSNorm(hidden_size=hidden_dim)
        self.mlp = SwiGLU(hidden_dim, intermediate_size, bias=False, _pack_weights=False)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, N, D] → [B, N, D]"""
        # GatedDeltaNet attention with pre-norm
        residual = x
        x = self.norm1(x)
        x, _, _ = self.attn(x)
        x = residual + x

        # SwiGLU FFN with pre-norm
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x


class GatedDeltaNetBackbone(nn.Module):
    """Action-conditioned GatedDeltaNet backbone for spatiotemporal prediction.

    Follows the same structure as TitansBackbone: per-timestep processing
    of 256 spatial tokens through encoder → action concat → GDN blocks → decoder.
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        action_dim: int = DEFAULT_ACTION_DIM,
        spatial_size: int = DEFAULT_SPATIAL_SIZE,
        n_layers: int = DEFAULT_N_LAYERS,
        num_heads: int = DEFAULT_NUM_HEADS,
        expand_k: float = DEFAULT_EXPAND_K,
        expand_v: float = DEFAULT_EXPAND_V,
        intermediate_size: int = DEFAULT_INTERMEDIATE_SIZE,
        conv_size: int = DEFAULT_CONV_SIZE,
        num_timesteps: int = DEFAULT_NUM_TIMESTEPS,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.spatial_size = spatial_size
        self.num_timesteps = num_timesteps

        N = spatial_size * spatial_size  # 256 patches

        # 1×1 Conv encoder: input_dim → hidden_dim
        self.encoder = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)

        # Action projection: hidden_dim + action_dim → hidden_dim
        self.action_proj = nn.Linear(hidden_dim + action_dim, hidden_dim)

        # Temporal position embedding (0 … num_timesteps inclusive)
        self.temporal_embed = nn.Embedding(num_timesteps + 1, hidden_dim)

        # Stacked GatedDeltaNet blocks
        self.blocks = nn.ModuleList([
            GatedDeltaNetBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                expand_k=expand_k,
                expand_v=expand_v,
                intermediate_size=intermediate_size,
                conv_size=conv_size,
                layer_idx=i,
            )
            for i in range(n_layers)
        ])

        # Final norm
        self.final_norm = nn.LayerNorm(hidden_dim)

        # 1×1 Conv decoder: hidden_dim → input_dim (residual prediction)
        self.decoder = nn.Conv2d(hidden_dim, input_dim, kernel_size=1)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def _tile_action(self, action: Tensor, H: int, W: int) -> Tensor:
        """Tile action spatially: [B, action_dim] → [B, action_dim, H, W]."""
        B, A = action.shape
        return action.view(B, A, 1, 1).expand(B, A, H, W)

    def forward(
        self,
        z: Tensor,
        actions: Tensor,
        states: Tensor | None = None,
        extrinsics: Tensor | None = None,
        target_timestep: int | None = None,
    ) -> Tensor:
        """Forward pass — predict next frames given context and actions.

        Args:
            z:        [B, T*N, D]  context features
            actions:  [B, T, action_dim]
            states:   unused (API compat)
            extrinsics: unused (API compat)
            target_timestep: temporal position override for jump prediction

        Returns:
            Predicted features [B, T*N, D]
        """
        B = z.shape[0]
        N = self.spatial_size * self.spatial_size
        D = self.input_dim
        H = W = self.spatial_size
        T = z.shape[1] // N

        # Reshape to spatial: [B, T, D, H, W]
        z_spatial = z.reshape(B, T, N, D).permute(0, 1, 3, 2).reshape(B, T, D, H, W)

        predictions: list[Tensor] = []

        for t in range(T):
            current_z = z_spatial[:, t]  # [B, D, H, W]

            # Encode: [B, D, H, W] → [B, hidden_dim, H, W]
            feat = self.encoder(current_z)

            # Tile action: [B, action_dim, H, W]
            action_tiled = self._tile_action(actions[:, t], H, W)

            # Concat + project: [B, hidden_dim+action_dim, H, W] → [B, N, hidden_dim]
            combined = torch.cat([feat, action_tiled], dim=1)  # [B, hid+act, H, W]
            combined = combined.reshape(B, self.hidden_dim + self.action_dim, N)
            combined = combined.permute(0, 2, 1)  # [B, N, hid+act]
            tokens = self.action_proj(combined)    # [B, N, hidden_dim]

            # Add temporal position embedding
            if target_timestep is not None:
                time_pos = min(target_timestep - 1, self.num_timesteps)
            else:
                time_pos = min(t, self.num_timesteps)
            time_pos_idx = torch.tensor([time_pos], dtype=torch.long, device=z.device)
            tokens = tokens + self.temporal_embed(time_pos_idx)

            # Process through GatedDeltaNet blocks
            for block in self.blocks:
                tokens = block(tokens)
            tokens = self.final_norm(tokens)

            # Decode residual: [B, N, hidden_dim] → [B, D, H, W]
            dec_in = tokens.permute(0, 2, 1).reshape(B, self.hidden_dim, H, W)
            delta = self.decoder(dec_in)

            # Residual prediction
            next_z = current_z + delta
            predictions.append(next_z)

        # Stack → [B, T, D, H, W] → [B, T*N, D]
        pred_stack = torch.stack(predictions, dim=1)
        pred_stack = pred_stack.reshape(B, T, D, N).permute(0, 1, 3, 2)
        return pred_stack.reshape(B, T * N, D)
