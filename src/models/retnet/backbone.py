"""RetNet action-conditioned backbone for spatiotemporal prediction.

Wraps the Retentive Network architecture (Sun et al., 2023) from torchscale
into the same encoder → action-tile → process → residual-decode pattern
used by all CL models in this repository.

Architecture per timestep:
    1. 1×1 Conv Encoder: 1024 → embed_dim
    2. Spatial action tiling + linear projection
    3. Temporal position embedding
    4. RetNetRelPos (exponential decay + rotary for retention)
    5. N stacked RetNetDecoderLayers (RMSNorm → MultiScaleRetention → RMSNorm → GLU)
    6. 1×1 Conv Decoder: embed_dim → 1024, output = z_t + delta

Reference: Sun et al., "Retentive Network: A Successor to Transformer for Large Language Models", 2023
"""

from __future__ import annotations

import logging
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rms_norm import RMSNorm
from .multiscale_retention import MultiScaleRetention
from .gate_linear_unit import GLU

log = logging.getLogger(__name__)

# ───────────────────────── Constants ─────────────────────────
DEFAULT_INPUT_DIM: Final = 1024
DEFAULT_EMBED_DIM: Final = 768
DEFAULT_VALUE_DIM: Final = 1280
DEFAULT_ACTION_DIM: Final = 2
DEFAULT_SPATIAL_SIZE: Final = 16
DEFAULT_N_LAYERS: Final = 6
DEFAULT_N_HEADS: Final = 4
DEFAULT_FFN_DIM: Final = 1280
DEFAULT_RECURRENT_CHUNK_SIZE: Final = 64
DEFAULT_NUM_TIMESTEPS: Final = 8
DEFAULT_DROPOUT: Final = 0.0
DEFAULT_LAYERNORM_EPS: Final = 1e-6


# ─────────────────────── RetNet-specific layers ──────────────


class RetNetRelPos(nn.Module):
    """Relative position encoding for RetNet with exponential decay.

    Simplified from torchscale/architecture/retnet.py — supports parallel
    and chunkwise_recurrent modes (no incremental/recurrent mode needed here).
    """

    def __init__(self, embed_dim: int, num_heads: int, recurrent_chunk_size: int = 64) -> None:
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-5 - torch.arange(num_heads, dtype=torch.float)))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        self.recurrent_chunk_size = recurrent_chunk_size

    def forward(self, slen: int, chunkwise_recurrent: bool = False):
        if chunkwise_recurrent:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])

            block_index = torch.arange(self.recurrent_chunk_size).to(self.decay)
            mask = torch.tril(torch.ones(self.recurrent_chunk_size, self.recurrent_chunk_size).to(self.decay))
            mask = torch.masked_fill(
                block_index[:, None] - block_index[None, :], ~mask.bool(), float("inf")
            )
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)

            value_inner_decay = mask[:, -1] / mask[:, -1].sum(dim=-1, keepdim=True)
            value_inner_decay = value_inner_decay.unsqueeze(-1)
            scale = mask.sum(dim=-1, keepdim=True).sqrt()
            inner_mask = mask / scale

            cross_decay = torch.exp(self.decay * self.recurrent_chunk_size)
            query_inner_decay = torch.exp(self.decay[:, None] * (block_index + 1))
            query_inner_decay = query_inner_decay[:, :, None] / (scale / mask[:, -1].sum(dim=-1)[:, None, None])
            cross_decay = cross_decay[:, None, None]
            retention_rel_pos = ((sin, cos), (inner_mask, cross_decay, query_inner_decay, value_inner_decay))
        else:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])
            mask = torch.tril(torch.ones(slen, slen).to(self.decay))
            mask = torch.masked_fill(
                index[:, None] - index[None, :], ~mask.bool(), float("inf")
            )
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos


class RetNetDecoderLayer(nn.Module):
    """Simplified RetNet decoder layer: RMSNorm → Retention → residual → RMSNorm → GLU → residual.

    Stripped of MoE, DropPath, deepnorm, and fairscale dependencies from the original.
    """

    def __init__(
        self,
        embed_dim: int,
        value_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        layernorm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout_module = nn.Dropout(dropout)

        self.retention = MultiScaleRetention(
            embed_dim=embed_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            layernorm_eps=layernorm_eps,
        )
        self.retention_layer_norm = RMSNorm(embed_dim, eps=layernorm_eps)

        self.ffn = GLU(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            activation_fn="swish",
            dropout=dropout,
            activation_dropout=dropout,
        )
        self.final_layer_norm = RMSNorm(embed_dim, eps=layernorm_eps)

    def forward(self, x: Tensor, retention_rel_pos=None, chunkwise_recurrent: bool = False) -> Tensor:
        # Retention with pre-norm + residual
        residual = x
        x = self.retention_layer_norm(x)
        x = self.retention(x, rel_pos=retention_rel_pos, chunkwise_recurrent=chunkwise_recurrent)
        x = self.dropout_module(x)
        x = residual + x

        # GLU FFN with pre-norm + residual
        residual = x
        x = self.final_layer_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x


# ─────────────────────── Backbone ───────────────────────────


class RetNetBackbone(nn.Module):
    """Action-conditioned RetNet backbone for spatiotemporal prediction.

    Follows the same structure as TitansBackbone: per-timestep processing
    of 256 spatial tokens through encoder → action concat → RetNet layers → decoder.
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        embed_dim: int = DEFAULT_EMBED_DIM,
        value_dim: int = DEFAULT_VALUE_DIM,
        action_dim: int = DEFAULT_ACTION_DIM,
        spatial_size: int = DEFAULT_SPATIAL_SIZE,
        n_layers: int = DEFAULT_N_LAYERS,
        n_heads: int = DEFAULT_N_HEADS,
        ffn_dim: int = DEFAULT_FFN_DIM,
        recurrent_chunk_size: int = DEFAULT_RECURRENT_CHUNK_SIZE,
        num_timesteps: int = DEFAULT_NUM_TIMESTEPS,
        dropout: float = DEFAULT_DROPOUT,
        layernorm_eps: float = DEFAULT_LAYERNORM_EPS,
        chunkwise_recurrent: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.spatial_size = spatial_size
        self.num_timesteps = num_timesteps
        self.chunkwise_recurrent = chunkwise_recurrent

        N = spatial_size * spatial_size  # 256 patches

        # 1×1 Conv encoder: input_dim → embed_dim
        self.encoder = nn.Conv2d(input_dim, embed_dim, kernel_size=1)

        # Action projection: embed_dim + action_dim → embed_dim
        self.action_proj = nn.Linear(embed_dim + action_dim, embed_dim)

        # Temporal position embedding (0 … num_timesteps inclusive)
        self.temporal_embed = nn.Embedding(num_timesteps + 1, embed_dim)

        # Relative position encoding for retention
        self.retnet_rel_pos = RetNetRelPos(
            embed_dim=embed_dim,
            num_heads=n_heads,
            recurrent_chunk_size=recurrent_chunk_size,
        )

        # Stacked RetNet decoder layers
        self.layers = nn.ModuleList([
            RetNetDecoderLayer(
                embed_dim=embed_dim,
                value_dim=value_dim,
                num_heads=n_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                layernorm_eps=layernorm_eps,
            )
            for _ in range(n_layers)
        ])

        # Final norm
        self.final_norm = nn.LayerNorm(embed_dim)

        # 1×1 Conv decoder: embed_dim → input_dim (residual prediction)
        self.decoder = nn.Conv2d(embed_dim, input_dim, kernel_size=1)
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

            # Encode: [B, D, H, W] → [B, embed_dim, H, W]
            feat = self.encoder(current_z)

            # Tile action: [B, action_dim, H, W]
            action_tiled = self._tile_action(actions[:, t], H, W)

            # Concat + project: [B, embed_dim+action_dim, H, W] → [B, N, embed_dim]
            combined = torch.cat([feat, action_tiled], dim=1)
            combined = combined.reshape(B, self.embed_dim + self.action_dim, N)
            combined = combined.permute(0, 2, 1)
            tokens = self.action_proj(combined)

            # Add temporal position embedding
            if target_timestep is not None:
                time_pos = min(target_timestep - 1, self.num_timesteps)
            else:
                time_pos = min(t, self.num_timesteps)
            time_pos_idx = torch.tensor([time_pos], dtype=torch.long, device=z.device)
            tokens = tokens + self.temporal_embed(time_pos_idx)

            # Compute retention relative positions for current sequence length
            retention_rel_pos = self.retnet_rel_pos(N, chunkwise_recurrent=self.chunkwise_recurrent)

            # Process through RetNet decoder layers
            for layer in self.layers:
                tokens = layer(tokens, retention_rel_pos=retention_rel_pos, chunkwise_recurrent=self.chunkwise_recurrent)
            tokens = self.final_norm(tokens)

            # Decode residual: [B, N, embed_dim] → [B, D, H, W]
            dec_in = tokens.permute(0, 2, 1).reshape(B, self.embed_dim, H, W)
            delta = self.decoder(dec_in)

            # Residual prediction
            next_z = current_z + delta
            predictions.append(next_z)

        # Stack → [B, T, D, H, W] → [B, T*N, D]
        pred_stack = torch.stack(predictions, dim=1)
        pred_stack = pred_stack.reshape(B, T, D, N).permute(0, 1, 3, 2)
        return pred_stack.reshape(B, T * N, D)
