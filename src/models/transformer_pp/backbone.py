"""Transformer++ action-conditioned backbone for spatiotemporal prediction.

Wraps the Transformer++ architecture (MHA with rotary embeddings, causal conv1d,
and GatedMLP) from Dao & Gu (2024) into the standard CL pipeline interface.

Architecture per timestep:
    1. 1×1 Conv Encoder: 1024 → d_model
    2. Spatial action tiling + linear projection
    3. Temporal position embedding
    4. N stacked TransformerPPBlocks (RMSNorm → MHA → RMSNorm → GatedMLP)
    5. 1×1 Conv Decoder: d_model → 1024, output = z_t + delta

Reference: Dao & Gu, "Transformers are SSMs", 2024 (Mamba-2 paper, Transformer++ baseline)
"""

from __future__ import annotations

import logging
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .mha import MHA
from .mlp import GatedMLP

log = logging.getLogger(__name__)

# ───────────────────────── Constants ─────────────────────────
DEFAULT_INPUT_DIM: Final = 1024
DEFAULT_D_MODEL: Final = 768
DEFAULT_ACTION_DIM: Final = 2
DEFAULT_SPATIAL_SIZE: Final = 16
DEFAULT_N_LAYERS: Final = 6
DEFAULT_N_HEADS: Final = 12
DEFAULT_D_CONV: Final = 4
DEFAULT_ROTARY_EMB_DIM: Final = 64
DEFAULT_MLP_HIDDEN: Final = 2048
DEFAULT_NUM_TIMESTEPS: Final = 8


class RMSNorm(nn.Module):
    """RMSNorm — pure PyTorch implementation (avoids mamba_ssm.ops.triton dependency)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        output = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (output * self.weight).type_as(x)


class TransformerPPBlock(nn.Module):
    """Single Transformer++ block: RMSNorm → MHA → residual → RMSNorm → GatedMLP → residual."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_conv: int = 4,
        rotary_emb_dim: int = 64,
        mlp_hidden: int = 2048,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mha = MHA(
            embed_dim=d_model,
            num_heads=n_heads,
            d_conv=d_conv,
            rotary_emb_dim=rotary_emb_dim,
            causal=False,  # Non-causal: spatial tokens have no temporal ordering
            layer_idx=layer_idx,
        )
        self.norm2 = RMSNorm(d_model)
        self.mlp = GatedMLP(
            in_features=d_model,
            hidden_features=mlp_hidden,
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, N, D] → [B, N, D]"""
        # MHA with pre-norm
        residual = x
        x = self.norm1(x)
        x = self.mha(x)
        x = residual + x

        # GatedMLP with pre-norm
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x


class TransformerPPBackbone(nn.Module):
    """Action-conditioned Transformer++ backbone for spatiotemporal prediction.

    Follows the same structure as TitansBackbone: per-timestep processing
    of 256 spatial tokens through encoder → action concat → Transformer++ blocks → decoder.
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        d_model: int = DEFAULT_D_MODEL,
        action_dim: int = DEFAULT_ACTION_DIM,
        spatial_size: int = DEFAULT_SPATIAL_SIZE,
        n_layers: int = DEFAULT_N_LAYERS,
        n_heads: int = DEFAULT_N_HEADS,
        d_conv: int = DEFAULT_D_CONV,
        rotary_emb_dim: int = DEFAULT_ROTARY_EMB_DIM,
        mlp_hidden: int = DEFAULT_MLP_HIDDEN,
        num_timesteps: int = DEFAULT_NUM_TIMESTEPS,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.action_dim = action_dim
        self.spatial_size = spatial_size
        self.num_timesteps = num_timesteps

        N = spatial_size * spatial_size  # 256 patches

        # 1×1 Conv encoder: input_dim → d_model
        self.encoder = nn.Conv2d(input_dim, d_model, kernel_size=1)

        # Action projection: d_model + action_dim → d_model
        self.action_proj = nn.Linear(d_model + action_dim, d_model)

        # Temporal position embedding (0 … num_timesteps inclusive)
        self.temporal_embed = nn.Embedding(num_timesteps + 1, d_model)

        # Stacked Transformer++ blocks
        self.blocks = nn.ModuleList([
            TransformerPPBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_conv=d_conv,
                rotary_emb_dim=rotary_emb_dim,
                mlp_hidden=mlp_hidden,
                layer_idx=i,
            )
            for i in range(n_layers)
        ])

        # Final norm
        self.final_norm = nn.LayerNorm(d_model)

        # 1×1 Conv decoder: d_model → input_dim (residual prediction)
        self.decoder = nn.Conv2d(d_model, input_dim, kernel_size=1)
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

            # Encode: [B, D, H, W] → [B, d_model, H, W]
            feat = self.encoder(current_z)

            # Tile action: [B, action_dim, H, W]
            action_tiled = self._tile_action(actions[:, t], H, W)

            # Concat + project: [B, d_model+action_dim, H, W] → [B, N, d_model]
            combined = torch.cat([feat, action_tiled], dim=1)
            combined = combined.reshape(B, self.d_model + self.action_dim, N)
            combined = combined.permute(0, 2, 1)
            tokens = self.action_proj(combined)

            # Add temporal position embedding
            if target_timestep is not None:
                time_pos = min(target_timestep - 1, self.num_timesteps)
            else:
                time_pos = min(t, self.num_timesteps)
            time_pos_idx = torch.tensor([time_pos], dtype=torch.long, device=z.device)
            tokens = tokens + self.temporal_embed(time_pos_idx)

            # Process through Transformer++ blocks
            for block in self.blocks:
                tokens = block(tokens)
            tokens = self.final_norm(tokens)

            # Decode residual: [B, N, d_model] → [B, D, H, W]
            dec_in = tokens.permute(0, 2, 1).reshape(B, self.d_model, H, W)
            delta = self.decoder(dec_in)

            # Residual prediction
            next_z = current_z + delta
            predictions.append(next_z)

        # Stack → [B, T, D, H, W] → [B, T*N, D]
        pred_stack = torch.stack(predictions, dim=1)
        pred_stack = pred_stack.reshape(B, T, D, N).permute(0, 1, 3, 2)
        return pred_stack.reshape(B, T * N, D)
