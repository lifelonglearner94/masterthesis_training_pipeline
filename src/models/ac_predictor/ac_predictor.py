# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from functools import partial
from typing import Final

import torch
import torch.nn as nn

from src.models.ac_predictor.utils.modules import ACBlock as Block
from src.models.ac_predictor.utils.modules import build_action_block_causal_attention_mask
from src.models.ac_predictor.utils.tensors import trunc_normal_

logger = logging.getLogger(__name__)

# Constants
ACTION_TOKENS_WITH_EXTRINSICS: Final = 3
ACTION_TOKENS_WITHOUT_EXTRINSICS: Final = 2


class VisionTransformerPredictorAC(nn.Module):
    """Action Conditioned Vision Transformer Predictor.

    This predictor operates on PRE-ENCODED features from V-JEPA2 encoder.
    The temporal dimension has already been reduced by tubelet encoding,
    so num_timesteps represents the encoded temporal dimension.
    """

    def __init__(
        self,
        img_size: tuple[int, int] | int = (256, 256),
        patch_size: int = 16,
        num_timesteps: int = 8,
        embed_dim: int = 768,
        predictor_embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        init_std: float = 0.02,
        uniform_power: bool = True,
        use_silu: bool = False,
        wide_silu: bool = True,
        is_frame_causal: bool = True,
        use_activation_checkpointing: bool = False,
        use_rope: bool = True,
        action_embed_dim: int = 7,
        use_extrinsics: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.is_frame_causal = is_frame_causal
        self.use_extrinsics = use_extrinsics

        # Map input to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        self.state_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        self.extrinsics_encoder = nn.Linear(
            action_embed_dim - 1, predictor_embed_dim, bias=True
        )

        # Determine positional embedding
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_timesteps = num_timesteps

        self.grid_height = img_size[0] // self.patch_size
        self.grid_width = img_size[1] // self.patch_size
        self.use_activation_checkpointing = use_activation_checkpointing

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Attention Blocks
        self.use_rope = use_rope
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # Normalize & project back to input dimension
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # Initialize weights
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        # Build attention mask
        attn_mask = None
        if self.is_frame_causal:
            grid_depth = self.num_timesteps
            grid_height = self.img_height // self.patch_size
            grid_width = self.img_width // self.patch_size
            add_tokens = (
                ACTION_TOKENS_WITH_EXTRINSICS
                if use_extrinsics
                else ACTION_TOKENS_WITHOUT_EXTRINSICS
            )
            attn_mask = build_action_block_causal_attention_mask(
                grid_depth, grid_height, grid_width, add_tokens=add_tokens
            )
        self.attn_mask = attn_mask

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize module weights.

        Args:
            m: PyTorch module to initialize.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self) -> None:
        """Rescale attention and MLP weights for stable training."""

        def rescale(param: torch.Tensor, layer_id: int) -> None:
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        target_timestep: int | None = None,
    ) -> torch.Tensor:
        """Forward pass through the AC predictor.

        Args:
            x: Context tokens [B, T*N, D] where T is num_timesteps, N = H*W patches
            actions: Action sequences [B, T, action_dim]
            states: State sequences [B, T, action_dim]
            extrinsics: Optional extrinsic parameters [B, T, action_dim-1]
            target_timestep: Target frame index for jump prediction (0-indexed
                into the features tensor). When set, RoPE temporal positions
                are overridden so the output predicts frame ``target_timestep``.

        Returns:
            Predicted features [B, T*N, D]
        """
        logger.debug(f"    [AC_PREDICTOR] Input x: {x.shape}")
        logger.debug(
            f"    [AC_PREDICTOR] Actions: {actions.shape}, States: {states.shape}"
        )

        # Map tokens to predictor dimensions
        x = self.predictor_embed(x)
        logger.debug(
            f"    [AC_PREDICTOR] After predictor_embed: {x.shape}, "
            f"min={x.min().item():.4f}, max={x.max().item():.4f}"
        )

        batch_size, num_context, dim = x.size()
        timesteps = num_context // (self.grid_height * self.grid_width)
        logger.debug(
            f"    [AC_PREDICTOR] B={batch_size}, N_ctxt={num_context}, "
            f"D={dim}, T={timesteps}"
        )

        # Validate input temporal dimension
        expected_patches = self.grid_height * self.grid_width
        if num_context % expected_patches != 0:
            raise ValueError(
                f"Input token count ({num_context}) is not divisible by "
                f"patches_per_frame ({expected_patches}). "
                "Check that input features match model config."
            )
        if timesteps > self.num_timesteps:
            raise ValueError(
                f"Input has {timesteps} timesteps but model was configured "
                f"for max {self.num_timesteps}. "
                "Either reduce input sequence length or increase model.num_timesteps."
            )

        # Interleave action tokens
        s = self.state_encoder(states).unsqueeze(2)
        a = self.action_encoder(actions).unsqueeze(2)
        logger.debug(
            f"    [AC_PREDICTOR] State embedding: {s.shape}, "
            f"Action embedding: {a.shape}"
        )

        x = x.view(batch_size, timesteps, self.grid_height * self.grid_width, dim)
        if self.use_extrinsics:
            e = self.extrinsics_encoder(extrinsics).unsqueeze(2)
            x = torch.cat([a, s, e, x], dim=2).flatten(1, 2)
        else:
            x = torch.cat([a, s, x], dim=2).flatten(1, 2)
        logger.debug(f"    [AC_PREDICTOR] After interleaving action tokens: {x.shape}")

        cond_tokens = (
            ACTION_TOKENS_WITH_EXTRINSICS
            if self.use_extrinsics
            else ACTION_TOKENS_WITHOUT_EXTRINSICS
        )

        # Clone attn_mask to prevent inference tensor conflicts during TTA backward pass
        attn_mask = (
            self.attn_mask[: x.size(1), : x.size(1)]
            .to(x.device, non_blocking=True)
            .clone()
        )
        logger.debug(f"    [AC_PREDICTOR] Attention mask shape: {attn_mask.shape}")

        # Forward pass through transformer blocks
        logger.debug(
            f"    [AC_PREDICTOR] >>> Processing {len(self.predictor_blocks)} "
            f"transformer blocks >>>"
        )
        for i, blk in enumerate(self.predictor_blocks):
            logger.debug(
                f"    [AC_PREDICTOR] Block {i}/{len(self.predictor_blocks) - 1}: "
                f"input shape={x.shape}"
            )
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=timesteps,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=cond_tokens,
                    target_timestep=target_timestep,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=timesteps,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=cond_tokens,
                    target_timestep=target_timestep,
                )
            logger.debug(
                f"    [AC_PREDICTOR] Block {i} output: shape={x.shape}, "
                f"min={x.min().item():.4f}, max={x.max().item():.4f}, "
                f"mean={x.mean().item():.4f}"
            )

        logger.debug(f"    [AC_PREDICTOR] <<< Finished transformer blocks <<<")

        # Split out action and frame tokens
        x = x.view(
            batch_size,
            timesteps,
            cond_tokens + self.grid_height * self.grid_width,
            dim,
        )
        x = x[:, :, cond_tokens:, :].flatten(1, 2)
        logger.debug(f"    [AC_PREDICTOR] After removing action tokens: {x.shape}")

        x = self.predictor_norm(x)
        logger.debug(
            f"    [AC_PREDICTOR] After predictor_norm: min={x.min().item():.4f}, "
            f"max={x.max().item():.4f}"
        )
        x = self.predictor_proj(x)
        logger.debug(
            f"    [AC_PREDICTOR] After predictor_proj (output): {x.shape}, "
            f"min={x.min().item():.4f}, max={x.max().item():.4f}"
        )

        return x


def vit_ac_predictor(**kwargs) -> VisionTransformerPredictorAC:
    """Create a Vision Transformer AC Predictor with default configuration.

    Args:
        **kwargs: Configuration options to override defaults.

    Returns:
        Configured VisionTransformerPredictorAC model instance.
    """
    defaults = {
        "mlp_ratio": 4,
        "qkv_bias": True,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
    }
    merged_kwargs = {**defaults, **kwargs}
    return VisionTransformerPredictorAC(**merged_kwargs)
