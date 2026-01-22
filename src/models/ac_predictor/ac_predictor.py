# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial

import torch
import torch.nn as nn

from src.models.ac_predictor.utils.modules import ACBlock as Block
from src.models.ac_predictor.utils.modules import build_action_block_causal_attention_mask
from src.models.ac_predictor.utils.tensors import trunc_normal_


class VisionTransformerPredictorAC(nn.Module):
    """Action Conditioned Vision Transformer Predictor.

    This predictor operates on PRE-ENCODED features from V-JEPA2 encoder.
    The temporal dimension has already been reduced by tubelet encoding,
    so num_timesteps represents the encoded temporal dimension.
    """

    def __init__(
        self,
        img_size=(256, 256),
        patch_size=16,
        num_timesteps=8,
        embed_dim=768,
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=True,
        use_silu=False,
        wide_silu=True,
        is_frame_causal=True,
        use_activation_checkpointing=False,
        use_rope=True,
        action_embed_dim=7,
        use_extrinsics=False,
        **kwargs
    ):
        super().__init__()
        self.is_frame_causal = is_frame_causal
        self.use_extrinsics = use_extrinsics

        # Map input to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        self.state_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        self.extrinsics_encoder = nn.Linear(action_embed_dim - 1, predictor_embed_dim, bias=True)

        # Determine positional embedding
        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        # --
        self.num_timesteps = num_timesteps
        self.is_video = num_timesteps > 1

        self.grid_height = img_size[0] // self.patch_size
        self.grid_width = img_size[1] // self.patch_size
        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Position embedding
        self.uniform_power = uniform_power

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

        # ------ initialize weights
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        attn_mask = None
        if self.is_frame_causal:
            grid_depth = self.num_timesteps  # Already encoded temporal dimension
            grid_height = self.img_height // self.patch_size
            grid_width = self.img_width // self.patch_size
            attn_mask = build_action_block_causal_attention_mask(
                grid_depth, grid_height, grid_width, add_tokens=3 if use_extrinsics else 2
            )
        self.attn_mask = attn_mask

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, x, actions, states, extrinsics=None):
        """Forward pass through the AC predictor.

        Args:
            x: Context tokens [B, T*N, D] where T is num_timesteps, N = H*W patches
            actions: Action sequences [B, T, action_dim]
            states: State sequences [B, T, action_dim]
            extrinsics: Optional extrinsic parameters [B, T, action_dim-1]

        Returns:
            Predicted features [B, T*N, D]
        """
        import logging
        log = logging.getLogger(__name__)

        log.debug(f"    [AC_PREDICTOR] Input x: {x.shape}")
        log.debug(f"    [AC_PREDICTOR] Actions: {actions.shape}, States: {states.shape}")

        # Map tokens to predictor dimensions
        x = self.predictor_embed(x)
        log.debug(f"    [AC_PREDICTOR] After predictor_embed: {x.shape}, min={x.min().item():.4f}, max={x.max().item():.4f}")

        B, N_ctxt, D = x.size()
        T = N_ctxt // (self.grid_height * self.grid_width)
        log.debug(f"    [AC_PREDICTOR] B={B}, N_ctxt={N_ctxt}, D={D}, T={T}")

        # Validate input temporal dimension
        expected_patches = self.grid_height * self.grid_width
        if N_ctxt % expected_patches != 0:
            raise ValueError(
                f"Input token count ({N_ctxt}) is not divisible by patches_per_frame "
                f"({expected_patches}). Check that input features match model config."
            )
        if T > self.num_timesteps:
            raise ValueError(
                f"Input has {T} timesteps but model was configured for max {self.num_timesteps}. "
                f"Either reduce input sequence length or increase model.num_timesteps."
            )
        T = N_ctxt // (self.grid_height * self.grid_width)

        # Interleave action tokens
        s = self.state_encoder(states).unsqueeze(2)
        a = self.action_encoder(actions).unsqueeze(2)
        log.debug(f"    [AC_PREDICTOR] State embedding: {s.shape}, Action embedding: {a.shape}")

        x = x.view(B, T, self.grid_height * self.grid_width, D)  # [B, T, H*W, D]
        if self.use_extrinsics:
            e = self.extrinsics_encoder(extrinsics).unsqueeze(2)
            x = torch.cat([a, s, e, x], dim=2).flatten(1, 2)  # [B, T*(H*W+3), D]
        else:
            x = torch.cat([a, s, x], dim=2).flatten(1, 2)  # [B, T*(H*W+2), D]
        log.debug(f"    [AC_PREDICTOR] After interleaving action tokens: {x.shape}")

        cond_tokens = 3 if self.use_extrinsics else 2
        attn_mask = self.attn_mask[: x.size(1), : x.size(1)].to(x.device, non_blocking=True)
        log.debug(f"    [AC_PREDICTOR] Attention mask shape: {attn_mask.shape}")

        # Fwd prop through transformer blocks
        log.debug(f"    [AC_PREDICTOR] >>> Processing {len(self.predictor_blocks)} transformer blocks >>>")
        for i, blk in enumerate(self.predictor_blocks):
            log.debug(f"    [AC_PREDICTOR] Block {i}/{len(self.predictor_blocks)-1}: input shape={x.shape}")
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=cond_tokens,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=cond_tokens,
                )
            log.debug(f"    [AC_PREDICTOR] Block {i} output: shape={x.shape}, min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}")

        log.debug(f"    [AC_PREDICTOR] <<< Finished transformer blocks <<<")

        # Split out action and frame tokens
        x = x.view(B, T, cond_tokens + self.grid_height * self.grid_width, D)  # [B, T, K+H*W, D]
        x = x[:, :, cond_tokens:, :].flatten(1, 2)
        log.debug(f"    [AC_PREDICTOR] After removing action tokens: {x.shape}")

        x = self.predictor_norm(x)
        log.debug(f"    [AC_PREDICTOR] After predictor_norm: min={x.min().item():.4f}, max={x.max().item():.4f}")
        x = self.predictor_proj(x)
        log.debug(f"    [AC_PREDICTOR] After predictor_proj (output): {x.shape}, min={x.min().item():.4f}, max={x.max().item():.4f}")

        return x


def vit_ac_predictor(**kwargs):
    # Set defaults, but allow kwargs to override
    defaults = {
        "mlp_ratio": 4,
        "qkv_bias": True,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
    }
    # Merge defaults with kwargs (kwargs take precedence)
    merged_kwargs = {**defaults, **kwargs}
    model = VisionTransformerPredictorAC(**merged_kwargs)
    return model
