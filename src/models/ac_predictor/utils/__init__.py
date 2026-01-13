# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Utility modules for AC Predictor."""

from src.models.ac_predictor.utils.modules import (
    ACBlock,
    ACRoPEAttention,
    Attention,
    Block,
    CrossAttention,
    CrossAttentionBlock,
    DropPath,
    MLP,
    RoPEAttention,
    SwiGLUFFN,
    build_action_block_causal_attention_mask,
    rotate_queries_or_keys,
)
from src.models.ac_predictor.utils.pos_embs import (
    get_1d_sincos_pos_embed,
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed,
)
from src.models.ac_predictor.utils.tensors import repeat_interleave_batch, trunc_normal_

__all__ = [
    # modules
    "ACBlock",
    "ACRoPEAttention",
    "Attention",
    "Block",
    "CrossAttention",
    "CrossAttentionBlock",
    "DropPath",
    "MLP",
    "RoPEAttention",
    "SwiGLUFFN",
    "build_action_block_causal_attention_mask",
    "rotate_queries_or_keys",
    # pos_embs
    "get_1d_sincos_pos_embed",
    "get_1d_sincos_pos_embed_from_grid",
    "get_2d_sincos_pos_embed",
    "get_3d_sincos_pos_embed",
    # tensors
    "repeat_interleave_batch",
    "trunc_normal_",
]
