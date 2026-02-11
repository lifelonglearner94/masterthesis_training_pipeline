# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Positional embedding utilities for vision transformers.

This module provides functions to generate sinusoidal positional embeddings
for 1D, 2D, and 3D grids, commonly used in Vision Transformer (ViT) and
Masked Autoencoder (MAE) architectures.
"""

import numpy as np
import numpy.typing as npt


def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    grid_depth: int,
    cls_token: bool = False,
    uniform_power: bool = False,
) -> npt.NDArray[np.float64]:
    """Generate 3D sinusoidal positional embeddings.

    Creates positional embeddings for a 3D grid (depth × height × width) using
    sinusoidal functions. The embeddings are constructed by concatenating
    position encodings along each dimension.

    Args:
        embed_dim: The embedding dimension for each position.
        grid_size: The height and width of the spatial grid.
        grid_depth: The depth of the grid.
        cls_token: Prepend a zero vector for a [CLS] token.
        uniform_power: Distribute embedding dimensions evenly across all
            three dimensions. When False, uses a 2:1:1 ratio
            (depth:height:width).

    Returns:
        Position embeddings of shape [grid_depth*grid_size*grid_size, embed_dim]
        or [1+grid_depth*grid_size*grid_size, embed_dim] if cls_token=True.
    """
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(
        grid_h, grid_d, grid_w
    )  # order of meshgrid is very important for indexing as [d,h,w]

    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim / 6) * 2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)  # (T*H*W, D1)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)  # (T*H*W, D2)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)  # (T*H*W, D3)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
) -> npt.NDArray[np.float64]:
    """Generate 2D sinusoidal positional embeddings.

    Creates positional embeddings for a 2D grid (height × width) using
    sinusoidal functions. The embeddings are constructed by concatenating
    position encodings along each dimension.

    Args:
        embed_dim: The embedding dimension for each position.
        grid_size: The height and width of the spatial grid.
        cls_token: Prepend a zero vector for a [CLS] token.

    Returns:
        Position embeddings of shape [grid_size*grid_size, embed_dim]
        or [1+grid_size*grid_size, embed_dim] if cls_token=True.
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)  # order of meshgrid is very important for indexing as [h, w]

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)  # (H*W, D/2)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
) -> npt.NDArray[np.float64]:
    """Generate 1D sinusoidal positional embeddings.

    Creates positional embeddings for a 1D grid using sinusoidal functions.
    This implements the sinusoidal position encoding from the
    "Attention Is All You Need" paper.

    Args:
        embed_dim: The embedding dimension for each position.
        grid_size: The length of the 1D grid.
        cls_token: Prepend a zero vector for a [CLS] token.

    Returns:
        Position embeddings of shape [grid_size, embed_dim]
        or [1+grid_size, embed_dim] if cls_token=True.
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int,
    pos: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Generate sinusoidal positional embeddings from a position grid.

    This function implements the sinusoidal position encoding from the
    "Attention Is All You Need" paper. It uses sine and cosine functions
    of different frequencies to encode positional information.

    Args:
        embed_dim: The embedding dimension for each position.
        pos: Position values to encode, shape (M,).

    Returns:
        Positional embeddings of shape (M, embed_dim), where M is the number
        of positions.

    Raises:
        ValueError: If embed_dim is not an even number.
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = np.array(pos).reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
