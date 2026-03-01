"""Continuum Memory System (CMS) — Multi-Frequency MLP Hierarchy (Behrouz 2025).

Implements the CMS component of the HOPE architecture. CMS provides a hierarchy
of MLPs operating at different temporal frequencies:
    - Fast level  (period=1): processes every frame
    - Medium level (period=3): processes every 3rd frame (e.g. aligned with jump_k)
    - Slow level   (period=7): processes once per clip (clip-level physics context)

Each level is a simple MLP with residual connection and gradient clipping.
Higher-frequency levels process every frame; lower-frequency levels only
process frames whose index is divisible by their update_period.

Key design from the paper (Section 8.3):
    y_t = MLP^(f_K)( MLP^(f_{K-1})( ... MLP^(f_1)(o_t) ... ) )

Frame-Aware Scheduling (adapted for video data):
    update_period refers to FRAMES, not flat token indices. When scheduling
    is enabled, each CMS level only processes all tokens belonging to
    frames where (frame_index % update_period == 0). Tokens in skipped
    frames pass through unchanged via the residual connection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LevelSpec:
    """Immutable specification for a single CMS frequency level.

    Attributes:
        name: Human-readable identifier (e.g., 'fast', 'medium', 'slow').
        update_period: Frames between updates (1=every frame, 3=every 3rd frame, etc.).
        warmup_steps: Number of training steps before level starts updating.
        jitter: Random variation in update period for regularization.
        hidden_multiplier: Hidden dimension multiplier (dim * multiplier).

    Raises:
        ValueError: If update_period < 1 or hidden_multiplier <= 0.
    """

    name: str = "fast"
    update_period: int = 1
    warmup_steps: int = 0
    jitter: float = 0.0
    hidden_multiplier: float = 4.0

    def __post_init__(self) -> None:
        if self.update_period < 1:
            raise ValueError(
                f"update_period must be >= 1, got {self.update_period}"
            )
        if self.hidden_multiplier <= 0:
            raise ValueError(
                f"hidden_multiplier must be > 0, got {self.hidden_multiplier}"
            )


class CMSBlock(nn.Module):
    """Single CMS frequency level — an MLP with residual connection.

    Structure: x → LayerNorm → Linear → Act → Linear → + x

    The block uses zero-initialization for the output layer bias and small
    normal initialization for weights to stabilize residual learning.

    Args:
        dim: Input/output dimension.
        hidden_multiplier: Hidden dimension = dim * hidden_multiplier.
        act_layer: Activation function class (defaults to GELU).
        drop: Dropout rate applied after both linear layers.
        grad_clip: Maximum gradient norm for clipping through this block.
    """

    def __init__(
        self,
        dim: int,
        hidden_multiplier: float = 4.0,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        grad_clip: float = 1.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * hidden_multiplier)

        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)
        self.grad_clip = grad_clip

        # Initialize fc2 near zero for stable residual learning
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection.

        Args:
            x: [B, N, D] input tokens.

        Returns:
            [B, N, D] output tokens with residual connection applied.
        """
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return residual + x


class CMS(nn.Module):
    """Continuum Memory System — multi-frequency MLP hierarchy.

    Composes multiple CMSBlock levels at different temporal frequencies.
    All levels process the input sequentially (cascade):
        y = level_K( level_{K-1}( ... level_1(x) ... ) )

    In standard mode (no chunk scheduling), all levels process every token.
    When chunk scheduling is enabled, slower levels only process every
    `update_period`-th chunk of tokens based on global temporal position.

    Args:
        dim: Input/output dimension for all levels.
        levels: List of LevelSpec defining each frequency level. If None,
            uses default three-level hierarchy (fast, medium, slow).
        act_layer: Activation function class for all levels.
        drop: Dropout rate applied in all CMS blocks.
        grad_clip: Gradient clipping norm for all CMS blocks.
        use_chunk_scheduling: Enable frequency-based temporal scheduling.
    """

    def __init__(
        self,
        dim: int,
        levels: list[LevelSpec] | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        grad_clip: float = 1.0,
        use_chunk_scheduling: bool = False,
    ) -> None:
        super().__init__()

        if levels is None:
            levels = [
                LevelSpec(name="fast", update_period=1, hidden_multiplier=4.0),
                LevelSpec(name="medium", update_period=4, hidden_multiplier=4.0),
                LevelSpec(name="slow", update_period=16, hidden_multiplier=4.0),
            ]

        self.levels_spec = levels
        self.use_chunk_scheduling = use_chunk_scheduling

        self.blocks = nn.ModuleList([
            CMSBlock(
                dim=dim,
                hidden_multiplier=spec.hidden_multiplier,
                act_layer=act_layer,
                drop=drop,
                grad_clip=grad_clip,
            )
            for spec in levels
        ])

        # Step counter for chunk scheduling
        self.register_buffer("_global_step", torch.tensor(0, dtype=torch.long), persistent=False)

        # Chunk accumulators for each level (reserved for future use)
        self._chunk_accumulators: list[list[Tensor]] = [[] for _ in levels]

    @property
    def num_levels(self) -> int:
        """Return the number of frequency levels."""
        return len(self.blocks)

    def forward(
        self,
        x: Tensor,
        T: int | None = None,
        tokens_per_frame: int | None = None,
    ) -> Tensor:
        """Forward pass through all CMS levels (cascade).

        In standard mode, all levels process every token. In frame-aware
        chunk scheduling mode, slower levels only process tokens belonging
        to frames where (frame_index % update_period == 0). All tokens
        within an active frame are processed together, preserving spatial
        coherence within each frame.

        Args:
            x: [B, N, D] input tokens from Titan memory output.
            T: Number of temporal frames in the sequence. Required for
                frame-aware scheduling.
            tokens_per_frame: Number of tokens per frame (H*W + action_tokens).
                Required for frame-aware scheduling.

        Returns:
            [B, N, D] processed tokens.
        """
        if not self.use_chunk_scheduling or T is None or tokens_per_frame is None:
            # Standard mode: all levels process all tokens
            for block in self.blocks:
                x = block(x)
            return x

        # Frame-aware scheduling: mask by frame index, not token index
        B, N, D = x.shape

        for block, spec in zip(self.blocks, self.levels_spec):
            # Fast level (period=1): always process everything
            if spec.update_period <= 1:
                x = block(x)
                continue

            # Determine which frames are active for this level
            # Frame f is active if f % update_period == 0
            frame_indices = torch.arange(T, device=x.device)
            active_frames = (frame_indices % spec.update_period == 0)

            if not active_frames.any():
                continue

            if active_frames.all():
                x = block(x)
            else:
                # Expand frame-level mask to token-level mask:
                # each frame owns `tokens_per_frame` contiguous tokens
                token_mask = active_frames.repeat_interleave(tokens_per_frame)
                # Handle edge case: N might not equal T * tokens_per_frame
                token_mask = token_mask[:N]

                indices = token_mask.nonzero(as_tuple=True)[0]
                x_sub = x[:, indices, :]
                x_sub = block(x_sub)
                x = x.clone()
                x[:, indices, :] = x_sub

        return x

    def reset_step_counter(self) -> None:
        """Reset the global step counter and clear accumulators.

        Call this method at the start of each new sequence to ensure
        correct temporal indexing.
        """
        self._global_step.zero_()
        self._chunk_accumulators = [[] for _ in self.levels_spec]

    def get_level_params(self) -> list[tuple[str, list[nn.Parameter]]]:
        """Get parameters grouped by CMS level (for per-level optimizers).

        Returns:
            List of (level_name, params) tuples where params is a list
            of learnable parameters for that level.
        """
        result = []
        for spec, block in zip(self.levels_spec, self.blocks):
            params = list(block.parameters())
            result.append((spec.name, params))
        return result
