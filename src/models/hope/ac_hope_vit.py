"""AC-HOPE-ViT — Adaptive Continuum Vision Transformer (Novel Architecture).

Fuses the V-JEPA2 AC predictor's I/O pipeline with the HOPE architecture
(Self-Modifying Titans + CMS) from Behrouz 2025.

This predictor operates on PRE-ENCODED V-JEPA2 features:
    Input:  [B, T*N, D=1024]  encoded feature maps + [B, T, action_dim] actions
    Output: [B, T*N, D=1024]  predicted next-frame features

Architecture (3 stages):
    Stage 1 — Input & Embedding (AC_ViT Legacy):
        - predictor_embed: 1024 → 384
        - action_encoder / state_encoder: action_dim → 384
        - Token interleaving: 1792 → 1806 tokens (256 image + 2 cond per frame)

    Stage 2 — HOPE-ViT Backbone (NEW):
        24× HOPEBlock, each containing:
            Phase A: Self-Modifying Titan (replaces attention)
            Phase B: CMS multi-frequency MLPs (replaces FFN)

    Stage 3 — Output & Decoding (AC_ViT Legacy):
        - Remove action/state tokens: 1806 → 1792
        - predictor_norm: LayerNorm(384)
        - predictor_proj: 384 → 1024

The forward() method signature matches VisionTransformerPredictorAC exactly
so the Lightning module wrapper can be reused.
"""

from __future__ import annotations

import logging
import math
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor

from src.models.ac_predictor.utils.modules import build_action_block_causal_attention_mask
from src.models.ac_predictor.utils.tensors import trunc_normal_
from src.models.hope.cms import LevelSpec
from src.models.hope.hope_block import HOPEBlock, HOPEBlockConfig

log = logging.getLogger(__name__)


class ACHOPEViT(nn.Module):
    """Action-Conditioned HOPE Vision Transformer Predictor.

    Drop-in replacement for VisionTransformerPredictorAC that uses
    HOPE blocks (Titan + CMS) instead of standard transformer blocks
    (Attention + MLP).

    Args:
        img_size: Input image size (H, W).
        patch_size: Patch size used by the encoder.
        num_timesteps: Number of encoded temporal frames (T).
        embed_dim: Encoder embedding dimension (input/output, typically 1024).
        predictor_embed_dim: Internal predictor dimension (typically 384).
        depth: Number of HOPE blocks.
        num_heads: Number of attention heads (used for RoPE dim splitting).
        action_embed_dim: Dimension of action/state vectors.
        use_rope: Whether to use 3D RoPE in HOPE blocks.
        is_frame_causal: Whether to use frame-causal attention masking.
        use_activation_checkpointing: Trade compute for memory.
        titan_hidden_multiplier: Titan memory MLP hidden dim multiplier.
        titan_layers: Number of layers in Titan memory MLP.
        titan_activation: Activation function for Titan memory.
        titan_grad_clip_inner: Gradient clip for inner-loop (DGD) gradients.
        cms_level_specs: List of CMS level specifications.
        cms_use_chunk_scheduling: Whether CMS uses chunk-based scheduling.
        self_mod_dim: Hidden dimension for the SelfModifier MLP.
        surprise_threshold: Minimum surprise to trigger memory update.
        drop_rate: Dropout rate.
        drop_path_rate: Stochastic depth rate.
        log_hope_diagnostics: Whether to collect HOPE diagnostic metrics.
        init_std: Standard deviation for weight initialization.
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (256, 256),
        patch_size: int = 16,
        num_timesteps: int = 8,
        embed_dim: int = 1024,
        predictor_embed_dim: int = 384,
        depth: int = 24,
        num_heads: int = 16,
        action_embed_dim: int = 7,
        use_rope: bool = True,
        is_frame_causal: bool = True,
        use_activation_checkpointing: bool = False,
        use_extrinsics: bool = False,
        # HOPE-specific parameters
        titan_hidden_multiplier: int = 4,
        titan_layers: int = 2,
        titan_activation: str = "gelu",
        titan_grad_clip_inner: float = 1.0,
        cms_level_specs: list[dict] | None = None,
        cms_use_chunk_scheduling: bool = False,
        self_mod_dim: int = 64,
        surprise_threshold: float = 0.0,
        # Regularization
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        # Diagnostics (Criticism §1)
        log_hope_diagnostics: bool = True,
        # Initialization
        init_std: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__()
        self.is_frame_causal = is_frame_causal
        self.use_extrinsics = use_extrinsics
        self.log_hope_diagnostics = log_hope_diagnostics
        self.use_activation_checkpointing = use_activation_checkpointing

        # ─── Stage 1: Input & Embedding (same as AC_ViT) ───
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        self.state_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        self.extrinsics_encoder = nn.Linear(action_embed_dim - 1, predictor_embed_dim, bias=True)

        # Grid parameters
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_timesteps = num_timesteps
        self.grid_height = img_size[0] // patch_size
        self.grid_width = img_size[1] // patch_size

        # ─── Stage 2: HOPE-ViT Backbone ───

        # Parse CMS level specs from config dicts or use defaults
        if cms_level_specs is not None:
            cms_levels = [
                LevelSpec(
                    name=spec.get("name", f"level_{i}"),
                    update_period=spec.get("update_period", 1),
                    warmup_steps=spec.get("warmup_steps", 0),
                    jitter=spec.get("jitter", 0.0),
                    hidden_multiplier=spec.get("hidden_multiplier", 4.0),
                )
                for i, spec in enumerate(cms_level_specs)
            ]
        else:
            cms_levels = [
                LevelSpec(name="fast", update_period=1, hidden_multiplier=4.0),
                LevelSpec(name="medium", update_period=4, hidden_multiplier=4.0),
                LevelSpec(name="slow", update_period=16, hidden_multiplier=4.0),
            ]

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.hope_blocks = nn.ModuleList([
            HOPEBlock(
                HOPEBlockConfig(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    titan_hidden_multiplier=titan_hidden_multiplier,
                    titan_layers=titan_layers,
                    titan_activation=titan_activation,
                    titan_grad_clip_inner=titan_grad_clip_inner,
                    cms_levels=cms_levels,
                    cms_use_chunk_scheduling=cms_use_chunk_scheduling,
                    self_mod_dim=self_mod_dim,
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    surprise_threshold=surprise_threshold,
                    drop_path=dpr[i],
                    drop=drop_rate,
                )
            )
            for i in range(depth)
        ])

        # ─── Stage 3: Output & Decoding (same as AC_ViT) ───
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # ─── Causal attention mask ───
        attn_mask = None
        if self.is_frame_causal:
            grid_depth = self.num_timesteps
            grid_height = self.grid_height
            grid_width = self.grid_width
            attn_mask = build_action_block_causal_attention_mask(
                grid_depth, grid_height, grid_width,
                add_tokens=3 if use_extrinsics else 2,
            )
        self.attn_mask = attn_mask

        # ─── Initialize weights ───
        self.init_std = init_std
        self.apply(self._init_weights)

        # Store config for diagnostics logging (Criticism §1)
        self._config_summary = {
            "depth": depth,
            "predictor_embed_dim": predictor_embed_dim,
            "num_heads": num_heads,
            "titan_hidden_multiplier": titan_hidden_multiplier,
            "titan_layers": titan_layers,
            "titan_grad_clip_inner": titan_grad_clip_inner,
            "cms_levels": len(cms_levels),
            "use_rope": use_rope,
            "self_mod_dim": self_mod_dim,
            "surprise_threshold": surprise_threshold,
            "total_params": sum(p.numel() for p in self.parameters()),
        }

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_config_summary(self) -> dict:
        """Get configuration summary for logging (Criticism §1).

        Returns all key parameters that could affect training stability:
        - Architecture dimensions
        - Titan memory settings (inner-loop grad clip, hidden mult)
        - CMS level count
        - RoPE on/off
        - Total parameter count
        """
        return self._config_summary

    def forward(
        self,
        x: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through AC-HOPE-ViT predictor.

        Signature-compatible with VisionTransformerPredictorAC.forward().

        Args:
            x: Context tokens [B, T*N, D] where T=num_timesteps, N=H*W patches.
            actions: Action sequences [B, T, action_dim].
            states: State sequences [B, T, action_dim].
            extrinsics: Optional extrinsic parameters [B, T, action_dim-1].

        Returns:
            Predicted features [B, T*N, D].
        """
        log.debug(f"    [AC_HOPE_ViT] Input x: {x.shape}")
        log.debug(f"    [AC_HOPE_ViT] Actions: {actions.shape}, States: {states.shape}")

        # ─── Stage 1: Embed & Interleave ───
        x = self.predictor_embed(x)
        B, N_ctxt, D = x.size()
        T = N_ctxt // (self.grid_height * self.grid_width)

        # Validate input
        expected_patches = self.grid_height * self.grid_width
        if N_ctxt % expected_patches != 0:
            raise ValueError(
                f"Input token count ({N_ctxt}) not divisible by patches_per_frame "
                f"({expected_patches}). Check input features vs model config."
            )
        if T > self.num_timesteps:
            raise ValueError(
                f"Input has {T} timesteps but model configured for max {self.num_timesteps}."
            )

        # Interleave action/state tokens
        s = self.state_encoder(states).unsqueeze(2)   # [B, T, 1, D]
        a = self.action_encoder(actions).unsqueeze(2)  # [B, T, 1, D]

        x = x.view(B, T, self.grid_height * self.grid_width, D)  # [B, T, H*W, D]
        if self.use_extrinsics and extrinsics is not None:
            e = self.extrinsics_encoder(extrinsics).unsqueeze(2)
            x = torch.cat([a, s, e, x], dim=2).flatten(1, 2)  # [B, T*(H*W+3), D]
        else:
            x = torch.cat([a, s, x], dim=2).flatten(1, 2)  # [B, T*(H*W+2), D]

        cond_tokens = 3 if self.use_extrinsics else 2

        log.debug(f"    [AC_HOPE_ViT] After interleaving: {x.shape}")

        # Prepare causal mask
        attn_mask = self.attn_mask[:x.size(1), :x.size(1)].to(x.device, non_blocking=True).clone()

        # ─── Stage 2: HOPE Backbone ───
        log.debug(f"    [AC_HOPE_ViT] >>> Processing {len(self.hope_blocks)} HOPE blocks >>>")
        for i, blk in enumerate(self.hope_blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    None,      # mask
                    attn_mask,
                    T,
                    self.grid_height,
                    self.grid_width,
                    cond_tokens,
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
            log.debug(
                f"    [AC_HOPE_ViT] Block {i} output: shape={x.shape}, "
                f"min={x.min().item():.4f}, max={x.max().item():.4f}"
            )

        log.debug(f"    [AC_HOPE_ViT] <<< Finished HOPE blocks <<<")

        # ─── Stage 3: Decode ───
        # Remove action/state tokens
        x = x.view(B, T, cond_tokens + self.grid_height * self.grid_width, D)
        x = x[:, :, cond_tokens:, :].flatten(1, 2)  # [B, T*H*W, D]

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)

        log.debug(f"    [AC_HOPE_ViT] Output: {x.shape}")

        return x

    def reset_all_memories(self) -> None:
        """Reset all HOPE block memory states (call between sequences)."""
        for blk in self.hope_blocks:
            blk.reset_memory_state()

    def get_all_diagnostics(self) -> dict[str, float]:
        """Aggregate diagnostics from all HOPE blocks (Criticism §1).

        Returns averaged diagnostics across all blocks.
        """
        if not self.log_hope_diagnostics:
            return {}

        aggregated: dict[str, float] = {}
        n_blocks = len(self.hope_blocks)

        for i, blk in enumerate(self.hope_blocks):
            block_diag = blk.get_diagnostics()
            for key, val in block_diag.items():
                if key not in aggregated:
                    aggregated[key] = 0.0
                aggregated[key] += val / n_blocks  # Average across blocks

        return aggregated

    def get_parameter_groups(self) -> list[dict]:
        """Get parameter groups for optimizer with different learning rates.

        Returns 3 groups:
            1. Titan memory parameters (lower LR recommended)
            2. CMS parameters (normal LR)
            3. Projection/embedding parameters (normal LR)

        This allows the optimizer to use per-group learning rates,
        which is important because Titan memories use inner-loop
        optimization and may need gentler outer-loop updates.
        """
        titan_params = []
        cms_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(m in name for m in ["M_k.", "M_v.", "M_eta.", "M_alpha.", "M_memory."]):
                titan_params.append(param)
            elif "cms." in name:
                cms_params.append(param)
            else:
                other_params.append(param)

        return [
            {"params": titan_params, "group_name": "titan"},
            {"params": cms_params, "group_name": "cms"},
            {"params": other_params, "group_name": "projections"},
        ]


def ac_hope_vit(**kwargs) -> ACHOPEViT:
    """Factory function for ACHOPEViT (mirrors vit_ac_predictor)."""
    model = ACHOPEViT(**kwargs)
    return model
