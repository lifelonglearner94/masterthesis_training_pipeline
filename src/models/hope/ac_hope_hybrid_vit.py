"""AC-HOPE-Hybrid-ViT — Attention + Titan Memory + CMS (Phase 8).

Hybrid architecture that combines the ViT baseline's self-attention with
HOPE's in-context Titan memory adaptation. This is NOT a replacement of
attention (like the original HOPE), but an AUGMENTATION: attention handles
token interaction, Titan memory handles in-context CL adaptation.

Architecture (3 stages, matching AC-HOPE-ViT's I/O contract):
    Stage 1 — Input & Embedding (from ViT baseline):
        - predictor_embed: 1024 → 384
        - action_encoder / state_encoder: action_dim → 384
        - Token interleaving + learnable temporal embeddings

    Stage 2 — Hybrid Backbone (NEW):
        12× HybridBlock, each containing:
            Phase A: ACRoPEAttention (multi-head self-attention + 3D RoPE)
            Phase B: Titan Memory (single M_memory + simplified η/α + DGD)
            Phase C: CMS (multi-frequency MLPs)

    Stage 3 — Output & Decoding (from ViT baseline):
        - Remove action/state tokens
        - predictor_norm: LayerNorm(384)
        - predictor_proj: 384 → 1024

Key differences from original HOPE (ac_hope_vit.py):
    1. Self-attention RETAINED (not replaced by Titan)
    2. Only 1 Titan memory per block (not 5)
    3. η/α are learned parameters (not full Titan memories)
    4. RoPE works in attention; temporal embeddings are additive
    5. depth=12 (not 5) — deeper hierarchy enabled by lower per-block cost

The forward() signature is identical to VisionTransformerPredictorAC and
ACHOPEViT, so the Lightning module wrapper can be reused.
"""

from __future__ import annotations

import logging
from typing import Final

import torch
import torch.nn as nn
from torch import Tensor

from src.models.ac_predictor.utils.modules import build_action_block_causal_attention_mask
from src.models.ac_predictor.utils.tensors import trunc_normal_
from src.models.hope.cms import LevelSpec
from src.models.hope.hybrid_block import HybridBlock, HybridBlockConfig
from src.models.hope.titan_memory import _backward_grad_clip

log = logging.getLogger(__name__)

# Type aliases
type DiagnosticMetrics = dict[str, float]
type ParameterGroups = list[dict]
type BlockConfig = dict

# Constants
COND_TOKENS_WITHOUT_EXTRINSICS: Final = 2
COND_TOKENS_WITH_EXTRINSICS: Final = 3
DEFAULT_INIT_STD: Final = 0.02
DEFAULT_DROP_RATE: Final = 0.0
DEFAULT_DROP_PATH_RATE: Final = 0.0


class ACHOPEHybridViT(nn.Module):
    """Action-Conditioned HOPE Hybrid Vision Transformer Predictor.

    Combines self-attention (ViT) with Titan memory (HOPE) for CL.

    Args:
        img_size: Input image size (H, W).
        patch_size: Patch size used by the encoder.
        num_timesteps: Number of encoded temporal frames (T).
        embed_dim: Encoder embedding dimension (input/output, typically 1024).
        predictor_embed_dim: Internal predictor dimension (typically 384).
        depth: Number of Hybrid blocks (default 12).
        num_heads: Number of attention heads.
        action_embed_dim: Dimension of action/state vectors.
        is_frame_causal: Whether to use frame-causal attention masking.
        use_activation_checkpointing: Trade compute for memory.
        use_extrinsics: Whether to use extrinsic parameters.
        titan_hidden_multiplier: Titan memory MLP hidden dim multiplier.
        titan_layers: Number of layers in Titan memory MLP.
        titan_activation: Activation function for Titan memory.
        titan_grad_clip_inner: Gradient clip for inner-loop (DGD) gradients.
        titan_grad_clip_backward: Max gradient norm for DGD backward path.
        cms_level_specs: List of CMS level specifications.
        cms_use_chunk_scheduling: Whether CMS uses chunk-based scheduling.
        titan_detach_interval: Detach Titan computation graph every N steps.
        surprise_threshold: Minimum surprise to trigger memory update.
        use_longterm_memory: Whether to use persistent cross-clip memory.
        longterm_hidden_multiplier: Hidden dim multiplier for M_longterm.
        longterm_lr_scale: DGD learning rate scale for M_longterm.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        log_hope_diagnostics: Whether to collect diagnostic metrics.
        init_std: Standard deviation for weight initialization.
    """

    def __init__(
        self,
        img_size: tuple[int, int] | int = (256, 256),
        patch_size: int = 16,
        num_timesteps: int = 8,
        embed_dim: int = 1024,
        predictor_embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 16,
        action_embed_dim: int = 7,
        is_frame_causal: bool = True,
        use_activation_checkpointing: bool = False,
        use_extrinsics: bool = False,
        # Titan memory settings (simplified)
        titan_hidden_multiplier: int = 2,
        titan_layers: int = 2,
        titan_activation: str = "gelu",
        titan_grad_clip_inner: float = 1.0,
        titan_grad_clip_backward: float = 1.0,
        cms_level_specs: list[BlockConfig] | None = None,
        cms_use_chunk_scheduling: bool = False,
        titan_detach_interval: int = 0,
        surprise_threshold: float = 0.0,
        # Longterm memory (optional, for CL)
        use_longterm_memory: bool = False,
        longterm_hidden_multiplier: int = 2,
        longterm_lr_scale: float = 0.1,
        # Attention settings
        qkv_bias: bool = True,
        attn_drop_rate: float = 0.0,
        # Regularization
        drop_rate: float = DEFAULT_DROP_RATE,
        drop_path_rate: float = DEFAULT_DROP_PATH_RATE,
        # Diagnostics
        log_hope_diagnostics: bool = True,
        # Initialization
        init_std: float = DEFAULT_INIT_STD,
        **kwargs,
    ) -> None:
        super().__init__()
        self.is_frame_causal = is_frame_causal
        self.use_extrinsics = use_extrinsics
        self.log_hope_diagnostics = log_hope_diagnostics
        self.use_activation_checkpointing = use_activation_checkpointing
        self.init_std = init_std
        self.titan_grad_clip_backward = titan_grad_clip_backward

        # ─── Stage 1: Input & Embedding (same as ViT baseline) ───
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        self.state_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        self.extrinsics_encoder = nn.Linear(
            action_embed_dim - 1, predictor_embed_dim, bias=True
        )

        # Grid parameters
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_timesteps = num_timesteps
        self.grid_height = img_size[0] // patch_size
        self.grid_width = img_size[1] // patch_size
        self.patches_per_frame = self.grid_height * self.grid_width

        # ─── Learnable temporal embeddings (additive, coexist with RoPE) ───
        # RoPE handles relative positions in attention (Phase A)
        # These embeddings give global frame identity to ALL phases (A, B, C)
        self.frame_pos_embed = nn.Embedding(num_timesteps + 2, predictor_embed_dim)
        self.target_pos_embed = nn.Embedding(num_timesteps + 2, predictor_embed_dim)

        # ─── Stage 2: Hybrid Backbone ───
        cond_tok_count = (
            COND_TOKENS_WITH_EXTRINSICS
            if use_extrinsics
            else COND_TOKENS_WITHOUT_EXTRINSICS
        )

        cms_levels = self._build_cms_levels(cms_level_specs)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.hybrid_blocks = nn.ModuleList([
            HybridBlock(
                HybridBlockConfig(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop_rate,
                    proj_drop=drop_rate,
                    grid_size=self.grid_height,
                    titan_hidden_multiplier=titan_hidden_multiplier,
                    titan_layers=titan_layers,
                    titan_activation=titan_activation,
                    titan_grad_clip_inner=titan_grad_clip_inner,
                    titan_grad_clip_backward=titan_grad_clip_backward,
                    titan_detach_interval=titan_detach_interval,
                    surprise_threshold=surprise_threshold,
                    cms_levels=cms_levels,
                    cms_use_chunk_scheduling=cms_use_chunk_scheduling,
                    use_longterm_memory=use_longterm_memory,
                    longterm_hidden_multiplier=longterm_hidden_multiplier,
                    longterm_lr_scale=longterm_lr_scale,
                    drop_path=dpr[i],
                    drop=drop_rate,
                ),
                titan_detach_interval=titan_detach_interval,
            )
            for i in range(depth)
        ])

        # ─── Stage 3: Output & Decoding (same as ViT baseline) ───
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # ─── Causal attention mask ───
        self.attn_mask = self._build_attention_mask()

        # ─── Initialize weights ───
        self.apply(self._init_weights)
        self._rescale_attention_blocks()

        # ─── Config summary for diagnostics ───
        self._use_longterm_memory = use_longterm_memory
        self._config_summary = {
            "architecture": "hybrid",
            "depth": depth,
            "predictor_embed_dim": predictor_embed_dim,
            "num_heads": num_heads,
            "titan_hidden_multiplier": titan_hidden_multiplier,
            "titan_layers": titan_layers,
            "titan_grad_clip_inner": titan_grad_clip_inner,
            "titan_grad_clip_backward": titan_grad_clip_backward,
            "cms_levels": len(cms_levels),
            "has_attention": True,
            "has_rope": True,
            "has_temporal_embeddings": True,
            "use_longterm_memory": use_longterm_memory,
            "longterm_hidden_multiplier": longterm_hidden_multiplier if use_longterm_memory else None,
            "longterm_lr_scale": longterm_lr_scale if use_longterm_memory else None,
            "total_params": sum(p.numel() for p in self.parameters()),
        }

    def _build_cms_levels(
        self, cms_level_specs: list[BlockConfig] | None
    ) -> list[LevelSpec]:
        """Build CMS level specifications from config or use defaults."""
        if cms_level_specs is not None:
            return [
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
            return [
                LevelSpec(name="fast", update_period=1, hidden_multiplier=2.0),
                LevelSpec(name="medium", update_period=3, hidden_multiplier=2.5),
                LevelSpec(name="slow", update_period=7, hidden_multiplier=3.0),
            ]

    def _build_attention_mask(self) -> Tensor | None:
        """Build causal attention mask if frame-causal mode is enabled."""
        if not self.is_frame_causal:
            return None

        return build_action_block_causal_attention_mask(
            self.num_timesteps,
            self.grid_height,
            self.grid_width,
            add_tokens=(
                COND_TOKENS_WITH_EXTRINSICS
                if self.use_extrinsics
                else COND_TOKENS_WITHOUT_EXTRINSICS
            ),
        )

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize module weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_attention_blocks(self) -> None:
        """Rescale attention output projection weights for stable deep training.

        Same rescaling as the original ViT: divide by sqrt(2 * layer_id).
        This prevents signal explosion in deep residual networks.
        """
        import math
        for layer_id, blk in enumerate(self.hybrid_blocks):
            blk.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def get_config_summary(self) -> dict[str, int | float | bool | str | None]:
        """Get configuration summary for logging."""
        return self._config_summary

    # ─── Forward pass ───

    def forward(
        self,
        x: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
        target_timestep: int | None = None,
    ) -> Tensor:
        """Forward pass through AC-HOPE-Hybrid-ViT predictor.

        Signature-compatible with VisionTransformerPredictorAC.forward().

        Args:
            x: Context tokens [B, T*N, D] where T=num_timesteps, N=H*W patches.
            actions: Action sequences [B, T, action_dim].
            states: State sequences [B, T, action_dim].
            extrinsics: Optional extrinsic parameters [B, T, action_dim-1].
            target_timestep: Target frame index for jump prediction.

        Returns:
            Predicted features [B, T*N, D].
        """
        # ─── Stage 1: Embed & Interleave ───
        x = self.predictor_embed(x)
        B, N_ctxt, D = x.size()
        T = N_ctxt // self.patches_per_frame

        if N_ctxt % self.patches_per_frame != 0:
            raise ValueError(
                f"Input token count ({N_ctxt}) not divisible by patches_per_frame "
                f"({self.patches_per_frame})."
            )
        if T > self.num_timesteps:
            raise ValueError(
                f"Input has {T} timesteps but model configured for max {self.num_timesteps}."
            )

        x = self._interleave_conditioning_tokens(
            x, actions, states, extrinsics, B, T, D
        )

        cond_tokens = (
            COND_TOKENS_WITH_EXTRINSICS
            if self.use_extrinsics
            else COND_TOKENS_WITHOUT_EXTRINSICS
        )

        # Prepare causal mask
        attn_mask = self._prepare_attention_mask(x)

        # ─── Stage 2: Hybrid Backbone ───
        x = self._process_hybrid_blocks(x, attn_mask, T, cond_tokens, target_timestep)

        # ─── Stage 3: Decode ───
        x = self._decode_output(x, B, T, D, cond_tokens)

        return x

    def _interleave_conditioning_tokens(
        self,
        x: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None,
        B: int, T: int, D: int,
    ) -> Tensor:
        """Interleave action/state tokens with image tokens."""
        s = self.state_encoder(states).unsqueeze(2)   # [B, T, 1, D]
        a = self.action_encoder(actions).unsqueeze(2)  # [B, T, 1, D]

        x = x.view(B, T, self.patches_per_frame, D)

        if self.use_extrinsics and extrinsics is not None:
            e = self.extrinsics_encoder(extrinsics).unsqueeze(2)
            return torch.cat([a, s, e, x], dim=2).flatten(1, 2)
        else:
            return torch.cat([a, s, x], dim=2).flatten(1, 2)

    def _prepare_attention_mask(self, x: Tensor) -> Tensor:
        """Prepare attention mask for current sequence length."""
        seq_len = x.size(1)
        return self.attn_mask[:seq_len, :seq_len].to(
            device=x.device, non_blocking=True
        ).clone()

    def _process_hybrid_blocks(
        self, x: Tensor, attn_mask: Tensor, T: int, cond_tokens: int,
        target_timestep: int | None = None,
    ) -> Tensor:
        """Process tokens through Hybrid backbone blocks."""
        tokens_per_frame = cond_tokens + self.patches_per_frame

        # ─── Inject learnable temporal position information ───
        if target_timestep is not None:
            # Jump prediction mode
            frame_0_emb = self.frame_pos_embed(
                torch.zeros(1, dtype=torch.long, device=x.device)
            )
            x = x + frame_0_emb.unsqueeze(0)
            target_emb = self.target_pos_embed(
                torch.tensor(target_timestep, dtype=torch.long, device=x.device)
            )
            x = x + target_emb.unsqueeze(0).unsqueeze(0)
        else:
            # Teacher-forcing mode
            frame_indices = torch.arange(T, dtype=torch.long, device=x.device)
            frame_embs = self.frame_pos_embed(frame_indices)
            frame_embs = frame_embs.unsqueeze(0).unsqueeze(2)
            frame_embs = frame_embs.expand(-1, -1, tokens_per_frame, -1)
            frame_embs = frame_embs.reshape(1, T * tokens_per_frame, -1)
            x = x + frame_embs

        for i, blk in enumerate(self.hybrid_blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, None, attn_mask, T,
                    self.grid_height, self.grid_width,
                    cond_tokens, target_timestep,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x, mask=None, attn_mask=attn_mask,
                    T=T, H=self.grid_height, W=self.grid_width,
                    action_tokens=cond_tokens,
                    target_timestep=target_timestep,
                )

            # Block-level backward gradient clipping
            if self.titan_grad_clip_backward > 0 and self.training:
                x = _backward_grad_clip(x, self.titan_grad_clip_backward)

        return x

    def _decode_output(
        self, x: Tensor, B: int, T: int, D: int, cond_tokens: int
    ) -> Tensor:
        """Remove conditioning tokens and project to output dimension."""
        x = x.view(B, T, cond_tokens + self.patches_per_frame, D)
        x = x[:, :, cond_tokens:, :].flatten(1, 2)

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)
        return x

    # ─── Memory management (CL-compatible API) ───

    def reset_all_memories(self) -> None:
        """Reset all clip-level memory states (call between clips).

        M_longterm is NOT reset — it persists across clips by design.
        """
        for blk in self.hybrid_blocks:
            blk.reset_memory_state()

    def reset_all_longterm_memories(self) -> None:
        """Explicitly reset all longterm memories."""
        for blk in self.hybrid_blocks:
            blk.reset_longterm_memory()

    def freeze_all_inner_loops(self) -> None:
        """Freeze DGD memory updates (for CL evaluation)."""
        for blk in self.hybrid_blocks:
            blk.freeze_inner_loop = True
        log.info("Hybrid inner-loop DGD frozen for all blocks (eval mode)")

    def unfreeze_all_inner_loops(self) -> None:
        """Unfreeze DGD memory updates."""
        for blk in self.hybrid_blocks:
            blk.freeze_inner_loop = False
        log.info("Hybrid inner-loop DGD unfrozen for all blocks (train mode)")

    # ─── CL Phase 9: Attention freeze & Titan reset at task boundaries ───

    def freeze_attention(self) -> None:
        """Freeze all attention parameters (QKV, proj, norm1) across blocks.

        Used during CL task training to preserve the universal spatial-temporal
        token mixing learned during base training. The attention layers encode
        general-purpose RoPE-based positional reasoning that should not drift
        when adapting to new distribution shifts.
        """
        frozen_count = 0
        for blk in self.hybrid_blocks:
            for name, param in blk.named_parameters():
                if any(p in name for p in ("attn.", "norm1.")):
                    param.requires_grad = False
                    frozen_count += 1
        log.info(
            f"Attention frozen: {frozen_count} parameter tensors set to "
            f"requires_grad=False across {len(self.hybrid_blocks)} blocks"
        )

    def unfreeze_attention(self) -> None:
        """Unfreeze all attention parameters (restore trainability)."""
        unfrozen_count = 0
        for blk in self.hybrid_blocks:
            for name, param in blk.named_parameters():
                if any(p in name for p in ("attn.", "norm1.")):
                    param.requires_grad = True
                    unfrozen_count += 1
        log.info(
            f"Attention unfrozen: {unfrozen_count} parameter tensors restored "
            f"across {len(self.hybrid_blocks)} blocks"
        )

    def reset_titan_for_new_task(self) -> None:
        """Reset ALL Titan memories (clip-level AND longterm) for a new CL task.

        Called at task boundaries to give Titan a clean slate for each new
        distribution shift. This prevents stale memory states from interfering
        with outer-loop gradient descent during task adaptation.

        Unlike reset_all_memories() (which preserves longterm), this resets
        everything — the model starts each task with fresh meta-learned
        initial memory weights.
        """
        for blk in self.hybrid_blocks:
            blk.M_memory.reset_active_weights()
            blk.M_memory.reset_diagnostics()
            blk.cms.reset_step_counter()
            if blk.use_longterm_memory:
                blk.M_longterm.reset_active_weights()
                blk.M_longterm.reset_diagnostics()
        log.info(
            "Titan memories fully reset (clip-level + longterm) for new CL task"
        )

    def get_aux_loss(self) -> Tensor:
        """Return zero aux loss (hybrid doesn't need M_k/M_v aux loss).

        In the original HOPE, M_k and M_v had no direct gradient path from
        the outer loss, requiring an auxiliary loss. In the hybrid, the
        memory key/value come from simple Linear projections that DO get
        direct gradients through the main loss. No aux loss needed.
        """
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def get_all_diagnostics(self) -> DiagnosticMetrics:
        """Aggregate diagnostics from all blocks."""
        if not self.log_hope_diagnostics:
            return {}

        aggregated: DiagnosticMetrics = {}
        n_blocks = len(self.hybrid_blocks)

        for blk in self.hybrid_blocks:
            for key, val in blk.get_diagnostics().items():
                aggregated[key] = aggregated.get(key, 0.0) + val / n_blocks

        return aggregated

    def get_parameter_groups(self) -> ParameterGroups:
        """Get parameter groups for optimizer with different learning rates.

        Returns 4 groups:
            1. **attention** — self-attention params (QKV, proj, norm1) per block.
               Matches ViT baseline LR for comparable token-mixing capacity.
            2. **titan** — Titan memory params (M_memory, M_longterm, η/α, projections,
               gate). Lower LR since these are inner-loop augmentation.
            3. **cms** — CMS MLP parameters. Moderate LR.
            4. **projections** — embedding, decoding, action/state encoders,
               remaining norms. Matches attention LR.
        """
        attention_params: list[Tensor] = []
        titan_params: list[Tensor] = []
        cms_params: list[Tensor] = []
        other_params: list[Tensor] = []

        titan_patterns = {
            "M_memory.", "M_longterm.", "mem_q_proj.", "mem_k_proj.",
            "mem_v_proj.", "mem_out_proj.", "eta_base", "alpha_base",
            "longterm_gate.",
        }

        attention_patterns = {
            "attn.", "norm1.",
        }

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(pattern in name for pattern in titan_patterns):
                titan_params.append(param)
            elif "cms." in name:
                cms_params.append(param)
            elif any(pattern in name for pattern in attention_patterns):
                attention_params.append(param)
            else:
                other_params.append(param)

        return [
            {"params": attention_params, "group_name": "attention"},
            {"params": titan_params, "group_name": "titan"},
            {"params": cms_params, "group_name": "cms"},
            {"params": other_params, "group_name": "projections"},
        ]


def ac_hope_hybrid_vit(**kwargs) -> ACHOPEHybridViT:
    """Factory function for ACHOPEHybridViT."""
    return ACHOPEHybridViT(**kwargs)
