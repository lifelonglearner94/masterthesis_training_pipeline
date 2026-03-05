"""AC-DNH-HOPE-Hybrid-ViT — Attention + Dynamic Nested Hierarchy + Dynamic CMS (Phase 11).

Extends the Phase 8 ACHOPEHybridViT with DNH-style dynamic memory hierarchy.
The architecture is identical EXCEPT:
    - Phase B: DynamicNestedHierarchy (2–5 SMM levels) replaces single TitanMemory
    - Phase C: DynamicCMS (3–5 CMSBlock levels) replaces static CMS

Architecture (3 stages, same I/O contract as Phase 8 / ViT baseline):
    Stage 1 — Input & Embedding (from ViT baseline):
        - predictor_embed: 1024 → 384
        - action_encoder / state_encoder: action_dim → 384
        - Token interleaving + learnable temporal embeddings

    Stage 2 — DNH Hybrid Backbone (NEW):
        12× DNHHybridBlock, each containing:
            Phase A: ACRoPEAttention (unchanged from Phase 8)
            Phase B: DynamicNestedHierarchy (2–5 self-modifying memories)
            Phase C: DynamicCMS (3–5 frequency levels)

    Stage 3 — Output & Decoding (from ViT baseline):
        - Remove action/state tokens
        - predictor_norm: LayerNorm(384)
        - predictor_proj: 384 → 1024

Structural evolution is managed externally by StructuralEvolutionController,
invoked from the Lightning module's training_step.

The forward() signature is identical to ACHOPEHybridViT.forward() so the
Lightning module wrapper can be swapped in without pipeline changes.
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
from src.models.hope.dnh_hybrid_block import DNHHybridBlock, DNHHybridBlockConfig
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


class ACDNHHOPEHybridViT(nn.Module):
    """Action-Conditioned DNH-HOPE Hybrid Vision Transformer Predictor (Phase 11).

    Combines self-attention (ViT) with dynamic nested hierarchy of self-modifying
    memories (DNH-HOPE) for continual learning.

    Args:
        img_size: Input image size (H, W).
        patch_size: Patch size used by the encoder.
        num_timesteps: Number of encoded temporal frames (T).
        embed_dim: Encoder embedding dimension (input/output, typically 1024).
        predictor_embed_dim: Internal predictor dimension (typically 384).
        depth: Number of DNH Hybrid blocks (default 12).
        num_heads: Number of attention heads.
        action_embed_dim: Dimension of action/state vectors.
        is_frame_causal: Whether to use frame-causal attention masking.
        use_activation_checkpointing: Trade compute for memory.
        use_extrinsics: Whether to use extrinsic parameters.
        dnh_L_init: Initial number of SMM levels per block (default 2).
        dnh_L_max: Maximum SMM levels per block (default 5).
        dnh_L_min: Minimum SMM levels per block (default 2).
        titan_hidden_multiplier: Titan memory MLP hidden dim multiplier.
        titan_layers: Number of layers in Titan memory MLP.
        titan_activation: Activation function for Titan memory.
        titan_grad_clip_inner: Gradient clip for inner-loop DGD.
        titan_grad_clip_backward: Max gradient norm for DGD backward path.
        meta_hidden_dim: Hidden dim for SMM meta-networks.
        cms_level_specs: List of CMS level specifications.
        cms_use_chunk_scheduling: Whether CMS uses chunk-based scheduling.
        cms_L_max: Maximum CMS frequency levels.
        cms_L_min: Minimum CMS frequency levels.
        titan_detach_interval: Detach computation graph every N steps.
        surprise_threshold: Minimum surprise to trigger memory update.
        use_longterm_memory: Whether to use persistent cross-clip memory.
        longterm_hidden_multiplier: Hidden dim multiplier for longterm memory.
        longterm_lr_scale: DGD learning rate scale for longterm memory.
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
        # DNH settings
        dnh_L_init: int = 2,
        dnh_L_max: int = 5,
        dnh_L_min: int = 2,
        # Titan memory settings
        titan_hidden_multiplier: int = 2,
        titan_layers: int = 2,
        titan_activation: str = "gelu",
        titan_grad_clip_inner: float = 1.0,
        titan_grad_clip_backward: float = 1.0,
        meta_hidden_dim: int | None = None,
        # CMS settings
        cms_level_specs: list[BlockConfig] | None = None,
        cms_use_chunk_scheduling: bool = False,
        cms_L_max: int = 5,
        cms_L_min: int = 2,
        # Scheduling
        titan_detach_interval: int = 0,
        surprise_threshold: float = 0.0,
        # Longterm memory
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
        self.frame_pos_embed = nn.Embedding(num_timesteps + 2, predictor_embed_dim)
        self.target_pos_embed = nn.Embedding(num_timesteps + 2, predictor_embed_dim)

        # ─── Stage 2: DNH Hybrid Backbone ───
        cond_tok_count = (
            COND_TOKENS_WITH_EXTRINSICS
            if use_extrinsics
            else COND_TOKENS_WITHOUT_EXTRINSICS
        )

        cms_levels = self._build_cms_levels(cms_level_specs)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.dnh_blocks = nn.ModuleList([
            DNHHybridBlock(
                DNHHybridBlockConfig(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop_rate,
                    proj_drop=drop_rate,
                    grid_size=self.grid_height,
                    dnh_L_init=dnh_L_init,
                    dnh_L_max=dnh_L_max,
                    dnh_L_min=dnh_L_min,
                    titan_hidden_multiplier=titan_hidden_multiplier,
                    titan_layers=titan_layers,
                    titan_activation=titan_activation,
                    titan_grad_clip_inner=titan_grad_clip_inner,
                    titan_grad_clip_backward=titan_grad_clip_backward,
                    titan_detach_interval=titan_detach_interval,
                    surprise_threshold=surprise_threshold,
                    meta_hidden_dim=meta_hidden_dim,
                    cms_levels=cms_levels,
                    cms_use_chunk_scheduling=cms_use_chunk_scheduling,
                    cms_L_max=cms_L_max,
                    cms_L_min=cms_L_min,
                    use_longterm_memory=use_longterm_memory,
                    longterm_hidden_multiplier=longterm_hidden_multiplier,
                    longterm_lr_scale=longterm_lr_scale,
                    drop_path=dpr[i],
                    drop=drop_rate,
                ),
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
            "architecture": "dnh_hybrid",
            "depth": depth,
            "predictor_embed_dim": predictor_embed_dim,
            "num_heads": num_heads,
            "dnh_L_init": dnh_L_init,
            "dnh_L_max": dnh_L_max,
            "titan_hidden_multiplier": titan_hidden_multiplier,
            "titan_layers": titan_layers,
            "titan_grad_clip_inner": titan_grad_clip_inner,
            "titan_grad_clip_backward": titan_grad_clip_backward,
            "cms_levels": len(cms_levels),
            "cms_L_max": cms_L_max,
            "has_attention": True,
            "has_rope": True,
            "has_temporal_embeddings": True,
            "has_dnh": True,
            "use_longterm_memory": use_longterm_memory,
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
        """Rescale attention output projection weights for stable deep training."""
        import math
        for layer_id, blk in enumerate(self.dnh_blocks):
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
        """Forward pass through AC-DNH-HOPE-Hybrid-ViT predictor.

        Signature-compatible with ACHOPEHybridViT.forward().

        Args:
            x: Context tokens [B, T*N, D].
            actions: Action sequences [B, T, action_dim].
            states: State sequences [B, T, action_dim].
            extrinsics: Optional extrinsic parameters.
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

        attn_mask = self._prepare_attention_mask(x)

        # ─── Stage 2: DNH Hybrid Backbone ───
        x = self._process_dnh_blocks(x, attn_mask, T, cond_tokens, target_timestep)

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
        s = self.state_encoder(states).unsqueeze(2)
        a = self.action_encoder(actions).unsqueeze(2)

        x = x.view(B, T, self.patches_per_frame, D)

        if self.use_extrinsics and extrinsics is not None:
            e = self.extrinsics_encoder(extrinsics).unsqueeze(2)
            return torch.cat([a, s, e, x], dim=2).flatten(1, 2)
        else:
            return torch.cat([a, s, x], dim=2).flatten(1, 2)

    def _prepare_attention_mask(self, x: Tensor) -> Tensor | None:
        """Prepare attention mask for current sequence length."""
        if self.attn_mask is None:
            return None
        seq_len = x.size(1)
        return self.attn_mask[:seq_len, :seq_len].to(
            device=x.device, non_blocking=True
        ).clone()

    def _process_dnh_blocks(
        self, x: Tensor, attn_mask: Tensor, T: int, cond_tokens: int,
        target_timestep: int | None = None,
    ) -> Tensor:
        """Process tokens through DNH Hybrid backbone blocks."""
        tokens_per_frame = cond_tokens + self.patches_per_frame

        # ─── Inject learnable temporal position information ───
        if target_timestep is not None:
            frame_0_emb = self.frame_pos_embed(
                torch.zeros(1, dtype=torch.long, device=x.device)
            )
            x = x + frame_0_emb.unsqueeze(0)
            target_emb = self.target_pos_embed(
                torch.tensor(target_timestep, dtype=torch.long, device=x.device)
            )
            x = x + target_emb.unsqueeze(0).unsqueeze(0)
        else:
            frame_indices = torch.arange(T, dtype=torch.long, device=x.device)
            frame_embs = self.frame_pos_embed(frame_indices)
            frame_embs = frame_embs.unsqueeze(0).unsqueeze(2)
            frame_embs = frame_embs.expand(-1, -1, tokens_per_frame, -1)
            frame_embs = frame_embs.reshape(1, T * tokens_per_frame, -1)
            x = x + frame_embs

        for i, blk in enumerate(self.dnh_blocks):
            if self.use_activation_checkpointing:
                # Phase A (attention) — pure, checkpoint-safe
                x = torch.utils.checkpoint.checkpoint(
                    blk._phase_a, x, None, attn_mask, T,
                    self.grid_height, self.grid_width,
                    cond_tokens, target_timestep,
                    use_reentrant=False,
                )

                # Phase B (DNH memory) — stateful DGD, NOT checkpointable
                y = blk.norm2(x)
                y = blk.dnh(y)
                x = x + blk.drop_path(y)

                # Phase C (CMS) — pure, checkpoint-safe
                x = torch.utils.checkpoint.checkpoint(
                    blk._phase_c, x, T,
                    self.grid_height, self.grid_width,
                    cond_tokens,
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
        """Reset all clip-level memory states."""
        for blk in self.dnh_blocks:
            blk.reset_memory_state()

    def reset_all_longterm_memories(self) -> None:
        """Explicitly reset all longterm memories."""
        for blk in self.dnh_blocks:
            blk.reset_longterm_memory()

    def freeze_all_inner_loops(self) -> None:
        """Freeze DGD memory updates (for CL evaluation)."""
        for blk in self.dnh_blocks:
            blk.dnh.freeze_inner_loops()
        log.info("DNH inner-loop DGD frozen for all blocks (eval mode)")

    def unfreeze_all_inner_loops(self) -> None:
        """Unfreeze DGD memory updates."""
        for blk in self.dnh_blocks:
            blk.dnh.unfreeze_inner_loops()
        log.info("DNH inner-loop DGD unfrozen for all blocks (train mode)")

    # ─── CL Phase 9 compatible: Attention freeze & Titan reset ───

    def freeze_attention(self) -> None:
        """Freeze all attention parameters across blocks."""
        frozen_count = 0
        for blk in self.dnh_blocks:
            for name, param in blk.named_parameters():
                if any(p in name for p in ("attn.", "norm1.")):
                    param.requires_grad = False
                    frozen_count += 1
        log.info(
            f"Attention frozen: {frozen_count} params across {len(self.dnh_blocks)} blocks"
        )

    def unfreeze_attention(self) -> None:
        """Unfreeze all attention parameters."""
        unfrozen_count = 0
        for blk in self.dnh_blocks:
            for name, param in blk.named_parameters():
                if any(p in name for p in ("attn.", "norm1.")):
                    param.requires_grad = True
                    unfrozen_count += 1
        log.info(
            f"Attention unfrozen: {unfrozen_count} params across {len(self.dnh_blocks)} blocks"
        )

    def reset_titan_for_new_task(self) -> None:
        """Reset clip-level memories for a new CL task, preserving longterm."""
        for blk in self.dnh_blocks:
            blk.dnh.clear_all_levels()
            blk.dynamic_cms.reset_step_counter()
            if blk.dnh.use_longterm_memory:
                blk.dnh.detach_all_levels()
        log.info(
            "DNH memories cleared for new CL task, longterm preserved (detached)"
        )

    def get_aux_loss(self) -> Tensor:
        """Return zero aux loss (DNH hybrid doesn't need aux loss)."""
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def get_all_diagnostics(self) -> DiagnosticMetrics:
        """Aggregate diagnostics from all blocks."""
        if not self.log_hope_diagnostics:
            return {}

        aggregated: DiagnosticMetrics = {}
        n_blocks = len(self.dnh_blocks)

        for blk in self.dnh_blocks:
            for key, val in blk.get_diagnostics().items():
                aggregated[key] = aggregated.get(key, 0.0) + val / n_blocks

        return aggregated

    def get_structural_summary(self) -> dict[str, list]:
        """Get current hierarchy structure across all blocks.

        Returns dict with:
            - num_memory_levels: [int] per block
            - memory_frequencies: [[float]] per block
            - num_cms_levels: [int] per block
        """
        summary: dict[str, list] = {
            "num_memory_levels": [],
            "memory_frequencies": [],
            "num_cms_levels": [],
        }
        for blk in self.dnh_blocks:
            summary["num_memory_levels"].append(blk.dnh.num_levels)
            summary["memory_frequencies"].append(blk.dnh.frequencies)
            summary["num_cms_levels"].append(blk.dynamic_cms.num_levels)
        return summary

    def step_structural_evolution(
        self,
        evolution_controllers: list,
        meta_loss: float,
    ) -> dict[str, int | float]:
        """Execute structural evolution step on all blocks.

        Args:
            evolution_controllers: List of StructuralEvolutionController,
                one per block.
            meta_loss: Current meta-loss scalar.

        Returns:
            Aggregated evolution events dict.
        """
        all_events: dict[str, int | float] = {}
        for i, (blk, ctrl) in enumerate(
            zip(self.dnh_blocks, evolution_controllers)
        ):
            events = ctrl.step(blk.dnh, meta_loss)
            for key, val in events.items():
                all_events[f"block_{i}/{key}"] = val
        return all_events

    def get_parameter_groups(self) -> ParameterGroups:
        """Get parameter groups for optimizer with different learning rates.

        Returns 5 groups:
            1. attention — self-attention params (QKV, proj, norm1)
            2. dnh_memory — SMM weights, meta-networks, η/α, level contexts
            3. dnh_controller — frequency parameters
            4. cms — DynamicCMS MLP parameters
            5. projections — embedding, decoding, action/state encoders, norms
        """
        attention_params: list[Tensor] = []
        dnh_memory_params: list[Tensor] = []
        dnh_controller_params: list[Tensor] = []
        cms_params: list[Tensor] = []
        other_params: list[Tensor] = []

        # Patterns for classification
        dnh_memory_patterns = {
            "dnh.levels.", "dnh.mem_q_proj.", "dnh.mem_k_proj.",
            "dnh.mem_v_proj.", "dnh.mem_out_proj.", "dnh.M_longterm.",
            "dnh.longterm_gate.",
        }
        dnh_controller_patterns = {"dnh.freq_raw."}
        attention_patterns = {"attn.", "norm1."}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(pattern in name for pattern in dnh_controller_patterns):
                dnh_controller_params.append(param)
            elif any(pattern in name for pattern in dnh_memory_patterns):
                dnh_memory_params.append(param)
            elif "dynamic_cms." in name:
                cms_params.append(param)
            elif any(pattern in name for pattern in attention_patterns):
                attention_params.append(param)
            else:
                other_params.append(param)

        return [
            {"params": attention_params, "group_name": "attention"},
            {"params": dnh_memory_params, "group_name": "dnh_memory"},
            {"params": dnh_controller_params, "group_name": "dnh_controller"},
            {"params": cms_params, "group_name": "cms"},
            {"params": other_params, "group_name": "projections"},
        ]


def ac_dnh_hope_hybrid_vit(**kwargs) -> ACDNHHOPEHybridViT:
    """Factory function for ACDNHHOPEHybridViT."""
    return ACDNHHOPEHybridViT(**kwargs)
