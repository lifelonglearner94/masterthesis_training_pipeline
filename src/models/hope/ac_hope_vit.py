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
from typing import Final

import torch
import torch.nn as nn
from torch import Tensor

from src.models.ac_predictor.utils.modules import build_action_block_causal_attention_mask
from src.models.ac_predictor.utils.tensors import trunc_normal_
from src.models.hope.cms import LevelSpec
from src.models.hope.hope_block import HOPEBlock, HOPEBlockConfig
from src.models.hope.titan_memory import _backward_grad_clip

log = logging.getLogger(__name__)

# Type aliases for clarity
type DiagnosticMetrics = dict[str, float]
type ParameterGroups = list[dict]
type BlockConfig = dict

# Constants
COND_TOKENS_WITHOUT_EXTRINSICS: Final = 2
COND_TOKENS_WITH_EXTRINSICS: Final = 3
DEFAULT_INIT_STD: Final = 0.02
DEFAULT_DROP_RATE: Final = 0.0
DEFAULT_DROP_PATH_RATE: Final = 0.0


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
        use_extrinsics: Whether to use extrinsic parameters.
        titan_hidden_multiplier: Titan memory MLP hidden dim multiplier.
        titan_layers: Number of layers in Titan memory MLP.
        titan_activation: Activation function for Titan memory.
        titan_grad_clip_inner: Gradient clip for inner-loop (DGD) gradients.
        titan_grad_clip_backward: Max gradient norm for DGD backward path (default 1.0).
        cms_level_specs: List of CMS level specifications.
        cms_use_chunk_scheduling: Whether CMS uses chunk-based scheduling.
        chunk_size: Chunk size for intra-sequence Titan updates (0 = full sequence).
        titan_detach_interval: Detach Titan computation graph every N steps (0 = never).
        surprise_threshold: Minimum surprise to trigger memory update.
        drop_rate: Dropout rate.
        drop_path_rate: Stochastic depth rate.
        log_hope_diagnostics: Whether to collect HOPE diagnostic metrics.
        init_std: Standard deviation for weight initialization.
    """

    def __init__(
        self,
        img_size: tuple[int, int] | int = (256, 256),
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
        titan_grad_clip_backward: float = 1.0,
        cms_level_specs: list[BlockConfig] | None = None,
        cms_use_chunk_scheduling: bool = False,
        chunk_size: int = 0,
        titan_detach_interval: int = 0,
        surprise_threshold: float = 0.0,
        # Spatial mixing (Phase C)
        use_spatial_mixing: bool = False,
        # Regularization
        drop_rate: float = DEFAULT_DROP_RATE,
        drop_path_rate: float = DEFAULT_DROP_PATH_RATE,
        # Diagnostics (Criticism §1)
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

        # ─── Stage 1: Input & Embedding (same as AC_ViT) ───
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

        # ─── Learnable temporal embeddings (Phase 5 fixes) ───
        # These replace RoPE's temporal function with embeddings that work with MLPs.
        # - frame_pos_embed: tells each token WHICH frame it belongs to (input position)
        # - target_pos_embed: tells the model WHICH frame to predict (jump target)
        # Without these, HOPE with use_rope=False has NO temporal position info at all,
        # and jump prediction cannot distinguish between different target frames.
        self.frame_pos_embed = nn.Embedding(num_timesteps + 2, predictor_embed_dim)
        self.target_pos_embed = nn.Embedding(num_timesteps + 2, predictor_embed_dim)

        # ─── Spatial mixing tokens count ───
        cond_tok_count = (
            COND_TOKENS_WITH_EXTRINSICS
            if use_extrinsics
            else COND_TOKENS_WITHOUT_EXTRINSICS
        )
        spatial_mixing_tokens = self.patches_per_frame + cond_tok_count
        self._use_spatial_mixing = use_spatial_mixing

        # ─── Stage 2: HOPE-ViT Backbone ───
        cms_levels = self._build_cms_levels(cms_level_specs)

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
                    titan_grad_clip_backward=titan_grad_clip_backward,
                    cms_levels=cms_levels,
                    cms_use_chunk_scheduling=cms_use_chunk_scheduling,
                    chunk_size=chunk_size,
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    surprise_threshold=surprise_threshold,
                    use_spatial_mixing=use_spatial_mixing,
                    spatial_mixing_tokens=spatial_mixing_tokens,
                    drop_path=dpr[i],
                    drop=drop_rate,
                ),
                titan_detach_interval=titan_detach_interval,
            )
            for i in range(depth)
        ])

        # ─── Stage 3: Output & Decoding (same as AC_ViT) ───
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # ─── Causal attention mask ───
        self.attn_mask = self._build_attention_mask()

        # ─── Initialize weights ───
        self.apply(self._init_weights)

        # Store config for diagnostics logging (Criticism §1)
        self._config_summary = {
            "depth": depth,
            "predictor_embed_dim": predictor_embed_dim,
            "num_heads": num_heads,
            "titan_hidden_multiplier": titan_hidden_multiplier,
            "titan_layers": titan_layers,
            "titan_grad_clip_inner": titan_grad_clip_inner,
            "titan_grad_clip_backward": titan_grad_clip_backward,
            "cms_levels": len(cms_levels),
            "use_rope": use_rope,
            "chunk_size": chunk_size,
            "titan_detach_interval": titan_detach_interval,
            "surprise_threshold": surprise_threshold,
            "use_spatial_mixing": use_spatial_mixing,
            "total_params": sum(p.numel() for p in self.parameters()),
        }

    def _build_cms_levels(
        self, cms_level_specs: list[BlockConfig] | None
    ) -> list[LevelSpec]:
        """Build CMS level specifications from config or use defaults.

        Args:
            cms_level_specs: Optional list of level specification dicts.

        Returns:
            List of LevelSpec objects for CMS.
        """
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
                LevelSpec(name="fast", update_period=1, hidden_multiplier=4.0),
                LevelSpec(name="medium", update_period=4, hidden_multiplier=4.0),
                LevelSpec(name="slow", update_period=16, hidden_multiplier=4.0),
            ]

    def _build_attention_mask(self) -> Tensor | None:
        """Build causal attention mask if frame-causal mode is enabled.

        Returns:
            Causal attention mask tensor or None.
        """
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
        """Initialize module weights.

        Args:
            m: Module to initialize.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_config_summary(self) -> dict[str, int | float | bool | str]:
        """Get configuration summary for logging (Criticism §1).

        Returns all key parameters that could affect training stability:
        - Architecture dimensions
        - Titan memory settings (inner-loop grad clip, hidden mult)
        - CMS level count
        - RoPE on/off
        - Total parameter count

        Returns:
            Dictionary containing configuration summary.
        """
        return self._config_summary

    def forward(
        self,
        x: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
        target_timestep: int | None = None,
    ) -> Tensor:
        """Forward pass through AC-HOPE-ViT predictor.

        Signature-compatible with VisionTransformerPredictorAC.forward().

        Args:
            x: Context tokens [B, T*N, D] where T=num_timesteps, N=H*W patches.
            actions: Action sequences [B, T, action_dim].
            states: State sequences [B, T, action_dim].
            extrinsics: Optional extrinsic parameters [B, T, action_dim-1].
            target_timestep: Target frame index for jump prediction.

        Returns:
            Predicted features [B, T*N, D].

        Raises:
            ValueError: If input token count is not divisible by patches per frame,
                or if input timesteps exceed configured maximum.
        """
        log.debug(f"    [AC_HOPE_ViT] Input x: {x.shape}")
        log.debug(f"    [AC_HOPE_ViT] Actions: {actions.shape}, States: {states.shape}")

        # ─── Stage 1: Embed & Interleave ───
        x = self.predictor_embed(x)
        B, N_ctxt, D = x.size()
        T = N_ctxt // self.patches_per_frame

        # Validate input
        if N_ctxt % self.patches_per_frame != 0:
            raise ValueError(
                f"Input token count ({N_ctxt}) not divisible by patches_per_frame "
                f"({self.patches_per_frame}). Check input features vs model config."
            )
        if T > self.num_timesteps:
            raise ValueError(
                f"Input has {T} timesteps but model configured for max {self.num_timesteps}."
            )

        # Interleave action/state tokens
        x = self._interleave_conditioning_tokens(
            x, actions, states, extrinsics, B, T, D
        )

        cond_tokens = (
            COND_TOKENS_WITH_EXTRINSICS
            if self.use_extrinsics
            else COND_TOKENS_WITHOUT_EXTRINSICS
        )

        log.debug(f"    [AC_HOPE_ViT] After interleaving: {x.shape}")

        # Prepare causal mask
        attn_mask = self._prepare_attention_mask(x)

        # ─── Stage 2: HOPE Backbone ───
        x = self._process_hope_blocks(x, attn_mask, T, cond_tokens, target_timestep=target_timestep)

        # ─── Stage 3: Decode ───
        x = self._decode_output(x, B, T, D, cond_tokens)

        log.debug(f"    [AC_HOPE_ViT] Output: {x.shape}")

        return x

    def _interleave_conditioning_tokens(
        self,
        x: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None,
        B: int,
        T: int,
        D: int,
    ) -> Tensor:
        """Interleave action/state/extrinsic tokens with image tokens.

        Args:
            x: Embedded image tokens [B, T*N, D].
            actions: Action sequences [B, T, action_dim].
            states: State sequences [B, T, action_dim].
            extrinsics: Optional extrinsic parameters [B, T, action_dim-1].
            B: Batch size.
            T: Number of timesteps.
            D: Embedding dimension.

        Returns:
            Interleaved tokens [B, T*(N+cond), D].
        """
        s = self.state_encoder(states).unsqueeze(2)  # [B, T, 1, D]
        a = self.action_encoder(actions).unsqueeze(2)  # [B, T, 1, D]

        x = x.view(B, T, self.patches_per_frame, D)  # [B, T, H*W, D]

        if self.use_extrinsics and extrinsics is not None:
            e = self.extrinsics_encoder(extrinsics).unsqueeze(2)
            return torch.cat([a, s, e, x], dim=2).flatten(1, 2)
        else:
            return torch.cat([a, s, x], dim=2).flatten(1, 2)

    def _prepare_attention_mask(self, x: Tensor) -> Tensor:
        """Prepare attention mask for current sequence length.

        Args:
            x: Input tokens tensor.

        Returns:
            Attention mask tensor on the same device as x.
        """
        seq_len = x.size(1)
        return self.attn_mask[:seq_len, :seq_len].to(
            device=x.device, non_blocking=True
        ).clone()

    def _process_hope_blocks(
        self, x: Tensor, attn_mask: Tensor, T: int, cond_tokens: int,
        target_timestep: int | None = None,
    ) -> Tensor:
        """Process tokens through HOPE backbone blocks.

        Args:
            x: Input tokens.
            attn_mask: Attention mask.
            T: Number of timesteps.
            cond_tokens: Number of conditioning tokens.
            target_timestep: Target frame index for jump prediction.

        Returns:
            Processed tokens after all HOPE blocks.
        """
        log.debug(f"    [AC_HOPE_ViT] >>> Processing {len(self.hope_blocks)} HOPE blocks >>>")

        # ─── Inject learnable temporal position information ───
        tokens_per_frame = cond_tokens + self.patches_per_frame
        if target_timestep is not None:
            # Jump prediction mode: input is frame 0, predict frame τ.
            # Add frame-0 position to input tokens.
            frame_0_emb = self.frame_pos_embed(
                torch.zeros(1, dtype=torch.long, device=x.device)
            )  # [1, D]
            x = x + frame_0_emb.unsqueeze(0)  # broadcast [1, 1, D] → [B, N, D]
            # Add target timestep embedding so the model knows WHICH frame to predict.
            target_emb = self.target_pos_embed(
                torch.tensor(target_timestep, dtype=torch.long, device=x.device)
            )  # [D]
            x = x + target_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, D] → [B, N, D]
        else:
            # Teacher-forcing mode: add per-frame position embeddings.
            # Each frame's tokens get the embedding for that frame index.
            frame_indices = torch.arange(T, dtype=torch.long, device=x.device)
            frame_embs = self.frame_pos_embed(frame_indices)  # [T, D]
            # Expand: [T, D] → [1, T, 1, D] → [1, T, tpf, D] → [1, T*tpf, D]
            frame_embs = frame_embs.unsqueeze(0).unsqueeze(2)  # [1, T, 1, D]
            frame_embs = frame_embs.expand(-1, -1, tokens_per_frame, -1)  # [1, T, tpf, D]
            frame_embs = frame_embs.reshape(1, T * tokens_per_frame, -1)  # [1, N, D]
            x = x + frame_embs  # broadcast over batch

        for i, blk in enumerate(self.hope_blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    None,  # mask
                    attn_mask,
                    T,
                    self.grid_height,
                    self.grid_width,
                    cond_tokens,
                    target_timestep,
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
                    target_timestep=target_timestep,
                )

            # Level 3: block-level backward gradient clipping.
            # Prevents gradient compounding across blocks by bounding the
            # backward gradient norm of the block output.  Without this,
            # the gradient flowing backward through the INPUT path of
            # each Titan memory (grad_key = grad_output @ active_w) is
            # unclipped and compounds exponentially across 6 blocks:
            #   block_5: O(0.06) → block_4: O(3.8e9) → block_3: O(1.6e19) → Inf
            # With this clip, each block sees at most grad_clip_backward from downstream.
            if self.titan_grad_clip_backward > 0 and self.training:
                x = _backward_grad_clip(x, self.titan_grad_clip_backward)

            log.debug(
                f"    [AC_HOPE_ViT] Block {i} output: shape={x.shape}, "
                f"min={x.min().item():.4f}, max={x.max().item():.4f}"
            )

        log.debug(f"    [AC_HOPE_ViT] <<< Finished HOPE blocks <<<")
        return x

    def _decode_output(
        self, x: Tensor, B: int, T: int, D: int, cond_tokens: int
    ) -> Tensor:
        """Remove conditioning tokens and project to output dimension.

        Args:
            x: Tokens after HOPE blocks.
            B: Batch size.
            T: Number of timesteps.
            D: Embedding dimension.
            cond_tokens: Number of conditioning tokens to remove.

        Returns:
            Decoded output tokens [B, T*H*W, D_out].
        """
        # Remove action/state tokens
        x = x.view(B, T, cond_tokens + self.patches_per_frame, D)
        x = x[:, :, cond_tokens:, :].flatten(1, 2)  # [B, T*H*W, D]

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)
        return x

    def reset_all_memories(self) -> None:
        """Reset all HOPE block memory states (call between sequences)."""
        for blk in self.hope_blocks:
            blk.reset_memory_state()

    def freeze_all_inner_loops(self) -> None:
        """Freeze all HOPE blocks' inner-loop DGD memory updates.

        Memories are still read (retrieval) but never written. Use this
        for CL evaluation phases where no weight/memory updates are allowed.
        """
        for blk in self.hope_blocks:
            blk.freeze_inner_loop = True
        log.info("HOPE inner-loop DGD frozen for all blocks (eval mode)")

    def unfreeze_all_inner_loops(self) -> None:
        """Unfreeze all HOPE blocks' inner-loop DGD memory updates."""
        for blk in self.hope_blocks:
            blk.freeze_inner_loop = False
        log.info("HOPE inner-loop DGD unfrozen for all blocks (train mode)")

    def get_aux_loss(self) -> Tensor:
        """Aggregate auxiliary M_k/M_v retrieval-quality loss from all blocks.

        This loss provides gradient flow to M_k and M_v parameters which
        otherwise receive no gradients under first-order meta-learning
        (FOMAML). Should be added to the outer loss with a small weight.

        Returns:
            Scalar tensor with accumulated auxiliary loss.
        """
        device = next(self.parameters()).device
        total = torch.tensor(0.0, device=device)
        for blk in self.hope_blocks:
            total = total + blk._aux_loss.to(device)
        return total

    def get_all_diagnostics(self) -> DiagnosticMetrics:
        """Aggregate diagnostics from all HOPE blocks (Criticism §1).

        Returns averaged diagnostics across all blocks.

        Returns:
            Dictionary of diagnostic metric names to average values.
        """
        if not self.log_hope_diagnostics:
            return {}

        aggregated: DiagnosticMetrics = {}
        n_blocks = len(self.hope_blocks)

        for blk in self.hope_blocks:
            block_diag = blk.get_diagnostics()
            for key, val in block_diag.items():
                aggregated[key] = aggregated.get(key, 0.0) + val / n_blocks

        return aggregated

    def get_parameter_groups(self) -> ParameterGroups:
        """Get parameter groups for optimizer with different learning rates.

        Returns 3 groups:
            1. Titan memory parameters (lower LR recommended)
            2. CMS parameters (normal LR)
            3. Projection/embedding parameters (normal LR)

        This allows the optimizer to use per-group learning rates,
        which is important because Titan memories use inner-loop
        optimization and may need gentler outer-loop updates.

        Returns:
            List of parameter group dictionaries with 'params' and 'group_name' keys.
        """
        titan_params: list[Tensor] = []
        cms_params: list[Tensor] = []
        other_params: list[Tensor] = []

        titan_patterns = {"M_k.", "M_v.", "M_eta.", "M_alpha.", "M_memory."}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(pattern in name for pattern in titan_patterns):
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
    """Factory function for ACHOPEViT (mirrors vit_ac_predictor).

    Args:
        **kwargs: Arguments to pass to ACHOPEViT constructor.

    Returns:
        Instantiated ACHOPEViT model.
    """
    return ACHOPEViT(**kwargs)
