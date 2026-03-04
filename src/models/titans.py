"""Titans MAC (Memory as a Context) model for action-conditioned prediction.

Integrates the Titans architecture from "Learn to Memorize at Test Time"
into the existing CL pipeline. The model uses:
    - Neural Memory Module (NMM): A small MLP whose weights are updated
      at test time via surprise-driven gradient descent.
    - Persistent Memory (PM): Learnable context tokens prepended to attention.
    - MAC Layer: Combines short-term attention, long-term NMM, and PM.

The architecture is taken from the original Titans implementation
(temp/titans-lmm/) and adapted to the action-conditioned prediction
interface used by all CL models in this repository.

Architecture:
    - 1×1 Conv Encoder: 1024 → hidden_dim
    - Action Embedding: spatial tiling + concat
    - Stacked MACTitanLayers (attention + NMM + PM)
    - 1×1 Conv Decoder: hidden_dim → 1024, output = z_t + delta

Reference: Ali et al., "Titans: Learning to Memorize at Test Time", 2024
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Final

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call
from torch.nn.functional import normalize

from src.models.mixins import ACPredictorLossMixin


# ───────────────────────── Constants ─────────────────────────
DEFAULT_INPUT_DIM: Final = 1024
DEFAULT_HIDDEN_DIM: Final = 768
DEFAULT_ACTION_DIM: Final = 2
DEFAULT_SPATIAL_SIZE: Final = 16
DEFAULT_PM_LEN: Final = 4
DEFAULT_N_LAYERS: Final = 4
DEFAULT_N_LAYERS_NMM: Final = 2
DEFAULT_ALPHA: Final = 0.999
DEFAULT_ETA: Final = 0.8
DEFAULT_THETA: Final = 0.3
DEFAULT_NUM_TIMESTEPS: Final = 8
DEFAULT_T_TEACHER: Final = 7
DEFAULT_JUMP_K: Final = 7
DEFAULT_LOSS_WEIGHT_TEACHER: Final = 1.0
DEFAULT_LOSS_WEIGHT_JUMP: Final = 1.0
DEFAULT_LOSS_TYPE: Final = "l1"
DEFAULT_HUBER_DELTA: Final = 1.0
DEFAULT_LEARNING_RATE: Final = 4.25e-4
DEFAULT_WEIGHT_DECAY: Final = 0.04
DEFAULT_BETAS: Final = (0.9, 0.999)
DEFAULT_WARMUP_EPOCHS: Final = 10
DEFAULT_MAX_EPOCHS: Final = 100
DEFAULT_WARMUP_PCT: Final = 0.085
DEFAULT_CONSTANT_PCT: Final = 0.83
DEFAULT_DECAY_PCT: Final = 0.085
DEFAULT_WARMUP_START_LR: Final = 7.5e-5
LAYER_NORM_EPS: Final = 1e-6

# Type aliases
type HiddenState = tuple[Tensor, Tensor]
type CurriculumParams = dict[str, float | int]
type TestResult = dict[str, Any]


log = logging.getLogger(__name__)


# =============================================================================
# Core Titans architecture (adapted from temp/titans-lmm/)
# =============================================================================


class NeuralMemory(nn.Module):
    """Neural Memory Module — a small MLP whose weights serve as long-term memory.

    The NMM is updated at every forward step via surprise-gated momentum:
        surprise[t] = η · surprise[t-1] − θ · ∇_loss
        θ_NMM[t+1]  = α · θ_NMM[t]   + surprise[t]

    This implements the *test-time training* (TTT) inner loop from the Titans
    paper (Algorithm 1).

    Args:
        emb_dim: Embedding / key-value dimension.
        n_layers: Number of hidden layers in the MLP.
        hidden_dim: Width of hidden layers.
        alpha: Weight decay towards initial weights (momentum term).
        eta: Surprise momentum.
        theta: Surprise learning rate.
    """

    def __init__(
        self,
        emb_dim: int = 16,
        n_layers: int = 2,
        hidden_dim: int = 32,
        alpha: float = DEFAULT_ALPHA,
        eta: float = DEFAULT_ETA,
        theta: float = DEFAULT_THETA,
    ) -> None:
        super().__init__()

        # Build MLP layers
        if n_layers == 1:
            self.layers = nn.ModuleList([nn.Linear(emb_dim, emb_dim)])
        else:
            layers: list[nn.Module] = []
            layers.append(nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.SiLU()))
            for _ in range(n_layers - 2):
                layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
            layers.append(nn.Sequential(nn.Linear(hidden_dim, emb_dim)))
            self.layers = nn.ModuleList(layers)

        # Key / value projections
        self.K = nn.Linear(emb_dim, emb_dim, bias=False)
        self.V = nn.Linear(emb_dim, emb_dim, bias=False)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.V.weight)

        self.alpha = alpha
        self.eta = eta
        self.theta = theta

        self.silu = nn.SiLU()
        self.surprise: dict[str, Tensor] = {}

    def retrieve(self, x: Tensor) -> Tensor:
        """Retrieve from the NMM using current parameters (functional call)."""
        return functional_call(self, dict(self.named_parameters()), x)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def update(self, x: Tensor) -> tuple[float, dict[str, Tensor]]:
        """Surprise-gated weight update (inner loop / TTT step).

        Args:
            x: Input tokens used to compute keys/values for the NMM loss.

        Returns:
            (loss_value, updated_params_dict)

        Note:
            Uses ``torch.enable_grad()`` internally so the NMM inner loop
            works even when the outer context is ``torch.no_grad()``
            (e.g. during validation/test).
        """
        z = x.detach()

        with torch.enable_grad():
            # Ensure temporary copies require grad for autograd
            keys = normalize(self.silu(self.K(z)))
            vals = self.silu(self.V(z))

            # Propagate keys through the MLP to get predicted values
            pred = keys
            for layer in self.layers:
                pred = layer(pred)

            # Associative recall loss: ||M(keys) − vals||²
            loss = ((pred - vals) ** 2).mean(dim=0).sum()

            grads = torch.autograd.grad(loss, self.parameters(), create_graph=False)

        updated_params: dict[str, Tensor] = {}
        for (name, param), grad in zip(self.named_parameters(), grads):
            if self.surprise.get(name) is None:
                self.surprise[name] = torch.zeros_like(grad)
            self.surprise[name] = self.surprise[name] * self.eta - self.theta * grad
            # Don't update K/V projections (only the MLP body)
            if name[0] in ("K", "V"):
                updated_params[name] = param.data
            else:
                updated_params[name] = self.alpha * param.data + self.surprise[name]
            param.data = updated_params[name]

        return loss.item(), updated_params

    def reset_surprise(self) -> None:
        """Reset surprise accumulators (call between sequences)."""
        self.surprise = {}


class MACTitanLayer(nn.Module):
    """Single MAC (Memory as a Context) layer.

    Combines persistent memory, NMM retrieval, and causal self-attention:
        1. Retrieve from NMM → long-term context
        2. Prepend persistent memory tokens + NMM tokens to input
        3. Self-attention over the concatenated sequence
        4. Update NMM via surprise-gated gradient step
        5. Gate output with post-update NMM retrieval

    Args:
        hidden_dim: Feature dimension per token.
        seq_len: Number of input tokens in the short-term context.
        pm_len: Number of persistent memory tokens.
        n_layers_nmm: Depth of the NMM MLP.
        alpha, eta, theta: NMM hyper-parameters.
    """

    def __init__(
        self,
        hidden_dim: int,
        seq_len: int,
        pm_len: int,
        n_layers_nmm: int = 2,
        alpha: float = DEFAULT_ALPHA,
        eta: float = DEFAULT_ETA,
        theta: float = DEFAULT_THETA,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.pm_len = pm_len
        self.hidden_dim = hidden_dim

        # Persistent memory
        self.persistent_memory = nn.Parameter(torch.randn(pm_len, hidden_dim))

        # Attention core
        self.att_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=2,
            activation=nn.SiLU(),
            batch_first=True,
        )

        # Query projection for NMM retrieval
        self.Q = nn.Linear(hidden_dim, hidden_dim)

        # Neural Memory Module
        self.nm_module = NeuralMemory(
            emb_dim=hidden_dim,
            n_layers=n_layers_nmm,
            hidden_dim=2 * hidden_dim,
            alpha=alpha,
            eta=eta,
            theta=theta,
        )

        # Per-token projection (replaces the giant flatten→linear that doesn't
        # scale to large seq_len).  After attention over [PM, NMM, input], we
        # extract only the input-position tokens and project per-token.
        self.final_layer = nn.Linear(hidden_dim, hidden_dim)

        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, seq_len, hidden_dim]
        Returns:
            [B, seq_len, hidden_dim]
        """
        B = x.shape[0]

        # 1. Retrieve from NMM
        queries = self.silu(normalize(self.Q(x.reshape(-1, self.hidden_dim))))
        nmm_vals = self.nm_module.retrieve(queries).view(B, -1, self.hidden_dim)

        # 2. Prepend persistent memory + NMM tokens
        pm = self.persistent_memory.unsqueeze(0).expand(B, -1, -1)
        x_cat = torch.cat([pm, nmm_vals, x], dim=1)  # [B, pm_len+2*seq_len, hidden_dim]

        # 3. Attention over the full concatenated sequence
        att_out = self.silu(self.att_layer(x_cat))

        # 4. Extract only the input-position tokens (last seq_len tokens)
        #    PM and NMM tokens served as context; we discard their outputs.
        x_out = att_out[:, self.pm_len + self.seq_len :, :]  # [B, seq_len, hidden_dim]
        x_flat = self.final_layer(x_out).reshape(-1, self.hidden_dim)

        # 5. Update NMM
        _, new_params = self.nm_module.update(x_flat)

        # 6. Gate with post-update retrieval
        y = functional_call(self.nm_module, new_params, normalize(self.Q(x_flat)))
        gated = (x_flat * self.sigmoid(y)).view(B, self.seq_len, self.hidden_dim)

        return gated

    def reset_memory(self) -> None:
        """Reset NMM surprise state between sequences."""
        self.nm_module.reset_surprise()


class TitansBackbone(nn.Module):
    """Action-conditioned MAC-Titan backbone for spatiotemporal prediction.

    Processes V-JEPA2 features ([B, T*N, D]) with actions, using the Titans
    MAC architecture for temporal modelling and residual prediction.

    Pipeline per timestep:
        1. Encode features: Conv1x1 1024 → hidden_dim
        2. Tile & concat action
        3. Flatten spatial dim into token sequence
        4. Process through stacked MACTitanLayers
        5. Decode residual: Conv1x1 hidden_dim → 1024
        6. z_{t+1} = z_t + delta

    Args:
        input_dim:    Feature dimension (1024 for V-JEPA2).
        hidden_dim:   Internal working dimension.
        action_dim:   Action vector size.
        spatial_size:  Spatial grid side (16 → 256 patches).
        pm_len:       Number of persistent memory tokens.
        n_layers:     Number of stacked MAC layers.
        n_layers_nmm: Depth of NMM MLP.
        alpha, eta, theta: NMM hyper-parameters.
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        action_dim: int = DEFAULT_ACTION_DIM,
        spatial_size: int = DEFAULT_SPATIAL_SIZE,
        pm_len: int = DEFAULT_PM_LEN,
        n_layers: int = DEFAULT_N_LAYERS,
        n_layers_nmm: int = DEFAULT_N_LAYERS_NMM,
        alpha: float = DEFAULT_ALPHA,
        eta: float = DEFAULT_ETA,
        theta: float = DEFAULT_THETA,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.spatial_size = spatial_size
        self.pm_len = pm_len

        N = spatial_size * spatial_size  # 256 patches

        # 1×1 Conv encoder: input_dim → hidden_dim
        self.encoder = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)

        # Each timestep produces hidden_dim + action_dim channels after concat
        # We project back to hidden_dim before feeding into MAC layers
        self.action_proj = nn.Linear(hidden_dim + action_dim, hidden_dim)

        # Stacked MAC layers — seq_len = N (spatial patches per frame)
        self.mac_layers = nn.ModuleList([
            MACTitanLayer(
                hidden_dim=hidden_dim,
                seq_len=N,
                pm_len=pm_len,
                n_layers_nmm=n_layers_nmm,
                alpha=alpha,
                eta=eta,
                theta=theta,
            )
            for _ in range(n_layers)
        ])

        # 1×1 Conv decoder: hidden_dim → input_dim (residual)
        self.decoder = nn.Conv2d(hidden_dim, input_dim, kernel_size=1)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

        self.silu = nn.SiLU()

    def _tile_action(self, action: Tensor, H: int, W: int) -> Tensor:
        """Tile action spatially: [B, action_dim] → [B, action_dim, H, W]."""
        B, A = action.shape
        return action.view(B, A, 1, 1).expand(B, A, H, W)

    def reset_memory(self) -> None:
        """Reset NMM surprise across all MAC layers."""
        for layer in self.mac_layers:
            layer.reset_memory()

    def forward(
        self,
        z: Tensor,
        actions: Tensor,
        states: Tensor | None = None,
        extrinsics: Tensor | None = None,
    ) -> Tensor:
        """Forward pass — predict next frames given context and actions.

        Args:
            z:        [B, T*N, D]  context features
            actions:  [B, T, action_dim]
            states:   unused (API compat)
            extrinsics: unused (API compat)

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

        # Reset NMM state for each new batch
        self.reset_memory()

        predictions: list[Tensor] = []

        for t in range(T):
            current_z = z_spatial[:, t]  # [B, D, H, W]

            # Encode: [B, D, H, W] → [B, hidden_dim, H, W]
            feat = self.encoder(current_z)

            # Tile action: [B, action_dim, H, W]
            action_tiled = self._tile_action(actions[:, t], H, W)

            # Concat + project: [B, hidden_dim+action_dim, H, W] → [B, N, hidden_dim]
            combined = torch.cat([feat, action_tiled], dim=1)  # [B, hid+act, H, W]
            combined = combined.reshape(B, self.hidden_dim + self.action_dim, N)
            combined = combined.permute(0, 2, 1)  # [B, N, hid+act]
            tokens = self.action_proj(combined)    # [B, N, hidden_dim]

            # Process through MAC layers (residual connections)
            for mac_layer in self.mac_layers:
                tokens = tokens + self.silu(mac_layer(tokens))

            # Decode residual: [B, N, hidden_dim] → [B, D, H, W]
            dec_in = tokens.permute(0, 2, 1).reshape(B, self.hidden_dim, H, W)
            delta = self.decoder(dec_in)  # [B, D, H, W]

            # Residual prediction
            next_z = current_z + delta
            predictions.append(next_z)

        # Stack → [B, T, D, H, W] → [B, T*N, D]
        pred_stack = torch.stack(predictions, dim=1)  # [B, T, D, H, W]
        pred_stack = pred_stack.reshape(B, T, D, N).permute(0, 1, 3, 2)  # [B, T, N, D]
        return pred_stack.reshape(B, T * N, D)


# =============================================================================
# Lightning Module (follows BaselineLitModule pattern exactly)
# =============================================================================


class TitansLitModule(ACPredictorLossMixin, L.LightningModule):
    """PyTorch Lightning module for Titans MAC model.

    Wraps :class:`TitansBackbone` into the standard CL pipeline interface.
    Uses the same loss computation, curriculum schedule, and evaluation
    protocol as all other CL models via :class:`ACPredictorLossMixin`.

    Expected batch format (identical to ConvLSTM baseline):
        - features: [B, T+1, N, D]  pre-computed V-JEPA2 encoder features
        - actions:  [B, T, action_dim]
        - states:   [B, T, action_dim]

    Where T=8, N=256, D=1024.
    """

    def __init__(
        self,
        # Architecture
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        action_dim: int = DEFAULT_ACTION_DIM,
        spatial_size: int = DEFAULT_SPATIAL_SIZE,
        pm_len: int = DEFAULT_PM_LEN,
        n_layers: int = DEFAULT_N_LAYERS,
        n_layers_nmm: int = DEFAULT_N_LAYERS_NMM,
        alpha: float = DEFAULT_ALPHA,
        eta: float = DEFAULT_ETA,
        theta: float = DEFAULT_THETA,
        num_timesteps: int = DEFAULT_NUM_TIMESTEPS,
        # Loss (same as other CL models)
        T_teacher: int = DEFAULT_T_TEACHER,
        jump_k: int = DEFAULT_JUMP_K,
        loss_weight_teacher: float = DEFAULT_LOSS_WEIGHT_TEACHER,
        loss_weight_jump: float = DEFAULT_LOSS_WEIGHT_JUMP,
        normalize_reps: bool = True,
        loss_type: str = DEFAULT_LOSS_TYPE,
        huber_delta: float = DEFAULT_HUBER_DELTA,
        # Optimizer
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        betas: tuple[float, float] = DEFAULT_BETAS,
        warmup_epochs: int = DEFAULT_WARMUP_EPOCHS,
        max_epochs: int = DEFAULT_MAX_EPOCHS,
        # Iteration-based LR schedule
        use_iteration_scheduler: bool = False,
        curriculum_schedule: list[dict[str, float | int]] | None = None,
        warmup_pct: float = DEFAULT_WARMUP_PCT,
        constant_pct: float = DEFAULT_CONSTANT_PCT,
        decay_pct: float = DEFAULT_DECAY_PCT,
        warmup_start_lr: float = DEFAULT_WARMUP_START_LR,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.num_timesteps = num_timesteps

        # Build backbone
        self.model = TitansBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            spatial_size=spatial_size,
            pm_len=pm_len,
            n_layers=n_layers,
            n_layers_nmm=n_layers_nmm,
            alpha=alpha,
            eta=eta,
            theta=theta,
        )

        # Loss hyperparameters (required by mixin)
        self.T_teacher = T_teacher
        self.jump_k = jump_k
        self.loss_weight_teacher = loss_weight_teacher
        self.loss_weight_jump = loss_weight_jump
        self.normalize_reps = normalize_reps

        from src.models.mixins.loss_mixin import VALID_LOSS_TYPES
        if loss_type not in VALID_LOSS_TYPES:
            raise ValueError(
                f"loss_type must be one of {VALID_LOSS_TYPES} (got '{loss_type}')."
            )
        self.loss_type = loss_type
        self.huber_delta = huber_delta

        # Optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        # Iteration-based LR schedule
        self.use_iteration_scheduler = use_iteration_scheduler
        self.warmup_pct = warmup_pct
        self.constant_pct = constant_pct
        self.decay_pct = decay_pct
        self.warmup_start_lr = warmup_start_lr

        # Grid size (required by mixin)
        self.patches_per_frame = spatial_size * spatial_size

        # Curriculum
        self.curriculum_schedule = curriculum_schedule
        if curriculum_schedule:
            self._validate_curriculum_schedule(curriculum_schedule)

        # Test results storage
        self._test_results: list[TestResult] = []

    # ── Curriculum helpers (same as BaselineLitModule) ──────────────────

    def _validate_curriculum_schedule(self, schedule: list[dict]) -> None:
        """Validate curriculum schedule format."""
        if (
            not isinstance(schedule, Sequence)
            or isinstance(schedule, str)
            or len(schedule) == 0
        ):
            raise ValueError("curriculum_schedule must be a non-empty list")

        for i, phase in enumerate(schedule):
            if not isinstance(phase, Mapping):
                raise ValueError(f"Phase {i} must be a dict, got {type(phase)}")
            if "epoch" not in phase:
                raise ValueError(f"Phase {i} must have 'epoch' key")
            if not isinstance(phase["epoch"], int) or phase["epoch"] < 0:
                raise ValueError(
                    f"Phase {i} 'epoch' must be a non-negative integer"
                )
            if "jump_k" in phase and phase["jump_k"] > self.num_timesteps:
                raise ValueError(
                    f"Phase {i}: jump_k ({phase['jump_k']}) exceeds "
                    f"num_timesteps ({self.num_timesteps})"
                )

        epochs = [p["epoch"] for p in schedule]
        if epochs != sorted(epochs):
            raise ValueError("curriculum_schedule phases must be sorted by epoch")

    def _get_curriculum_params_for_epoch(self, epoch: int) -> CurriculumParams:
        if not self.curriculum_schedule:
            return {}
        applicable_phase = None
        for phase in self.curriculum_schedule:
            if phase["epoch"] <= epoch:
                applicable_phase = phase
            else:
                break
        if applicable_phase is None:
            return {}
        return {k: v for k, v in applicable_phase.items() if k != "epoch"}

    def on_train_epoch_start(self) -> None:
        """Update curriculum parameters at the start of each epoch."""
        if not self.curriculum_schedule:
            return
        epoch = self.current_epoch
        params = self._get_curriculum_params_for_epoch(epoch)
        if not params:
            return

        changes: list[str] = []
        if "jump_k" in params and params["jump_k"] != self.jump_k:
            old_val = self.jump_k
            self.jump_k = int(params["jump_k"])
            changes.append(f"jump_k: {old_val} → {self.jump_k}")
        if (
            "loss_weight_teacher" in params
            and params["loss_weight_teacher"] != self.loss_weight_teacher
        ):
            old_val = self.loss_weight_teacher
            self.loss_weight_teacher = float(params["loss_weight_teacher"])
            changes.append(f"loss_weight_teacher: {old_val} → {self.loss_weight_teacher}")
        if (
            "loss_weight_jump" in params
            and params["loss_weight_jump"] != self.loss_weight_jump
        ):
            old_val = self.loss_weight_jump
            self.loss_weight_jump = float(params["loss_weight_jump"])
            changes.append(f"loss_weight_jump: {old_val} → {self.loss_weight_jump}")

        if changes:
            log.info(f"[Curriculum] Epoch {epoch}: {', '.join(changes)}")

        self.log("curriculum/jump_k", float(self.jump_k), sync_dist=True)
        self.log("curriculum/loss_weight_teacher", self.loss_weight_teacher, sync_dist=True)
        self.log("curriculum/loss_weight_jump", self.loss_weight_jump, sync_dist=True)

    # ── Forward / step predictor ───────────────────────────────────────

    def forward(
        self,
        features: Tensor,
        actions: Tensor,
        states: Tensor | None = None,
        extrinsics: Tensor | None = None,
    ) -> Tensor:
        return self.model(features, actions, states, extrinsics)

    def _step_predictor(
        self,
        z: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
        target_timestep: int | None = None,
    ) -> Tensor:
        """Single predictor step with optional layer normalization.

        Required by ACPredictorLossMixin.
        Note: ``target_timestep`` is accepted for interface compatibility
        but ignored (no RoPE in Titans backbone).
        """
        z_pred = self.model(z, actions, states, extrinsics)
        if self.normalize_reps:
            z_pred = F.layer_norm(z_pred, (z_pred.size(-1),), eps=LAYER_NORM_EPS)
        return z_pred

    # ── Training / validation / test ───────────────────────────────────

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        features = batch["features"]
        actions = batch["actions"]
        states = batch["states"]
        extrinsics = batch.get("extrinsics", None)
        clip_names = batch.get(
            "clip_names", [f"clip_{batch_idx}_{i}" for i in range(features.shape[0])]
        )

        if features.dim() == 3:
            B, total_tokens, D = features.shape
            T_plus_1 = total_tokens // self.patches_per_frame
            features = features.reshape(B, T_plus_1, self.patches_per_frame, D)
        else:
            B = features.shape[0]

        loss_teacher = self._compute_teacher_forcing_loss(
            features, actions, states, extrinsics
        )
        loss_jump, per_timestep_losses, per_sample_losses = (
            self._compute_jump_loss_per_timestep(
                features, actions, states, extrinsics
            )
        )

        loss = (
            self.loss_weight_teacher * loss_teacher
            + self.loss_weight_jump * loss_jump
        )

        bs = features.shape[0]
        self.log("test/loss_teacher", loss_teacher, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("test/loss_jump", loss_jump, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("test/loss", loss, prog_bar=True, sync_dist=True, batch_size=bs)

        for step, step_loss in enumerate(per_timestep_losses):
            T = self.num_timesteps
            k = min(self.jump_k, T)
            tau_min = T + 1 - k
            target_frame = tau_min + step
            self.log(f"test/loss_jump_tau_{target_frame}", step_loss.mean(), sync_dist=True, batch_size=bs)

        per_sample_jump_losses = per_sample_losses[0]
        T = self.num_timesteps
        k = min(self.jump_k, T)
        tau_min = T + 1 - k
        for i in range(B):
            clip_result = {
                "clip_name": clip_names[i] if i < len(clip_names) else f"unknown_{batch_idx}_{i}",
                "loss_jump": per_sample_jump_losses[i].item(),
                "loss_teacher": loss_teacher.item(),
                "per_timestep_losses": {
                    f"tau_{tau_min + s}": per_timestep_losses[s][i].item()
                    for s in range(len(per_timestep_losses))
                },
            }
            self._test_results.append(clip_result)

        return loss

    def on_test_epoch_start(self) -> None:
        self._test_results = []

    def on_test_epoch_end(self) -> None:
        if not self._test_results:
            log.warning("No test results to aggregate")
            return

        num_clips = len(self._test_results)
        jump_losses = [r["loss_jump"] for r in self._test_results]

        mean_loss = sum(jump_losses) / num_clips
        sorted_losses = sorted(jump_losses)
        median_loss = sorted_losses[num_clips // 2]
        min_loss = min(jump_losses)
        max_loss = max(jump_losses)
        std_loss = (sum((x - mean_loss) ** 2 for x in jump_losses) / num_clips) ** 0.5

        print("\n" + "=" * 70)
        print("            TITANS TEST RESULTS SUMMARY")
        print("=" * 70)
        print(f"\n📊 AGGREGATE STATISTICS (over {num_clips} clips)")
        print("-" * 50)
        print(f"  Jump Prediction Loss ({self.loss_type}):")
        print(f"    Mean:   {mean_loss:.6f}")
        print(f"    Median: {median_loss:.6f}")
        print(f"    Std:    {std_loss:.6f}")
        print(f"    Min:    {min_loss:.6f}")
        print(f"    Max:    {max_loss:.6f}")
        print("=" * 70)

        self.log("test/final_mean_loss_jump", mean_loss, sync_dist=True)
        self.log("test/final_median_loss_jump", median_loss, sync_dist=True)

    # ── Optimizer ──────────────────────────────────────────────────────

    def configure_optimizers(self) -> dict:
        """Configure optimizer and LR scheduler (matches other CL models)."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )

        if self.use_iteration_scheduler:
            if self.trainer is not None and self.trainer.estimated_stepping_batches:
                total_iters = int(self.trainer.estimated_stepping_batches)
            else:
                import warnings
                warnings.warn(
                    "Could not get total iterations from trainer. "
                    "LR schedule may not work correctly.",
                    UserWarning,
                )
                total_iters = 10000

            warmup_iters = int(self.warmup_pct * total_iters)
            constant_iters = int(self.constant_pct * total_iters)
            decay_iters = int(self.decay_pct * total_iters)

            computed_total = warmup_iters + constant_iters + decay_iters
            if computed_total != total_iters:
                constant_iters += total_iters - computed_total

            warmup_end = warmup_iters
            constant_end = warmup_end + constant_iters

            warmup_start_factor = self.warmup_start_lr / self.learning_rate

            def lr_lambda_iter(step: int) -> float:
                if step < warmup_end:
                    progress = step / warmup_iters
                    return warmup_start_factor + (1.0 - warmup_start_factor) * progress
                elif step < constant_end:
                    return 1.0
                elif step < total_iters:
                    progress = (step - constant_end) / decay_iters
                    return 1.0 - progress
                else:
                    return 0.0

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_iter)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
            def lr_lambda_epoch(epoch: int) -> float:
                if epoch < self.warmup_epochs:
                    return epoch / self.warmup_epochs
                progress = (epoch - self.warmup_epochs) / (
                    self.max_epochs - self.warmup_epochs
                )
                return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_epoch)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
