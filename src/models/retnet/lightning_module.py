"""PyTorch Lightning module for RetNet in the CL pipeline.

Wraps :class:`RetNetBackbone` into the standard CL pipeline interface.
Uses the same loss computation, curriculum schedule, and evaluation protocol
as all other CL models via :class:`ACPredictorLossMixin`.

Usage:
    uv run src/cl_train.py experiment=cl_retnet paths.data_dir=/path/to/clips
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Final

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.mixins import ACPredictorLossMixin
from .backbone import RetNetBackbone

# ───────────────────────── Constants ─────────────────────────
DEFAULT_INPUT_DIM: Final = 1024
DEFAULT_EMBED_DIM: Final = 768
DEFAULT_VALUE_DIM: Final = 1280
DEFAULT_ACTION_DIM: Final = 2
DEFAULT_SPATIAL_SIZE: Final = 16
DEFAULT_N_LAYERS: Final = 6
DEFAULT_N_HEADS: Final = 4
DEFAULT_FFN_DIM: Final = 1280
DEFAULT_RECURRENT_CHUNK_SIZE: Final = 64
DEFAULT_NUM_TIMESTEPS: Final = 8
DEFAULT_DROPOUT: Final = 0.0
DEFAULT_LAYERNORM_EPS: Final = 1e-6
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
LAYER_NORM_EPS_OUTPUT: Final = 1e-6

# Type aliases
type CurriculumParams = dict[str, float | int]
type TestResult = dict[str, Any]

log = logging.getLogger(__name__)


class RetNetLitModule(ACPredictorLossMixin, L.LightningModule):
    """PyTorch Lightning module for RetNet model.

    Wraps :class:`RetNetBackbone` into the standard CL pipeline interface.

    Expected batch format (identical to other CL models):
        - features: [B, T+1, N, D]  pre-computed V-JEPA2 encoder features
        - actions:  [B, T, action_dim]
        - states:   [B, T, action_dim]

    Where T=8, N=256, D=1024.
    """

    def __init__(
        self,
        # Architecture
        input_dim: int = DEFAULT_INPUT_DIM,
        embed_dim: int = DEFAULT_EMBED_DIM,
        value_dim: int = DEFAULT_VALUE_DIM,
        action_dim: int = DEFAULT_ACTION_DIM,
        spatial_size: int = DEFAULT_SPATIAL_SIZE,
        n_layers: int = DEFAULT_N_LAYERS,
        n_heads: int = DEFAULT_N_HEADS,
        ffn_dim: int = DEFAULT_FFN_DIM,
        recurrent_chunk_size: int = DEFAULT_RECURRENT_CHUNK_SIZE,
        num_timesteps: int = DEFAULT_NUM_TIMESTEPS,
        dropout: float = DEFAULT_DROPOUT,
        layernorm_eps: float = DEFAULT_LAYERNORM_EPS,
        chunkwise_recurrent: bool = True,
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
        self.model = RetNetBackbone(
            input_dim=input_dim,
            embed_dim=embed_dim,
            value_dim=value_dim,
            action_dim=action_dim,
            spatial_size=spatial_size,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            recurrent_chunk_size=recurrent_chunk_size,
            num_timesteps=num_timesteps,
            dropout=dropout,
            layernorm_eps=layernorm_eps,
            chunkwise_recurrent=chunkwise_recurrent,
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

    # ── Curriculum helpers (same as TitansLitModule) ───────────────────

    def _validate_curriculum_schedule(self, schedule: list[dict]) -> None:
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
        target_timestep: int | None = None,
    ) -> Tensor:
        return self.model(features, actions, states, extrinsics, target_timestep=target_timestep)

    def _step_predictor(
        self,
        z: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
        target_timestep: int | None = None,
    ) -> Tensor:
        """Required by ACPredictorLossMixin."""
        z_pred = self.model(z, actions, states, extrinsics, target_timestep=target_timestep)
        if self.normalize_reps:
            z_pred = F.layer_norm(z_pred, (z_pred.size(-1),), eps=LAYER_NORM_EPS_OUTPUT)
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
        print("            RETNET TEST RESULTS SUMMARY")
        print("=" * 70)
        print(f"\n  AGGREGATE STATISTICS (over {num_clips} clips)")
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
