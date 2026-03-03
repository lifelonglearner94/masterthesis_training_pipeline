"""Lightning Module for the AC-HOPE-Hybrid-ViT architecture (Phase 8).

Wraps ACHOPEHybridViT with the same training / evaluation pipeline as
ACPredictorModule and ACHOPEModule:
    - Teacher-forcing + jump prediction losses (via ACPredictorLossMixin)
    - Test Time Adaptation (via TTAMixin)
    - Curriculum learning schedule
    - Iteration-based LR scheduler (V-JEPA2 paper)
    - HOPE-specific diagnostic logging

The module uses the same data contract as ACPredictorModule:
    Input batch:
        - features: [B, T+1, N, D]  pre-computed V-JEPA2 features
        - actions:   [B, T, action_dim]
        - states:    [B, T, action_dim]
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.hope.ac_hope_hybrid_vit import ac_hope_hybrid_vit
from src.models.mixins import ACPredictorLossMixin, TTAMixin

log = logging.getLogger(__name__)

type TestResultsDict = dict[str, Any]


class InvalidConfigurationError(ValueError):
    """Raised when module configuration is invalid."""


class ACHOPEHybridModule(TTAMixin, ACPredictorLossMixin, L.LightningModule):
    """PyTorch Lightning module for AC-HOPE-Hybrid-ViT predictor.

    Structurally parallel to ACHOPEModule but uses the Hybrid architecture
    (Attention + Titan Memory + CMS) instead of pure HOPE blocks.

    Inherits:
        - ACPredictorLossMixin: teacher-forcing + jump prediction loss
        - TTAMixin: test-time adaptation
    """

    DEFAULT_DIAGNOSTICS_LOG_INTERVAL: int = 50
    DEFAULT_TOTAL_ITERS_FALLBACK: int = 10000
    LAYER_NORM_EPS: float = 1e-6

    def __init__(
        self,
        # Model architecture
        img_size: tuple[int, int] = (256, 256),
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
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        # Titan memory settings
        titan_hidden_multiplier: int = 2,
        titan_layers: int = 2,
        titan_activation: str = "gelu",
        titan_grad_clip_inner: float = 1.0,
        titan_grad_clip_backward: float = 1.0,
        cms_level_specs: list[dict] | None = None,
        cms_use_chunk_scheduling: bool = False,
        titan_detach_interval: int = 0,
        surprise_threshold: float = 0.0,
        log_hope_diagnostics: bool = True,
        diagnostics_log_interval: int = DEFAULT_DIAGNOSTICS_LOG_INTERVAL,
        # Attention settings
        qkv_bias: bool = True,
        # Longterm memory
        use_longterm_memory: bool = False,
        longterm_hidden_multiplier: int = 2,
        longterm_lr_scale: float = 0.1,
        # Optimizer: per-group LR/WD scaling
        attention_lr_scale: float = 1.0,
        titan_lr_scale: float = 0.1,
        cms_lr_scale: float = 0.6,
        projections_lr_scale: float = 1.0,
        titan_weight_decay: float | None = None,
        aux_loss_weight: float = 0.0,  # Not needed for hybrid
        # Loss settings
        T_teacher: int = 7,
        jump_k: int = 3,
        loss_weight_teacher: float = 1.0,
        loss_weight_jump: float = 1.0,
        normalize_reps: bool = True,
        loss_type: str = "l1",
        huber_delta: float = 1.0,
        # Optimizer settings
        learning_rate: float = 4.25e-4,
        weight_decay: float = 0.04,
        betas: tuple[float, float] = (0.9, 0.999),
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        optimizer_type: str = "adamw",
        # Iteration-based LR schedule
        use_iteration_scheduler: bool = False,
        curriculum_schedule: list[dict] | None = None,
        warmup_pct: float = 0.085,
        constant_pct: float = 0.83,
        decay_pct: float = 0.085,
        warmup_start_lr: float = 7.5e-5,
        # TTA settings
        tta_enabled: bool = False,
        tta_lr: float = 1e-4,
        tta_grad_clip: float = 1.0,
        tta_reset_per_clip: bool = True,
        tta_adaptation_horizon: int = 1,
        tta_optimizer_type: str = "adam",
        tta_optimizer_betas: tuple[float, float] = (0.9, 0.999),
        tta_mode: str = "jump",
        tta_num_adaptation_steps: int = 1,
        tta_adapt_layers: str = "layernorm",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self._init_tta(
            tta_enabled=tta_enabled,
            tta_lr=tta_lr,
            tta_grad_clip=tta_grad_clip,
            tta_reset_per_clip=tta_reset_per_clip,
            tta_adaptation_horizon=tta_adaptation_horizon,
            tta_optimizer_type=tta_optimizer_type,
            tta_optimizer_betas=tta_optimizer_betas,
            tta_mode=tta_mode,
            tta_num_adaptation_steps=tta_num_adaptation_steps,
            tta_adapt_layers=tta_adapt_layers,
        )

        self.num_timesteps = num_timesteps

        self.model = ac_hope_hybrid_vit(
            img_size=img_size,
            patch_size=patch_size,
            num_timesteps=num_timesteps,
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=depth,
            num_heads=num_heads,
            action_embed_dim=action_embed_dim,
            is_frame_causal=is_frame_causal,
            use_activation_checkpointing=use_activation_checkpointing,
            use_extrinsics=use_extrinsics,
            titan_hidden_multiplier=titan_hidden_multiplier,
            titan_layers=titan_layers,
            titan_activation=titan_activation,
            titan_grad_clip_inner=titan_grad_clip_inner,
            titan_grad_clip_backward=titan_grad_clip_backward,
            cms_level_specs=cms_level_specs,
            cms_use_chunk_scheduling=cms_use_chunk_scheduling,
            titan_detach_interval=titan_detach_interval,
            surprise_threshold=surprise_threshold,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            log_hope_diagnostics=log_hope_diagnostics,
            use_longterm_memory=use_longterm_memory,
            longterm_hidden_multiplier=longterm_hidden_multiplier,
            longterm_lr_scale=longterm_lr_scale,
            qkv_bias=qkv_bias,
        )

        self.T_teacher = T_teacher
        self.jump_k = jump_k
        self._validate_temporal_dimensions(T_teacher, jump_k)

        self.loss_weight_teacher = loss_weight_teacher
        self.loss_weight_jump = loss_weight_jump
        self.normalize_reps = normalize_reps

        from src.models.mixins.loss_mixin import VALID_LOSS_TYPES
        if loss_type not in VALID_LOSS_TYPES:
            raise InvalidConfigurationError(
                f"loss_type must be one of {VALID_LOSS_TYPES} (got '{loss_type}')."
            )
        self.loss_type = loss_type
        self.huber_delta = huber_delta

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.attention_lr_scale = attention_lr_scale
        self.titan_lr_scale = titan_lr_scale
        self.cms_lr_scale = cms_lr_scale
        self.projections_lr_scale = projections_lr_scale
        self.titan_weight_decay = titan_weight_decay
        self.aux_loss_weight = aux_loss_weight
        self.optimizer_type = optimizer_type

        self.use_iteration_scheduler = use_iteration_scheduler
        self.warmup_pct = warmup_pct
        self.constant_pct = constant_pct
        self.decay_pct = decay_pct
        self.warmup_start_lr = warmup_start_lr

        self.grid_height = img_size[0] // patch_size
        self.grid_width = img_size[1] // patch_size
        self.patches_per_frame = self.grid_height * self.grid_width

        self.curriculum_schedule = curriculum_schedule
        if curriculum_schedule:
            self._validate_curriculum_schedule(curriculum_schedule)

        self.log_hope_diagnostics = log_hope_diagnostics
        self._diagnostics_log_interval = diagnostics_log_interval

        self._test_results: list[TestResultsDict] = []

        # CL compatibility flag
        self.skip_aux_loss: bool = False

    def _validate_temporal_dimensions(self, T_teacher: int, jump_k: int) -> None:
        """Validate temporal dimensions are within bounds."""
        max_steps = self.num_timesteps - 1
        if T_teacher > max_steps:
            warnings.warn(
                f"T_teacher ({T_teacher}) exceeds available prediction steps ({max_steps}).",
                UserWarning, stacklevel=2,
            )
        if jump_k > self.num_timesteps:
            warnings.warn(
                f"jump_k ({jump_k}) exceeds num_timesteps ({self.num_timesteps}).",
                UserWarning, stacklevel=2,
            )

    # ─── Forward pass ───

    def forward(
        self, features: Tensor, actions: Tensor, states: Tensor,
        extrinsics: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through the Hybrid predictor."""
        return self.model(features, actions, states, extrinsics)

    def _step_predictor(
        self, z: Tensor, actions: Tensor, states: Tensor,
        extrinsics: Tensor | None = None,
        target_timestep: int | None = None,
    ) -> Tensor:
        """Single predictor step with optional layer normalization."""
        z_pred = self.model(z, actions, states, extrinsics, target_timestep=target_timestep)
        if self.normalize_reps:
            z_pred = F.layer_norm(z_pred, (z_pred.size(-1),), eps=self.LAYER_NORM_EPS)
        return z_pred

    # ─── Training / Validation ───

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Training step with memory reset and diagnostic logging."""
        self.model.reset_all_memories()
        loss = self._shared_step(batch, "train")

        # Hybrid doesn't need aux loss, but log for consistency
        if self.log_hope_diagnostics and batch_idx % self._diagnostics_log_interval == 0:
            self._log_diagnostics("train")

        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Validation step."""
        self.model.reset_all_memories()
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Test step with per-clip analysis."""
        features = batch["features"]
        actions = batch["actions"]
        states = batch["states"]
        extrinsics = batch.get("extrinsics", None)
        clip_names = batch.get(
            "clip_names",
            [f"clip_{batch_idx}_{i}" for i in range(features.shape[0])],
        )

        if features.dim() == 3:
            B, total_tokens, D = features.shape
            T_plus_1 = total_tokens // self.patches_per_frame
            features = features.reshape(B, T_plus_1, self.patches_per_frame, D)
        else:
            B = features.shape[0]

        if self.tta_enabled:
            return self._test_step_with_tta(
                features, actions, states, extrinsics, clip_names, batch_idx
            )

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

        T = self.num_timesteps
        k = min(self.jump_k, T)
        tau_min = T + 1 - k
        for step, step_loss in enumerate(per_timestep_losses):
            target_frame = tau_min + step
            self.log(f"test/loss_jump_tau_{target_frame}", step_loss.mean(), sync_dist=True, batch_size=bs)

        per_sample_jump_losses = per_sample_losses[0]
        for i in range(B):
            clip_result: TestResultsDict = {
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

    def _test_step_with_tta(
        self, features: Tensor, actions: Tensor, states: Tensor,
        extrinsics: Tensor | None, clip_names: list[str], batch_idx: int,
    ) -> Tensor:
        """Test step with TTA."""
        B, T_plus_1, N, D = features.shape
        T = T_plus_1 - 1
        k = min(self.jump_k, T)
        T_pred = k

        all_losses: list[float] = []
        all_per_timestep_losses: list[list[float]] = [[] for _ in range(T_pred)]

        for i in range(B):
            sf = features[i:i+1]
            sa = actions[i:i+1]
            ss = states[i:i+1]
            se = extrinsics[i:i+1] if extrinsics is not None else None
            cn = clip_names[i] if i < len(clip_names) else f"unknown_{batch_idx}_{i}"

            if self.tta_mode == "jump":
                predictions, targets, tta_stats = self._tta_process_clip_jump(
                    sf, sa, ss, se, num_adaptation_steps=self.tta_num_adaptation_steps,
                )
            else:
                predictions, targets, tta_stats = self._tta_process_clip_sequential(
                    sf, sa, ss, se,
                )

            sample_loss = self._compute_loss(predictions, targets)
            all_losses.append(sample_loss.item())

            per_step_losses: dict[str, float] = {}
            tau_min = T + 1 - k
            for step in range(T_pred):
                step_pred = predictions[:, step*N:(step+1)*N, :]
                step_target = targets[:, step*N:(step+1)*N, :]
                step_loss = self._compute_loss(step_pred, step_target)
                per_step_losses[f"tau_{tau_min + step}"] = step_loss.item()
                all_per_timestep_losses[step].append(step_loss.item())

            self._test_results.append({
                "clip_name": cn,
                "loss_jump": sample_loss.item(),
                "loss_teacher": 0.0,
                "per_timestep_losses": per_step_losses,
                "tta_stats": tta_stats,
            })

        loss_jump = sum(all_losses) / len(all_losses)
        self.log("test/loss_jump", loss_jump, prog_bar=True, sync_dist=True, batch_size=B)
        self.log("test/loss", loss_jump, prog_bar=True, sync_dist=True, batch_size=B)
        return torch.tensor(loss_jump, device=features.device)

    # ─── Curriculum Learning ───

    def _validate_curriculum_schedule(self, schedule: list[dict]) -> None:
        """Validate curriculum schedule format."""
        if not isinstance(schedule, Sequence) or isinstance(schedule, str) or len(schedule) == 0:
            raise InvalidConfigurationError("curriculum_schedule must be a non-empty list")
        for i, phase in enumerate(schedule):
            if not isinstance(phase, Mapping):
                raise InvalidConfigurationError(f"Phase {i} must be a dict")
            if "epoch" not in phase:
                raise InvalidConfigurationError(f"Phase {i} must have 'epoch' key")
            if not isinstance(phase["epoch"], int) or phase["epoch"] < 0:
                raise InvalidConfigurationError(f"Phase {i} 'epoch' must be non-negative int")
        epochs = [p["epoch"] for p in schedule]
        if epochs != sorted(epochs):
            raise InvalidConfigurationError("curriculum phases must be sorted by epoch")

    def _get_curriculum_params_for_epoch(self, epoch: int) -> dict[str, Any]:
        """Get curriculum parameters for given epoch."""
        if not self.curriculum_schedule:
            return {}
        applicable = None
        for phase in self.curriculum_schedule:
            if phase["epoch"] <= epoch:
                applicable = phase
            else:
                break
        if applicable is None:
            return {}
        return {k: v for k, v in applicable.items() if k != "epoch"}

    def on_train_epoch_start(self) -> None:
        """Update curriculum parameters at epoch start."""
        if not self.curriculum_schedule or getattr(self, "disable_curriculum", False):
            return
        epoch = self.current_epoch
        params = self._get_curriculum_params_for_epoch(epoch)
        if not params:
            return
        changes: list[str] = []
        if "loss_weight_teacher" in params and params["loss_weight_teacher"] != self.loss_weight_teacher:
            old = self.loss_weight_teacher
            self.loss_weight_teacher = params["loss_weight_teacher"]
            changes.append(f"loss_weight_teacher: {old} → {self.loss_weight_teacher}")
        if "loss_weight_jump" in params and params["loss_weight_jump"] != self.loss_weight_jump:
            old = self.loss_weight_jump
            self.loss_weight_jump = params["loss_weight_jump"]
            changes.append(f"loss_weight_jump: {old} → {self.loss_weight_jump}")
        if changes:
            log.info(f"[Curriculum] Epoch {epoch}: {', '.join(changes)}")

    # ─── Diagnostics ───

    def _log_diagnostics(self, stage: str) -> None:
        """Log diagnostic metrics."""
        diag = self.model.get_all_diagnostics()
        for key, val in diag.items():
            self.log(f"{stage}/{key}", val, sync_dist=True)

    def on_fit_start(self) -> None:
        """Log configuration summary at training start."""
        config = self.model.get_config_summary()
        log.info(f"[AC-HOPE-Hybrid-ViT] Configuration: {config}")
        if self.logger:
            for key, val in config.items():
                if isinstance(val, (int, float)):
                    self.logger.experiment.log({f"config/{key}": float(val)})

    # ─── Test epoch hooks ───

    def on_test_epoch_start(self) -> None:
        """Clear test results and configure TTA."""
        self._test_results = []
        if self.tta_enabled:
            self._tta_configure_params()
            self._tta_original_ln_state = self._tta_save_ln_state()
            self._tta_all_clip_stats = []
            self._tta_optimizer = self._tta_create_optimizer()
            self._tta_clip_stats = {"adaptation_losses": [], "num_adaptations": 0}

    def on_test_epoch_end(self) -> None:
        """Aggregate and log test results."""
        if not self._test_results:
            log.warning("No test results to aggregate")
            return

        num_clips = len(self._test_results)
        jump_losses = [r["loss_jump"] for r in self._test_results]
        mean_loss = sum(jump_losses) / num_clips
        sorted_losses = sorted(jump_losses)
        median_loss = sorted_losses[num_clips // 2]

        per_timestep_stats: dict[str, dict[str, float]] = {}
        for step_key in self._test_results[0]["per_timestep_losses"].keys():
            step_losses = [r["per_timestep_losses"][step_key] for r in self._test_results]
            per_timestep_stats[step_key] = {
                "mean": sum(step_losses) / num_clips,
                "min": min(step_losses),
                "max": max(step_losses),
            }

        worst_clips = sorted(self._test_results, key=lambda x: x["loss_jump"], reverse=True)[:10]
        best_clips = sorted(self._test_results, key=lambda x: x["loss_jump"])[:5]

        print(f"\n{'='*70}")
        print("         AC-HOPE-HYBRID-ViT TEST RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"\n  Aggregate (over {num_clips} clips)")
        print(f"    Mean:   {mean_loss:.6f}")
        print(f"    Median: {median_loss:.6f}")
        for step_key, stats in per_timestep_stats.items():
            print(f"    {step_key}: mean={stats['mean']:.6f}")
        print(f"{'='*70}")

        # Export
        output_dir = Path(".")
        if self.trainer and hasattr(self.trainer, "log_dir") and self.trainer.log_dir:
            output_dir = Path(self.trainer.log_dir)
        results_file = output_dir / "test_results.json"
        try:
            results_file.parent.mkdir(parents=True, exist_ok=True)
            results_file.write_text(json.dumps({
                "model_type": "AC-HOPE-Hybrid-ViT",
                "summary": {
                    "num_clips": num_clips,
                    "jump_loss": {"mean": mean_loss, "median": median_loss},
                    "per_timestep": per_timestep_stats,
                },
                "worst_clips": worst_clips,
                "best_clips": best_clips,
                "all_clips": self._test_results,
            }, indent=2, default=str))
        except OSError as e:
            log.warning(f"Failed to export results: {e}")

        self.log("test/final_mean_loss_jump", mean_loss, sync_dist=True)
        self.log("test/final_median_loss_jump", median_loss, sync_dist=True)

    # ─── Optimizer ───

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure AdamW optimizer with per-group learning rates."""
        param_groups = self.model.get_parameter_groups()

        optimizer_groups: list[dict[str, Any]] = []
        for group in param_groups:
            name = group.get("group_name", "default")
            params = group["params"]
            if not params:
                continue

            if name == "attention":
                lr = self.learning_rate * self.attention_lr_scale
                wd = self.weight_decay
            elif name == "titan":
                lr = self.learning_rate * self.titan_lr_scale
                wd = self.titan_weight_decay if self.titan_weight_decay is not None else self.weight_decay
            elif name == "cms":
                lr = self.learning_rate * self.cms_lr_scale
                wd = self.weight_decay
            else:  # projections
                lr = self.learning_rate * self.projections_lr_scale
                wd = self.weight_decay

            optimizer_groups.append({"params": params, "lr": lr, "weight_decay": wd})
            n_params = sum(p.numel() for p in params)
            log.info(f"[Optimizer] Group '{name}': {n_params:,} params, LR={lr:.2e}, WD={wd}")

        optimizer = torch.optim.AdamW(optimizer_groups, betas=self.betas)

        if self.use_iteration_scheduler:
            return self._configure_iteration_scheduler(optimizer)
        else:
            return self._configure_epoch_scheduler(optimizer)

    def _configure_iteration_scheduler(self, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        """Configure iteration-based LR scheduler."""
        if self.trainer is not None and self.trainer.estimated_stepping_batches:
            total_iters = int(self.trainer.estimated_stepping_batches)
        else:
            warnings.warn("Could not get total iterations from trainer.", UserWarning, stacklevel=2)
            total_iters = self.DEFAULT_TOTAL_ITERS_FALLBACK

        warmup_iters = int(self.warmup_pct * total_iters)
        constant_iters = int(self.constant_pct * total_iters)
        decay_iters = int(self.decay_pct * total_iters)
        computed_total = warmup_iters + constant_iters + decay_iters
        if computed_total != total_iters:
            constant_iters += total_iters - computed_total

        warmup_end = warmup_iters
        constant_end = warmup_end + constant_iters
        warmup_start_factor = self.warmup_start_lr / self.learning_rate

        log.info(
            f"[LR Schedule] Total: {total_iters}, warmup: {warmup_iters}, "
            f"constant: {constant_iters}, decay: {decay_iters}"
        )

        def lr_lambda_iter(step: int) -> float:
            if step < warmup_end:
                progress = step / max(warmup_iters, 1)
                return warmup_start_factor + (1.0 - warmup_start_factor) * progress
            elif step < constant_end:
                return 1.0
            elif step < total_iters:
                progress = (step - constant_end) / max(decay_iters, 1)
                return 1.0 - progress
            return 0.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_iter)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def _configure_epoch_scheduler(self, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        """Configure epoch-based LR scheduler with cosine annealing."""
        def lr_lambda_epoch(epoch: int) -> float:
            if epoch < self.warmup_epochs:
                return epoch / max(self.warmup_epochs, 1)
            progress = (epoch - self.warmup_epochs) / max(self.max_epochs - self.warmup_epochs, 1)
            return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_epoch)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
