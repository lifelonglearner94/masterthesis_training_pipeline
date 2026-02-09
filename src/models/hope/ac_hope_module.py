"""Lightning Module for the AC-HOPE-ViT architecture.

Wraps ACHOPEViT with the same training / evaluation pipeline as ACPredictorModule:
    - Teacher-forcing + rollout losses (via ACPredictorLossMixin)
    - Test Time Adaptation (via TTAMixin)
    - Curriculum learning schedule
    - Iteration-based LR scheduler (V-JEPA2 paper)
    - HOPE-specific diagnostic logging (Criticism §1)

The module uses the same data contract:
    Input batch:
        - features: [B, T+1, N, D]  pre-computed V-JEPA2 features
        - actions:   [B, T, action_dim]
        - states:    [B, T, action_dim]

All loss computation and training logic is inherited from ACPredictorLossMixin,
ensuring scientific comparability between the original AC predictor and AC-HOPE-ViT.
"""

from __future__ import annotations

import logging
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.hope.ac_hope_vit import ac_hope_vit
from src.models.mixins import ACPredictorLossMixin, TTAMixin

log = logging.getLogger(__name__)

# Type alias
TestResultsDict = dict[str, Any]


class ACHOPEModule(TTAMixin, ACPredictorLossMixin, L.LightningModule):
    """PyTorch Lightning module for AC-HOPE-ViT predictor.

    This is structurally parallel to ACPredictorModule but uses the HOPE
    architecture (Self-Modifying Titans + CMS) instead of standard
    Transformer blocks.

    Inherits:
        - ACPredictorLossMixin: teacher-forcing + rollout loss computation
        - TTAMixin: test-time adaptation with LayerNorm parameter tuning

    Args:
        See ACHOPEViT and ACPredictorModule for parameter documentation.
    """

    def __init__(
        self,
        # Model architecture (Stage 1 & 3 — same as AC_ViT)
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
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        # HOPE-specific architecture (Stage 2)
        titan_hidden_multiplier: int = 4,
        titan_layers: int = 2,
        titan_activation: str = "gelu",
        titan_grad_clip_inner: float = 1.0,
        cms_level_specs: list[dict] | None = None,
        cms_use_chunk_scheduling: bool = False,
        chunk_size: int = 0,
        titan_detach_interval: int = 0,
        surprise_threshold: float = 0.0,
        log_hope_diagnostics: bool = True,
        # Optimizer: per-group LR/WD scaling
        titan_lr_scale: float = 0.2,  # Titan params LR = learning_rate * titan_lr_scale
        cms_lr_scale: float = 1.0,    # CMS params LR = learning_rate * cms_lr_scale
        titan_weight_decay: float | None = None,  # Separate WD for Titan (avoids double reg with inner α)
        # Loss settings (same as ACPredictorModule)
        T_teacher: int = 7,
        T_rollout: int = 2,
        context_frames: int = 1,
        loss_weight_teacher: float = 1.0,
        loss_weight_rollout: float = 1.0,
        normalize_reps: bool = True,
        loss_exp: float = 1.0,
        # Optimizer settings (same as ACPredictorModule)
        learning_rate: float = 4.25e-4,
        weight_decay: float = 0.04,
        betas: tuple[float, float] = (0.9, 0.999),
        warmup_epochs: int = 10,
        max_epochs: int = 100,
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
        tta_mode: str = "full_rollout",
        tta_num_adaptation_steps: int = 1,
        tta_adapt_layers: str = "layernorm",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Initialize TTA configuration (from TTAMixin)
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

        # ─── Build the HOPE model ───
        self.model = ac_hope_vit(
            img_size=img_size,
            patch_size=patch_size,
            num_timesteps=num_timesteps,
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=depth,
            num_heads=num_heads,
            action_embed_dim=action_embed_dim,
            use_rope=use_rope,
            is_frame_causal=is_frame_causal,
            use_activation_checkpointing=use_activation_checkpointing,
            use_extrinsics=use_extrinsics,
            titan_hidden_multiplier=titan_hidden_multiplier,
            titan_layers=titan_layers,
            titan_activation=titan_activation,
            titan_grad_clip_inner=titan_grad_clip_inner,
            cms_level_specs=cms_level_specs,
            cms_use_chunk_scheduling=cms_use_chunk_scheduling,
            chunk_size=chunk_size,
            titan_detach_interval=titan_detach_interval,
            surprise_threshold=surprise_threshold,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            log_hope_diagnostics=log_hope_diagnostics,
        )

        # ─── Loss hyperparameters (same as ACPredictorModule) ───
        self.T_teacher = T_teacher
        self.T_rollout = T_rollout
        self.context_frames = context_frames

        # Validate temporal dimensions
        max_prediction_steps = num_timesteps - 1
        if T_teacher > max_prediction_steps:
            import warnings
            warnings.warn(
                f"T_teacher ({T_teacher}) exceeds available prediction steps "
                f"({max_prediction_steps}). Will be clamped at runtime.",
                UserWarning,
            )
        if T_rollout > max_prediction_steps:
            import warnings
            warnings.warn(
                f"T_rollout ({T_rollout}) exceeds available prediction steps "
                f"({max_prediction_steps}). Will be clamped at runtime.",
                UserWarning,
            )

        self.loss_weight_teacher = loss_weight_teacher
        self.loss_weight_rollout = loss_weight_rollout
        self.normalize_reps = normalize_reps

        if loss_exp <= 0:
            raise ValueError(
                f"loss_exp must be positive (got {loss_exp}). "
                "Use 1.0 for L1, 2.0 for L2."
            )
        self.loss_exp = loss_exp

        # ─── Optimizer hyperparameters ───
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.titan_lr_scale = titan_lr_scale
        self.cms_lr_scale = cms_lr_scale
        self.titan_weight_decay = titan_weight_decay

        # Iteration-based LR schedule
        self.use_iteration_scheduler = use_iteration_scheduler
        self.warmup_pct = warmup_pct
        self.constant_pct = constant_pct
        self.decay_pct = decay_pct
        self.warmup_start_lr = warmup_start_lr

        # Grid size for reshaping
        self.grid_height = img_size[0] // patch_size
        self.grid_width = img_size[1] // patch_size
        self.patches_per_frame = self.grid_height * self.grid_width

        # Curriculum learning
        self.curriculum_schedule = curriculum_schedule
        if curriculum_schedule:
            self._validate_curriculum_schedule(curriculum_schedule)

        # HOPE diagnostics
        self.log_hope_diagnostics = log_hope_diagnostics
        self._diagnostics_log_interval = 50  # Log every N steps

        # Test results storage
        self._test_results: list[TestResultsDict] = []

    # ─── Forward pass ───

    def forward(
        self,
        features: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through the AC-HOPE-ViT predictor."""
        return self.model(features, actions, states, extrinsics)

    def _step_predictor(
        self,
        z: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
    ) -> Tensor:
        """Single predictor step with optional layer normalization."""
        z_pred = self.model(z, actions, states, extrinsics)
        if self.normalize_reps:
            z_pred = F.layer_norm(z_pred, (z_pred.size(-1),), eps=1e-6)
        return z_pred

    # ─── Training / Validation ───
    # Loss computation methods are inherited from ACPredictorLossMixin:
    #   _compute_loss, _compute_teacher_forcing_loss, _compute_rollout_loss,
    #   _compute_rollout_loss_per_timestep, _shared_step

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Training step with HOPE diagnostic logging.

        Resets all Titan memory active weights before each forward pass so
        that DGD self-modification operates on fresh detached clones, avoiding
        in-place modification errors on the outer autograd graph.
        """
        # Reset memory state before forward — creates detached active weight clones
        self.model.reset_all_memories()

        loss = self._shared_step(batch, "train")

        # Log HOPE diagnostics periodically (Criticism §1)
        if self.log_hope_diagnostics and batch_idx % self._diagnostics_log_interval == 0:
            self._log_hope_diagnostics("train")

        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Validation step."""
        # Reset memory state before forward
        self.model.reset_all_memories()
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Test step with detailed per-clip analysis.

        Mirrors ACPredictorModule.test_step() for scientific comparability.
        """
        features = batch["features"]
        actions = batch["actions"]
        states = batch["states"]
        extrinsics = batch.get("extrinsics", None)
        clip_names = batch.get(
            "clip_names",
            [f"clip_{batch_idx}_{i}" for i in range(features.shape[0])],
        )

        # Reshape if needed
        if features.dim() == 3:
            B, total_tokens, D = features.shape
            T_plus_1 = total_tokens // self.patches_per_frame
            features = features.reshape(B, T_plus_1, self.patches_per_frame, D)
        else:
            B = features.shape[0]

        # TTA support
        if self.tta_enabled:
            return self._test_step_with_tta(
                features, actions, states, extrinsics, clip_names, batch_idx
            )

        # Standard test step
        loss_teacher = self._compute_teacher_forcing_loss(
            features, actions, states, extrinsics
        )
        loss_rollout, per_timestep_losses, per_sample_losses = (
            self._compute_rollout_loss_per_timestep(
                features, actions, states, extrinsics
            )
        )
        loss = (
            self.loss_weight_teacher * loss_teacher
            + self.loss_weight_rollout * loss_rollout
        )

        self.log("test/loss_teacher", loss_teacher, prog_bar=True, sync_dist=True)
        self.log("test/loss_rollout", loss_rollout, prog_bar=True, sync_dist=True)
        self.log("test/loss", loss, prog_bar=True, sync_dist=True)

        for step, step_loss in enumerate(per_timestep_losses):
            predicted_frame = self.context_frames + step
            self.log(
                f"test/loss_step_{predicted_frame}",
                step_loss.mean(),
                sync_dist=True,
            )

        per_sample_rollout_losses = per_sample_losses[0]
        for i in range(B):
            clip_result: TestResultsDict = {
                "clip_name": (
                    clip_names[i]
                    if i < len(clip_names)
                    else f"unknown_{batch_idx}_{i}"
                ),
                "loss_rollout": per_sample_rollout_losses[i].item(),
                "loss_teacher": loss_teacher.item(),
                "per_timestep_losses": {
                    f"step_{self.context_frames + s}": per_timestep_losses[s][i].item()
                    for s in range(len(per_timestep_losses))
                },
            }
            self._test_results.append(clip_result)

        return loss

    def _test_step_with_tta(
        self,
        features: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None,
        clip_names: list[str],
        batch_idx: int,
    ) -> Tensor:
        """Test step with TTA — mirrors ACPredictorModule implementation."""
        B, T_plus_1, N, D = features.shape
        C = self.context_frames
        T_pred = min(T_plus_1 - C, self.T_rollout)

        all_losses = []
        all_per_timestep_losses: list[list[float]] = [[] for _ in range(T_pred)]

        for i in range(B):
            sample_features = features[i : i + 1]
            sample_actions = actions[i : i + 1]
            sample_states = states[i : i + 1]
            sample_extrinsics = (
                extrinsics[i : i + 1] if extrinsics is not None else None
            )
            clip_name = (
                clip_names[i]
                if i < len(clip_names)
                else f"unknown_{batch_idx}_{i}"
            )

            if self.tta_mode == "full_rollout":
                predictions, targets, tta_stats = (
                    self._tta_process_clip_full_rollout(
                        sample_features,
                        sample_actions,
                        sample_states,
                        sample_extrinsics,
                        num_adaptation_steps=self.tta_num_adaptation_steps,
                    )
                )
            else:
                predictions, targets, tta_stats = (
                    self._tta_process_clip_sequential(
                        sample_features,
                        sample_actions,
                        sample_states,
                        sample_extrinsics,
                    )
                )

            sample_loss = self._compute_loss(predictions, targets)
            all_losses.append(sample_loss.item())

            per_step_losses = {}
            for step in range(T_pred):
                step_pred = predictions[:, step * N : (step + 1) * N, :]
                step_target = targets[:, step * N : (step + 1) * N, :]
                step_loss = self._compute_loss(step_pred, step_target)
                per_step_losses[f"step_{C + step}"] = step_loss.item()
                all_per_timestep_losses[step].append(step_loss.item())

            clip_result: TestResultsDict = {
                "clip_name": clip_name,
                "loss_rollout": sample_loss.item(),
                "loss_teacher": 0.0,
                "per_timestep_losses": per_step_losses,
                "tta_stats": tta_stats,
            }
            self._test_results.append(clip_result)

        loss_rollout = sum(all_losses) / len(all_losses)
        self.log(
            "test/loss_rollout",
            loss_rollout,
            prog_bar=True,
            sync_dist=True,
        )
        self.log("test/loss", loss_rollout, prog_bar=True, sync_dist=True)

        return torch.tensor(loss_rollout, device=features.device)

    # ─── Curriculum Learning ───

    def _validate_curriculum_schedule(self, schedule: list[dict]) -> None:
        """Validate curriculum schedule format."""
        from collections.abc import Mapping, Sequence

        if (
            not isinstance(schedule, Sequence)
            or isinstance(schedule, str)
            or len(schedule) == 0
        ):
            raise ValueError("curriculum_schedule must be a non-empty list")

        max_prediction_steps = self.num_timesteps - 1

        for i, phase in enumerate(schedule):
            if not isinstance(phase, Mapping):
                raise ValueError(f"Phase {i} must be a dict, got {type(phase)}")
            if "epoch" not in phase:
                raise ValueError(f"Phase {i} must have 'epoch' key")
            if not isinstance(phase["epoch"], int) or phase["epoch"] < 0:
                raise ValueError(
                    f"Phase {i} 'epoch' must be a non-negative integer"
                )
            if "T_rollout" in phase and phase["T_rollout"] > max_prediction_steps:
                raise ValueError(
                    f"Phase {i}: T_rollout ({phase['T_rollout']}) exceeds "
                    f"max prediction steps ({max_prediction_steps})"
                )

        epochs = [p["epoch"] for p in schedule]
        if epochs != sorted(epochs):
            raise ValueError(
                "curriculum_schedule phases must be sorted by epoch"
            )

    def _get_curriculum_params_for_epoch(self, epoch: int) -> dict:
        """Get curriculum parameters for the given epoch."""
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
        """Update curriculum parameters at epoch start."""
        if not self.curriculum_schedule:
            return

        epoch = self.current_epoch
        params = self._get_curriculum_params_for_epoch(epoch)

        if not params:
            return

        changes = []

        if "T_rollout" in params and params["T_rollout"] != self.T_rollout:
            old_val = self.T_rollout
            self.T_rollout = params["T_rollout"]
            changes.append(f"T_rollout: {old_val} → {self.T_rollout}")

        if (
            "loss_weight_teacher" in params
            and params["loss_weight_teacher"] != self.loss_weight_teacher
        ):
            old_val = self.loss_weight_teacher
            self.loss_weight_teacher = params["loss_weight_teacher"]
            changes.append(
                f"loss_weight_teacher: {old_val} → {self.loss_weight_teacher}"
            )

        if (
            "loss_weight_rollout" in params
            and params["loss_weight_rollout"] != self.loss_weight_rollout
        ):
            old_val = self.loss_weight_rollout
            self.loss_weight_rollout = params["loss_weight_rollout"]
            changes.append(
                f"loss_weight_rollout: {old_val} → {self.loss_weight_rollout}"
            )

        if changes:
            log.info(f"[Curriculum] Epoch {epoch}: {', '.join(changes)}")

        self.log("curriculum/T_rollout", float(self.T_rollout), sync_dist=True)
        self.log(
            "curriculum/loss_weight_teacher",
            self.loss_weight_teacher,
            sync_dist=True,
        )
        self.log(
            "curriculum/loss_weight_rollout",
            self.loss_weight_rollout,
            sync_dist=True,
        )

    # ─── HOPE Diagnostics (Criticism §1) ───

    def _log_hope_diagnostics(self, stage: str) -> None:
        """Log HOPE-specific diagnostic metrics for monitoring training stability.

        Logs:
            - Titan inner-loop gradient norms (detects exploding/vanishing inner gradients)
            - Surprise values (monitors memory utilization)
            - Memory parameter norms (detects weight explosion)
            - Model configuration summary (logged once at start)
        """
        diag = self.model.get_all_diagnostics()
        for key, val in diag.items():
            self.log(f"{stage}/{key}", val, sync_dist=True)

    def on_fit_start(self) -> None:
        """Log configuration summary at training start (Criticism §1)."""
        config = self.model.get_config_summary()
        log.info(f"[AC-HOPE-ViT] Configuration: {config}")

        # Log scalar config values to wandb for reproducibility
        # Note: self.log() is not allowed in on_fit_start, use logger directly
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
            self._tta_clip_stats = {
                "adaptation_losses": [],
                "num_adaptations": 0,
            }
            log.info(
                f"[TTA] Test epoch started. LR={self.tta_lr}, "
                f"grad_clip={self.tta_grad_clip}, "
                f"reset_per_clip={self.tta_reset_per_clip}"
            )

    def on_test_epoch_end(self) -> None:
        """Aggregate and log test results (mirrors ACPredictorModule)."""
        import json
        from pathlib import Path

        if not self._test_results:
            log.warning("No test results to aggregate")
            return

        num_clips = len(self._test_results)
        rollout_losses = [r["loss_rollout"] for r in self._test_results]

        mean_loss = sum(rollout_losses) / num_clips
        sorted_losses = sorted(rollout_losses)
        median_loss = sorted_losses[num_clips // 2]
        min_loss = min(rollout_losses)
        max_loss = max(rollout_losses)
        std_loss = (
            sum((x - mean_loss) ** 2 for x in rollout_losses) / num_clips
        ) ** 0.5

        # Per-timestep statistics
        num_ts = len(self._test_results[0]["per_timestep_losses"])
        per_timestep_stats = {}
        for step_key in self._test_results[0]["per_timestep_losses"].keys():
            step_losses = [
                r["per_timestep_losses"][step_key] for r in self._test_results
            ]
            per_timestep_stats[step_key] = {
                "mean": sum(step_losses) / num_clips,
                "min": min(step_losses),
                "max": max(step_losses),
            }

        worst_clips = sorted(
            self._test_results, key=lambda x: x["loss_rollout"], reverse=True
        )[:10]
        best_clips = sorted(
            self._test_results, key=lambda x: x["loss_rollout"]
        )[:5]

        print("\n" + "=" * 70)
        print("              AC-HOPE-ViT TEST RESULTS SUMMARY")
        print("=" * 70)
        print(f"\n📊 AGGREGATE STATISTICS (over {num_clips} clips)")
        print("-" * 50)
        print(f"  Rollout Loss (L1):")
        print(f"    Mean:   {mean_loss:.6f}")
        print(f"    Median: {median_loss:.6f}")
        print(f"    Std:    {std_loss:.6f}")
        print(f"    Min:    {min_loss:.6f}")
        print(f"    Max:    {max_loss:.6f}")

        print(
            f"\n📈 PER-TIMESTEP BREAKDOWN (Context: {self.context_frames} frames)"
        )
        print("-" * 50)
        print(f"  {'Frame':<10} {'Mean Loss':<12} {'Min':<12} {'Max':<12}")
        for step_key, stats in per_timestep_stats.items():
            frame_num = step_key.replace("step_", "z_")
            print(
                f"  {frame_num:<10} {stats['mean']:<12.6f} "
                f"{stats['min']:<12.6f} {stats['max']:<12.6f}"
            )

        print(f"\n🔴 WORST-PERFORMING CLIPS (top 10)")
        for i, clip in enumerate(worst_clips, 1):
            print(
                f"  {i:2d}. {clip['clip_name']:<20} loss={clip['loss_rollout']:.6f}"
            )

        print(f"\n🟢 BEST-PERFORMING CLIPS (top 5)")
        for i, clip in enumerate(best_clips, 1):
            print(
                f"  {i:2d}. {clip['clip_name']:<20} loss={clip['loss_rollout']:.6f}"
            )
        print("=" * 70)

        # Export to JSON
        output_dir = Path(".")
        if self.trainer and hasattr(self.trainer, "log_dir") and self.trainer.log_dir:
            output_dir = Path(self.trainer.log_dir)

        results_file = output_dir / "test_results.json"
        export_data = {
            "model_type": "AC-HOPE-ViT",
            "summary": {
                "num_clips": num_clips,
                "context_frames": self.context_frames,
                "T_rollout": self.T_rollout,
                "rollout_loss": {
                    "mean": mean_loss,
                    "median": median_loss,
                    "std": std_loss,
                    "min": min_loss,
                    "max": max_loss,
                },
                "per_timestep": per_timestep_stats,
            },
            "worst_clips": worst_clips,
            "best_clips": best_clips,
            "all_clips": self._test_results,
        }

        try:
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
            print(f"\n💾 Results exported to: {results_file}")
        except Exception as e:
            log.warning(f"Failed to export results: {e}")

        self.log("test/final_mean_loss_rollout", mean_loss, sync_dist=True)
        self.log("test/final_median_loss_rollout", median_loss, sync_dist=True)

    # ─── Optimizer ───

    def configure_optimizers(self) -> dict:
        """Configure optimizer with per-group learning rates.

        Uses AdamW with 3 parameter groups:
            1. Titan memories: LR * titan_lr_scale (default 0.5×)
            2. CMS blocks: LR * cms_lr_scale (default 1.0×)
            3. Projections/embeddings: LR (full)

        This is important because Titan memories use inner-loop optimization
        (DGD), so their outer-loop learning rate should be more conservative
        to avoid instability (Criticism §1).
        """
        param_groups = self.model.get_parameter_groups()

        optimizer_groups = []
        for group in param_groups:
            name = group.get("group_name", "default")
            params = group["params"]
            if not params:
                continue

            if name == "titan":
                lr = self.learning_rate * self.titan_lr_scale
                # Titan memories have inner-loop α (weight decay) via DGD,
                # so outer weight decay should be reduced to avoid double
                # regularization. Use titan_weight_decay if set, else default.
                wd = self.titan_weight_decay if self.titan_weight_decay is not None else self.weight_decay
            elif name == "cms":
                lr = self.learning_rate * self.cms_lr_scale
                wd = self.weight_decay
            else:
                lr = self.learning_rate
                wd = self.weight_decay

            optimizer_groups.append({
                "params": params,
                "lr": lr,
                "weight_decay": wd,
            })

            n_params = sum(p.numel() for p in params)
            log.info(
                f"[Optimizer] Group '{name}': {n_params:,} params, "
                f"LR={lr:.2e}, WD={wd}"
            )

        optimizer = torch.optim.AdamW(
            optimizer_groups,
            betas=self.betas,
        )

        if self.use_iteration_scheduler:
            # Iteration-based: Warmup → Constant → Decay (V-JEPA2 paper)
            if (
                self.trainer is not None
                and self.trainer.estimated_stepping_batches
            ):
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

            log.info(
                f"[LR Schedule] Total iters: {total_iters}, "
                f"warmup: {warmup_iters}, constant: {constant_iters}, "
                f"decay: {decay_iters}"
            )

            def lr_lambda_iter(step: int) -> float:
                if step < warmup_end:
                    progress = step / max(warmup_iters, 1)
                    return warmup_start_factor + (
                        1.0 - warmup_start_factor
                    ) * progress
                elif step < constant_end:
                    return 1.0
                elif step < total_iters:
                    progress = (step - constant_end) / max(decay_iters, 1)
                    return 1.0 - progress
                else:
                    return 0.0

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda_iter
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
            # Epoch-based: Cosine annealing with linear warmup
            def lr_lambda_epoch(epoch: int) -> float:
                if epoch < self.warmup_epochs:
                    return epoch / max(self.warmup_epochs, 1)
                progress = (epoch - self.warmup_epochs) / max(
                    self.max_epochs - self.warmup_epochs, 1
                )
                return 0.5 * (
                    1.0
                    + torch.cos(torch.tensor(torch.pi * progress)).item()
                )

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda_epoch
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
