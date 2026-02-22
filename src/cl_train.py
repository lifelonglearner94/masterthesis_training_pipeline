"""Continual Learning Training Pipeline.

Orchestrates a full CL experiment:
    1. Base Training (5000 clips, configurable epochs)
    2. Sequential Tasks (1000 clips each, 5 tasks)
    3. Full Evaluation after each phase (fixed eval clips, no weight updates)

Creates separate W&B runs for each phase, grouped under a single experiment group.

Computes CL metrics: FWT, BWT, ExperienceForgetting, StreamForgetting, Top1_L1_Exp/Stream
via the R-matrix approach.

Usage:
    uv run src/cl_train.py experiment=cl_ac_vit paths.data_dir=/path/to/clips
    uv run src/cl_train.py experiment=cl_ac_hope paths.data_dir=/path/to/clips
"""

import pyrootutils

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from dotenv import load_dotenv

load_dotenv()

import copy
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import omegaconf
import torch
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from src.utils.cl_metrics import ContinualLearningMetricsTracker
from src.utils.device_utils import log_device_info

# Fix for PyTorch 2.6+ security update
torch.serialization.add_safe_globals([
    omegaconf.listconfig.ListConfig,
    omegaconf.dictconfig.DictConfig,
    omegaconf.base.ContainerMetadata,
    Any,
])

log = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def create_wandb_logger(
    cfg: DictConfig,
    run_name: str,
    group: str,
    job_type: str,
    tags: list[str] | None = None,
    save_dir: str | None = None,
) -> WandbLogger:
    """Create a fresh WandbLogger for a CL phase.

    Each phase gets its own W&B run, all grouped under the same experiment group.
    """
    wandb_cfg = cfg.get("logger", {}).get("wandb", {})
    project = wandb_cfg.get(
        "project", os.environ.get("WANDB_PROJECT", "hope-vs-vit-research")
    )
    entity = wandb_cfg.get("entity", os.environ.get("WANDB_ENTITY", None))

    return WandbLogger(
        project=project,
        entity=entity if entity != "null" else None,
        name=run_name,
        group=group,
        job_type=job_type,
        tags=tags or [],
        save_dir=save_dir or str(Path(cfg.paths.output_dir) / "wandb"),
        log_model=False,
    )


def create_trainer(
    cfg: DictConfig,
    wandb_logger: WandbLogger,
    max_epochs: int,
    output_dir: str,
    enable_checkpointing: bool = True,
    log_every_n_steps: int = 50,
    gradient_clip_val: float | None = None,
    precision: str | int = "16-mixed",
) -> Trainer:
    """Create a Trainer for a single CL phase."""
    callbacks: list[Callback] = [RichProgressBar()]

    if enable_checkpointing:
        callbacks.append(
            ModelCheckpoint(
                dirpath=f"{output_dir}/checkpoints",
                filename="checkpoint_e{epoch:04d}",
                save_last=True,
                save_top_k=-1,
                every_n_epochs=max(1, max_epochs // 5),
                save_weights_only=False,
            )
        )

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        precision=precision,
        deterministic=cfg.get("deterministic", True),
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm="norm",
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=log_every_n_steps,
        enable_checkpointing=enable_checkpointing,
        check_val_every_n_epoch=1,
    )

    return trainer


def create_datamodule(
    cfg: DictConfig,
    clip_start: int,
    clip_end: int,
    batch_size: int | None = None,
    val_split: float = 0.0,
    shuffle_test: bool = False,
) -> LightningDataModule:
    """Create a DataModule for a specific clip range.

    Uses Hydra instantiation with overrides for the clip range.
    """
    data_cfg = copy.deepcopy(cfg.data)
    data_cfg.clip_start = clip_start
    data_cfg.clip_end = clip_end
    data_cfg.val_split = val_split
    if batch_size is not None:
        data_cfg.batch_size = batch_size
    data_cfg.shuffle_test = shuffle_test

    return hydra.utils.instantiate(data_cfg)


def instantiate_model(cfg: DictConfig) -> LightningModule:
    """Instantiate the model from config."""
    return hydra.utils.instantiate(cfg.model)


def load_checkpoint_weights(
    model: LightningModule, ckpt_path: str
) -> None:
    """Load model weights from a checkpoint (weights only, no optimizer state)."""
    log.info(f"Loading checkpoint weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    log.info("Checkpoint weights loaded successfully")


def save_model_checkpoint(model: LightningModule, path: str) -> None:
    """Save a minimal checkpoint with just model weights."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, path)
    log.info(f"Model checkpoint saved to: {path}")


def finish_wandb(logger: WandbLogger) -> None:
    """Safely finish a W&B run."""
    try:
        if logger.experiment is not None:
            import wandb
            wandb.finish()
    except Exception as e:
        log.warning(f"Failed to finish W&B run: {e}")


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_all_tasks(
    model: LightningModule,
    cfg: DictConfig,
    tracker: ContinualLearningMetricsTracker,
    train_exp_id: int,
    phase_name: str,
    wandb_group: str,
    output_dir: str,
    is_hope: bool = False,
) -> dict[str, float]:
    """Run full evaluation on all data partitions (base + tasks).

    No weight updates! For HOPE, inner-loop DGD is frozen.

    Args:
        model: The model to evaluate.
        cfg: Full Hydra config.
        tracker: CL metrics tracker to update.
        train_exp_id: Current training experience index.
        phase_name: Human-readable phase name for W&B run.
        wandb_group: W&B group name.
        output_dir: Directory for evaluation outputs.
        is_hope: Whether this is a HOPE model (needs inner-loop freezing).

    Returns:
        Dictionary of all CL metrics computed after this evaluation.
    """
    cl_cfg = cfg.cl
    tasks = cl_cfg.tasks
    eval_clips = cl_cfg.eval.clips_per_task

    # Build list of all evaluation partitions
    eval_partitions = []

    # Base training partition eval subset
    base_end = cl_cfg.base_training.clip_end
    base_eval_start = base_end - eval_clips
    eval_partitions.append({
        "name": "base",
        "clip_start": base_eval_start,
        "clip_end": base_end,
        "exp_id": 0,
    })

    # Task partitions eval subsets
    for i, task in enumerate(tasks):
        task_eval_start = task.clip_end - eval_clips
        eval_partitions.append({
            "name": task.name,
            "clip_start": task_eval_start,
            "clip_end": task.clip_end,
            "exp_id": i + 1,
        })

    # Create W&B logger for this evaluation phase
    eval_tags = list(cfg.get("tags", [])) + ["eval", phase_name]
    wandb_logger = create_wandb_logger(
        cfg,
        run_name=f"eval_{phase_name}",
        group=wandb_group,
        job_type="evaluation",
        tags=eval_tags,
        save_dir=f"{output_dir}/eval_{phase_name}",
    )

    eval_trainer = create_trainer(
        cfg,
        wandb_logger,
        max_epochs=0,
        output_dir=f"{output_dir}/eval_{phase_name}",
        enable_checkpointing=False,
        log_every_n_steps=1,
        precision=OmegaConf.select(cfg, "trainer.precision", default="16-mixed"),
    )

    # Freeze HOPE inner loops for pure inference
    if is_hope:
        model.model.freeze_all_inner_loops()

    # Evaluate on each partition
    model.eval()
    for partition in eval_partitions:
        log.info(
            f"  Evaluating on '{partition['name']}' "
            f"(clips {partition['clip_start']}-{partition['clip_end']})"
        )

        eval_dm = create_datamodule(
            cfg,
            clip_start=partition["clip_start"],
            clip_end=partition["clip_end"],
            batch_size=cfg.data.get("batch_size", 64),
            val_split=0.0,
        )

        # Temporarily disable TTA for evaluation
        tta_was_enabled = getattr(model, "tta_enabled", False)
        if tta_was_enabled:
            model.tta_enabled = False

        eval_trainer.test(model=model, datamodule=eval_dm)

        # Restore TTA state
        if tta_was_enabled:
            model.tta_enabled = tta_was_enabled

        # Collect the loss metric
        loss = eval_trainer.callback_metrics.get("test/loss")
        if loss is None:
            loss = eval_trainer.callback_metrics.get("test/loss_jump")
        if loss is not None:
            loss_val = loss.item() if hasattr(loss, "item") else float(loss)
            tracker.update(train_exp_id, partition["exp_id"], loss_val)
            log.info(
                f"  R[{train_exp_id}, {partition['exp_id']}] = {loss_val:.6f}"
            )
        else:
            log.warning(
                f"  No loss metric found for partition '{partition['name']}'"
            )

    # Unfreeze HOPE inner loops after evaluation
    if is_hope:
        model.model.unfreeze_all_inner_loops()

    # Compute and log CL metrics
    cl_metrics = tracker.compute_all_metrics(train_exp_id)
    log.info(f"CL Metrics after {phase_name}:")
    for k, v in cl_metrics.items():
        log.info(f"  {k}: {v:.6f}")

    # Log CL metrics to W&B
    if wandb_logger.experiment is not None:
        wandb_logger.experiment.log(cl_metrics)
        # Log R-matrix as a table
        wandb_logger.experiment.log({
            "cl/R_matrix": tracker.R_matrix.tolist(),
        })

    finish_wandb(wandb_logger)

    return cl_metrics


# =============================================================================
# Phase Runners
# =============================================================================


def run_base_training(
    cfg: DictConfig,
    wandb_group: str,
    output_dir: str,
) -> tuple[LightningModule, str]:
    """Run base training phase.

    Returns:
        Tuple of (trained model, checkpoint path).
    """
    cl_cfg = cfg.cl
    base_cfg = cl_cfg.base_training

    log.info("=" * 70)
    log.info("PHASE 0: BASE TRAINING")
    log.info(f"  Clips: {base_cfg.clip_start} - {base_cfg.clip_end}")
    log.info(f"  Epochs: {base_cfg.max_epochs}")
    log.info("=" * 70)

    # Instantiate model
    model = instantiate_model(cfg)

    # Create datamodule for base training
    dm = create_datamodule(
        cfg,
        clip_start=base_cfg.clip_start,
        clip_end=base_cfg.clip_end,
        val_split=base_cfg.get("val_split", 0.1),
    )

    # Create W&B logger
    train_tags = list(cfg.get("tags", [])) + ["base_training"]
    wandb_logger = create_wandb_logger(
        cfg,
        run_name="base_training",
        group=wandb_group,
        job_type="base_training",
        tags=train_tags,
        save_dir=f"{output_dir}/base_training",
    )

    # Create trainer
    trainer = create_trainer(
        cfg,
        wandb_logger,
        max_epochs=base_cfg.max_epochs,
        output_dir=f"{output_dir}/base_training",
        gradient_clip_val=OmegaConf.select(cfg, "trainer.gradient_clip_val", default=1.01),
        precision=OmegaConf.select(cfg, "trainer.precision", default="16-mixed"),
    )

    # Train
    trainer.fit(model=model, datamodule=dm)

    # Save checkpoint
    ckpt_dir = f"{output_dir}/checkpoints"
    ckpt_path = f"{ckpt_dir}/base_training_final.ckpt"
    save_model_checkpoint(model, ckpt_path)

    finish_wandb(wandb_logger)

    return model, ckpt_path


def run_task_training_tta(
    cfg: DictConfig,
    model: LightningModule,
    task_idx: int,
    task_cfg: DictConfig,
    wandb_group: str,
    output_dir: str,
) -> tuple[LightningModule, str]:
    """Run task training for AC-ViT using TTA (test-time adaptation).

    Processes all task clips with TTA enabled and tta_reset_per_clip=False,
    so weight updates accumulate across clips. The adapted weights become
    the model state for the next task.

    Returns:
        Tuple of (adapted model, checkpoint path).
    """
    tta_cfg = cfg.cl.tta

    log.info("=" * 70)
    log.info(f"TASK {task_idx}: {task_cfg.name} (TTA mode)")
    log.info(f"  Clips: {task_cfg.clip_start} - {task_cfg.clip_end}")
    log.info(f"  TTA LR: {tta_cfg.tta_lr}, Steps: {tta_cfg.tta_num_adaptation_steps}")
    log.info("=" * 70)

    # Configure TTA on the model
    model.tta_enabled = True
    model.tta_mode = tta_cfg.get("tta_mode", "jump")
    model.tta_lr = tta_cfg.tta_lr
    model.tta_grad_clip = tta_cfg.tta_grad_clip
    model.tta_reset_per_clip = False  # CRITICAL: accumulate across clips
    model.tta_num_adaptation_steps = tta_cfg.tta_num_adaptation_steps
    model.tta_adapt_layers = tta_cfg.get("tta_adapt_layers", "layernorm")
    model.tta_optimizer_type = tta_cfg.get("tta_optimizer_type", "adamw")
    model.tta_optimizer_betas = tuple(tta_cfg.get("tta_optimizer_betas", [0.9, 0.999]))

    # Eval clips are reserved at the end of each task range
    eval_clips = cfg.cl.eval.clips_per_task
    train_clip_end = task_cfg.clip_end - eval_clips

    # Create datamodule (TTA processes batch_size=1)
    dm = create_datamodule(
        cfg,
        clip_start=task_cfg.clip_start,
        clip_end=train_clip_end,
        batch_size=1,
        val_split=0.0,
    )

    # Create W&B logger
    train_tags = list(cfg.get("tags", [])) + [f"task_{task_idx}", task_cfg.name, "tta"]
    wandb_logger = create_wandb_logger(
        cfg,
        run_name=f"task_{task_idx}_{task_cfg.name}",
        group=wandb_group,
        job_type="task_training_tta",
        tags=train_tags,
        save_dir=f"{output_dir}/task_{task_idx}",
    )

    # Create trainer for TTA (test mode, 0 epochs train)
    trainer = create_trainer(
        cfg,
        wandb_logger,
        max_epochs=0,
        output_dir=f"{output_dir}/task_{task_idx}",
        enable_checkpointing=False,
        log_every_n_steps=1,
        precision=OmegaConf.select(cfg, "trainer.precision", default="32"),
    )

    # Run TTA — this processes clips and accumulates weight updates
    trainer.test(model=model, datamodule=dm)

    # Save adapted model
    ckpt_path = f"{output_dir}/checkpoints/task_{task_idx}_{task_cfg.name}.ckpt"
    save_model_checkpoint(model, ckpt_path)

    # Disable TTA after task training
    model.tta_enabled = False

    finish_wandb(wandb_logger)

    return model, ckpt_path


def run_task_training_finetune(
    cfg: DictConfig,
    model: LightningModule,
    task_idx: int,
    task_cfg: DictConfig,
    wandb_group: str,
    output_dir: str,
) -> tuple[LightningModule, str]:
    """Run task training for HOPE using regular fine-tuning.

    Trains the model normally (gradient descent) on task clips.

    Returns:
        Tuple of (fine-tuned model, checkpoint path).
    """
    task_train_cfg = cfg.cl.task_training

    log.info("=" * 70)
    log.info(f"TASK {task_idx}: {task_cfg.name} (finetune mode)")
    log.info(f"  Clips: {task_cfg.clip_start} - {task_cfg.clip_end}")
    log.info(f"  Epochs: {task_train_cfg.max_epochs}")
    log.info("=" * 70)

    # Eval clips reserved at end of range
    eval_clips = cfg.cl.eval.clips_per_task
    train_clip_end = task_cfg.clip_end - eval_clips

    # Create datamodule for task training
    dm = create_datamodule(
        cfg,
        clip_start=task_cfg.clip_start,
        clip_end=train_clip_end,
        val_split=task_train_cfg.get("val_split", 0.0),
    )

    # Create W&B logger
    train_tags = list(cfg.get("tags", [])) + [f"task_{task_idx}", task_cfg.name, "finetune"]
    wandb_logger = create_wandb_logger(
        cfg,
        run_name=f"task_{task_idx}_{task_cfg.name}",
        group=wandb_group,
        job_type="task_training_finetune",
        tags=train_tags,
        save_dir=f"{output_dir}/task_{task_idx}",
    )

    # Create trainer
    trainer = create_trainer(
        cfg,
        wandb_logger,
        max_epochs=task_train_cfg.max_epochs,
        output_dir=f"{output_dir}/task_{task_idx}",
        gradient_clip_val=OmegaConf.select(cfg, "trainer.gradient_clip_val", default=3.0),
        precision=OmegaConf.select(cfg, "trainer.precision", default="32"),
    )

    # Train on task data
    trainer.fit(model=model, datamodule=dm)

    # Save checkpoint
    ckpt_path = f"{output_dir}/checkpoints/task_{task_idx}_{task_cfg.name}.ckpt"
    save_model_checkpoint(model, ckpt_path)

    finish_wandb(wandb_logger)

    return model, ckpt_path


# =============================================================================
# Joint Training (Upper Bound)
# =============================================================================


def run_joint_training(
    cfg: DictConfig,
    wandb_group: str,
    output_dir: str,
) -> tuple[LightningModule, str]:
    """Run joint (i.i.d.) training on ALL data simultaneously.

    This is the Upper Bound: the sequential constraint is removed and the
    model sees base + all task clips mixed together. The result is the
    theoretical best performance this architecture can achieve.

    Returns:
        Tuple of (trained model, checkpoint path).
    """
    cl_cfg = cfg.cl
    joint_cfg = cl_cfg.joint_training

    # Compute overall clip range: min start across base+tasks to max end
    all_starts = [cl_cfg.base_training.clip_start] + [
        t.clip_start for t in cl_cfg.tasks
    ]
    all_ends = [cl_cfg.base_training.clip_end] + [
        t.clip_end for t in cl_cfg.tasks
    ]
    clip_start = min(all_starts)
    clip_end = max(all_ends)

    log.info("=" * 70)
    log.info("UPPER BOUND: JOINT TRAINING (i.i.d.)")
    log.info(f"  Clips: {clip_start} - {clip_end} (all data mixed)")
    log.info(f"  Epochs: {joint_cfg.max_epochs}")
    log.info("=" * 70)

    # Instantiate model
    model = instantiate_model(cfg)

    # Create datamodule spanning ALL clips
    dm = create_datamodule(
        cfg,
        clip_start=clip_start,
        clip_end=clip_end,
        val_split=joint_cfg.get("val_split", 0.1),
    )

    # Create W&B logger
    train_tags = list(cfg.get("tags", [])) + ["joint_training", "upper_bound"]
    wandb_logger = create_wandb_logger(
        cfg,
        run_name="joint_training",
        group=wandb_group,
        job_type="joint_training",
        tags=train_tags,
        save_dir=f"{output_dir}/joint_training",
    )

    # Create trainer
    trainer = create_trainer(
        cfg,
        wandb_logger,
        max_epochs=joint_cfg.max_epochs,
        output_dir=f"{output_dir}/joint_training",
        gradient_clip_val=OmegaConf.select(
            cfg, "trainer.gradient_clip_val", default=1.01
        ),
        precision=OmegaConf.select(cfg, "trainer.precision", default="16-mixed"),
    )

    # Train on all data
    trainer.fit(model=model, datamodule=dm)

    # Save checkpoint
    ckpt_dir = f"{output_dir}/checkpoints"
    ckpt_path = f"{ckpt_dir}/joint_training_final.ckpt"
    save_model_checkpoint(model, ckpt_path)

    finish_wandb(wandb_logger)

    return model, ckpt_path


# =============================================================================
# Main Pipeline
# =============================================================================


def _log_final_summary(
    cfg: DictConfig,
    tracker: ContinualLearningMetricsTracker,
    wandb_group: str,
    output_dir: str,
) -> None:
    """Print R-matrix, save JSON, and create a summary W&B run."""
    import numpy as np

    log.info("\n" + "=" * 70)
    log.info("CONTINUAL LEARNING PIPELINE COMPLETE")
    log.info("=" * 70)

    # Print R-matrix
    log.info("\nR-matrix (performance of task j after training experience i):")
    np.set_printoptions(precision=4, suppress=True)
    log.info(f"\n{tracker.R_matrix}")

    # Print final CL metrics for each stage
    for exp_id in range(tracker.num_experiences):
        if not np.isnan(tracker.R_matrix[exp_id]).all():
            metrics = tracker.compute_all_metrics(exp_id)
            log.info(f"\nMetrics after experience {exp_id}:")
            for k, v in metrics.items():
                log.info(f"  {k}: {v:.6f}")

    # Save tracker to JSON
    tracker_path = f"{output_dir}/cl_metrics.json"
    tracker.save_json(tracker_path)
    log.info(f"\nCL metrics saved to: {tracker_path}")

    # Create a final summary W&B run with all metrics
    final_logger = create_wandb_logger(
        cfg,
        run_name="cl_summary",
        group=wandb_group,
        job_type="summary",
        tags=list(cfg.get("tags", [])) + ["summary"],
        save_dir=f"{output_dir}/summary",
    )
    if final_logger.experiment is not None:
        final_logger.experiment.log({
            "cl/R_matrix_final": tracker.R_matrix.tolist(),
        })
        final_metrics = tracker.compute_all_metrics(tracker.num_experiences - 1)
        final_logger.experiment.log(final_metrics)
    finish_wandb(final_logger)


def _run_sequential_pipeline(cfg: DictConfig) -> None:
    """Run the standard sequential CL pipeline (base → task1 → ... → taskN).

    Used for: AC-ViT+TTA, AC-HOPE, and Lower Bound (naive finetuning).
    """
    cl_cfg = cfg.cl
    output_dir = str(cfg.paths.output_dir)
    wandb_group = cl_cfg.wandb_group
    task_training_mode = cl_cfg.task_training_mode
    is_hope = task_training_mode == "finetune" and "hope" in cfg.get("task_name", "")
    num_tasks = len(cl_cfg.tasks)

    log.info("=" * 70)
    log.info("CONTINUAL LEARNING PIPELINE (Sequential)")
    log.info(f"  Task name: {cfg.get('task_name', 'unknown')}")
    log.info(f"  Training mode: {task_training_mode}")
    log.info(f"  Number of tasks: {num_tasks}")
    log.info(f"  W&B group: {wandb_group}")
    log.info(f"  Output dir: {output_dir}")
    log.info("=" * 70)

    tracker = ContinualLearningMetricsTracker(
        num_tasks=num_tasks, higher_is_better=False
    )

    # PHASE 0: BASE TRAINING
    model, base_ckpt = run_base_training(cfg, wandb_group, output_dir)

    log.info("\n--- Evaluation after Base Training ---")
    evaluate_all_tasks(
        model=model,
        cfg=cfg,
        tracker=tracker,
        train_exp_id=0,
        phase_name="after_base",
        wandb_group=wandb_group,
        output_dir=output_dir,
        is_hope=is_hope,
    )

    # PHASES 1-N: SEQUENTIAL TASKS
    for task_idx_0based, task in enumerate(cl_cfg.tasks):
        task_idx = task_idx_0based + 1
        exp_id = task_idx

        if task_training_mode == "tta":
            model, ckpt_path = run_task_training_tta(
                cfg=cfg,
                model=model,
                task_idx=task_idx,
                task_cfg=task,
                wandb_group=wandb_group,
                output_dir=output_dir,
            )
        elif task_training_mode == "finetune":
            model, ckpt_path = run_task_training_finetune(
                cfg=cfg,
                model=model,
                task_idx=task_idx,
                task_cfg=task,
                wandb_group=wandb_group,
                output_dir=output_dir,
            )
        else:
            raise ValueError(f"Unknown task_training_mode: {task_training_mode}")

        log.info(f"\n--- Evaluation after Task {task_idx} ({task.name}) ---")
        evaluate_all_tasks(
            model=model,
            cfg=cfg,
            tracker=tracker,
            train_exp_id=exp_id,
            phase_name=f"after_task_{task_idx}",
            wandb_group=wandb_group,
            output_dir=output_dir,
            is_hope=is_hope,
        )

    _log_final_summary(cfg, tracker, wandb_group, output_dir)


def _run_joint_pipeline(cfg: DictConfig) -> None:
    """Run the joint (i.i.d.) training pipeline (Upper Bound).

    Trains on ALL data simultaneously, then evaluates on each partition.
    The R-matrix has a single row (experience 0 = joint training).
    """
    cl_cfg = cfg.cl
    output_dir = str(cfg.paths.output_dir)
    wandb_group = cl_cfg.wandb_group
    num_tasks = len(cl_cfg.tasks)

    log.info("=" * 70)
    log.info("UPPER BOUND: JOINT TRAINING PIPELINE")
    log.info(f"  Number of eval partitions: {num_tasks + 1}")
    log.info(f"  W&B group: {wandb_group}")
    log.info("=" * 70)

    # For the upper bound, we use a single-row R-matrix:
    # Row 0 = performance on each partition after joint training.
    tracker = ContinualLearningMetricsTracker(
        num_tasks=num_tasks, higher_is_better=False
    )

    # JOINT TRAINING on all data
    model, ckpt_path = run_joint_training(cfg, wandb_group, output_dir)

    # FULL EVALUATION on all partitions
    log.info("\n--- Evaluation after Joint Training ---")
    evaluate_all_tasks(
        model=model,
        cfg=cfg,
        tracker=tracker,
        train_exp_id=0,
        phase_name="after_joint",
        wandb_group=wandb_group,
        output_dir=output_dir,
        is_hope=False,
    )

    _log_final_summary(cfg, tracker, wandb_group, output_dir)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """Run the full Continual Learning pipeline.

    Supports two pipeline modes via ``cl.pipeline_mode``:
        - ``"sequential"`` (default): Base → Task 1 → ... → Task N with eval after each.
        - ``"joint"``: Train on all data i.i.d. (upper bound), then evaluate.

    Args:
        cfg: Hydra config with 'cl' section for CL pipeline settings.
    """
    log_device_info()
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    pipeline_mode = cfg.cl.get("pipeline_mode", "sequential")

    if pipeline_mode == "sequential":
        _run_sequential_pipeline(cfg)
    elif pipeline_mode == "joint":
        _run_joint_pipeline(cfg)
    else:
        raise ValueError(
            f"Unknown pipeline_mode: '{pipeline_mode}'. "
            f"Expected 'sequential' or 'joint'."
        )


if __name__ == "__main__":
    main()
