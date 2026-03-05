"""Continual Learning Benchmark Training Pipeline.

Orchestrates CL experiments on standard image-classification benchmarks:
    - Split CIFAR-100  (class-incremental)
    - Permuted MNIST   (domain-incremental)

Follows the same structure as ``cl_train.py`` but uses classification
(cross-entropy + accuracy) instead of video-feature prediction.

Workflow:
    1. Task 0 (base training)  — train on the first task's data
    2. Tasks 1..N-1             — fine-tune sequentially
    3. Full evaluation after every phase (on all tasks seen so far)
    4. Compute CL metrics: Avg Accuracy, BWT, FWT, Forgetting

Usage:
    uv run src/cl_benchmark_train.py experiment=cl_split_cifar100
    uv run src/cl_benchmark_train.py experiment=cl_permuted_mnist
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
import numpy as np
import omegaconf
import torch
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

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
# Accuracy-Based CL Metrics Tracker
# =============================================================================


class BenchmarkCLMetricsTracker:
    """Track CL metrics via an R-matrix where R[i, j] = test accuracy on
    task j after training on task i.

    Computes:
        - Average Accuracy (AA): mean accuracy on all seen tasks after final training
        - Backward Transfer (BWT): average change in accuracy on previous tasks
        - Forward Transfer (FWT): accuracy on unseen tasks before any training on them
        - Average Forgetting (AF): average max-drop in accuracy per previous task
    """

    def __init__(self, num_tasks: int) -> None:
        self.num_tasks = num_tasks
        # R[i, j] = accuracy on task j after training on task i
        self.R = np.full((num_tasks, num_tasks), np.nan)

    def update(self, train_task: int, eval_task: int, accuracy: float) -> None:
        self.R[train_task, eval_task] = accuracy
        log.info(f"R[{train_task}, {eval_task}] = {accuracy:.4f}")

    def average_accuracy(self) -> float:
        """Average accuracy on all tasks after final task training."""
        last_row = self.R[-1, :]
        valid = ~np.isnan(last_row)
        if not valid.any():
            return float("nan")
        return float(np.mean(last_row[valid]))

    def backward_transfer(self) -> float:
        """BWT = (1/(T-1)) * sum_{j<T} [R[T-1, j] - R[j, j]]."""
        T = self.num_tasks
        if T < 2:
            return 0.0
        bwt_vals = []
        for j in range(T - 1):
            if not np.isnan(self.R[T - 1, j]) and not np.isnan(self.R[j, j]):
                bwt_vals.append(self.R[T - 1, j] - self.R[j, j])
        return float(np.mean(bwt_vals)) if bwt_vals else float("nan")

    def forward_transfer(self) -> float:
        """FWT = (1/(T-1)) * sum_{j>0} [R[j-1, j] - baseline].

        Baseline = random chance (not always available, so we use 0 here).
        """
        T = self.num_tasks
        if T < 2:
            return 0.0
        fwt_vals = []
        for j in range(1, T):
            if not np.isnan(self.R[j - 1, j]):
                fwt_vals.append(self.R[j - 1, j])
        return float(np.mean(fwt_vals)) if fwt_vals else float("nan")

    def average_forgetting(self) -> float:
        """AF = (1/(T-1)) * sum_{j<T} max_{l in [j..T-2]} [R[l,j] - R[T-1,j]]."""
        T = self.num_tasks
        if T < 2:
            return 0.0
        forgetting_vals = []
        for j in range(T - 1):
            best = -np.inf
            for l in range(j, T - 1):
                if not np.isnan(self.R[l, j]):
                    best = max(best, self.R[l, j])
            if best > -np.inf and not np.isnan(self.R[T - 1, j]):
                forgetting_vals.append(best - self.R[T - 1, j])
        return float(np.mean(forgetting_vals)) if forgetting_vals else float("nan")

    def compute_all(self) -> dict[str, float]:
        return {
            "cl/average_accuracy": self.average_accuracy(),
            "cl/backward_transfer": self.backward_transfer(),
            "cl/forward_transfer": self.forward_transfer(),
            "cl/average_forgetting": self.average_forgetting(),
        }


# =============================================================================
# Helpers
# =============================================================================


def create_wandb_logger(
    cfg: DictConfig,
    run_name: str,
    group: str,
    job_type: str,
    tags: list[str] | None = None,
    save_dir: str | None = None,
) -> WandbLogger:
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
    log_every_n_steps: int = 10,
    gradient_clip_val: float | None = None,
    precision: str | int = "32",
    num_sanity_val_steps: int | None = None,
    limit_val_batches: int | float = 1.0,
) -> Trainer:
    callbacks: list[Callback] = [RichProgressBar()]
    if enable_checkpointing:
        callbacks.append(
            ModelCheckpoint(
                dirpath=f"{output_dir}/checkpoints",
                filename="best",
                save_last=False,
                monitor="val/acc",
                mode="max",
                save_top_k=1,
            )
        )
    return Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        precision=precision,
        deterministic=cfg.get("deterministic", True),
        gradient_clip_val=gradient_clip_val,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=log_every_n_steps,
        enable_checkpointing=enable_checkpointing,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=num_sanity_val_steps,
        limit_val_batches=limit_val_batches,
    )


def finish_wandb(logger: WandbLogger) -> None:
    try:
        if logger.experiment is not None:
            import wandb
            wandb.finish()
    except Exception as e:
        log.warning(f"Failed to finish W&B run: {e}")


def save_model_checkpoint(model: LightningModule, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, path)
    log.info(f"Checkpoint saved: {path}")


def load_checkpoint_weights(model: LightningModule, ckpt_path: str) -> None:
    log.info(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)


def _get_best_checkpoint_path(trainer: Trainer) -> str | None:
    """Extract the best checkpoint path from a finished Trainer's ModelCheckpoint."""
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
            return cb.best_model_path
    return None


# =============================================================================
# DataModule Factory
# =============================================================================


def create_datamodule_for_task(
    cfg: DictConfig,
    task_id: int,
    task_classes: list[int] | None = None,
    all_classes_so_far: list[int] | None = None,
    val_split: float | None = None,
    for_eval: bool = False,
) -> LightningDataModule:
    """Create a DataModule for a specific benchmark task.

    Dispatches to the correct DataModule based on ``cfg.data._target_``.

    Args:
        cfg: Full Hydra config.
        task_id: Task index (0-based).
        task_classes: Class IDs for this task (Split CIFAR-100 only).
        all_classes_so_far: Union of classes from all tasks up to now.
        val_split: Override validation split. If None, uses config default.
        for_eval: If True, sets up for test-only evaluation.
    """
    data_cfg = copy.deepcopy(cfg.data)
    OmegaConf.set_struct(data_cfg, False)
    target = OmegaConf.select(data_cfg, "_target_", default="")

    if "PermutedMNIST" in target:
        data_cfg.task_id = task_id
        if val_split is not None:
            data_cfg.val_split = val_split
        return hydra.utils.instantiate(data_cfg)

    elif "SplitCIFAR100" in target:
        if task_classes is not None:
            data_cfg.task_classes = task_classes
        if all_classes_so_far is not None:
            data_cfg.all_classes_so_far = all_classes_so_far
        if val_split is not None:
            data_cfg.val_split = val_split
        return hydra.utils.instantiate(data_cfg)

    else:
        raise ValueError(
            f"Unsupported data target for benchmarks: {target}. "
            f"Expected SplitCIFAR100DataModule or PermutedMNISTDataModule."
        )


# =============================================================================
# Task Definitions
# =============================================================================


def build_task_schedule(cfg: DictConfig) -> list[dict[str, Any]]:
    """Build the task schedule from the CL config.

    For Split CIFAR-100: each task has a disjoint set of class IDs.
    For Permuted MNIST: each task has a unique permutation index.
    """
    cl_cfg = cfg.cl
    target = OmegaConf.select(cfg.data, "_target_", default="")
    tasks: list[dict[str, Any]] = []

    if "SplitCIFAR100" in target:
        num_tasks = cl_cfg.get("num_tasks", 10)
        classes_per_task = cl_cfg.get("classes_per_task", 10)
        total_classes = num_tasks * classes_per_task

        # Deterministic class ordering (shuffled by seed)
        rng = np.random.default_rng(cfg.get("seed", 42))
        all_class_ids = rng.permutation(total_classes).tolist()

        for t in range(num_tasks):
            start = t * classes_per_task
            end = start + classes_per_task
            classes = sorted(all_class_ids[start:end])
            tasks.append({
                "task_id": t,
                "name": f"classes_{classes[0]}_{classes[-1]}",
                "class_ids": classes,
            })

    elif "PermutedMNIST" in target:
        num_tasks = cl_cfg.get("num_tasks", 10)
        for t in range(num_tasks):
            tasks.append({
                "task_id": t,
                "name": f"perm_{t}" if t > 0 else "original",
            })

    else:
        raise ValueError(f"Cannot build task schedule for: {target}")

    return tasks


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_on_all_tasks(
    model: LightningModule,
    cfg: DictConfig,
    task_schedule: list[dict[str, Any]],
    train_task_id: int,
    tracker: BenchmarkCLMetricsTracker,
    wandb_logger: WandbLogger,
) -> None:
    """Evaluate model on all tasks seen so far and update the R-matrix."""
    target = OmegaConf.select(cfg.data, "_target_", default="")

    for eval_task in task_schedule:
        eval_id = eval_task["task_id"]

        if "SplitCIFAR100" in target:
            # Evaluate on this task's classes only
            dm = create_datamodule_for_task(
                cfg, task_id=eval_id,
                task_classes=eval_task["class_ids"],
                all_classes_so_far=eval_task["class_ids"],
                val_split=0.0,
                for_eval=True,
            )
        else:
            dm = create_datamodule_for_task(
                cfg, task_id=eval_id, val_split=0.0, for_eval=True,
            )

        eval_trainer = Trainer(
            accelerator="auto",
            devices=1,
            precision=OmegaConf.select(cfg, "trainer.precision", default="32"),
            deterministic=cfg.get("deterministic", True),
            logger=False,
            enable_checkpointing=False,
        )

        model.eval()
        results = eval_trainer.test(model=model, datamodule=dm)
        acc = results[0].get("test/acc", float("nan")) if results else float("nan")
        tracker.update(train_task_id, eval_id, acc)

        log.info(f"  Eval task {eval_id} ({eval_task['name']}): acc={acc:.4f}")


# =============================================================================
# Phase Runners
# =============================================================================


def run_task(
    cfg: DictConfig,
    model: LightningModule,
    task: dict[str, Any],
    task_schedule: list[dict[str, Any]],
    tracker: BenchmarkCLMetricsTracker,
    wandb_group: str,
    output_dir: str,
    max_epochs: int,
    val_split: float = 0.1,
    learning_rate_override: float | None = None,
) -> LightningModule:
    """Train model on a single task and evaluate on all tasks seen so far."""
    task_id = task["task_id"]
    task_name = task["name"]

    log.info("=" * 70)
    log.info(f"TASK {task_id}: {task_name}")
    log.info(f"  Max epochs: {max_epochs}")
    log.info("=" * 70)

    target = OmegaConf.select(cfg.data, "_target_", default="")

    # Build cumulative class list for Split CIFAR-100
    all_classes = []
    for t in task_schedule[:task_id + 1]:
        if "class_ids" in t:
            all_classes.extend(t["class_ids"])
    all_classes = sorted(set(all_classes)) if all_classes else None

    # Create DataModule
    if "SplitCIFAR100" in target:
        dm = create_datamodule_for_task(
            cfg, task_id=task_id,
            task_classes=task["class_ids"],
            all_classes_so_far=all_classes,
            val_split=val_split,
        )
    else:
        dm = create_datamodule_for_task(
            cfg, task_id=task_id, val_split=val_split,
        )

    # Override LR if specified (for fine-tuning tasks)
    if learning_rate_override is not None:
        model.learning_rate = learning_rate_override

    # W&B logger
    tags = list(cfg.get("tags", [])) + [f"task_{task_id}", task_name]
    wandb_logger = create_wandb_logger(
        cfg,
        run_name=f"task_{task_id}_{task_name}",
        group=wandb_group,
        job_type="task_training",
        tags=tags,
        save_dir=f"{output_dir}/task_{task_id}",
    )

    # Sync scheduler max_epochs with per-task epoch budget
    model.max_epochs = max_epochs

    no_val = val_split == 0.0
    trainer = create_trainer(
        cfg,
        wandb_logger,
        max_epochs=max_epochs,
        output_dir=f"{output_dir}/task_{task_id}",
        gradient_clip_val=OmegaConf.select(cfg, "trainer.gradient_clip_val", default=None),
        precision=OmegaConf.select(cfg, "trainer.precision", default="32"),
        num_sanity_val_steps=0 if no_val else None,
        limit_val_batches=0 if no_val else 1.0,
    )

    trainer.fit(model=model, datamodule=dm)

    # ── Best-epoch model selection ──
    # Reload the checkpoint with the highest val/acc (not last epoch).
    # This aligns with standard CL benchmark protocol: evaluate the
    # best-performing model from each task's training phase.
    best_ckpt = _get_best_checkpoint_path(trainer)
    if best_ckpt and os.path.isfile(best_ckpt):
        log.info(f"Loading best checkpoint for eval: {best_ckpt}")
        load_checkpoint_weights(model, best_ckpt)
    else:
        log.warning("No best checkpoint found — evaluating last-epoch model.")

    # Save final task checkpoint (best-epoch weights)
    ckpt_path = f"{output_dir}/checkpoints/task_{task_id}_{task_name}.ckpt"
    save_model_checkpoint(model, ckpt_path)

    # Evaluate on all tasks
    log.info(f"\n--- Evaluation after Task {task_id} ({task_name}) ---")
    evaluate_on_all_tasks(
        model=model,
        cfg=cfg,
        task_schedule=task_schedule,
        train_task_id=task_id,
        tracker=tracker,
        wandb_logger=wandb_logger,
    )

    finish_wandb(wandb_logger)
    return model


# =============================================================================
# Main Pipeline
# =============================================================================


def _run_single_seed(
    cfg: DictConfig,
    seed: int,
    run_idx: int,
    num_runs: int,
) -> dict[str, Any]:
    """Run a complete CL benchmark for one seed.  Returns metrics + R-matrix."""
    L.seed_everything(seed, workers=True)

    cl_cfg = cfg.cl
    base_output_dir = str(cfg.paths.output_dir)
    output_dir = f"{base_output_dir}/seed_{seed}" if num_runs > 1 else base_output_dir
    os.makedirs(output_dir, exist_ok=True)

    wandb_group = cl_cfg.get(
        "wandb_group",
        f"cl_benchmark_{cfg.get('task_name', 'unknown')}_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    if num_runs > 1:
        wandb_group = f"{wandb_group}_s{seed}"

    task_schedule = build_task_schedule(cfg)
    num_tasks = len(task_schedule)

    log.info("=" * 70)
    log.info(f"CONTINUAL LEARNING BENCHMARK (seed={seed}, run {run_idx+1}/{num_runs})")
    log.info(f"  Task name: {cfg.get('task_name', 'unknown')}")
    log.info(f"  Number of tasks: {num_tasks}")
    log.info(f"  W&B group: {wandb_group}")
    log.info(f"  Output dir: {output_dir}")
    log.info("=" * 70)

    tracker = BenchmarkCLMetricsTracker(num_tasks=num_tasks)

    # Instantiate model (fresh for each seed)
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Base training settings
    base_epochs = cl_cfg.get("base_epochs", cl_cfg.get("epochs_per_task", 50))
    task_epochs = cl_cfg.get("task_epochs", cl_cfg.get("epochs_per_task", 50))
    task_lr = cl_cfg.get("task_learning_rate", None)
    val_split = cl_cfg.get("val_split", 0.1)

    for task in task_schedule:
        epochs = base_epochs if task["task_id"] == 0 else task_epochs
        lr_override = task_lr if task["task_id"] > 0 else None

        model = run_task(
            cfg=cfg,
            model=model,
            task=task,
            task_schedule=task_schedule,
            tracker=tracker,
            wandb_group=wandb_group,
            output_dir=output_dir,
            max_epochs=epochs,
            val_split=val_split,
            learning_rate_override=lr_override,
        )

    # ─── Per-seed Summary ───
    cl_metrics = tracker.compute_all()
    log.info("\n" + "=" * 70)
    log.info(f"CL METRICS (seed={seed})")
    log.info("=" * 70)
    for k, v in cl_metrics.items():
        log.info(f"  {k}: {v:.4f}")

    # Log summary to W&B
    summary_logger = create_wandb_logger(
        cfg,
        run_name=f"summary_s{seed}",
        group=wandb_group,
        job_type="summary",
        tags=list(cfg.get("tags", [])) + ["summary", f"seed_{seed}"],
        save_dir=f"{output_dir}/summary",
    )
    if summary_logger.experiment is not None:
        summary_logger.experiment.log(cl_metrics)
        summary_logger.experiment.log({"cl/R_matrix": tracker.R.tolist()})
    finish_wandb(summary_logger)

    # Save per-seed results JSON
    results_path = f"{output_dir}/cl_results.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "seed": seed,
            "R_matrix": tracker.R.tolist(),
            "metrics": cl_metrics,
            "task_schedule": [
                {k: v for k, v in t.items() if not isinstance(v, np.ndarray)}
                for t in task_schedule
            ],
        }, f, indent=2, default=str)
    log.info(f"Results saved to: {results_path}")

    return {"seed": seed, "metrics": cl_metrics, "R_matrix": tracker.R}


def _run_sequential_benchmark(cfg: DictConfig) -> None:
    """Sequential CL benchmark with multi-seed averaging.

    Paper protocol (Anbar Jafari et al., 2025): "All results are averaged
    over three independent runs, with standard deviations reported."

    Seeds are specified via ``cl.seeds`` (list) or fall back to the single
    ``seed`` field for backward compatibility.
    """
    cl_cfg = cfg.cl
    seeds: list[int] = list(cl_cfg.get("seeds", [cfg.get("seed", 42)]))
    num_runs = len(seeds)

    log.info(f"Running {num_runs} seed(s): {seeds}")

    all_results: list[dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        result = _run_single_seed(cfg, seed=seed, run_idx=i, num_runs=num_runs)
        all_results.append(result)

    # ─── Aggregate across seeds ───
    metric_keys = list(all_results[0]["metrics"].keys())
    agg: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        vals = [r["metrics"][key] for r in all_results if not np.isnan(r["metrics"][key])]
        agg[key] = {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }

    log.info("\n" + "=" * 70)
    log.info(f"FINAL CL METRICS (averaged over {num_runs} seeds)")
    log.info("=" * 70)
    for key, stats in agg.items():
        log.info(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    # Save aggregated results
    output_dir = str(cfg.paths.output_dir)
    agg_path = f"{output_dir}/cl_results_aggregated.json"
    os.makedirs(os.path.dirname(agg_path), exist_ok=True)
    with open(agg_path, "w") as f:
        json.dump({
            "seeds": seeds,
            "num_runs": num_runs,
            "aggregated_metrics": {k: v for k, v in agg.items()},
            "per_seed": [
                {"seed": r["seed"], "metrics": r["metrics"], "R_matrix": r["R_matrix"].tolist()}
                for r in all_results
            ],
        }, f, indent=2, default=str)
    log.info(f"Aggregated results saved to: {agg_path}")

    # Log aggregate to W&B
    wandb_group = cl_cfg.get(
        "wandb_group",
        f"cl_benchmark_{cfg.get('task_name', 'unknown')}_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    agg_logger = create_wandb_logger(
        cfg,
        run_name="aggregate",
        group=wandb_group,
        job_type="aggregate",
        tags=list(cfg.get("tags", [])) + ["aggregate"],
        save_dir=f"{output_dir}/aggregate",
    )
    if agg_logger.experiment is not None:
        flat_metrics = {f"{k}/mean": v["mean"] for k, v in agg.items()}
        flat_metrics.update({f"{k}/std": v["std"] for k, v in agg.items()})
        agg_logger.experiment.log(flat_metrics)
    finish_wandb(agg_logger)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """Entry point for CL benchmark experiments.

    Args:
        cfg: Hydra config with 'cl' section for benchmark settings.
    """
    log_device_info()
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    pipeline_mode = cfg.cl.get("pipeline_mode", "sequential")

    if pipeline_mode == "sequential":
        _run_sequential_benchmark(cfg)
    else:
        raise ValueError(
            f"Unknown pipeline_mode: '{pipeline_mode}'. "
            f"Benchmarks currently only support 'sequential'."
        )


if __name__ == "__main__":
    main()
