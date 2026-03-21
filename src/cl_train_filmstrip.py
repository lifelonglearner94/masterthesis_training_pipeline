"""Continual Learning Training Pipeline — with Filmstrip Visualization.

Thin wrapper around ``src/cl_train.py`` that adds qualitative PCA→RGB
filmstrip generation after base training and after every sequential task.

The training logic is **identical** to ``cl_train.py`` — only the evaluation
hooks are extended.  The original file is untouched.

Usage:
    uv run src/cl_train_filmstrip.py experiment=cl_ac_hope_phase8_hybrid_filmstrip paths.data_dir=/path
    uv run src/cl_train_filmstrip.py experiment=cl_gated_delta_net_filmstrip paths.data_dir=/path
"""

import pyrootutils

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from dotenv import load_dotenv

load_dotenv()

import logging

import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from src.cl_train import (
    _create_replay_buffer,
    _log_final_summary,
    create_datamodule,
    evaluate_all_tasks,
    instantiate_model,
    load_checkpoint_weights,
    run_base_training,
    run_task_training_finetune,
    run_task_training_finetune_with_replay,
    run_task_training_tta,
)
from src.utils.cl_metrics import ContinualLearningMetricsTracker
from src.utils.device_utils import log_device_info
from src.utils.filmstrip_rollout import (
    fit_pca_on_base_clips,
    generate_filmstrips_for_phase,
)

log = logging.getLogger(__name__)


def _ensure_cuda(model):
    """Ensure model is on CUDA — required for Triton/fla kernels (GatedDeltaNet)."""
    if torch.cuda.is_available():
        model = model.cuda()
    return model


# =============================================================================
# Sequential Pipeline + Filmstrips
# =============================================================================


def _run_sequential_pipeline_with_filmstrip(cfg: DictConfig) -> None:
    """Run the standard sequential CL pipeline with filmstrip generation.

    Identical to ``cl_train._run_sequential_pipeline`` but calls
    :func:`generate_filmstrips_for_phase` after every evaluation phase
    (after base, after task 1, …, after task N).
    """
    cl_cfg = cfg.cl
    output_dir = str(cfg.paths.output_dir)
    wandb_group = cl_cfg.wandb_group
    task_training_mode = cl_cfg.task_training_mode
    is_hope = task_training_mode == "finetune" and "hope" in cfg.get("task_name", "")
    num_tasks = len(cl_cfg.tasks)

    log.info("=" * 70)
    log.info("CL PIPELINE + FILMSTRIP VISUALIZATION")
    log.info(f"  Task name:     {cfg.get('task_name', 'unknown')}")
    log.info(f"  Training mode: {task_training_mode}")
    log.info(f"  Num tasks:     {num_tasks}")
    log.info(f"  W&B group:     {wandb_group}")
    log.info(f"  Output dir:    {output_dir}")
    log.info("=" * 70)

    tracker = ContinualLearningMetricsTracker(
        num_tasks=num_tasks, higher_is_better=False
    )

    # ── PHASE 0: BASE TRAINING ─────────────────────────────────────────
    resume_ckpt = cl_cfg.get("resume_from_base_checkpoint", None)
    if resume_ckpt:
        log.info("Skipping base training — loading checkpoint: %s", resume_ckpt)
        model = instantiate_model(cfg)
        load_checkpoint_weights(model, resume_ckpt)
    else:
        model, _base_ckpt = run_base_training(cfg, wandb_group, output_dir)

    model = _ensure_cuda(model)

    # ── FIT PCA (once on base eval clips, reused for every filmstrip) ──
    log.info("\n--- Fitting PCA on base evaluation clips ---")
    eval_clips = cl_cfg.eval.clips_per_task
    base_eval_start = cl_cfg.base_training.clip_end - eval_clips
    pca_dm = create_datamodule(
        cfg,
        clip_start=base_eval_start,
        clip_end=cl_cfg.base_training.clip_end,
        batch_size=32,
        val_split=0.0,
    )
    # Force num_workers=0 to avoid DataLoader deadlocks on rental GPUs
    pca_dm.num_workers = 0
    pca_dm.persistent_workers = False
    filmstrip_cfg = cl_cfg.get("filmstrip", {})
    pca = fit_pca_on_base_clips(
        pca_dm,
        n_components=4,
        max_clips=filmstrip_cfg.get("pca_max_clips", 50),
    )

    # ── EVAL + FILMSTRIP after base ────────────────────────────────────
    log.info("\n--- Evaluation after Base Training ---")
    model = _ensure_cuda(model)
    evaluate_all_tasks(
        model=model, cfg=cfg, tracker=tracker, train_exp_id=0,
        phase_name="after_base", wandb_group=wandb_group,
        output_dir=output_dir, is_hope=is_hope,
    )
    model = _ensure_cuda(model)  # re-enforce after Lightning Trainer.test()
    generate_filmstrips_for_phase(
        model=model, cfg=cfg, pca=pca,
        phase_name="after_base", output_dir=output_dir,
    )

    # ── Replay buffer (Phase 10) if configured ─────────────────────────
    use_replay = cl_cfg.get("replay", None) is not None
    replay_buffer = None
    if use_replay:
        log.info("\n--- Initializing Experience Replay Buffer ---")
        replay_buffer = _create_replay_buffer(cfg)

    # ── PHASES 1–N: SEQUENTIAL TASKS ──────────────────────────────────
    for task_idx_0based, task in enumerate(cl_cfg.tasks):
        task_idx = task_idx_0based + 1

        # --- train ---
        if task_training_mode == "tta":
            model, _ckpt = run_task_training_tta(
                cfg=cfg, model=model, task_idx=task_idx, task_cfg=task,
                wandb_group=wandb_group, output_dir=output_dir,
            )
        elif task_training_mode == "finetune":
            if use_replay and replay_buffer is not None:
                model, _ckpt = run_task_training_finetune_with_replay(
                    cfg=cfg, model=model, task_idx=task_idx, task_cfg=task,
                    wandb_group=wandb_group, output_dir=output_dir,
                    replay_buffer=replay_buffer,
                )
            else:
                model, _ckpt = run_task_training_finetune(
                    cfg=cfg, model=model, task_idx=task_idx, task_cfg=task,
                    wandb_group=wandb_group, output_dir=output_dir,
                )
        else:
            raise ValueError(f"Unknown task_training_mode: {task_training_mode}")

        # --- eval ---
        log.info(f"\n--- Evaluation after Task {task_idx} ({task.name}) ---")
        model = _ensure_cuda(model)
        evaluate_all_tasks(
            model=model, cfg=cfg, tracker=tracker, train_exp_id=task_idx,
            phase_name=f"after_task_{task_idx}", wandb_group=wandb_group,
            output_dir=output_dir, is_hope=is_hope,
        )

        # --- filmstrip ---
        model = _ensure_cuda(model)  # re-enforce after Lightning Trainer.test()
        generate_filmstrips_for_phase(
            model=model, cfg=cfg, pca=pca,
            phase_name=f"after_task_{task_idx}", output_dir=output_dir,
        )

    _log_final_summary(cfg, tracker, wandb_group, output_dir)


# =============================================================================
# Hydra entry point
# =============================================================================


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """Run the CL pipeline with filmstrip generation."""
    log_device_info()
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    _run_sequential_pipeline_with_filmstrip(cfg)


if __name__ == "__main__":
    main()
