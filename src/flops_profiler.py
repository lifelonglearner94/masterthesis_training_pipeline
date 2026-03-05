#!/usr/bin/env python3
"""
Iso-FLOP / Computational Cost Profiler for all CL experiment architectures.

Measures per-step FLOPs, wall-clock time, memory, and parameter counts for
every model architecture, then extrapolates to full training costs.

Usage:
    uv run python src/flops_profiler.py [--device cuda|cpu] [--batch-size 4]
                                        [--warmup 3] [--repeats 10]
                                        [--models MODEL1,MODEL2,...]

Example:
    uv run python src/flops_profiler.py --device cuda --batch-size 4
    uv run python src/flops_profiler.py --models ac_vit,titans,retnet

The profiler estimates:
  1. Forward-only FLOPs (inference / evaluation cost)
  2. Forward + Backward FLOPs (training cost per step)
  3. Wall-clock time per step (forward, backward, total)
  4. Peak memory per step
  5. Parameter counts (total, trainable)
  6. Extrapolated cost per epoch (given dataset size) and per full CL pipeline
"""

import argparse
import gc
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================================
# Model factory — instantiates each backbone with experiment-matching params
# ============================================================================

def _make_ac_vit():
    """AC-ViT (24-layer transformer baseline, used in upper/lower bound)."""
    from models.ac_predictor import ACPredictorModule
    module = ACPredictorModule(
        img_size=(256, 256), patch_size=16, num_timesteps=8,
        embed_dim=1024, predictor_embed_dim=384, depth=24,
        num_heads=16, mlp_ratio=4.0, action_embed_dim=2,
        use_rope=True, is_frame_causal=True,
        use_activation_checkpointing=False, use_extrinsics=False,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
        # Loss (not used for profiling, but required)
        T_teacher=7, jump_k=3, loss_weight_teacher=1.0,
        loss_weight_jump=1.0, normalize_reps=True, loss_type="l1",
        # Optimizer (not used for profiling)
        learning_rate=4.25e-4, weight_decay=0.04,
    )
    return module.model  # Return raw backbone


def _make_ac_hope():
    """AC-HOPE-ViT (Phase 7: 5 blocks, Titan + CMS, no attention)."""
    from models.hope import ACHOPEModule
    module = ACHOPEModule(
        img_size=(256, 256), patch_size=16, num_timesteps=8,
        embed_dim=1024, predictor_embed_dim=384, depth=6,
        num_heads=16, action_embed_dim=2,
        use_rope=True, is_frame_causal=True,
        use_activation_checkpointing=False, use_extrinsics=False,
        drop_rate=0.0, drop_path_rate=0.1,
        # HOPE-specific
        titan_hidden_multiplier=2, titan_layers=2,
        titan_activation="gelu", titan_grad_clip_inner=1.0,
        titan_grad_clip_backward=1.0,
        cms_level_specs=[
            {"name": "fast", "update_period": 1, "hidden_multiplier": 4.0, "warmup_steps": 0},
            {"name": "medium", "update_period": 4, "hidden_multiplier": 4.0, "warmup_steps": 0},
            {"name": "slow", "update_period": 16, "hidden_multiplier": 4.0, "warmup_steps": 0},
        ],
        cms_use_chunk_scheduling=False, chunk_size=1,
        titan_detach_interval=1, surprise_threshold=0.0,
        log_hope_diagnostics=False, use_spatial_mixing=False,
        use_longterm_memory=False,
        # Loss
        T_teacher=7, jump_k=3, loss_weight_teacher=1.0,
        loss_weight_jump=1.0, normalize_reps=True, loss_type="l1",
        # Optimizer
        learning_rate=1.5e-4, weight_decay=0.04,
    )
    return module.model


def _make_ac_hope_hybrid():
    """AC-HOPE-Hybrid-ViT (Phase 8: 12 blocks, Attention + Titan + CMS)."""
    from models.hope import ACHOPEHybridModule
    module = ACHOPEHybridModule(
        img_size=(256, 256), patch_size=16, num_timesteps=8,
        embed_dim=1024, predictor_embed_dim=384, depth=12,
        num_heads=16, action_embed_dim=2,
        is_frame_causal=True, use_activation_checkpointing=False,
        use_extrinsics=False, drop_rate=0.0, drop_path_rate=0.1,
        # Hybrid-specific
        qkv_bias=True, attn_drop_rate=0.0,
        titan_hidden_multiplier=2, titan_layers=2,
        titan_activation="gelu", titan_grad_clip_inner=1.0,
        titan_grad_clip_backward=1.0,
        titan_detach_interval=2, surprise_threshold=0.1,
        use_longterm_memory=False,  # profiling without longterm
        cms_level_specs=[
            {"name": "fast", "update_period": 1, "hidden_multiplier": 2.0, "warmup_steps": 0},
            {"name": "medium", "update_period": 3, "hidden_multiplier": 2.5, "warmup_steps": 0},
            {"name": "slow", "update_period": 7, "hidden_multiplier": 3.0, "warmup_steps": 0},
        ],
        cms_use_chunk_scheduling=True,
        log_hope_diagnostics=False,
        # Loss
        T_teacher=7, jump_k=3, loss_weight_teacher=1.0,
        loss_weight_jump=1.0, normalize_reps=True, loss_type="l1",
        # Optimizer
        learning_rate=1.5e-4, weight_decay=0.04,
    )
    return module.model


def _make_ac_dnh_hope_hybrid():
    """AC-DNH-HOPE-Hybrid-ViT (Phase 11: DNH with structural evolution)."""
    from models.hope import ACDNHHOPEHybridModule
    module = ACDNHHOPEHybridModule(
        img_size=(256, 256), patch_size=16, num_timesteps=8,
        embed_dim=1024, predictor_embed_dim=384, depth=12,
        num_heads=16, action_embed_dim=2,
        is_frame_causal=True, use_activation_checkpointing=False,
        use_extrinsics=False, drop_rate=0.0, drop_path_rate=0.1,
        # DNH-specific
        qkv_bias=True, attn_drop_rate=0.0,
        dnh_L_init=2, dnh_L_max=5, dnh_L_min=2,
        titan_hidden_multiplier=1, titan_layers=2,
        titan_activation="gelu", titan_grad_clip_inner=1.0,
        titan_grad_clip_backward=1.0,
        titan_detach_interval=2, surprise_threshold=0.1,
        meta_hidden_dim=192,
        use_longterm_memory=False,
        cms_level_specs=[
            {"name": "fast", "update_period": 1, "hidden_multiplier": 1.5, "warmup_steps": 0},
            {"name": "medium", "update_period": 3, "hidden_multiplier": 2.0, "warmup_steps": 0},
            {"name": "slow", "update_period": 7, "hidden_multiplier": 2.5, "warmup_steps": 0},
        ],
        cms_use_chunk_scheduling=True,
        cms_L_max=5, cms_L_min=2,
        log_hope_diagnostics=False,
        # Structural evolution (required parameters)
        dnh_tau_add=0.5, dnh_epsilon_prune=0.01,
        dnh_gamma_freq=0.1, dnh_eta_freq=0.01,
        dnh_beta_momentum=0.9, dnh_evolution_interval=50,
        dnh_evolution_warmup=200,
        dnh_enable_addition=True, dnh_enable_pruning=True,
        dnh_enable_freq_modulation=True,
        dnh_meta_lambda=0.01, dnh_meta_mu=0.001,
        # Loss
        T_teacher=7, jump_k=3, loss_weight_teacher=1.0,
        loss_weight_jump=1.0, normalize_reps=True, loss_type="l1",
        # Optimizer
        learning_rate=1.5e-4, weight_decay=0.04,
    )
    return module.model


def _make_titans():
    """Titans MAC (4 layers, attention + NMM + persistent memory)."""
    from models.titans import TitansLitModule
    module = TitansLitModule(
        input_dim=1024, hidden_dim=768, action_dim=2,
        spatial_size=16, pm_len=4, n_layers=4,
        n_layers_nmm=2, num_timesteps=8,
        alpha=0.999, eta=0.8, theta=0.3,
        T_teacher=7, jump_k=3, loss_weight_teacher=1.0,
        loss_weight_jump=1.0, normalize_reps=True, loss_type="l1",
        learning_rate=4.25e-4, weight_decay=0.04,
    )
    return module.model


def _make_retnet():
    """RetNet (6 layers, multi-scale retention + GLU)."""
    from models.retnet import RetNetLitModule
    module = RetNetLitModule(
        input_dim=1024, embed_dim=768, value_dim=1280,
        action_dim=2, spatial_size=16, n_layers=6,
        n_heads=4, ffn_dim=1280, recurrent_chunk_size=64,
        chunkwise_recurrent=True, num_timesteps=8,
        T_teacher=7, jump_k=3, loss_weight_teacher=1.0,
        loss_weight_jump=1.0, normalize_reps=True, loss_type="l1",
        learning_rate=4.25e-4, weight_decay=0.04,
    )
    return module.model


def _make_gated_delta_net():
    """GatedDeltaNet (5 layers, gated delta rule linear attention + SwiGLU)."""
    from models.gated_delta_net import GatedDeltaNetLitModule
    module = GatedDeltaNetLitModule(
        input_dim=1024, hidden_dim=768, action_dim=2,
        spatial_size=16, n_layers=5, num_heads=12,
        expand_k=0.75, expand_v=1.5, intermediate_size=2048,
        conv_size=4, num_timesteps=8,
        T_teacher=7, jump_k=3, loss_weight_teacher=1.0,
        loss_weight_jump=1.0, normalize_reps=True, loss_type="l1",
        learning_rate=4.25e-4, weight_decay=0.04,
    )
    return module.model


def _make_transformer_pp():
    """Transformer++ (6 layers, MHA+RoPE + GatedMLP + depthwise conv)."""
    from models.transformer_pp import TransformerPPLitModule
    module = TransformerPPLitModule(
        input_dim=1024, d_model=768, action_dim=2,
        spatial_size=16, n_layers=6, n_heads=12,
        d_conv=4, rotary_emb_dim=64, mlp_hidden=2048,
        num_timesteps=8,
        T_teacher=7, jump_k=3, loss_weight_teacher=1.0,
        loss_weight_jump=1.0, normalize_reps=True, loss_type="l1",
        learning_rate=4.25e-4, weight_decay=0.04,
    )
    return module.model


# Registry of all models
MODEL_REGISTRY = {
    "ac_vit":             ("AC-ViT (baseline)",         _make_ac_vit),
    "ac_hope":            ("AC-HOPE-ViT",               _make_ac_hope),
    "ac_hope_hybrid":     ("AC-HOPE-Hybrid (Ph8)",      _make_ac_hope_hybrid),
    "ac_dnh_hope_hybrid": ("AC-DNH-HOPE-Hybrid (Ph11)", _make_ac_dnh_hope_hybrid),
    "titans":             ("Titans MAC",                 _make_titans),
    "retnet":             ("RetNet",                     _make_retnet),
    "gated_delta_net":    ("GatedDeltaNet",              _make_gated_delta_net),
    "transformer_pp":     ("Transformer++",              _make_transformer_pp),
}

# Training schedule for each experiment (for extrapolation)
# Format: (base_epochs, base_clips, task_epochs, task_clips, num_tasks)
TRAINING_SCHEDULE = {
    "ac_vit":             (40, 5000, 10, 1000, 5),
    "ac_hope":            (65, 5000, 10, 1000, 5),
    "ac_hope_hybrid":     (40, 5000, 10, 1000, 5),
    "ac_dnh_hope_hybrid": (40, 5000, 10, 1000, 5),
    "titans":             (40, 5000, 10, 1000, 5),
    "retnet":             (40, 5000, 10, 1000, 5),
    "gated_delta_net":    (40, 5000, 10, 1000, 5),
    "transformer_pp":     (40, 5000, 10, 1000, 5),
}

# Batch sizes used in actual experiments
BATCH_SIZES = {
    "ac_vit": 16, "ac_hope": 16, "ac_hope_hybrid": 16,
    "ac_dnh_hope_hybrid": 8, "titans": 16, "retnet": 16,
    "gated_delta_net": 8, "transformer_pp": 16,
}


# ============================================================================
# Measurement utilities
# ============================================================================

def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_dummy_inputs(batch_size: int, device: torch.device, dtype: torch.dtype):
    """Create dummy inputs matching CL experiment shapes: [B, T*N, D]."""
    T, N, D = 8, 256, 1024
    action_dim = 2
    features = torch.randn(batch_size, T * N, D, device=device, dtype=dtype)
    actions = torch.randn(batch_size, T, action_dim, device=device, dtype=dtype)
    states = torch.randn(batch_size, T, action_dim, device=device, dtype=dtype)
    return features, actions, states


@contextmanager
def cuda_memory_tracker(device):
    """Track peak GPU memory on a CUDA device."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    yield
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_forward_flops(model: nn.Module, inputs: tuple, device: torch.device):
    """
    Measure forward-pass FLOPs using torch.profiler.
    Returns estimated FLOPs (int) or None if measurement fails.
    """
    features, actions, states = inputs

    try:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
            ] + ([torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else []),
            with_flops=True,
            record_shapes=True,
        ) as prof:
            with torch.no_grad():
                _ = model(features, actions, states)

        # Sum up all estimated FLOPs from profiler events
        total_flops = 0
        for evt in prof.key_averages():
            if evt.flops and evt.flops > 0:
                total_flops += evt.flops
        return total_flops if total_flops > 0 else None
    except Exception as e:
        print(f"    [WARN] torch.profiler FLOPs measurement failed: {e}")
        return None


def measure_training_step(
    model: nn.Module,
    inputs: tuple,
    device: torch.device,
    warmup: int = 3,
    repeats: int = 10,
):
    """
    Measure wall-clock time and memory for a full training step (fwd + bwd).
    Returns dict with timing stats.
    """
    features, actions, states = inputs
    target = features.clone()  # Dummy target for loss

    # Create a simple L1 loss
    loss_fn = nn.L1Loss()

    # Warmup
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        out = model(features, actions, states)
        loss = loss_fn(out, target)
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Measure forward only
    fwd_times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(features, actions, states)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fwd_times.append(time.perf_counter() - t0)

    # Measure forward + backward
    fwd_bwd_times = []
    for _ in range(repeats):
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out = model(features, actions, states)
        loss = loss_fn(out, target)
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fwd_bwd_times.append(time.perf_counter() - t0)

    peak_mem = 0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GiB

    return {
        "fwd_mean_ms": 1000 * sum(fwd_times) / len(fwd_times),
        "fwd_std_ms": 1000 * (sum((t - sum(fwd_times)/len(fwd_times))**2 for t in fwd_times) / len(fwd_times)) ** 0.5,
        "fwd_bwd_mean_ms": 1000 * sum(fwd_bwd_times) / len(fwd_bwd_times),
        "fwd_bwd_std_ms": 1000 * (sum((t - sum(fwd_bwd_times)/len(fwd_bwd_times))**2 for t in fwd_bwd_times) / len(fwd_bwd_times)) ** 0.5,
        "peak_memory_gib": peak_mem,
    }


def measure_training_flops(model: nn.Module, inputs: tuple, device: torch.device):
    """
    Measure forward+backward FLOPs using torch.profiler.
    Returns estimated FLOPs or None.
    """
    features, actions, states = inputs
    target = features.clone()
    loss_fn = nn.L1Loss()

    try:
        model.zero_grad(set_to_none=True)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
            ] + ([torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else []),
            with_flops=True,
            record_shapes=True,
        ) as prof:
            out = model(features, actions, states)
            loss = loss_fn(out, target)
            loss.backward()

        total_flops = 0
        for evt in prof.key_averages():
            if evt.flops and evt.flops > 0:
                total_flops += evt.flops
        return total_flops if total_flops > 0 else None
    except Exception as e:
        print(f"    [WARN] torch.profiler training FLOPs measurement failed: {e}")
        return None


# ============================================================================
# Formatting helpers
# ============================================================================

def fmt_flops(flops):
    """Format FLOPs with appropriate SI prefix."""
    if flops is None:
        return "N/A"
    if flops >= 1e12:
        return f"{flops/1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f} MFLOPs"
    else:
        return f"{flops:.0f} FLOPs"


def fmt_params(n):
    """Format parameter count."""
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def fmt_time(ms):
    """Format time in ms."""
    if ms >= 1000:
        return f"{ms/1000:.2f}s"
    return f"{ms:.1f}ms"


def fmt_hours(seconds):
    """Format seconds as hours."""
    return f"{seconds / 3600:.1f}h"


# ============================================================================
# Main profiling loop
# ============================================================================

def profile_all(
    models_to_profile: list[str],
    device_str: str = "cuda",
    batch_size: int = 4,
    warmup: int = 3,
    repeats: int = 10,
    use_wandb: bool = False,
    wandb_project: str = "flops-profiler",
    wandb_name: str | None = None,
):
    """Profile all selected models and print comparison table."""
    device = torch.device(device_str)
    dtype = torch.float32

    # ── W&B initialization ──────────────────────────────────────────────
    wb_run = None
    if use_wandb:
        if not _WANDB_AVAILABLE:
            print("[WARN] wandb not installed, skipping W&B logging.")
        else:
            gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            wb_run = wandb.init(
                project=wandb_project,
                name=wandb_name or f"flops-profiler-{gpu_name.replace(' ', '_')}",
                config={
                    "device": device_str,
                    "gpu": gpu_name,
                    "batch_size": batch_size,
                    "warmup": warmup,
                    "repeats": repeats,
                    "models": models_to_profile,
                },
                tags=["profiling"],
            )
            print(f"  W&B run: {wb_run.url}")

    print("=" * 90)
    print("  ISO-FLOP PROFILER — Computational Cost Comparison")
    print("=" * 90)
    print(f"  Device:     {device}")
    if device.type == "cuda":
        print(f"  GPU:        {torch.cuda.get_device_name(device)}")
        print(f"  VRAM:       {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence:   [B, {8*256}, 1024]  (T=8, N=256 patches, D=1024)")
    print(f"  Warmup:     {warmup} iterations")
    print(f"  Repeats:    {repeats} iterations")
    print(f"  Precision:  float32")
    print("=" * 90)
    print()

    results = {}

    for model_key in models_to_profile:
        if model_key not in MODEL_REGISTRY:
            print(f"[SKIP] Unknown model: {model_key}")
            continue

        display_name, factory_fn = MODEL_REGISTRY[model_key]
        print(f"{'─' * 60}")
        print(f"  Profiling: {display_name}")
        print(f"{'─' * 60}")

        # Instantiate
        try:
            model = factory_fn()
            model = model.to(device=device, dtype=dtype)
            model.train()  # BN/dropout in training mode
        except Exception as e:
            print(f"  [ERROR] Failed to instantiate: {e}")
            print()
            continue

        # Count params
        total_params, train_params = count_parameters(model)
        print(f"  Parameters: {fmt_params(total_params)} total, {fmt_params(train_params)} trainable")

        # Create inputs
        inputs = make_dummy_inputs(batch_size, device, dtype)

        # Forward FLOPs
        print("  Measuring forward FLOPs...")
        fwd_flops = measure_forward_flops(model, inputs, device)
        print(f"  Forward FLOPs: {fmt_flops(fwd_flops)}")

        # Training FLOPs (fwd + bwd)
        print("  Measuring training FLOPs (fwd+bwd)...")
        train_flops = measure_training_flops(model, inputs, device)
        print(f"  Training FLOPs: {fmt_flops(train_flops)}")

        # Wall-clock timing
        print(f"  Timing ({warmup} warmup + {repeats} measured iterations)...")
        timing = measure_training_step(model, inputs, device, warmup, repeats)
        print(f"  Forward:  {fmt_time(timing['fwd_mean_ms'])} ± {fmt_time(timing['fwd_std_ms'])}")
        print(f"  Fwd+Bwd:  {fmt_time(timing['fwd_bwd_mean_ms'])} ± {fmt_time(timing['fwd_bwd_std_ms'])}")
        if timing["peak_memory_gib"] > 0:
            print(f"  Peak VRAM: {timing['peak_memory_gib']:.2f} GiB")

        # Extrapolation
        actual_bs = BATCH_SIZES.get(model_key, 16)
        schedule = TRAINING_SCHEDULE.get(model_key, (40, 5000, 10, 1000, 5))
        base_epochs, base_clips, task_epochs, task_clips, num_tasks = schedule

        # Scale timing to actual batch size
        time_scale = actual_bs / batch_size
        ms_per_step = timing["fwd_bwd_mean_ms"] * time_scale

        base_steps_per_epoch = base_clips / actual_bs
        task_steps_per_epoch = task_clips / actual_bs

        total_base_steps = base_steps_per_epoch * base_epochs
        total_task_steps = task_steps_per_epoch * task_epochs * num_tasks
        total_steps = total_base_steps + total_task_steps

        base_time_s = total_base_steps * ms_per_step / 1000
        task_time_s = total_task_steps * ms_per_step / 1000
        total_time_s = base_time_s + task_time_s

        # FLOPs extrapolation
        total_train_flops = None
        if train_flops:
            flops_per_actual_step = train_flops * time_scale
            total_train_flops = flops_per_actual_step * total_steps

        print(f"\n  ── Extrapolated CL Pipeline (bs={actual_bs}) ──")
        print(f"  Base training:  {total_base_steps:.0f} steps × {base_epochs} epochs = ~{fmt_hours(base_time_s)}")
        print(f"  Task finetuning: {total_task_steps:.0f} steps × {num_tasks} tasks = ~{fmt_hours(task_time_s)}")
        print(f"  Total wall-clock estimate: ~{fmt_hours(total_time_s)}")
        if total_train_flops:
            print(f"  Total training FLOPs:      ~{fmt_flops(total_train_flops)}")

        results[model_key] = {
            "name": display_name,
            "total_params": total_params,
            "train_params": train_params,
            "fwd_flops": fwd_flops,
            "train_flops": train_flops,
            "fwd_ms": timing["fwd_mean_ms"],
            "fwd_bwd_ms": timing["fwd_bwd_mean_ms"],
            "peak_gib": timing["peak_memory_gib"],
            "total_steps": total_steps,
            "estimated_hours": total_time_s / 3600,
            "total_train_flops": total_train_flops,
        }

        # ── Log per-model metrics to W&B ────────────────────────────────
        if wb_run is not None:
            prefix = f"models/{model_key}"
            wb_run.summary[f"{prefix}/total_params"] = total_params
            wb_run.summary[f"{prefix}/trainable_params"] = train_params
            wb_run.summary[f"{prefix}/fwd_flops"] = fwd_flops
            wb_run.summary[f"{prefix}/train_flops"] = train_flops
            wb_run.summary[f"{prefix}/fwd_ms"] = timing["fwd_mean_ms"]
            wb_run.summary[f"{prefix}/fwd_std_ms"] = timing["fwd_std_ms"]
            wb_run.summary[f"{prefix}/fwd_bwd_ms"] = timing["fwd_bwd_mean_ms"]
            wb_run.summary[f"{prefix}/fwd_bwd_std_ms"] = timing["fwd_bwd_std_ms"]
            wb_run.summary[f"{prefix}/peak_memory_gib"] = timing["peak_memory_gib"]
            wb_run.summary[f"{prefix}/estimated_hours"] = total_time_s / 3600
            if total_train_flops:
                wb_run.summary[f"{prefix}/total_train_flops"] = total_train_flops

        # Cleanup
        del model, inputs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print()

    # ======================================================================
    # Summary comparison table
    # ======================================================================
    if len(results) < 2:
        return results

    print("\n" + "=" * 130)
    print("  COMPARISON TABLE")
    print("=" * 130)

    # Header
    header = f"{'Model':<30} {'Params':>8} {'Fwd FLOPs':>14} {'Train FLOPs':>14} {'Fwd (ms)':>10} {'Fwd+Bwd':>10} {'VRAM':>8} {'Est. Hours':>10}"
    print(header)
    print("─" * 130)

    # Find baseline for relative comparison
    baseline_key = "ac_vit" if "ac_vit" in results else list(results.keys())[0]
    baseline = results[baseline_key]

    for key in models_to_profile:
        if key not in results:
            continue
        r = results[key]

        # Relative to baseline
        fwd_rel = f"({r['fwd_bwd_ms']/baseline['fwd_bwd_ms']:.2f}×)" if baseline["fwd_bwd_ms"] > 0 else ""
        flops_rel = ""
        if r["train_flops"] and baseline["train_flops"]:
            flops_rel = f"({r['train_flops']/baseline['train_flops']:.2f}×)"

        row = (
            f"{r['name']:<30} "
            f"{fmt_params(r['train_params']):>8} "
            f"{fmt_flops(r['fwd_flops']):>14} "
            f"{fmt_flops(r['train_flops']):>14} "
            f"{r['fwd_ms']:>9.1f} "
            f"{r['fwd_bwd_ms']:>9.1f} "
            f"{r['peak_gib']:>7.2f} "
            f"{r['estimated_hours']:>9.1f}"
        )
        print(row)

    print("─" * 130)
    print(f"\n  Relative to {baseline['name']}:")
    for key in models_to_profile:
        if key not in results or key == baseline_key:
            continue
        r = results[key]
        parts = [f"  {r['name']:<30}"]
        if baseline["fwd_bwd_ms"] > 0:
            parts.append(f"time: {r['fwd_bwd_ms']/baseline['fwd_bwd_ms']:.2f}×")
        if r["train_flops"] and baseline["train_flops"]:
            parts.append(f"FLOPs: {r['train_flops']/baseline['train_flops']:.2f}×")
        if r["total_train_flops"] and baseline["total_train_flops"]:
            parts.append(f"total: {r['total_train_flops']/baseline['total_train_flops']:.2f}×")
        print("  ".join(parts))

    # Per-timestep cost analysis
    print(f"\n{'=' * 90}")
    print("  PER-TIMESTEP ANALYSIS")
    print(f"{'=' * 90}")
    print("  (Dividing total step cost by T=8 timesteps to get per-timestep cost)")
    print()
    T = 8
    for key in models_to_profile:
        if key not in results:
            continue
        r = results[key]
        per_ts_ms = r["fwd_bwd_ms"] / T
        per_ts_flops = r["train_flops"] / T if r["train_flops"] else None
        print(f"  {r['name']:<30}  {per_ts_ms:.1f} ms/timestep  {fmt_flops(per_ts_flops)}/timestep")

    print()
    print("NOTE: FLOPs from torch.profiler may undercount operations in custom")
    print("      kernels (Triton, GatedDeltaNet) and inner-loop gradient computations")
    print("      (HOPE DGD, Titans NMM). Wall-clock time is the more reliable metric")
    print("      for comparing relative costs across architectures.")
    print()

    # ── Log summary table to W&B ────────────────────────────────────────
    if wb_run is not None and len(results) >= 1:
        columns = [
            "Model", "Trainable Params", "Fwd FLOPs", "Train FLOPs",
            "Fwd (ms)", "Fwd+Bwd (ms)", "Peak VRAM (GiB)",
            "Est. CL Hours", "Total Train FLOPs",
        ]
        table = wandb.Table(columns=columns)
        for key in models_to_profile:
            if key not in results:
                continue
            r = results[key]
            table.add_data(
                r["name"],
                r["train_params"],
                r["fwd_flops"],
                r["train_flops"],
                round(r["fwd_ms"], 2),
                round(r["fwd_bwd_ms"], 2),
                round(r["peak_gib"], 3),
                round(r["estimated_hours"], 2),
                r["total_train_flops"],
            )
        wb_run.log({"comparison_table": table})

        # Log bar charts for quick visual comparison
        model_names = [results[k]["name"] for k in models_to_profile if k in results]
        fwd_bwd_times = [results[k]["fwd_bwd_ms"] for k in models_to_profile if k in results]
        train_flops_vals = [results[k]["train_flops"] or 0 for k in models_to_profile if k in results]
        param_counts = [results[k]["train_params"] for k in models_to_profile if k in results]

        wb_run.log({
            "fwd_bwd_ms": wandb.plot.bar(
                wandb.Table(
                    data=list(zip(model_names, fwd_bwd_times)),
                    columns=["Model", "Fwd+Bwd (ms)"],
                ),
                "Model", "Fwd+Bwd (ms)", title="Forward+Backward Time (ms)",
            ),
            "train_flops_chart": wandb.plot.bar(
                wandb.Table(
                    data=list(zip(model_names, train_flops_vals)),
                    columns=["Model", "Train FLOPs"],
                ),
                "Model", "Train FLOPs", title="Training FLOPs per Step",
            ),
            "trainable_params_chart": wandb.plot.bar(
                wandb.Table(
                    data=list(zip(model_names, param_counts)),
                    columns=["Model", "Trainable Params"],
                ),
                "Model", "Trainable Params", title="Trainable Parameters",
            ),
        })

        wb_run.finish()
        print(f"  W&B run finished: {wb_run.url}")

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Iso-FLOP profiler for CL experiment architectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to profile on (default: cuda if available)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Profiling batch size (smaller = less VRAM; results are scaled for comparison)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Number of warmup iterations before measurement",
    )
    parser.add_argument(
        "--repeats", type=int, default=10,
        help="Number of measured iterations for timing",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated list of models to profile. "
             f"Available: {','.join(MODEL_REGISTRY.keys())}. Default: all.",
    )
    parser.add_argument(
        "--wandb", action="store_true", default=False,
        help="Log results to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="flops-profiler",
        help="W&B project name (default: flops-profiler).",
    )
    parser.add_argument(
        "--wandb-name", type=str, default=None,
        help="W&B run name (default: auto-generated from GPU name).",
    )

    args = parser.parse_args()

    if args.models:
        models_to_profile = [m.strip() for m in args.models.split(",")]
    else:
        models_to_profile = list(MODEL_REGISTRY.keys())

    profile_all(
        models_to_profile=models_to_profile,
        device_str=args.device,
        batch_size=args.batch_size,
        warmup=args.warmup,
        repeats=args.repeats,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )


if __name__ == "__main__":
    main()
