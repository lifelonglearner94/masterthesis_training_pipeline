"""Qualitative filmstrip visualization for latent-space rollouts.

Generates PCA→RGB images comparing ground truth vs autoregressive predictions
to visualize physics retention vs forgetting after continual learning.

Pipeline ("DINO trick" for ViT latent visualisation):
    1. Autoregressive rollout: z₁ (GT) → ẑ₂ → ẑ₃ → … → ẑ₈
    2. PCA fitting: Fit PCA(D=1024→4) on base-dataset ground truth
    3. PCA→RGB: Discard PC0 (RoPE background drift), use PC1-3 as RGB
    4. Upsampling: Bicubic interpolation to 128×128
    5. Filmstrip assembly: Side-by-side GT vs Predicted grids

Tensor conventions:
    features: [B, T+1, N, D]  where T=8, N=256, D=1024
    actions:  [B, T,   action_dim]
    states:   [B, T,   action_dim]
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import Tensor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 1. Autoregressive Rollout
# ─────────────────────────────────────────────────────────────


@torch.no_grad()
def autoregressive_rollout(
    model,
    features: Tensor,
    actions: Tensor,
    states: Tensor,
    num_steps: int = 7,
) -> Tensor:
    """Autoregressive rollout producing [8, N, D] predictions.

    Protocol (true compounding-error test):
        Frame 0 : ground-truth z₁ (copied verbatim)
        Frame 1 : model(z₁, a₁)          — true action
        Frame 2…7: model(ẑₜ, 0)          — zero action, input = previous output

    Calls ``model.model(...)`` directly (the backbone), bypassing layer-norm
    so that predictions live in the same representation space as raw GT
    features, which is required for a consistent PCA colour mapping.

    Args:
        model: Lightning module (ACHOPEHybridModule or GatedDeltaNetLitModule).
        features: [1, T+1, N, D] — single clip.
        actions:  [1, T, action_dim].
        states:   [1, T, action_dim].
        num_steps: Number of autoregressive prediction steps (default 7).

    Returns:
        Tensor [8, N, D] on CPU — frame 0 is GT, frames 1–7 are predictions.
    """
    model.eval()
    device = next(model.parameters()).device
    features = features.to(device)
    actions = actions.to(device)
    states = states.to(device)

    # Reset clip-level memories (HOPE Hybrid); no-op for stateless models
    backbone = model.model
    if hasattr(backbone, "reset_all_memories"):
        backbone.reset_all_memories()

    _B, _Tp1, N, D = features.shape
    action_dim = actions.shape[-1]

    # Frame 0: verbatim ground truth
    rollout: list[Tensor] = [features[0, 0].cpu()]  # [N, D]

    # z_current: [1, N, D] — autoregressive state
    z_current = features[:, 0:1].reshape(1, N, D)

    for t in range(num_steps):
        # True action for the first step, zero afterwards
        if t == 0:
            a_t = actions[:, 0:1]  # [1, 1, action_dim]
            s_t = states[:, 0:1]
        else:
            a_t = torch.zeros(1, 1, action_dim, device=device, dtype=actions.dtype)
            s_t = torch.zeros(1, 1, action_dim, device=device, dtype=states.dtype)

        target_ts = t + 1  # temporal position of target frame

        # Call backbone directly (no layer-norm → same space as raw GT)
        z_pred = backbone(z_current, a_t, s_t, target_timestep=target_ts)  # [1, N, D]

        rollout.append(z_pred[0].cpu())
        z_current = z_pred.detach()

    return torch.stack(rollout, dim=0)  # [8, N, D]


# ─────────────────────────────────────────────────────────────
# 2. PCA Fitting
# ─────────────────────────────────────────────────────────────


def fit_pca_on_base_clips(
    datamodule,
    n_components: int = 4,
    max_clips: int = 50,
) -> PCA:
    """Fit PCA(D→4) on ground-truth features from evaluation clips.

    We fit 4 components so that PC0 (dominated by 3D-RoPE background
    drift) can be discarded, leaving PC1-PC3 as clean RGB channels
    (the "DINO trick").

    Args:
        datamodule: A LightningDataModule whose test_dataloader yields
            batches with ``features: [B, T+1, N, D]``.
        n_components: PCA target dimensionality (4 = 1 discarded + 3 RGB).
        max_clips: Maximum clips to use for fitting (memory bound).

    Returns:
        Fitted sklearn PCA object.
    """
    datamodule.setup("test")
    dl = datamodule.test_dataloader()

    all_features: list[np.ndarray] = []
    n_clips = 0
    for batch in dl:
        feats = batch["features"]  # [B, T+1, N, D]
        B = feats.shape[0]
        for b in range(B):
            clip_feats = feats[b].reshape(-1, feats.shape[-1])  # [(T+1)*N, D]
            all_features.append(clip_feats.float().numpy())
            n_clips += 1
            if n_clips >= max_clips:
                break
        if n_clips >= max_clips:
            break

    all_features_arr = np.concatenate(all_features, axis=0)  # [num_samples, D]
    log.info(
        f"Fitting PCA on {all_features_arr.shape[0]} vectors from {n_clips} clips"
    )

    pca = PCA(n_components=n_components)
    pca.fit(all_features_arr)
    log.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    return pca


# ─────────────────────────────────────────────────────────────
# 3. PCA → RGB conversion
# ─────────────────────────────────────────────────────────────


def latent_to_rgb(
    tensor: Tensor,
    pca: PCA,
    grid_h: int = 16,
    grid_w: int = 16,
    upsample_size: int = 128,
    per_frame_norm: bool = False,
) -> np.ndarray:
    """Convert latent tensor [T, N, D] → upsampled RGB images.

    Uses the "DINO trick": PCA projects to 4 components, PC0 (which
    captures the 3D-RoPE background drift) is discarded, and PC1-PC3
    are used as RGB channels.

    Steps:
        1. PCA: [T*N, D] → [T*N, 4]
        2. Discard PC0, keep PC1-PC3 as RGB
        3. Per-component normalisation to [0, 1]
        4. Reshape to [T, H, W, 3]
        5. Bicubic upsample to ``upsample_size``

    Args:
        per_frame_norm: If True, normalise each frame independently.
            Useful for predicted features whose magnitude differs
            from the GT features the PCA was fitted on.

    Returns:
        np.ndarray [T, upsample_size, upsample_size, 3]  float32 in [0, 1].
    """
    T, N, D = tensor.shape
    flat = tensor.reshape(-1, D).float().numpy()  # [T*N, D]
    pca_all = pca.transform(flat)  # [T*N, n_components]

    # DINO trick: discard PC0 (RoPE background drift), keep PC1-PC3
    if pca_all.shape[1] >= 4:
        pca_3d = pca_all[:, 1:4]  # [T*N, 3]
    else:
        # Fallback for legacy 3-component PCA objects
        pca_3d = pca_all[:, :3]

    # Normalise each component to [0, 1]
    if per_frame_norm:
        pca_3d = pca_3d.reshape(T, N, 3)
        for t in range(T):
            for c in range(3):
                c_min, c_max = pca_3d[t, :, c].min(), pca_3d[t, :, c].max()
                if c_max - c_min > 1e-8:
                    pca_3d[t, :, c] = (pca_3d[t, :, c] - c_min) / (c_max - c_min)
                else:
                    pca_3d[t, :, c] = 0.5
        pca_3d = pca_3d.reshape(T * N, 3)
    else:
        for c in range(3):
            c_min, c_max = pca_3d[:, c].min(), pca_3d[:, c].max()
            if c_max - c_min > 1e-8:
                pca_3d[:, c] = (pca_3d[:, c] - c_min) / (c_max - c_min)
            else:
                pca_3d[:, c] = 0.5

    images = pca_3d.reshape(T, grid_h, grid_w, 3).astype(np.float32)

    # Bicubic upsample (smoother than bilinear for low-res patch grids)
    images_t = torch.from_numpy(images).permute(0, 3, 1, 2)  # [T, 3, H, W]
    images_up = F.interpolate(
        images_t,
        size=(upsample_size, upsample_size),
        mode="bicubic",
        align_corners=False,
    )
    images_up = images_up.permute(0, 2, 3, 1).numpy()  # [T, H, W, 3]
    return np.clip(images_up, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────
# 4. Filmstrip rendering (matplotlib)
# ─────────────────────────────────────────────────────────────


def render_filmstrip(
    gt_images: np.ndarray,
    pred_images: np.ndarray,
    title: str = "",
    save_path: str | Path | None = None,
) -> np.ndarray:
    """Render a 2-row filmstrip: GT (top) vs Predicted (bottom).

    Args:
        gt_images:   [T, H, W, 3] float32 in [0, 1].
        pred_images: [T, H, W, 3] float32 in [0, 1].
        title: Figure suptitle.
        save_path: Where to save the PNG (parent dirs created automatically).

    Returns:
        np.ndarray of the rendered figure as an RGB image.
    """
    T = gt_images.shape[0]

    fig, axes = plt.subplots(2, T, figsize=(T * 1.6, 3.8))

    for t in range(T):
        axes[0, t].imshow(gt_images[t])
        axes[0, t].set_xticks([])
        axes[0, t].set_yticks([])
        axes[0, t].set_title(f"t={t}", fontsize=8)
        if t == 0:
            axes[0, t].set_ylabel("Ground Truth", fontsize=9, fontweight="bold")

        axes[1, t].imshow(pred_images[t])
        axes[1, t].set_xticks([])
        axes[1, t].set_yticks([])
        if t == 0:
            axes[1, t].set_ylabel("Predicted", fontsize=9, fontweight="bold")

    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        log.info(f"Filmstrip saved: {save_path}")

    # Convert figure canvas → numpy RGB array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[..., :3].copy()  # drop alpha
    plt.close(fig)
    return img


# ─────────────────────────────────────────────────────────────
# 5. High-level orchestrator
# ─────────────────────────────────────────────────────────────


def generate_filmstrips_for_phase(
    model,
    cfg,
    pca: PCA,
    phase_name: str,
    output_dir: str,
) -> None:
    """Generate one filmstrip per task partition after a CL phase.

    Picks the **first** evaluation clip from each partition
    (base + 5 tasks = 6 filmstrips per phase).

    Args:
        model: Trained Lightning module.
        cfg: Full Hydra config (needs ``cl`` section).
        pca: Pre-fitted PCA(3) object.
        phase_name: E.g. ``"after_base"`` or ``"after_task_3"``.
        output_dir: Root output directory for the experiment.
    """
    from src.cl_train import create_datamodule

    cl_cfg = cfg.cl
    filmstrip_cfg = cl_cfg.get("filmstrip", {})
    upsample_size = filmstrip_cfg.get("upsample_size", 128)
    eval_clips = cl_cfg.eval.clips_per_task

    # Build one-clip partitions
    partitions: list[dict] = []
    base_end = cl_cfg.base_training.clip_end
    base_eval_start = base_end - eval_clips
    partitions.append({"name": "base", "clip_start": base_eval_start, "clip_end": base_eval_start + 1})
    for task in cl_cfg.tasks:
        task_eval_start = task.clip_end - eval_clips
        partitions.append({"name": task.name, "clip_start": task_eval_start, "clip_end": task_eval_start + 1})

    filmstrip_dir = Path(output_dir) / "filmstrips" / phase_name
    filmstrip_dir.mkdir(parents=True, exist_ok=True)

    # Freeze HOPE inner loops for pure inference
    is_hope = hasattr(model, "model") and hasattr(model.model, "freeze_all_inner_loops")
    if is_hope:
        model.model.freeze_all_inner_loops()

    model.eval()

    for partition in partitions:
        log.info(
            f"  Filmstrip for '{partition['name']}' (clip {partition['clip_start']})"
        )

        dm = create_datamodule(
            cfg,
            clip_start=partition["clip_start"],
            clip_end=partition["clip_end"],
            batch_size=1,
            val_split=0.0,
        )
        dm.setup("test")
        dl = dm.test_dataloader()

        batch = next(iter(dl))
        features = batch["features"]  # [1, T+1, N, D]
        actions = batch["actions"]    # [1, T, action_dim]
        states = batch["states"]      # [1, T, action_dim]

        # Ground truth: first 8 frames
        gt_tensor = features[0, :8].cpu()  # [8, N, D]

        # Autoregressive rollout
        pred_tensor = autoregressive_rollout(
            model, features, actions, states, num_steps=7,
        )  # [8, N, D]

        # Save raw tensors (so we can re-render if the PCA images are bad)
        tensor_path = filmstrip_dir / f"{partition['name']}_tensors.pt"
        torch.save(
            {"gt": gt_tensor, "pred": pred_tensor},
            str(tensor_path),
        )
        log.info(f"  Raw tensors saved: {tensor_path}")

        # PCA → RGB (per-frame norm for predictions so every frame is visible
        # even when the backbone output has a different magnitude than raw GT)
        gt_rgb = latent_to_rgb(gt_tensor, pca, upsample_size=upsample_size)
        pred_rgb = latent_to_rgb(
            pred_tensor, pca, upsample_size=upsample_size, per_frame_norm=True,
        )

        # Render and save
        save_path = filmstrip_dir / f"{partition['name']}.png"
        title = f"{phase_name} | {partition['name']}"
        render_filmstrip(gt_rgb, pred_rgb, title=title, save_path=save_path)

    # Unfreeze HOPE inner loops
    if is_hope:
        model.model.unfreeze_all_inner_loops()

    log.info(f"All filmstrips for '{phase_name}' saved to: {filmstrip_dir}")
