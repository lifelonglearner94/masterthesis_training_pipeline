#!/usr/bin/env python3
"""Render 4-row filmstrip comparisons: Real RGB / Ground Truth / AC-HOPE Hybrid / GatedDeltaNet.

Loads raw .pt tensor dumps from both model runs, fits a shared PCA on the
GT features, and produces publication-ready comparison figures.

The top row shows the actual RGB video frames (every 2nd frame to match
the encoder tubelet stride of 2, yielding 8 frames from 16).

Follows the EXACT same PCA→RGB pipeline as filmstrip_rollout.latent_to_rgb:
  1. PCA(D→4), discard PC0 (RoPE drift), keep PC1-PC3 as RGB
  2. Per-component min-max normalisation to [0, 1]
  3. Reshape to 16×16×3
  4. Bicubic upsample

Usage:
    .venv/bin/python src/scripts/render_3row_comparison.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ── Paths ────────────────────────────────────────────────────
HYBRID_DIR = Path("outputs/outputs/2026-03-21/11-06-41_Hybrid/filmstrips")
GDN_DIR = Path("outputs/outputs/2026-03-21/11-27-09_GDN/filmstrips")
OUTPUT_DIR = Path("outputs/comparison_filmstrips")
RGB_FRAMES_DIR = Path("outputs/comparison_filmstrips/six_clips_filmstrip")

PHASES = ["after_base", "after_task_1", "after_task_2", "after_task_3", "after_task_4", "after_task_5"]
PARTITIONS = ["base", "dissipation_shift", "kinematics_shift", "scaling_shift", "compositional_ood", "discretization_shift"]

# Mapping from partition name → real-RGB clip folder
PARTITION_TO_CLIP = {
    "base": "clip_04900_Base",
    "scaling_shift": "clip_05900_Task1",
    "dissipation_shift": "clip_06900_Task2",
    "discretization_shift": "clip_07900_Task3",
    "kinematics_shift": "clip_08900_Task4",
    "compositional_ood": "clip_09900_Task5",
}

# Mapping from partition name → task label for filenames
PARTITION_TO_TASK = {
    "base": "base",
    "scaling_shift": "task1_scaling_shift",
    "dissipation_shift": "task2_dissipation_shift",
    "discretization_shift": "task3_discretization_shift",
    "kinematics_shift": "task4_kinematics_shift",
    "compositional_ood": "task5_compositional_ood",
}

GRID_H, GRID_W = 16, 16


def fit_shared_pca(hybrid_dir: Path, n_components: int = 4) -> PCA:
    """Fit PCA on GT features from ALL phases & partitions."""
    all_feats = []
    for phase in PHASES:
        for part in PARTITIONS:
            pt_path = hybrid_dir / phase / f"{part}_tensors.pt"
            if pt_path.exists():
                data = torch.load(str(pt_path), map_location="cpu", weights_only=True)
                gt = data["gt"]  # [8, 256, 1024]
                all_feats.append(gt.reshape(-1, gt.shape[-1]).float().numpy())
    all_feats_arr = np.concatenate(all_feats, axis=0)
    print(f"Fitting PCA on {all_feats_arr.shape[0]:,} vectors from {len(all_feats)} clips")
    pca = PCA(n_components=n_components)
    pca.fit(all_feats_arr)
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    return pca


def latent_to_rgb(
    tensor: torch.Tensor,
    pca: PCA,
    upsample_size: int = 128,
    per_frame_norm: bool = False,
) -> np.ndarray:
    """Convert latent [T, N, D] → [T, H, W, 3] RGB.

    Exact copy of filmstrip_rollout.latent_to_rgb logic.
    """
    T, N, D = tensor.shape
    flat = tensor.reshape(-1, D).float().numpy()
    pca_all = pca.transform(flat)

    if pca_all.shape[1] >= 4:
        pca_3d = pca_all[:, 1:4]
    else:
        pca_3d = pca_all[:, :3]

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

    images = pca_3d.reshape(T, GRID_H, GRID_W, 3).astype(np.float32)

    images_t = torch.from_numpy(images).permute(0, 3, 1, 2)
    images_up = F.interpolate(
        images_t, size=(upsample_size, upsample_size),
        mode="bicubic", align_corners=False,
    )
    images_up = images_up.permute(0, 2, 3, 1).numpy()
    return np.clip(images_up, 0.0, 1.0)


def load_real_rgb_frames(
    partition: str,
    rgb_dir: Path,
    upsample_size: int = 128,
) -> np.ndarray | None:
    """Load real RGB frames for a partition, taking every 2nd frame (tubelet stride).

    Returns [8, upsample_size, upsample_size, 3] float32 in [0, 1].
    """
    clip_name = PARTITION_TO_CLIP.get(partition)
    if clip_name is None:
        return None
    clip_rgb_dir = rgb_dir / clip_name / "rgb"
    if not clip_rgb_dir.exists():
        return None

    frames = []
    for idx in range(0, 16, 2):  # 0, 2, 4, ..., 14 → 8 frames
        fpath = clip_rgb_dir / f"frame_{idx:05d}.png"
        if not fpath.exists():
            return None
        img = Image.open(fpath).convert("RGB")
        img_np = np.asarray(img).astype(np.float32) / 255.0
        frames.append(img_np)

    frames_arr = np.stack(frames, axis=0)  # [8, H_orig, W_orig, 3]
    # Resize to match the latent upsample size
    frames_t = torch.from_numpy(frames_arr).permute(0, 3, 1, 2)  # [8, 3, H, W]
    frames_up = F.interpolate(
        frames_t, size=(upsample_size, upsample_size),
        mode="bicubic", align_corners=False,
    )
    frames_up = frames_up.permute(0, 2, 3, 1).numpy()  # [8, us, us, 3]
    return np.clip(frames_up, 0.0, 1.0)


def render_4row_filmstrip(
    real_rgb: np.ndarray | None,
    gt_images: np.ndarray,
    hybrid_images: np.ndarray,
    gdn_images: np.ndarray,
    title: str = "",
    save_path: Path | None = None,
    dpi: int = 150,
) -> np.ndarray:
    """Render a 4-row filmstrip: Real RGB / GT / AC-HOPE Hybrid / GatedDeltaNet."""
    T = gt_images.shape[0]

    if real_rgb is not None:
        n_rows = 4
        row_labels = ["Real RGB", "Ground Truth", "AC-HOPE Hybrid", "GatedDeltaNet"]
        row_data = [real_rgb, gt_images, hybrid_images, gdn_images]
    else:
        n_rows = 3
        row_labels = ["Ground Truth", "AC-HOPE Hybrid", "GatedDeltaNet"]
        row_data = [gt_images, hybrid_images, gdn_images]

    cell_w = 1.35
    cell_h = 1.35
    fig_w = T * cell_w + 1.4
    fig_h = n_rows * cell_h + 0.6

    fig, axes = plt.subplots(
        n_rows, T, figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.03, "hspace": 0.08},
    )

    for row in range(n_rows):
        for t in range(T):
            ax = axes[row, t]
            ax.imshow(row_data[row][t])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("#555555")
            if row == 0:
                ax.set_title(f"$t = {t}$", fontsize=9, pad=3)
            if t == 0:
                ax.set_ylabel(row_labels[row], fontsize=9, fontweight="bold",
                              rotation=90, labelpad=8, va="center")

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=0.99)

    fig.subplots_adjust(left=0.10, right=0.995, top=0.91, bottom=0.01)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none", pad_inches=0.05)
        print(f"  Saved: {save_path}")

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[..., :3].copy()
    plt.close(fig)
    return img


def main():
    parser = argparse.ArgumentParser(description="Render 3-row comparison filmstrips")
    parser.add_argument("--upsample", type=int, default=128)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 0.5,
        "figure.dpi": args.dpi,
    })

    print("=" * 60)
    print("4-Row Comparison Filmstrip (Real RGB + latent — matches filmstrip_rollout.py)")
    print(f"  upsample={args.upsample}, dpi={args.dpi}")
    print("=" * 60)

    print("\n[1/2] Fitting shared PCA...")
    pca = fit_shared_pca(HYBRID_DIR)

    print("\n[2/2] Generating filmstrips...")
    for phase in PHASES:
        for part in PARTITIONS:
            hybrid_pt = HYBRID_DIR / phase / f"{part}_tensors.pt"
            gdn_pt = GDN_DIR / phase / f"{part}_tensors.pt"

            if not hybrid_pt.exists() or not gdn_pt.exists():
                print(f"  SKIP {phase}/{part}")
                continue

            hybrid_data = torch.load(str(hybrid_pt), map_location="cpu", weights_only=True)
            gdn_data = torch.load(str(gdn_pt), map_location="cpu", weights_only=True)

            gt_tensor = hybrid_data["gt"]
            hybrid_pred = hybrid_data["pred"]
            gdn_pred = gdn_data["pred"]

            # Exact same logic as filmstrip_rollout: GT uses global norm,
            # predictions use per_frame_norm=True
            gt_rgb = latent_to_rgb(gt_tensor, pca, upsample_size=args.upsample)
            hybrid_rgb = latent_to_rgb(hybrid_pred, pca, upsample_size=args.upsample, per_frame_norm=True)
            gdn_rgb = latent_to_rgb(gdn_pred, pca, upsample_size=args.upsample, per_frame_norm=True)

            # Load real RGB frames (every 2nd frame to match tubelet stride)
            real_rgb = load_real_rgb_frames(part, RGB_FRAMES_DIR, upsample_size=args.upsample)

            phase_label = phase.replace("_", " ").title()
            part_label = part.replace("_", " ").title().replace("Ood", "OOD")
            title = f"{phase_label} | {part_label}"

            save_path = output_dir / phase / f"{PARTITION_TO_TASK[part]}.png"
            render_4row_filmstrip(
                real_rgb, gt_rgb, hybrid_rgb, gdn_rgb,
                title=title, save_path=save_path, dpi=args.dpi,
            )

    print(f"\nAll filmstrips saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
