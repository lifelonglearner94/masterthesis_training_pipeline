"""Pre-encode BAIR Robot Pushing dataset with V-JEPA2 ViT-Large.

Downloads the bair_robot_pushing_small dataset via tensorflow_datasets,
encodes each episode's first 16 frames through V-JEPA2 ViT-L (tubelet_size=2),
and saves the features + actions in the same clip_XXXXX format used by
PrecomputedFeaturesDataModule.

Output structure:
    <output_dir>/
    ├── clip_00000/         (BAIR train episode 0)
    │   ├── feature_maps/
    │   │   └── vjepa2_vitl16.npy   # [8, 256, 1024]
    │   └── actions_states/
    │       └── actions.npy          # [8, 4]
    ├── ...
    ├── clip_43263/         (BAIR train episode 43263)
    ├── clip_43264/         (BAIR test episode 0)
    └── clip_43519/         (BAIR test episode 255)

Requirements:
    pip install tensorflow-datasets tensorflow

Usage:
    python src/scripts/encode_bair.py --output_dir data/bair
    python src/scripts/encode_bair.py --output_dir data/bair --batch_size 16 --device cuda
    python src/scripts/encode_bair.py --output_dir data/bair --max_clips 1000  # quick test
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# V-JEPA2 ViT-L constants
NUM_INPUT_FRAMES = 16       # Frames to extract per episode
TUBELET_SIZE = 2            # V-JEPA2 temporal patch size
NUM_ENCODED_TIMESTEPS = NUM_INPUT_FRAMES // TUBELET_SIZE  # 8
TARGET_RESOLUTION = 256     # To get 16x16=256 patches with patch_size=16
PATCHES_PER_FRAME = (TARGET_RESOLUTION // 16) ** 2  # 256
EMBED_DIM = 1024            # ViT-L embedding dimension

# ImageNet normalization (standard for ViT models)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)

# BAIR dataset info
BAIR_TRAIN_SIZE = 43_264
BAIR_TEST_SIZE = 256
BAIR_ACTION_DIM = 4


def load_bair_dataset(tfds_data_dir: str | None = None):
    """Load BAIR robot pushing dataset via tensorflow_datasets.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    try:
        import tensorflow_datasets as tfds
    except ImportError:
        log.error(
            "tensorflow-datasets is required. Install with:\n"
            "  pip install tensorflow-datasets tensorflow"
        )
        sys.exit(1)

    log.info("Loading BAIR robot pushing dataset (this may download ~30GB on first run)...")
    kwargs = {}
    if tfds_data_dir:
        kwargs["data_dir"] = tfds_data_dir

    ds_train = tfds.load("bair_robot_pushing_small", split="train", **kwargs)
    ds_test = tfds.load("bair_robot_pushing_small", split="test", **kwargs)

    log.info(f"Train episodes: {len(ds_train)}, Test episodes: {len(ds_test)}")
    return ds_train, ds_test


def load_vjepa2_encoder(device: torch.device) -> torch.nn.Module:
    """Load V-JEPA2 ViT-Large encoder from torch.hub.

    Uses a two-phase approach: first let torch.hub download the repo,
    then manually load hubconf.py with the vjepa2 repo's ``src`` package
    on sys.path (ahead of the project's own ``src``).

    Returns:
        V-JEPA2 encoder model in eval mode.
    """
    import importlib
    import sys

    log.info("Loading V-JEPA2 ViT-Large from torch.hub...")

    # Phase 1: Download the repo via torch.hub (but don't load the model yet)
    hub_dir = torch.hub.get_dir()
    repo_dir = Path(hub_dir) / "facebookresearch_vjepa2_main"

    if not repo_dir.exists():
        log.info("Downloading vjepa2 repo...")
        torch.hub._validate_not_a_forked_repo = lambda *a, **k: True  # skip check
        torch.hub.download_url_to_file(
            "https://github.com/facebookresearch/vjepa2/zipball/main",
            str(Path(hub_dir) / "main.zip"),
        )
        import zipfile
        with zipfile.ZipFile(str(Path(hub_dir) / "main.zip"), "r") as z:
            # The zip contains a top-level directory like facebookresearch-vjepa2-<hash>
            top = z.namelist()[0].split("/")[0]
            z.extractall(hub_dir)
        extracted = Path(hub_dir) / top
        extracted.rename(repo_dir)

    # Phase 2: Temporarily hijack sys.path and sys.modules so that
    # `from src.hub.backbones import ...` resolves to the vjepa2 repo's src/
    saved_path = sys.path.copy()
    saved_src_module = sys.modules.pop("src", None)
    # Also remove any sub-modules of our project's src that are cached
    saved_src_submodules = {
        k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith("src.")
    }

    try:
        sys.path.insert(0, str(repo_dir))
        # Force Python to re-discover 'src' from the new path
        spec = importlib.util.spec_from_file_location(
            "hubconf", str(repo_dir / "hubconf.py")
        )
        hubconf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hubconf)
        model = hubconf.vjepa2_vit_large()
    finally:
        # Restore everything
        sys.path = saved_path
        # Remove vjepa2's src from sys.modules, restore ours
        for k in list(sys.modules):
            if k == "src" or k.startswith("src."):
                sys.modules.pop(k, None)
        if saved_src_module is not None:
            sys.modules["src"] = saved_src_module
        sys.modules.update(saved_src_submodules)

    model = model.to(device).eval()
    log.info("V-JEPA2 encoder loaded successfully.")
    return model


def preprocess_frames(frames_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """Preprocess BAIR frames for V-JEPA2 encoding.

    Args:
        frames_np: Raw frames [T, H, W, C] uint8 (64x64x3).
        device: Target device.

    Returns:
        Preprocessed video tensor [1, 3, T, 256, 256] float32 normalized.
    """
    # Take first NUM_INPUT_FRAMES frames
    frames = frames_np[:NUM_INPUT_FRAMES]  # [16, 64, 64, 3]

    # Convert to torch float [0, 1] and rearrange to [1, C, T, H, W]
    video = torch.from_numpy(frames).float() / 255.0  # [T, H, W, C]
    video = video.permute(3, 0, 1, 2).unsqueeze(0)     # [1, C, T, H, W]

    # Resize from 64x64 to 256x256 (bilinear interpolation)
    B, C, T, H, W = video.shape
    # Reshape to [B*T, C, H, W] for F.interpolate, then back
    video_flat = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    video_flat = F.interpolate(video_flat, size=(TARGET_RESOLUTION, TARGET_RESOLUTION), mode="bilinear", align_corners=False)
    video = video_flat.reshape(B, T, C, TARGET_RESOLUTION, TARGET_RESOLUTION).permute(0, 2, 1, 3, 4)
    # Now [1, 3, 16, 256, 256]

    # ImageNet normalization
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    video = video.to(device)
    video = (video - mean) / std

    return video


def downsample_actions(actions_np: np.ndarray) -> np.ndarray:
    """Downsample BAIR actions to match V-JEPA2 tubelet temporal resolution.

    Takes 16 raw actions and averages consecutive pairs to produce 8 actions
    aligned with the 8 encoded temporal tokens.

    Args:
        actions_np: Raw actions [T, 4] from BAIR episode.

    Returns:
        Downsampled actions [8, 4].
    """
    actions = actions_np[:NUM_INPUT_FRAMES].astype(np.float32)  # [16, 4]
    # Average consecutive pairs to match tubelet_size=2
    actions_downsampled = actions.reshape(NUM_ENCODED_TIMESTEPS, TUBELET_SIZE, BAIR_ACTION_DIM).mean(axis=1)
    return actions_downsampled  # [8, 4]


def save_clip(
    output_dir: Path,
    clip_id: int,
    features: np.ndarray,
    actions: np.ndarray,
) -> None:
    """Save encoded features and actions in the standard clip directory format.

    Args:
        output_dir: Base output directory.
        clip_id: Numeric clip ID.
        features: Encoded features [T_enc, N, D].
        actions: Downsampled actions [T_enc, action_dim].
    """
    clip_dir = output_dir / f"clip_{clip_id:05d}"
    fm_dir = clip_dir / "feature_maps"
    as_dir = clip_dir / "actions_states"
    fm_dir.mkdir(parents=True, exist_ok=True)
    as_dir.mkdir(parents=True, exist_ok=True)

    np.save(fm_dir / "vjepa2_vitl16.npy", features)
    np.save(as_dir / "actions.npy", actions)


@torch.no_grad()
def encode_dataset(
    dataset,
    encoder: torch.nn.Module,
    output_dir: Path,
    clip_id_offset: int,
    batch_size: int,
    device: torch.device,
    max_clips: int | None = None,
) -> int:
    """Encode a BAIR dataset split through V-JEPA2.

    Args:
        dataset: TFDS dataset iterator.
        encoder: V-JEPA2 encoder model.
        output_dir: Base output directory.
        clip_id_offset: Starting clip ID for this split.
        batch_size: Number of episodes to encode at once.
        device: Torch device.
        max_clips: Maximum number of clips to encode (None = all).

    Returns:
        Number of clips encoded.
    """
    import tensorflow as tf

    # Collect episodes into batches for efficient GPU encoding
    batch_frames = []
    batch_actions = []
    batch_clip_ids = []
    clips_encoded = 0
    total_clips = max_clips or len(dataset)
    t_start = time.time()

    for i, episode in enumerate(dataset):
        if max_clips is not None and i >= max_clips:
            break

        clip_id = clip_id_offset + i

        # Skip already encoded clips (resume support)
        clip_dir = output_dir / f"clip_{clip_id:05d}"
        if (clip_dir / "feature_maps" / "vjepa2_vitl16.npy").exists():
            clips_encoded += 1
            if clips_encoded % 1000 == 0:
                log.info(f"  Skipped {clips_encoded}/{total_clips} (already encoded)")
            continue

        # Extract frames and actions as numpy
        frames = episode["image_main"].numpy()  # [T, 64, 64, 3]
        actions = episode["action"].numpy()      # [T, 4]

        if frames.shape[0] < NUM_INPUT_FRAMES:
            log.warning(
                f"  Episode {i} has only {frames.shape[0]} frames "
                f"(need {NUM_INPUT_FRAMES}), skipping."
            )
            continue

        batch_frames.append(frames)
        batch_actions.append(actions)
        batch_clip_ids.append(clip_id)

        # Process batch when full
        if len(batch_frames) >= batch_size:
            _encode_and_save_batch(
                batch_frames, batch_actions, batch_clip_ids,
                encoder, output_dir, device,
            )
            clips_encoded += len(batch_frames)
            batch_frames.clear()
            batch_actions.clear()
            batch_clip_ids.clear()

            elapsed = time.time() - t_start
            rate = clips_encoded / elapsed if elapsed > 0 else 0
            log.info(
                f"  Encoded {clips_encoded}/{total_clips} clips "
                f"({rate:.1f} clips/s, elapsed {elapsed:.0f}s)"
            )

    # Process remaining episodes
    if batch_frames:
        _encode_and_save_batch(
            batch_frames, batch_actions, batch_clip_ids,
            encoder, output_dir, device,
        )
        clips_encoded += len(batch_frames)

    elapsed = time.time() - t_start
    log.info(f"  Finished: {clips_encoded} clips in {elapsed:.0f}s")
    return clips_encoded


def _encode_and_save_batch(
    batch_frames: list[np.ndarray],
    batch_actions: list[np.ndarray],
    batch_clip_ids: list[int],
    encoder: torch.nn.Module,
    output_dir: Path,
    device: torch.device,
) -> None:
    """Encode a batch of episodes and save results.

    Args:
        batch_frames: List of frame arrays [T, 64, 64, 3].
        batch_actions: List of action arrays [T, 4].
        batch_clip_ids: List of clip IDs.
        encoder: V-JEPA2 encoder.
        output_dir: Output directory.
        device: Torch device.
    """
    # Preprocess all frames in batch
    videos = []
    for frames in batch_frames:
        video = preprocess_frames(frames, device)  # [1, 3, 16, 256, 256]
        videos.append(video)

    video_batch = torch.cat(videos, dim=0)  # [B, 3, 16, 256, 256]

    # Encode through V-JEPA2
    features = encoder(video_batch)  # Expected: [B, T_enc * N, D] or [B, N_total, D]

    # Handle different output shapes
    features_np = features.cpu().float().numpy()

    if features_np.ndim == 2:
        # Single sample: [N_total, D]
        features_np = features_np[np.newaxis, ...]

    B = features_np.shape[0]

    for j in range(B):
        feat = features_np[j]  # [N_total, D]

        # Reshape to [T_enc, N_patches, D]
        if feat.ndim == 2:
            N_total, D = feat.shape
            T_enc = N_total // PATCHES_PER_FRAME
            if T_enc == 0:
                # May have different spatial resolution, try to infer
                T_enc = NUM_ENCODED_TIMESTEPS
                N_patches = N_total // T_enc
                log.warning(
                    f"  Unexpected feature shape [{N_total}, {D}]. "
                    f"Inferring T_enc={T_enc}, N_patches={N_patches}."
                )
                feat = feat.reshape(T_enc, N_patches, D)
            else:
                feat = feat.reshape(T_enc, PATCHES_PER_FRAME, D)
        elif feat.ndim == 3:
            # Already [T_enc, N, D] — use as-is
            pass

        # Limit to expected timesteps
        feat = feat[:NUM_ENCODED_TIMESTEPS]

        # Downsample actions
        actions_ds = downsample_actions(batch_actions[j])

        # Save
        save_clip(output_dir, batch_clip_ids[j], feat, actions_ds)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-encode BAIR Robot Pushing dataset with V-JEPA2 ViT-Large"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/bair",
        help="Output directory for encoded clips (default: data/bair)",
    )
    parser.add_argument(
        "--tfds_data_dir", type=str, default=None,
        help="Custom directory for tensorflow_datasets downloads (default: ~/tensorflow_datasets)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for V-JEPA2 encoding (default: 8, reduce if OOM)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for encoding (default: cuda if available)",
    )
    parser.add_argument(
        "--max_clips", type=int, default=None,
        help="Maximum number of clips to encode per split (default: all). Useful for testing.",
    )
    parser.add_argument(
        "--skip_train", action="store_true",
        help="Skip encoding of the training split.",
    )
    parser.add_argument(
        "--skip_test", action="store_true",
        help="Skip encoding of the test split.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    log.info(f"Output directory: {output_dir}")
    log.info(f"Device: {device}")
    log.info(f"Batch size: {args.batch_size}")

    # Load BAIR dataset
    ds_train, ds_test = load_bair_dataset(args.tfds_data_dir)

    # Load V-JEPA2 encoder
    encoder = load_vjepa2_encoder(device)

    # Encode train split → clip_00000 to clip_43263
    if not args.skip_train:
        log.info("=" * 60)
        log.info("Encoding BAIR TRAIN split")
        log.info(f"  Clip IDs: 0 to {BAIR_TRAIN_SIZE - 1}")
        log.info("=" * 60)
        n_train = encode_dataset(
            ds_train, encoder, output_dir,
            clip_id_offset=0,
            batch_size=args.batch_size,
            device=device,
            max_clips=args.max_clips,
        )
        log.info(f"Train encoding complete: {n_train} clips")

    # Encode test split → clip_43264 to clip_43519
    if not args.skip_test:
        log.info("=" * 60)
        log.info("Encoding BAIR TEST split")
        log.info(f"  Clip IDs: {BAIR_TRAIN_SIZE} to {BAIR_TRAIN_SIZE + BAIR_TEST_SIZE - 1}")
        log.info("=" * 60)
        n_test = encode_dataset(
            ds_test, encoder, output_dir,
            clip_id_offset=BAIR_TRAIN_SIZE,
            batch_size=args.batch_size,
            device=device,
            max_clips=args.max_clips,
        )
        log.info(f"Test encoding complete: {n_test} clips")

    # Summary
    log.info("=" * 60)
    log.info("BAIR encoding complete!")
    log.info(f"  Output: {output_dir}")
    log.info(f"  Train clips: clip_00000 to clip_{BAIR_TRAIN_SIZE - 1:05d}")
    log.info(f"  Test clips:  clip_{BAIR_TRAIN_SIZE:05d} to clip_{BAIR_TRAIN_SIZE + BAIR_TEST_SIZE - 1:05d}")
    log.info(f"  Feature shape: [{NUM_ENCODED_TIMESTEPS}, {PATCHES_PER_FRAME}, {EMBED_DIM}]")
    log.info(f"  Action shape:  [{NUM_ENCODED_TIMESTEPS}, {BAIR_ACTION_DIM}]")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
