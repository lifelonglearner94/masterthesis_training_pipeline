"""Experience Replay Buffer for Continual Learning.

Stores a fixed-size random subset of clips from past tasks and provides
them as a DataLoader for interleaved training with new task data.

Design decisions:
    - **Reservoir sampling**: Maintains a uniform random sample across all
      past experiences. Each clip has equal probability of being in the buffer,
      regardless of which task it came from.
    - **Pre-computed features**: Stores V-JEPA2 feature tensors directly —
      no re-encoding needed. Memory cost is ~40KB per clip (features + actions).
    - **Random sampling per step**: During task training, each step draws a
      fresh random mini-batch from the buffer. This prevents overfitting to
      the buffer contents (the user's exact concern).
    - **Growing buffer**: As tasks complete, new clips can be added to the
      buffer. The reservoir sampling ensures past tasks aren't over-represented.

Typical memory cost:
    500 clips × 40KB ≈ 20MB — negligible compared to model parameters.

Usage (in cl_train.py):
    buffer = ReplayBuffer(max_size=500, seed=42)
    buffer.populate_from_datamodule(base_datamodule, cfg)
    replay_dl = buffer.get_dataloader(batch_size=8)
"""

from __future__ import annotations

import logging
import random
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


type SampleDict = dict[str, Tensor | str]
type BatchDict = dict[str, Tensor]


class ReplayBuffer:
    """Fixed-size experience replay buffer with reservoir sampling.

    Maintains a random subset of clips from past training experiences.
    Uses reservoir sampling (Vitter 1985) to ensure uniform coverage
    across all past data regardless of insertion order.

    Args:
        max_size: Maximum number of clips to store.
        seed: Random seed for reproducible sampling.
        replay_ratio: Fraction of each training batch that comes from replay.
            E.g., 0.3 means 30% replay, 70% new task data.
    """

    def __init__(
        self,
        max_size: int = 500,
        seed: int = 42,
        replay_ratio: float = 0.3,
    ) -> None:
        self.max_size = max_size
        self.replay_ratio = replay_ratio
        self._rng = random.Random(seed)
        self._buffer: list[SampleDict] = []
        self._total_seen: int = 0
        self._task_counts: dict[str, int] = {}

    @property
    def size(self) -> int:
        """Current number of clips in the buffer."""
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        """Whether the buffer has any clips."""
        return len(self._buffer) == 0

    def add_sample(self, sample: SampleDict, task_name: str = "unknown") -> None:
        """Add a single clip to the buffer using reservoir sampling.

        If the buffer is not full, the clip is added directly.
        If the buffer is full, the clip replaces a random existing clip
        with probability max_size / (total_seen + 1), ensuring uniform
        coverage across all clips ever seen.

        Args:
            sample: Dictionary with 'features', 'actions', 'states' tensors.
            task_name: Name of the task this clip belongs to (for logging).
        """
        self._total_seen += 1

        # Store task provenance as metadata
        sample_with_meta = {
            k: v.detach().cpu().clone() if isinstance(v, Tensor) else v
            for k, v in sample.items()
        }
        sample_with_meta["_replay_task"] = task_name

        if len(self._buffer) < self.max_size:
            self._buffer.append(sample_with_meta)
            self._task_counts[task_name] = self._task_counts.get(task_name, 0) + 1
        else:
            # Reservoir sampling: replace with probability max_size / total_seen
            j = self._rng.randint(0, self._total_seen - 1)
            if j < self.max_size:
                # Track which task's clip we're removing
                old_task = self._buffer[j].get("_replay_task", "unknown")
                self._task_counts[old_task] = max(0, self._task_counts.get(old_task, 1) - 1)

                self._buffer[j] = sample_with_meta
                self._task_counts[task_name] = self._task_counts.get(task_name, 0) + 1

    def populate_from_datamodule(
        self,
        data_dir: str,
        clip_start: int,
        clip_end: int,
        task_name: str = "base",
        num_timesteps: int = 8,
        patches_per_frame: int = 256,
        feature_map_name: str = "vjepa2_vitl16",
        clip_prefix: str = "clip_",
        use_extrinsics: bool = False,
    ) -> int:
        """Populate buffer from a clip range using reservoir sampling.

        Creates a temporary dataset for the specified clip range and adds
        all clips through reservoir sampling. This ensures uniform coverage
        regardless of the order clips are processed.

        Args:
            data_dir: Root data directory.
            clip_start: Start of clip range (inclusive).
            clip_end: End of clip range (exclusive).
            task_name: Name for tracking task provenance.
            num_timesteps: Number of encoded timesteps.
            patches_per_frame: Patches per frame.
            feature_map_name: Feature map file name.
            clip_prefix: Prefix for clip directories.
            use_extrinsics: Whether to load extrinsics.

        Returns:
            Number of clips processed.
        """
        from src.datamodules.precomputed_features import PrecomputedFeaturesDataset

        dataset = PrecomputedFeaturesDataset(
            data_dir=data_dir,
            num_timesteps=num_timesteps,
            patches_per_frame=patches_per_frame,
            use_extrinsics=use_extrinsics,
            feature_map_name=feature_map_name,
            clip_prefix=clip_prefix,
            clip_start=clip_start,
            clip_end=clip_end,
        )

        n_clips = len(dataset)
        log.info(
            f"  [Replay] Populating from '{task_name}' "
            f"(clips {clip_start}-{clip_end}, {n_clips} clips)"
        )

        for i in range(n_clips):
            sample = dataset[i]
            self.add_sample(sample, task_name=task_name)

        log.info(
            f"  [Replay] Buffer: {self.size}/{self.max_size} clips "
            f"(seen {self._total_seen} total). "
            f"Task distribution: {dict(self._task_counts)}"
        )
        return n_clips

    def get_dataset(self) -> _ReplayDataset:
        """Get a Dataset view of the buffer for DataLoader creation."""
        return _ReplayDataset(self._buffer)

    def get_dataloader(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> DataLoader:
        """Create a DataLoader that randomly samples from the buffer.

        Uses shuffle=True so each epoch sees clips in random order,
        preventing overfitting to any particular ordering.

        Args:
            batch_size: Batch size for replay.
            num_workers: Number of worker processes.
            pin_memory: Whether to pin memory.

        Returns:
            DataLoader yielding replay batches.
        """
        from src.datamodules.precomputed_features import collate_fn

        dataset = self.get_dataset()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            drop_last=True,  # Avoid partial batches (mismatch with task batch size)
        )

    def get_summary(self) -> dict[str, Any]:
        """Get buffer summary for logging."""
        return {
            "buffer_size": self.size,
            "max_size": self.max_size,
            "total_seen": self._total_seen,
            "replay_ratio": self.replay_ratio,
            "task_distribution": dict(self._task_counts),
            "fill_pct": self.size / self.max_size * 100 if self.max_size > 0 else 0,
        }


class _ReplayDataset(Dataset):
    """Thin Dataset wrapper around the replay buffer's internal list."""

    def __init__(self, buffer: list[SampleDict]) -> None:
        self._buffer = buffer

    def __len__(self) -> int:
        return len(self._buffer)

    def __getitem__(self, idx: int) -> SampleDict:
        item = self._buffer[idx]
        # Return a copy without internal metadata
        return {
            k: v for k, v in item.items()
            if not k.startswith("_replay_")
        }
