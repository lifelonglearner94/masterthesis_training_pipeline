"""DataModule for Split CIFAR-100 Continual Learning Benchmark.

Splits CIFAR-100 (100 classes, 600 images each) into sequential tasks,
each containing a disjoint subset of classes. This is a standard
continual-learning benchmark for measuring catastrophic forgetting.

Default configuration: 10 tasks × 10 classes each.
Each task's train/test split follows the original CIFAR-100 split
(500 train + 100 test per class).

Usage:
    uv run src/cl_benchmark_train.py experiment=cl_split_cifar100
"""

from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100

from src.utils.device_utils import should_pin_memory


class SplitCIFAR100Dataset(Dataset):
    """Wrapper that filters CIFAR-100 to a subset of classes.

    Re-maps class labels to a contiguous range [0, num_classes_per_task)
    when ``remap_labels=True``, or keeps the original CIFAR-100 label space
    when ``remap_labels=False`` (useful for evaluation across all tasks).

    Args:
        cifar_dataset: Underlying CIFAR-100 dataset (train or test).
        class_ids: List of original CIFAR-100 class IDs to include.
        remap_labels: Whether to remap labels to [0, len(class_ids)).
    """

    def __init__(
        self,
        cifar_dataset: CIFAR100,
        class_ids: list[int],
        remap_labels: bool = False,
    ) -> None:
        self.cifar_dataset = cifar_dataset
        self.class_ids = sorted(class_ids)
        self.remap_labels = remap_labels

        # Build label mapping
        if remap_labels:
            self._label_map = {c: i for i, c in enumerate(self.class_ids)}
        else:
            self._label_map = {c: c for c in self.class_ids}

        # Pre-filter indices for the requested classes
        targets = np.array(cifar_dataset.targets)
        mask = np.isin(targets, self.class_ids)
        self._indices = np.where(mask)[0].tolist()

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        real_idx = self._indices[idx]
        image, label = self.cifar_dataset[real_idx]
        return image, self._label_map[label]


class SplitCIFAR100DataModule(L.LightningDataModule):
    """Lightning DataModule for the Split CIFAR-100 CL benchmark.

    Downloads CIFAR-100 automatically. On ``setup()``, creates train/val/test
    datasets filtered to the classes specified by ``task_classes``.

    Args:
        data_dir: Root directory for CIFAR-100 download.
        task_classes: List of class IDs for the *current* task.
            Set by the CL training pipeline before each task.
        all_classes_so_far: Union of all classes seen so far (for evaluation).
            If None, defaults to ``task_classes``.
        total_classes: Total number of classes across all tasks.
        remap_labels: Whether to remap labels per-task to [0, N).
        val_split: Fraction of train set to use for validation.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of data-loading worker processes.
        pin_memory: Whether to pin memory. Auto-detected if None.
        persistent_workers: Whether to keep workers alive between epochs.
    """

    def __init__(
        self,
        data_dir: str = "data",
        task_classes: list[int] | None = None,
        all_classes_so_far: list[int] | None = None,
        total_classes: int = 100,
        remap_labels: bool = False,
        val_split: float = 0.1,
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool | None = None,
        persistent_workers: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.task_classes = task_classes or list(range(total_classes))
        self.all_classes_so_far = all_classes_so_far or self.task_classes
        self.total_classes = total_classes
        self.remap_labels = remap_labels
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory if pin_memory is not None else should_pin_memory()
        self.persistent_workers = persistent_workers and num_workers > 0

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

        # Standard CIFAR-100 transforms
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761],
            ),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761],
            ),
        ])

    def prepare_data(self) -> None:
        """Download CIFAR-100 if not already present."""
        CIFAR100(root=str(self.data_dir), train=True, download=True)
        CIFAR100(root=str(self.data_dir), train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        """Create train/val/test datasets for the current task classes."""
        cifar_train = CIFAR100(
            root=str(self.data_dir), train=True, download=False,
            transform=self.train_transform,
        )
        cifar_test = CIFAR100(
            root=str(self.data_dir), train=False, download=False,
            transform=self.test_transform,
        )

        if stage == "fit" or stage is None:
            full_train = SplitCIFAR100Dataset(
                cifar_train, self.task_classes, remap_labels=self.remap_labels,
            )
            if self.val_split > 0.0:
                n_val = max(1, int(len(full_train) * self.val_split))
                n_train = len(full_train) - n_val
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    full_train, [n_train, n_val],
                    generator=torch.Generator().manual_seed(42),
                )
            else:
                self.train_dataset = full_train
                self.val_dataset = None

        if stage == "test" or stage is None:
            # Evaluate on ALL classes seen so far (standard CL eval protocol)
            self.test_dataset = SplitCIFAR100Dataset(
                cifar_test, self.all_classes_so_far, remap_labels=self.remap_labels,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
