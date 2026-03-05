"""DataModule for Permuted MNIST Continual Learning Benchmark.

Each task applies a fixed random pixel permutation to the 28×28 MNIST images,
creating a new input distribution while preserving the 10-class label space.
The first task (task 0) uses the *original* unpermuted MNIST.

This is a standard domain-incremental CL benchmark: the label space stays the
same, but the input distribution shifts with every task.

Default configuration: 10 tasks (1 original + 9 permuted).

Usage:
    uv run src/cl_benchmark_train.py experiment=cl_permuted_mnist
"""

from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

from src.utils.device_utils import should_pin_memory


class PermutedMNISTDataset(Dataset):
    """MNIST dataset with a fixed pixel permutation applied.

    If ``permutation`` is None, returns original (unpermuted) images.

    Args:
        mnist_dataset: Underlying MNIST dataset (train or test).
        permutation: 1-D index array of length 784 for pixel permutation.
            If None, no permutation is applied (task 0 = original MNIST).
    """

    def __init__(
        self,
        mnist_dataset: MNIST,
        permutation: np.ndarray | None = None,
    ) -> None:
        self.mnist_dataset = mnist_dataset
        self.permutation = permutation

    def __len__(self) -> int:
        return len(self.mnist_dataset)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        image, label = self.mnist_dataset[idx]
        if self.permutation is not None:
            # image shape: [1, 28, 28] → flatten → permute → reshape
            flat = image.view(-1)
            flat = flat[self.permutation]
            image = flat.view(1, 28, 28)
        return image, label


def generate_permutations(
    num_tasks: int,
    seed: int = 42,
) -> list[np.ndarray | None]:
    """Generate deterministic pixel permutations for each task.

    Task 0 always uses the identity (None = no permutation).
    Tasks 1..num_tasks-1 each get a unique random permutation.

    Args:
        num_tasks: Total number of tasks.
        seed: Random seed for reproducibility.

    Returns:
        List of permutation arrays (None for task 0).
    """
    rng = np.random.default_rng(seed)
    permutations: list[np.ndarray | None] = [None]  # Task 0: identity
    pixel_count = 28 * 28
    for _ in range(1, num_tasks):
        perm = rng.permutation(pixel_count)
        permutations.append(perm)
    return permutations


class PermutedMNISTDataModule(L.LightningDataModule):
    """Lightning DataModule for the Permuted MNIST CL benchmark.

    Downloads MNIST automatically. On ``setup()``, creates train/val/test
    datasets with the permutation corresponding to ``task_id``.

    Args:
        data_dir: Root directory for MNIST download.
        task_id: Current task index (0 = original MNIST, 1+ = permuted).
        num_tasks: Total number of tasks (used to pre-generate all permutations).
        seed: Random seed for permutation generation.
        val_split: Fraction of train set for validation.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of data-loading workers.
        pin_memory: Whether to pin memory. Auto-detected if None.
        persistent_workers: Keep workers alive between epochs.
    """

    def __init__(
        self,
        data_dir: str = "data",
        task_id: int = 0,
        num_tasks: int = 10,
        seed: int = 42,
        val_split: float = 0.1,
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool | None = None,
        persistent_workers: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.task_id = task_id
        self.num_tasks = num_tasks
        self.seed = seed
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory if pin_memory is not None else should_pin_memory()
        self.persistent_workers = persistent_workers and num_workers > 0

        # Pre-generate all permutations (deterministic, cheap)
        self.permutations = generate_permutations(num_tasks, seed)

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

        # Standard MNIST transform (just normalize)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def prepare_data(self) -> None:
        """Download MNIST if not already present."""
        MNIST(root=str(self.data_dir), train=True, download=True)
        MNIST(root=str(self.data_dir), train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        """Create train/val/test datasets with the current task's permutation."""
        perm = self.permutations[self.task_id]

        mnist_train = MNIST(
            root=str(self.data_dir), train=True, download=False,
            transform=self.transform,
        )
        mnist_test = MNIST(
            root=str(self.data_dir), train=False, download=False,
            transform=self.transform,
        )

        if stage == "fit" or stage is None:
            full_train = PermutedMNISTDataset(mnist_train, permutation=perm)
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
            self.test_dataset = PermutedMNISTDataset(mnist_test, permutation=perm)

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
