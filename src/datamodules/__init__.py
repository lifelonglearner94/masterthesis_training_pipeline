"""DataModules package.

Contains Lightning DataModules for various datasets.
"""

from src.datamodules.precomputed_features import PrecomputedFeaturesDataModule
from src.datamodules.split_cifar100 import SplitCIFAR100DataModule
from src.datamodules.permuted_mnist import PermutedMNISTDataModule

__all__ = [
    "PrecomputedFeaturesDataModule",
    "SplitCIFAR100DataModule",
    "PermutedMNISTDataModule",
]
