"""DataModules package.

Contains Lightning DataModules for various datasets.
"""

from src.datamodules.precomputed_features import PrecomputedFeaturesDataModule

__all__ = [
    "PrecomputedFeaturesDataModule",
]
