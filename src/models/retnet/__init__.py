"""RetNet model package for the CL pipeline.

This package provides a simplified RetNet implementation adapted from
torchscale/RetNet, integrated into the standard CL-pipeline interface.

Classes:
    RetNetBackbone: Core RetNet backbone (encoder → RetNetDecoderLayers → decoder).
    RetNetLitModule: PyTorch Lightning wrapper for CL training/evaluation.

Example:
    >>> from src.models.retnet import RetNetBackbone, RetNetLitModule
"""

from .backbone import RetNetBackbone
from .lightning_module import RetNetLitModule

__all__ = ["RetNetBackbone", "RetNetLitModule"]
