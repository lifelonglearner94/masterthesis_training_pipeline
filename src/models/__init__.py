"""Models package.

This package contains model implementations for the pipeline.
"""

import logging as _logging

from src.models.ac_predictor import ACPredictorModule, VisionTransformerPredictorAC, vit_ac_predictor
from src.models.baseline_convlstm import BaselineLitModule, ConvLSTMBaseline
from src.models.baseline_identity import IdentityBaseline, IdentityBaselineLitModule
from src.models.hope import ACHOPEModule, ACHOPEViT, ac_hope_vit
from src.models.titans import TitansBackbone, TitansLitModule

_log = _logging.getLogger(__name__)

# Comparison architectures — guarded imports (require optional 'comparison-architectures' deps).
# Hydra instantiate uses fully-qualified _target_ paths, so these convenience
# re-exports are not required for training; they're only for interactive use.
try:
    from src.models.gated_delta_net import GatedDeltaNetBackbone, GatedDeltaNetLitModule
except ImportError:
    _log.debug("GatedDeltaNet not available (install 'comparison-architectures' extras)")
    GatedDeltaNetBackbone = None  # type: ignore[assignment,misc]
    GatedDeltaNetLitModule = None  # type: ignore[assignment,misc]

try:
    from src.models.transformer_pp import TransformerPPBackbone, TransformerPPLitModule
except ImportError:
    _log.debug("Transformer++ not available (install 'comparison-architectures' extras)")
    TransformerPPBackbone = None  # type: ignore[assignment,misc]
    TransformerPPLitModule = None  # type: ignore[assignment,misc]

try:
    from src.models.retnet import RetNetBackbone, RetNetLitModule
except ImportError:
    _log.debug("RetNet not available (install 'comparison-architectures' extras)")
    RetNetBackbone = None  # type: ignore[assignment,misc]
    RetNetLitModule = None  # type: ignore[assignment,misc]

__all__ = [
    "ACPredictorModule",
    "VisionTransformerPredictorAC",
    "vit_ac_predictor",
    "BaselineLitModule",
    "ConvLSTMBaseline",
    "IdentityBaseline",
    "IdentityBaselineLitModule",
    "ACHOPEModule",
    "ACHOPEViT",
    "ac_hope_vit",
    "TitansBackbone",
    "TitansLitModule",
    "GatedDeltaNetBackbone",
    "GatedDeltaNetLitModule",
    "TransformerPPBackbone",
    "TransformerPPLitModule",
    "RetNetBackbone",
    "RetNetLitModule",
]
