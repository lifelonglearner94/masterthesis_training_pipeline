"""HOPE module — AC-HOPE-ViT and AC-HOPE-Hybrid-ViT architectures.

Novel architectures combining V-JEPA2 AC predictor I/O with the HOPE
(Self-Modifying Titans + CMS) backbone from Behrouz 2025.

- ACHOPEViT: Original architecture (Titan replaces attention)
- ACHOPEHybridViT: Phase 8 hybrid (Attention + Titan + CMS)
"""

from src.models.hope.ac_hope_hybrid_module import ACHOPEHybridModule
from src.models.hope.ac_hope_hybrid_vit import ACHOPEHybridViT, ac_hope_hybrid_vit
from src.models.hope.ac_hope_module import ACHOPEModule
from src.models.hope.ac_hope_vit import ACHOPEViT, ac_hope_vit
from src.models.hope.cms import CMS, CMSBlock, LevelSpec
from src.models.hope.hope_block import HOPEBlock, HOPEBlockConfig
from src.models.hope.hybrid_block import HybridBlock, HybridBlockConfig
from src.models.hope.titan_memory import TitanMemory, TitanMemoryConfig

__all__ = [
    # Original HOPE
    "ACHOPEModule",
    "ACHOPEViT",
    "ac_hope_vit",
    "HOPEBlock",
    "HOPEBlockConfig",
    # Hybrid (Phase 8)
    "ACHOPEHybridModule",
    "ACHOPEHybridViT",
    "ac_hope_hybrid_vit",
    "HybridBlock",
    "HybridBlockConfig",
    # Shared components
    "TitanMemory",
    "TitanMemoryConfig",
    "CMS",
    "CMSBlock",
    "LevelSpec",
]
