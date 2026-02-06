"""HOPE module — AC-HOPE-ViT architecture.

Novel architecture combining V-JEPA2 AC predictor I/O with the HOPE
(Self-Modifying Titans + CMS) backbone from Behrouz 2025.
"""

from src.models.hope.ac_hope_module import ACHOPEModule
from src.models.hope.ac_hope_vit import ACHOPEViT, ac_hope_vit
from src.models.hope.cms import CMS, CMSBlock, LevelSpec
from src.models.hope.hope_block import HOPEBlock, HOPEBlockConfig
from src.models.hope.titan_memory import TitanMemory, TitanMemoryConfig

__all__ = [
    "ACHOPEModule",
    "ACHOPEViT",
    "ac_hope_vit",
    "HOPEBlock",
    "HOPEBlockConfig",
    "TitanMemory",
    "TitanMemoryConfig",
    "CMS",
    "CMSBlock",
    "LevelSpec",
]
