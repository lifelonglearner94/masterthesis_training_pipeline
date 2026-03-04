"""DNH (Dynamic Nested Hierarchy) — Phase 11 extension of the HOPE architecture.

Implements the DNH-HOPE architecture from Jafari 2025:
    "Dynamic Nested Hierarchies: Pioneering Self-Evolution"

This subpackage extends the Phase 8 Hybrid architecture (Attention + Titan + CMS)
by replacing the single-level Titan memory with a *Dynamic Nested Hierarchy* of
self-modifying memories, and adding dynamic level management to CMS.

Key components:
    - SelfModifyingMemory: Titan memory augmented with a meta-network g_ψ
    - DynamicNestedHierarchy: Manages a variable-depth chain of SMMs (2–5 levels)
    - StructuralEvolutionController: Add/prune levels + frequency modulation
    - DynamicCMS: CMS with dynamic level add/prune capability
    - compute_meta_loss: Meta-loss for structural evolution decisions

All components are designed to be composable with the existing Phase 8 architecture,
preserving backward compatibility (Phase 8 experiments still use the unchanged
HybridBlock / ACHOPEHybridViT classes).
"""

from src.models.hope.dnh.dynamic_cms import DynamicCMS
from src.models.hope.dnh.dynamic_hierarchy import DynamicNestedHierarchy
from src.models.hope.dnh.meta_loss import compute_meta_loss
from src.models.hope.dnh.smm import SelfModifyingMemory, SMMConfig
from src.models.hope.dnh.structural_evolution import StructuralEvolutionController

__all__ = [
    "SelfModifyingMemory",
    "SMMConfig",
    "DynamicNestedHierarchy",
    "StructuralEvolutionController",
    "DynamicCMS",
    "compute_meta_loss",
]
