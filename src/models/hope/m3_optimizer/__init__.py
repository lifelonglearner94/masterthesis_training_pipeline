"""M3 Muon Optimizer — standalone Muon+AdamW hybrid and Deep Momentum family."""

from src.models.hope.m3_optimizer.deep_momentum import DeepMomentum, DeepMomentumState
from src.models.hope.m3_optimizer.hybrid import (
    HybridMuonAdamW,
    build_hybrid_muon_adamw,
    is_muon_candidate,
)
from src.models.hope.m3_optimizer.levels import (
    LevelClock,
    LevelConfig,
    LevelOptimizerManager,
    LevelSpec,
    LevelState,
    ensure_level_specs,
)
from src.models.hope.m3_optimizer.factory import build_deep_momentum

__all__ = [
    # Deep Momentum inner optimizer
    "DeepMomentum",
    "DeepMomentumState",
    # Hybrid outer optimizer
    "HybridMuonAdamW",
    "build_hybrid_muon_adamw",
    "is_muon_candidate",
    # Level scheduling
    "LevelClock",
    "LevelConfig",
    "LevelOptimizerManager",
    "LevelSpec",
    "LevelState",
    "ensure_level_specs",
    # Factory
    "build_deep_momentum",
]
