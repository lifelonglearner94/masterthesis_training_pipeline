"""Mixins for model implementations."""

from src.models.mixins.loss_mixin import ACPredictorLossMixin
from src.models.mixins.tta_mixin import TTAMixin

__all__ = ["ACPredictorLossMixin", "TTAMixin"]
