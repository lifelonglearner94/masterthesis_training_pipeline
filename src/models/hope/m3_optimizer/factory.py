"""Factory for building :class:`DeepMomentum` instances from config dicts."""

from __future__ import annotations

from typing import Any, Dict

from src.models.hope.m3_optimizer.deep_momentum import DeepMomentum


def build_deep_momentum(config: Dict[str, Any]) -> DeepMomentum:
    """Instantiate a :class:`DeepMomentum` from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Must contain ``"type": "deep_momentum"`` and an optional ``"params"``
        sub-dict with keyword arguments forwarded to :class:`DeepMomentum`.

    Returns
    -------
    DeepMomentum

    Raises
    ------
    ValueError
        If ``config["type"]`` is not ``"deep_momentum"``.

    Example
    -------
    >>> build_deep_momentum({
    ...     "type": "deep_momentum",
    ...     "params": {"variant": "muon", "beta": 0.95},
    ... })
    """
    opt_type = config.get("type", "deep_momentum").lower()
    if opt_type != "deep_momentum":
        raise ValueError(f"Unsupported optimizer type {opt_type}")
    params = config.get("params", {})
    return DeepMomentum(**params)
