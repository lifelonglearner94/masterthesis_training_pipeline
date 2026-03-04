"""Meta-loss computation for DNH structural evolution (Jafari 2025).

L_meta = L_task + λ ||ΔG_t||_1 + μ D_KL(p_t(x) || p_{t-1}(x))

Components:
    1. L_task: Standard task loss (L1 teacher-forcing + jump prediction).
       Computed externally by ACPredictorLossMixin.

    2. λ ||ΔG_t||_1: Structural change penalty — penalises add/prune events
       to encourage structural stability. Implemented as a simple count of
       structural changes multiplied by λ.

    3. μ D_KL(p_t || p_{t-1}): Representation drift penalty — penalises
       large changes in the hidden representation between steps. Since we
       don't have access to the full probability distribution, we approximate
       D_KL as the MSE between current and previous (detached) block outputs.

The meta-loss drives two things:
    (a) Outer-loop gradient descent via the task loss component
    (b) Structural evolution decisions via the full meta-loss scalar
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_meta_loss(
    task_loss: Tensor,
    structural_changes: int = 0,
    current_hidden: Tensor | None = None,
    previous_hidden: Tensor | None = None,
    lambda_structural: float = 0.01,
    mu_drift: float = 0.001,
) -> Tensor:
    """Compute the DNH meta-loss for structural evolution.

    Args:
        task_loss: Scalar task loss from ACPredictorLossMixin.
        structural_changes: Count of add+prune events in this step.
        current_hidden: [B, N, D] current backbone output (optional).
        previous_hidden: [B, N, D] previous backbone output, detached (optional).
        lambda_structural: Weight for structural change penalty.
        mu_drift: Weight for representation drift penalty.

    Returns:
        Scalar meta-loss tensor.
    """
    meta_loss = task_loss.clone()

    # Structural change penalty: λ ||ΔG||_1
    if structural_changes > 0:
        penalty = lambda_structural * structural_changes
        meta_loss = meta_loss + penalty

    # Representation drift penalty: μ D_KL ≈ μ MSE(current, previous)
    if current_hidden is not None and previous_hidden is not None:
        drift = F.mse_loss(current_hidden, previous_hidden.detach())
        meta_loss = meta_loss + mu_drift * drift

    return meta_loss


def compute_level_meta_losses(
    level_outputs: list[Tensor],
    target: Tensor,
) -> list[Tensor]:
    """Compute per-level meta-losses for frequency modulation decisions.

    Each level's meta-loss is evaluated by how well its output reconstructs
    the target. Used by the structural evolution controller to determine
    which levels are contributing and which should be pruned.

    Args:
        level_outputs: List of [B, N, D] outputs from each level.
        target: [B, N, D] ground truth target.

    Returns:
        List of scalar loss tensors, one per level.
    """
    losses = []
    for output in level_outputs:
        loss = F.l1_loss(output, target.detach())
        losses.append(loss)
    return losses
