"""Framework-agnostic Continual Learning metrics tracker.

Maintains an R-matrix where R[i, j] = performance on task j after training
on task i. Computes standard CL metrics: FWT, BWT, forgetting (experience
and stream level), and Top1 metrics.

Designed for loss-based evaluation (lower is better).

Reference: ``temp/cl_framework_agnostic_logic.py`` (original prototype).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class ContinualLearningMetricsTracker:
    """Track CL metrics via an R-matrix of task performances.

    R_matrix[i, j] = performance on task j after training on experience i.

    Experience indexing:
        - 0 = base training (5000 clips)
        - 1..num_tasks = sequential CL tasks

    So total experiences = 1 (base) + num_tasks.

    Args:
        num_tasks: Number of CL tasks (excluding base training).
        higher_is_better: True for accuracy, False for loss metrics.
    """

    def __init__(self, num_tasks: int, higher_is_better: bool = False) -> None:
        self.num_tasks = num_tasks
        self.higher_is_better = higher_is_better
        self.num_experiences = num_tasks + 1  # base + tasks

        # R_matrix[i, j] = metric on task j after training experience i
        self.R_matrix = np.full((self.num_experiences, self.num_experiences), np.nan)
        # Secondary R-matrix for jump loss (same shape)
        self.R_matrix_jump = np.full((self.num_experiences, self.num_experiences), np.nan)

    def update(self, train_exp_id: int, eval_exp_id: int, performance: float) -> None:
        """Record performance on eval_exp_id after training experience train_exp_id.

        Args:
            train_exp_id: Training experience index (0=base, 1..N=tasks).
            eval_exp_id: Evaluation task index (0=base, 1..N=tasks).
            performance: Metric value (e.g. L1 loss).
        """
        self.R_matrix[train_exp_id, eval_exp_id] = performance
        log.info(
            f"R[{train_exp_id}, {eval_exp_id}] = {performance:.6f}"
        )

    def update_jump(self, train_exp_id: int, eval_exp_id: int, performance: float) -> None:
        """Record jump loss on eval_exp_id after training experience train_exp_id.

        Args:
            train_exp_id: Training experience index (0=base, 1..N=tasks).
            eval_exp_id: Evaluation task index (0=base, 1..N=tasks).
            performance: Jump loss value.
        """
        self.R_matrix_jump[train_exp_id, eval_exp_id] = performance
        log.info(
            f"R_jump[{train_exp_id}, {eval_exp_id}] = {performance:.6f}"
        )

    def get_top1_exp(self, train_exp_id: int, eval_exp_id: int) -> float:
        """Top1_L1_Exp: Performance on a specific experience."""
        return float(self.R_matrix[train_exp_id, eval_exp_id])

    def get_top1_stream(self, train_exp_id: int) -> float:
        """Top1_L1_Stream: Average performance on all SEEN experiences so far.

        Evaluates experiences 0 up to and including train_exp_id.
        """
        seen = self.R_matrix[train_exp_id, : train_exp_id + 1]
        if np.isnan(seen).any():
            return float("nan")
        return float(np.mean(seen))

    def get_experience_forgetting(
        self, train_exp_id: int, eval_exp_id: int
    ) -> float:
        """ExperienceForgetting: Degradation relative to best past performance.

        For loss (lower=better): current_loss - best_past_loss (positive = forgetting).
        For accuracy (higher=better): best_past_acc - current_acc.
        """
        if train_exp_id <= eval_exp_id:
            return 0.0  # Can't forget a future/current task

        past = self.R_matrix[:train_exp_id, eval_exp_id]
        current = self.R_matrix[train_exp_id, eval_exp_id]

        if np.isnan(past).all() or np.isnan(current):
            return float("nan")

        if self.higher_is_better:
            return float(np.nanmax(past) - current)
        else:
            return float(current - np.nanmin(past))

    def get_stream_forgetting(self, train_exp_id: int) -> float:
        """StreamForgetting: Average forgetting across all previously seen experiences."""
        if train_exp_id == 0:
            return 0.0

        forgetting = [
            self.get_experience_forgetting(train_exp_id, j)
            for j in range(train_exp_id)
        ]
        valid = [f for f in forgetting if not np.isnan(f)]
        return float(np.mean(valid)) if valid else float("nan")

    def get_forward_transfer(self, train_exp_id: int) -> float:
        """Forward Transfer: Average performance on unseen future tasks.

        Measures zero-shot generalization to tasks not yet trained on.
        """
        if train_exp_id >= self.num_experiences - 1:
            return float("nan")  # No future tasks

        future = self.R_matrix[train_exp_id, train_exp_id + 1 :]
        if np.isnan(future).any():
            return float("nan")
        return float(np.mean(future))

    def get_backward_transfer(self, train_exp_id: int) -> float:
        """Backward Transfer (BWT) after López-Paz & Ranzato (2017).

        Measures how learning new tasks affects performance on previous tasks:

            BWT = (1 / (T-1)) * sum_{i=1}^{T-1} (R[T, i] - R[i, i])

        For loss metrics (lower=better): positive BWT means forgetting (loss increased).
        For accuracy (higher=better): negative BWT means forgetting.

        Requires at least 2 experiences (base + 1 task) to be meaningful.
        """
        if train_exp_id < 1:
            return float("nan")  # Need at least base + 1 task

        bwt_values = []
        for i in range(train_exp_id):
            r_final = self.R_matrix[train_exp_id, i]
            r_diag = self.R_matrix[i, i]
            if np.isnan(r_final) or np.isnan(r_diag):
                continue
            bwt_values.append(r_final - r_diag)

        if not bwt_values:
            return float("nan")
        return float(np.mean(bwt_values))

    def compute_all_metrics(self, train_exp_id: int) -> dict[str, float]:
        """Compute all CL metrics for the current training stage.

        Returns:
            Dictionary with metric names → values, suitable for W&B logging.
        """
        metrics: dict[str, float] = {
            "cl/Top1_L1_Stream": self.get_top1_stream(train_exp_id),
            "cl/StreamForgetting": self.get_stream_forgetting(train_exp_id),
            "cl/ForwardTransfer": self.get_forward_transfer(train_exp_id),
            "cl/BackwardTransfer": self.get_backward_transfer(train_exp_id),
        }

        # Per-experience metrics
        for j in range(self.num_experiences):
            val = self.R_matrix[train_exp_id, j]
            if not np.isnan(val):
                metrics[f"cl/Top1_L1_Exp_{j}"] = float(val)
                if j < train_exp_id:
                    fgt = self.get_experience_forgetting(train_exp_id, j)
                    metrics[f"cl/ExperienceForgetting_{j}"] = fgt

        return metrics

    def _compute_metrics_on_matrix(
        self, matrix: np.ndarray, train_exp_id: int, prefix: str
    ) -> dict[str, float]:
        """Compute all CL metrics on an arbitrary R-matrix with a given prefix.

        Internal helper to avoid code duplication between teacher and jump metrics.
        """
        # Top1 Stream: average over seen experiences
        seen = matrix[train_exp_id, : train_exp_id + 1]
        top1_stream = float(np.mean(seen)) if not np.isnan(seen).any() else float("nan")

        # Forward Transfer: average over unseen future tasks
        if train_exp_id < self.num_experiences - 1:
            future = matrix[train_exp_id, train_exp_id + 1 :]
            fwt = float(np.mean(future)) if not np.isnan(future).any() else float("nan")
        else:
            fwt = float("nan")

        # Backward Transfer
        bwt = float("nan")
        if train_exp_id >= 1:
            bwt_vals = []
            for i in range(train_exp_id):
                r_final = matrix[train_exp_id, i]
                r_diag = matrix[i, i]
                if not (np.isnan(r_final) or np.isnan(r_diag)):
                    bwt_vals.append(r_final - r_diag)
            if bwt_vals:
                bwt = float(np.mean(bwt_vals))

        # Stream Forgetting
        if train_exp_id == 0:
            stream_fgt = 0.0
        else:
            fgt_vals = []
            for j in range(train_exp_id):
                past = matrix[:train_exp_id, j]
                current = matrix[train_exp_id, j]
                if not (np.isnan(past).all() or np.isnan(current)):
                    if self.higher_is_better:
                        fgt_vals.append(float(np.nanmax(past) - current))
                    else:
                        fgt_vals.append(float(current - np.nanmin(past)))
            stream_fgt = float(np.mean(fgt_vals)) if fgt_vals else float("nan")

        metrics: dict[str, float] = {
            f"{prefix}/Top1_L1_Stream": top1_stream,
            f"{prefix}/StreamForgetting": stream_fgt,
            f"{prefix}/ForwardTransfer": fwt,
            f"{prefix}/BackwardTransfer": bwt,
        }

        # Per-experience metrics
        for j in range(self.num_experiences):
            val = matrix[train_exp_id, j]
            if not np.isnan(val):
                metrics[f"{prefix}/Top1_L1_Exp_{j}"] = float(val)
                if j < train_exp_id:
                    past = matrix[:train_exp_id, j]
                    current = matrix[train_exp_id, j]
                    if not (np.isnan(past).all() or np.isnan(current)):
                        if self.higher_is_better:
                            fgt = float(np.nanmax(past) - current)
                        else:
                            fgt = float(current - np.nanmin(past))
                        metrics[f"{prefix}/ExperienceForgetting_{j}"] = fgt

        return metrics

    def compute_all_metrics_jump(self, train_exp_id: int) -> dict[str, float]:
        """Compute all CL metrics for the jump loss R-matrix.

        Returns:
            Dictionary with metric names prefixed with ``cl_jump/``.
        """
        return self._compute_metrics_on_matrix(
            self.R_matrix_jump, train_exp_id, prefix="cl_jump"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize tracker state to a JSON-friendly dictionary."""
        return {
            "num_tasks": self.num_tasks,
            "higher_is_better": self.higher_is_better,
            "R_matrix": self.R_matrix.tolist(),
            "R_matrix_jump": self.R_matrix_jump.tolist(),
        }

    def save_json(self, path: str) -> None:
        """Save tracker state and all metrics to a JSON file."""
        data = self.to_dict()
        # Add computed metrics for each training stage
        data["metrics_per_stage"] = {}
        for i in range(self.num_experiences):
            if not np.isnan(self.R_matrix[i]).all():
                stage_metrics = self.compute_all_metrics(i)
                # Include jump metrics if any jump data exists
                if not np.isnan(self.R_matrix_jump[i]).all():
                    stage_metrics.update(self.compute_all_metrics_jump(i))
                data["metrics_per_stage"][f"after_exp_{i}"] = stage_metrics
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        log.info(f"CL metrics saved to {path}")

    def __repr__(self) -> str:
        filled = np.count_nonzero(~np.isnan(self.R_matrix))
        filled_jump = np.count_nonzero(~np.isnan(self.R_matrix_jump))
        total = self.R_matrix.size
        return (
            f"ContinualLearningMetricsTracker("
            f"num_tasks={self.num_tasks}, "
            f"filled={filled}/{total}, "
            f"filled_jump={filled_jump}/{total})"
        )
