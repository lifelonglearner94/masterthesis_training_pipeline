"""Tests for ContinualLearningMetricsTracker.

Validates BWT, FWT, forgetting, and dual R-matrix logic against
known reference values.
"""

from __future__ import annotations

import json
import math
import tempfile

import numpy as np
import pytest

from src.utils.cl_metrics import ContinualLearningMetricsTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker_2tasks() -> ContinualLearningMetricsTracker:
    """Tracker for 2 tasks (3 experiences: base + 2 tasks)."""
    return ContinualLearningMetricsTracker(num_tasks=2, higher_is_better=False)


@pytest.fixture
def tracker_5tasks() -> ContinualLearningMetricsTracker:
    """Tracker for 5 tasks (6 experiences: base + 5 tasks)."""
    return ContinualLearningMetricsTracker(num_tasks=5, higher_is_better=False)


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

class TestInit:
    def test_matrix_shape(self, tracker_5tasks: ContinualLearningMetricsTracker) -> None:
        assert tracker_5tasks.R_matrix.shape == (6, 6)
        assert tracker_5tasks.R_matrix_jump.shape == (6, 6)

    def test_matrices_initialised_to_nan(self, tracker_2tasks: ContinualLearningMetricsTracker) -> None:
        assert np.isnan(tracker_2tasks.R_matrix).all()
        assert np.isnan(tracker_2tasks.R_matrix_jump).all()


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_primary(self, tracker_2tasks: ContinualLearningMetricsTracker) -> None:
        tracker_2tasks.update(0, 0, 0.5)
        assert tracker_2tasks.R_matrix[0, 0] == pytest.approx(0.5)
        # Jump matrix untouched
        assert np.isnan(tracker_2tasks.R_matrix_jump[0, 0])

    def test_update_jump(self, tracker_2tasks: ContinualLearningMetricsTracker) -> None:
        tracker_2tasks.update_jump(1, 0, 0.3)
        assert tracker_2tasks.R_matrix_jump[1, 0] == pytest.approx(0.3)
        # Primary matrix untouched
        assert np.isnan(tracker_2tasks.R_matrix[1, 0])


# ---------------------------------------------------------------------------
# Backward Transfer (BWT) — López-Paz & Ranzato (2017)
# ---------------------------------------------------------------------------

class TestBackwardTransfer:
    def test_bwt_nan_for_base_only(self, tracker_2tasks: ContinualLearningMetricsTracker) -> None:
        """BWT undefined when only base training is done."""
        tracker_2tasks.update(0, 0, 0.5)
        assert math.isnan(tracker_2tasks.get_backward_transfer(0))

    def test_bwt_positive_means_forgetting_for_loss(self) -> None:
        """Loss went up → positive BWT → forgetting."""
        t = ContinualLearningMetricsTracker(num_tasks=1, higher_is_better=False)
        # R[0,0] = 0.3 (base loss right after base training)
        t.update(0, 0, 0.3)
        # R[1,0] = 0.5 (base loss after task 1 training — worse)
        t.update(1, 0, 0.5)
        t.update(1, 1, 0.2)
        bwt = t.get_backward_transfer(1)
        # BWT = R[1,0] - R[0,0] = 0.5 - 0.3 = 0.2 (positive = forgetting)
        assert bwt == pytest.approx(0.2)

    def test_bwt_negative_means_improvement_for_loss(self) -> None:
        """Loss went down on old task → negative BWT → positive transfer."""
        t = ContinualLearningMetricsTracker(num_tasks=1, higher_is_better=False)
        t.update(0, 0, 0.5)
        t.update(1, 0, 0.3)  # Improved!
        t.update(1, 1, 0.2)
        bwt = t.get_backward_transfer(1)
        assert bwt == pytest.approx(-0.2)

    def test_bwt_average_over_multiple_tasks(self, tracker_5tasks: ContinualLearningMetricsTracker) -> None:
        """BWT averages over all past experiences."""
        # After base training
        tracker_5tasks.update(0, 0, 0.50)
        # After task 1
        tracker_5tasks.update(1, 0, 0.55)
        tracker_5tasks.update(1, 1, 0.40)
        # After task 2
        tracker_5tasks.update(2, 0, 0.60)  # base: 0.60 - 0.50 = +0.10
        tracker_5tasks.update(2, 1, 0.45)  # task1: 0.45 - 0.40 = +0.05
        tracker_5tasks.update(2, 2, 0.30)

        bwt = tracker_5tasks.get_backward_transfer(2)
        expected = (0.10 + 0.05) / 2  # 0.075
        assert bwt == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Forward Transfer (FWT)
# ---------------------------------------------------------------------------

class TestForwardTransfer:
    def test_fwt_zero_shot(self, tracker_5tasks: ContinualLearningMetricsTracker) -> None:
        """FWT = mean loss on unseen tasks."""
        for j in range(6):
            tracker_5tasks.update(0, j, 0.1 * (j + 1))
        fwt = tracker_5tasks.get_forward_transfer(0)
        # future tasks: j=1..5 → losses 0.2, 0.3, 0.4, 0.5, 0.6
        assert fwt == pytest.approx(np.mean([0.2, 0.3, 0.4, 0.5, 0.6]))

    def test_fwt_nan_after_last_task(self, tracker_2tasks: ContinualLearningMetricsTracker) -> None:
        tracker_2tasks.update(2, 0, 0.1)
        tracker_2tasks.update(2, 1, 0.2)
        tracker_2tasks.update(2, 2, 0.3)
        assert math.isnan(tracker_2tasks.get_forward_transfer(2))


# ---------------------------------------------------------------------------
# Experience & Stream Forgetting
# ---------------------------------------------------------------------------

class TestForgetting:
    def test_experience_forgetting_positive(self) -> None:
        """Loss increase → positive forgetting."""
        t = ContinualLearningMetricsTracker(num_tasks=2, higher_is_better=False)
        t.update(0, 0, 0.3)
        t.update(1, 0, 0.35)
        t.update(1, 1, 0.2)
        fgt = t.get_experience_forgetting(1, 0)
        assert fgt == pytest.approx(0.05)

    def test_experience_forgetting_best_past(self) -> None:
        """Forgetting is computed relative to the best (lowest loss) past value."""
        t = ContinualLearningMetricsTracker(num_tasks=3, higher_is_better=False)
        t.update(0, 0, 0.5)
        t.update(1, 0, 0.3)  # Improved (this is the best)
        t.update(1, 1, 0.4)
        t.update(2, 0, 0.4)  # Relative to best=0.3 → forgetting=0.1
        t.update(2, 1, 0.5)
        t.update(2, 2, 0.2)
        fgt = t.get_experience_forgetting(2, 0)
        assert fgt == pytest.approx(0.1)

    def test_stream_forgetting_average(self) -> None:
        t = ContinualLearningMetricsTracker(num_tasks=2, higher_is_better=False)
        t.update(0, 0, 0.3)
        t.update(1, 0, 0.35)
        t.update(1, 1, 0.2)
        t.update(2, 0, 0.4)   # fgt(2,0) = 0.4-0.3 = 0.1
        t.update(2, 1, 0.25)  # fgt(2,1) = 0.25-0.2 = 0.05
        t.update(2, 2, 0.1)
        sfgt = t.get_stream_forgetting(2)
        assert sfgt == pytest.approx((0.1 + 0.05) / 2)


# ---------------------------------------------------------------------------
# Top1 Stream
# ---------------------------------------------------------------------------

class TestTop1Stream:
    def test_top1_stream_averages_seen(self) -> None:
        t = ContinualLearningMetricsTracker(num_tasks=2, higher_is_better=False)
        t.update(1, 0, 0.3)
        t.update(1, 1, 0.5)
        top1 = t.get_top1_stream(1)
        assert top1 == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Dual R-Matrix: compute_all_metrics_jump
# ---------------------------------------------------------------------------

class TestDualRMatrix:
    def test_jump_metrics_prefix(self) -> None:
        t = ContinualLearningMetricsTracker(num_tasks=1, higher_is_better=False)
        t.update_jump(0, 0, 0.5)
        t.update_jump(0, 1, 0.8)
        metrics = t.compute_all_metrics_jump(0)
        # All keys should start with "cl_jump/"
        for k in metrics:
            assert k.startswith("cl_jump/"), f"Key {k} missing cl_jump/ prefix"

    def test_jump_bwt_independent(self) -> None:
        """Jump BWT computed on jump matrix, not on the primary matrix."""
        t = ContinualLearningMetricsTracker(num_tasks=1, higher_is_better=False)
        # Primary matrix: no forgetting
        t.update(0, 0, 0.5)
        t.update(1, 0, 0.5)
        t.update(1, 1, 0.3)
        # Jump matrix: severe forgetting
        t.update_jump(0, 0, 0.2)
        t.update_jump(1, 0, 0.9)
        t.update_jump(1, 1, 0.1)

        primary = t.compute_all_metrics(1)
        jump = t.compute_all_metrics_jump(1)

        assert primary["cl/BackwardTransfer"] == pytest.approx(0.0)   # 0.5 - 0.5
        assert jump["cl_jump/BackwardTransfer"] == pytest.approx(0.7)  # 0.9 - 0.2


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_includes_both_matrices(self) -> None:
        t = ContinualLearningMetricsTracker(num_tasks=1)
        t.update(0, 0, 1.0)
        t.update_jump(0, 0, 2.0)
        d = t.to_dict()
        assert "R_matrix" in d
        assert "R_matrix_jump" in d
        assert d["R_matrix"][0][0] == pytest.approx(1.0)
        assert d["R_matrix_jump"][0][0] == pytest.approx(2.0)

    def test_save_json_includes_jump_metrics(self) -> None:
        t = ContinualLearningMetricsTracker(num_tasks=1)
        t.update(0, 0, 0.5)
        t.update(0, 1, 0.8)
        t.update_jump(0, 0, 0.3)
        t.update_jump(0, 1, 0.6)
        with tempfile.NamedTemporaryFile(suffix=".json", mode="r+", delete=False) as f:
            t.save_json(f.name)
            f.seek(0)
            data = json.load(f)
        stage = data["metrics_per_stage"]["after_exp_0"]
        assert "cl/ForwardTransfer" in stage
        assert "cl_jump/ForwardTransfer" in stage

    def test_save_json_bwt_present(self) -> None:
        t = ContinualLearningMetricsTracker(num_tasks=1)
        t.update(0, 0, 0.5)
        t.update(1, 0, 0.6)
        t.update(1, 1, 0.3)
        with tempfile.NamedTemporaryFile(suffix=".json", mode="r+", delete=False) as f:
            t.save_json(f.name)
            f.seek(0)
            data = json.load(f)
        stage = data["metrics_per_stage"]["after_exp_1"]
        assert "cl/BackwardTransfer" in stage
        assert stage["cl/BackwardTransfer"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# compute_all_metrics consistency
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    def test_bwt_in_compute_all_metrics(self) -> None:
        t = ContinualLearningMetricsTracker(num_tasks=1)
        t.update(0, 0, 0.5)
        t.update(1, 0, 0.7)
        t.update(1, 1, 0.3)
        metrics = t.compute_all_metrics(1)
        assert "cl/BackwardTransfer" in metrics
        assert metrics["cl/BackwardTransfer"] == pytest.approx(0.2)

    def test_bwt_nan_when_base_only(self) -> None:
        t = ContinualLearningMetricsTracker(num_tasks=2)
        t.update(0, 0, 0.5)
        metrics = t.compute_all_metrics(0)
        assert math.isnan(metrics["cl/BackwardTransfer"])
