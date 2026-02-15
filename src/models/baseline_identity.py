"""Identity Baseline ("Last Frame Copy") for action-conditioned prediction.

The simplest possible baseline: for each predicted timestep, copy the last
context frame. This ignores actions entirely and measures how much a learned
predictor improves over the trivial "nothing changes" assumption.

Used for scientific comparison — any useful predictor should beat this.

Usage:
    # Evaluate identity baseline on test clips (no training needed):
    uv run src/eval.py experiment=test_baseline_identity paths.data_dir=/path/to/clips

    # Custom clip range:
    uv run src/eval.py experiment=test_baseline_identity \\
        paths.data_dir=/path/to/clips data.clip_start=15000 data.clip_end=16000
"""

import logging
from typing import Any, Final

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.mixins import ACPredictorLossMixin

# Constants (same as other baselines for consistency)
DEFAULT_INPUT_DIM: Final = 1024
DEFAULT_SPATIAL_SIZE: Final = 16
DEFAULT_NUM_TIMESTEPS: Final = 8
DEFAULT_T_TEACHER: Final = 7
DEFAULT_T_ROLLOUT: Final = 7
DEFAULT_CONTEXT_FRAMES: Final = 1
DEFAULT_LOSS_WEIGHT_TEACHER: Final = 0.0
DEFAULT_LOSS_WEIGHT_ROLLOUT: Final = 1.0
DEFAULT_LOSS_EXP: Final = 1.0
LAYER_NORM_EPS: Final = 1e-6

# Type aliases
type TestResult = dict[str, Any]

log = logging.getLogger(__name__)


class IdentityBaseline(torch.nn.Module):
    """Identity baseline: copies the last input frame for all predicted timesteps.

    Given input features z = [z_0, z_1, ..., z_{T-1}] with shape [B, T*N, D],
    the output is z_{T-1} repeated T times: [z_{T-1}, z_{T-1}, ..., z_{T-1}].

    This model has NO learnable parameters.

    Args:
        patches_per_frame: Number of patches (tokens) per frame (N = H*W).
    """

    def __init__(self, patches_per_frame: int = DEFAULT_SPATIAL_SIZE ** 2) -> None:
        super().__init__()
        self.patches_per_frame = patches_per_frame

    def forward(
        self,
        z: Tensor,
        actions: Tensor,
        states: Tensor | None = None,
        extrinsics: Tensor | None = None,
    ) -> Tensor:
        """Copy the last input frame for every output position.

        Args:
            z: Input features [B, T*N, D] (flattened frames).
            actions: Action sequences [B, T, action_dim] — IGNORED.
            states: State sequences — IGNORED.
            extrinsics: Extrinsics — IGNORED.

        Returns:
            Predicted features [B, T*N, D] where every frame is a copy
            of the last input frame.
        """
        N = self.patches_per_frame
        B, total_tokens, D = z.shape
        T_input = total_tokens // N

        # Extract the last frame: [B, N, D]
        last_frame = z[:, -N:, :]

        # Repeat it T_input times to match expected output shape [B, T*N, D]
        output = last_frame.unsqueeze(1).expand(B, T_input, N, D)
        return output.reshape(B, T_input * N, D)


class IdentityBaselineLitModule(ACPredictorLossMixin, L.LightningModule):
    """PyTorch Lightning module for Identity Baseline (Last Frame Copy).

    A parameter-free baseline that predicts every future frame as a copy of
    the last observed context frame. Actions are completely ignored.

    This establishes a lower bound: any useful learned predictor must beat
    the identity baseline to demonstrate that it has learned meaningful
    action-conditioned dynamics.

    Uses the same loss computation and evaluation protocol as all other models
    via ACPredictorLossMixin, ensuring scientific comparability.

    Expected batch format:
        - features: [B, T+1, N, D] - Pre-computed V-JEPA2 encoder features
        - actions: [B, T, action_dim] - Action sequences (ignored by model)
        - states: [B, T, action_dim] - State sequences (ignored by model)

    Where:
        - T is the number of encoded timesteps (e.g., 8)
        - N is the number of patches per frame (256 = 16x16)
        - D is the embedding dimension (1024 for V-JEPA2)
    """

    def __init__(
        self,
        # Model architecture
        input_dim: int = DEFAULT_INPUT_DIM,
        spatial_size: int = DEFAULT_SPATIAL_SIZE,
        num_timesteps: int = DEFAULT_NUM_TIMESTEPS,
        # Loss settings (same interface as other models)
        T_teacher: int = DEFAULT_T_TEACHER,
        T_rollout: int = DEFAULT_T_ROLLOUT,
        context_frames: int = DEFAULT_CONTEXT_FRAMES,
        loss_weight_teacher: float = DEFAULT_LOSS_WEIGHT_TEACHER,
        loss_weight_rollout: float = DEFAULT_LOSS_WEIGHT_ROLLOUT,
        normalize_reps: bool = True,
        loss_exp: float = DEFAULT_LOSS_EXP,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.num_timesteps = num_timesteps
        self.patches_per_frame = spatial_size * spatial_size

        # Build the identity model (no learnable parameters)
        self.model = IdentityBaseline(patches_per_frame=self.patches_per_frame)

        # Loss hyperparameters (required by ACPredictorLossMixin)
        self.T_teacher = T_teacher
        self.T_rollout = T_rollout
        self.context_frames = context_frames
        self.loss_weight_teacher = loss_weight_teacher
        self.loss_weight_rollout = loss_weight_rollout
        self.normalize_reps = normalize_reps

        if loss_exp <= 0:
            raise ValueError(f"loss_exp must be positive (got {loss_exp})")
        self.loss_exp = loss_exp

        # Test results storage
        self._test_results: list[TestResult] = []

        log.info(
            f"Identity Baseline initialized (0 learnable parameters)\n"
            f"  patches_per_frame={self.patches_per_frame}, "
            f"context_frames={context_frames}, T_rollout={T_rollout}"
        )

    def forward(
        self,
        features: Tensor,
        actions: Tensor,
        states: Tensor | None = None,
        extrinsics: Tensor | None = None,
    ) -> Tensor:
        """Forward pass (last-frame copy).

        Args:
            features: Encoded features [B, T*N, D]
            actions: Action sequences [B, T, action_dim] — ignored
            states: State sequences — ignored
            extrinsics: Extrinsics — ignored

        Returns:
            Predicted features [B, T*N, D]
        """
        return self.model(features, actions, states, extrinsics)

    def _step_predictor(
        self,
        z: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
    ) -> Tensor:
        """Single predictor step with optional layer normalization.

        Required by ACPredictorLossMixin.

        Args:
            z: Input features [B, T*N, D]
            actions: Actions [B, T, action_dim] — ignored
            states: States [B, T, action_dim] — ignored
            extrinsics: Optional extrinsics — ignored

        Returns:
            Predicted features [B, T*N, D], optionally normalized
        """
        z_pred = self.model(z, actions, states, extrinsics)
        if self.normalize_reps:
            z_pred = F.layer_norm(z_pred, (z_pred.size(-1),), eps=LAYER_NORM_EPS)
        return z_pred

    # ---- Training is not needed, but provide stubs for completeness ----

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Training step (identity baseline has nothing to train)."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Validation step."""
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Test step with detailed per-clip analysis."""
        features = batch["features"]
        actions = batch["actions"]
        states = batch["states"]
        extrinsics = batch.get("extrinsics", None)
        clip_names = batch.get(
            "clip_names", [f"clip_{batch_idx}_{i}" for i in range(features.shape[0])]
        )

        # Reshape features if needed: [B, (T+1)*N, D] -> [B, T+1, N, D]
        if features.dim() == 3:
            B, total_tokens, D = features.shape
            T_plus_1 = total_tokens // self.patches_per_frame
            features = features.reshape(B, T_plus_1, self.patches_per_frame, D)
        else:
            B = features.shape[0]

        # Compute losses
        loss_teacher = self._compute_teacher_forcing_loss(
            features, actions, states, extrinsics
        )
        loss_rollout, per_timestep_losses, per_sample_losses = (
            self._compute_rollout_loss_per_timestep(
                features, actions, states, extrinsics
            )
        )

        loss = (
            self.loss_weight_teacher * loss_teacher
            + self.loss_weight_rollout * loss_rollout
        )

        # Log metrics
        self.log("test/loss_teacher", loss_teacher, prog_bar=True, sync_dist=True)
        self.log("test/loss_rollout", loss_rollout, prog_bar=True, sync_dist=True)
        self.log("test/loss", loss, prog_bar=True, sync_dist=True)

        for step, step_loss in enumerate(per_timestep_losses):
            predicted_frame = self.context_frames + step
            self.log(
                f"test/loss_step_{predicted_frame}",
                step_loss.mean(),
                sync_dist=True,
            )

        # Store per-clip results
        per_sample_rollout_losses = per_sample_losses[0]
        for i in range(B):
            clip_result = {
                "clip_name": (
                    clip_names[i]
                    if i < len(clip_names)
                    else f"unknown_{batch_idx}_{i}"
                ),
                "loss_rollout": per_sample_rollout_losses[i].item(),
                "loss_teacher": loss_teacher.item(),
                "per_timestep_losses": {
                    f"step_{self.context_frames + s}": per_timestep_losses[s][i].item()
                    for s in range(len(per_timestep_losses))
                },
            }
            self._test_results.append(clip_result)

        return loss

    def on_test_epoch_start(self) -> None:
        """Clear test results."""
        self._test_results = []

    def on_test_epoch_end(self) -> None:
        """Aggregate and output test results."""
        if not self._test_results:
            log.warning("No test results to aggregate")
            return

        num_clips = len(self._test_results)
        rollout_losses = [r["loss_rollout"] for r in self._test_results]

        mean_loss = sum(rollout_losses) / num_clips
        sorted_losses = sorted(rollout_losses)
        median_loss = sorted_losses[num_clips // 2]
        min_loss = min(rollout_losses)
        max_loss = max(rollout_losses)
        std_loss = (
            sum((x - mean_loss) ** 2 for x in rollout_losses) / num_clips
        ) ** 0.5

        print("\n" + "=" * 70)
        print("        IDENTITY BASELINE (LAST FRAME COPY) TEST RESULTS")
        print("=" * 70)
        print(f"\n📊 AGGREGATE STATISTICS (over {num_clips} clips)")
        print("-" * 50)
        print(f"  Rollout Loss (L1):")
        print(f"    Mean:   {mean_loss:.6f}")
        print(f"    Median: {median_loss:.6f}")
        print(f"    Std:    {std_loss:.6f}")
        print(f"    Min:    {min_loss:.6f}")
        print(f"    Max:    {max_loss:.6f}")
        print()

        # Per-timestep breakdown
        if self._test_results[0].get("per_timestep_losses"):
            step_keys = sorted(self._test_results[0]["per_timestep_losses"].keys())
            print(f"  Per-Timestep Breakdown:")
            for key in step_keys:
                step_losses = [
                    r["per_timestep_losses"][key] for r in self._test_results
                ]
                step_mean = sum(step_losses) / len(step_losses)
                print(f"    {key}: {step_mean:.6f}")

        print("=" * 70)

        self.log("test/final_mean_loss_rollout", mean_loss, sync_dist=True)
        self.log("test/final_median_loss_rollout", median_loss, sync_dist=True)

    def configure_optimizers(self) -> dict:
        """No optimizer needed (no trainable parameters).

        Returns a dummy optimizer in case Lightning requires one.
        """
        # Create a single dummy parameter so AdamW doesn't complain
        dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        optimizer = torch.optim.SGD([dummy], lr=0.0)
        return {"optimizer": optimizer}
