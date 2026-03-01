"""Tests for CMS frame-aware scheduling.

Verifies that update_period operates on frames (not flat token indices),
and that the fallback to standard mode works correctly.
"""

import pytest
import torch

from src.models.hope.cms import CMS, LevelSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dim() -> int:
    return 16


@pytest.fixture
def fast_only_cms(dim: int) -> CMS:
    """CMS with only a fast level (period=1)."""
    return CMS(
        dim=dim,
        levels=[LevelSpec(name="fast", update_period=1, hidden_multiplier=2.0)],
        use_chunk_scheduling=True,
    )


@pytest.fixture
def multi_level_cms(dim: int) -> CMS:
    """CMS with fast=1, medium=3, slow=7."""
    return CMS(
        dim=dim,
        levels=[
            LevelSpec(name="fast", update_period=1, hidden_multiplier=2.0),
            LevelSpec(name="medium", update_period=3, hidden_multiplier=2.0),
            LevelSpec(name="slow", update_period=7, hidden_multiplier=2.0),
        ],
        use_chunk_scheduling=True,
    )


@pytest.fixture
def scheduling_disabled_cms(dim: int) -> CMS:
    """CMS with scheduling disabled — all levels process all tokens."""
    return CMS(
        dim=dim,
        levels=[
            LevelSpec(name="fast", update_period=1, hidden_multiplier=2.0),
            LevelSpec(name="slow", update_period=7, hidden_multiplier=2.0),
        ],
        use_chunk_scheduling=False,
    )


# ---------------------------------------------------------------------------
# Tests: Frame-level masking correctness
# ---------------------------------------------------------------------------

class TestFrameAwareMasking:
    """Verify that update_period selects frames, not flat token indices."""

    def test_fast_processes_all_tokens(self, fast_only_cms: CMS, dim: int):
        """Fast level (period=1) should process every token."""
        B, T, tpf = 2, 4, 3
        x = torch.randn(B, T * tpf, dim)
        out = fast_only_cms(x, T=T, tokens_per_frame=tpf)
        # Output shape must match input
        assert out.shape == x.shape
        # Output should differ from input (MLP applied)
        assert not torch.allclose(out, x, atol=1e-6)

    def test_medium_processes_correct_frames(self, dim: int):
        """Medium (period=2): with T=4, should process frames 0, 2 only."""
        # Use a CMS where ONLY the medium level exists to isolate its effect
        cms = CMS(
            dim=dim,
            levels=[LevelSpec(name="medium", update_period=2, hidden_multiplier=2.0)],
            use_chunk_scheduling=True,
        )
        B, T, tpf = 1, 4, 3  # 4 frames × 3 tokens each = 12 tokens
        x = torch.randn(B, T * tpf, dim)

        out = cms(x, T=T, tokens_per_frame=tpf)

        # Frames 0 and 2 should be modified (indices 0-2 and 6-8)
        # Frames 1 and 3 should be unchanged (indices 3-5 and 9-11)
        assert not torch.allclose(out[:, 0:3, :], x[:, 0:3, :], atol=1e-6), \
            "Frame 0 should be processed"
        assert torch.allclose(out[:, 3:6, :], x[:, 3:6, :], atol=1e-6), \
            "Frame 1 should be skipped"
        assert not torch.allclose(out[:, 6:9, :], x[:, 6:9, :], atol=1e-6), \
            "Frame 2 should be processed"
        assert torch.allclose(out[:, 9:12, :], x[:, 9:12, :], atol=1e-6), \
            "Frame 3 should be skipped"

    def test_slow_processes_only_frame_zero(self, dim: int):
        """Slow (period=7): with T=7, should process only frame 0."""
        cms = CMS(
            dim=dim,
            levels=[LevelSpec(name="slow", update_period=7, hidden_multiplier=2.0)],
            use_chunk_scheduling=True,
        )
        B, T, tpf = 1, 7, 4  # 7 frames × 4 tokens each = 28 tokens
        x = torch.randn(B, T * tpf, dim)

        out = cms(x, T=T, tokens_per_frame=tpf)

        # Only frame 0 (tokens 0-3) should be modified
        assert not torch.allclose(out[:, 0:4, :], x[:, 0:4, :], atol=1e-6), \
            "Frame 0 should be processed"
        # All other frames should be unchanged
        assert torch.allclose(out[:, 4:, :], x[:, 4:, :], atol=1e-6), \
            "Frames 1-6 should be skipped"

    def test_period_larger_than_T_processes_only_frame_zero(self, dim: int):
        """Period=16 with T=7: only frame 0 should be processed (index 0 % 16 == 0)."""
        cms = CMS(
            dim=dim,
            levels=[LevelSpec(name="slow", update_period=16, hidden_multiplier=2.0)],
            use_chunk_scheduling=True,
        )
        B, T, tpf = 1, 7, 4
        x = torch.randn(B, T * tpf, dim)

        out = cms(x, T=T, tokens_per_frame=tpf)

        assert not torch.allclose(out[:, 0:4, :], x[:, 0:4, :], atol=1e-6)
        assert torch.allclose(out[:, 4:, :], x[:, 4:, :], atol=1e-6)

    def test_all_frames_active_when_period_equals_1(self, multi_level_cms: CMS, dim: int):
        """All tokens should be modified when period=1 for fast level."""
        B, T, tpf = 2, 7, 258  # Realistic: 256 patches + 2 action tokens
        x = torch.randn(B, T * tpf, dim)
        out = multi_level_cms(x, T=T, tokens_per_frame=tpf)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Tests: Backward compatibility / fallback
# ---------------------------------------------------------------------------

class TestFallbackBehavior:
    """Verify that scheduling gracefully degrades when T/tokens_per_frame are missing."""

    def test_no_T_falls_back_to_standard(self, multi_level_cms: CMS, dim: int):
        """Without T, all levels should process all tokens (standard mode)."""
        B, N = 2, 12
        x = torch.randn(B, N, dim)
        out = multi_level_cms(x, T=None, tokens_per_frame=3)
        assert out.shape == x.shape
        # All tokens should be modified (standard cascade)
        assert not torch.allclose(out, x, atol=1e-6)

    def test_no_tokens_per_frame_falls_back(self, multi_level_cms: CMS, dim: int):
        """Without tokens_per_frame, all levels should process all tokens."""
        B, N = 2, 12
        x = torch.randn(B, N, dim)
        out = multi_level_cms(x, T=4, tokens_per_frame=None)
        assert out.shape == x.shape
        assert not torch.allclose(out, x, atol=1e-6)

    def test_scheduling_disabled_processes_all(self, scheduling_disabled_cms: CMS, dim: int):
        """With use_chunk_scheduling=False, all levels process all tokens."""
        B, T, tpf = 1, 4, 3
        x = torch.randn(B, T * tpf, dim)
        out = scheduling_disabled_cms(x, T=T, tokens_per_frame=tpf)
        assert out.shape == x.shape
        assert not torch.allclose(out, x, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Ensure gradients flow through both processed and skipped tokens."""

    def test_gradients_flow_to_all_input_tokens(self, dim: int):
        """All input tokens should have gradients, even if their frame is skipped."""
        cms = CMS(
            dim=dim,
            levels=[LevelSpec(name="slow", update_period=7, hidden_multiplier=2.0)],
            use_chunk_scheduling=True,
        )
        B, T, tpf = 1, 7, 4
        x = torch.randn(B, T * tpf, dim, requires_grad=True)

        out = cms(x, T=T, tokens_per_frame=tpf)
        loss = out.sum()
        loss.backward()

        # All input tokens should have gradients (skipped tokens have grad=1
        # from the identity pass-through)
        assert x.grad is not None
        assert x.grad.shape == x.shape
        # No zero gradients — even skipped frames get gradient 1.0 from identity
        assert (x.grad.abs() > 0).all(), \
            "All tokens should have non-zero gradients"

    def test_gradients_flow_to_cms_parameters(self, dim: int):
        """CMS block parameters should receive gradients from active frames."""
        cms = CMS(
            dim=dim,
            levels=[LevelSpec(name="medium", update_period=2, hidden_multiplier=2.0)],
            use_chunk_scheduling=True,
        )
        B, T, tpf = 1, 4, 3
        x = torch.randn(B, T * tpf, dim)

        out = cms(x, T=T, tokens_per_frame=tpf)
        loss = out.sum()
        loss.backward()

        for name, param in cms.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert param.grad.abs().sum() > 0, f"Parameter {name} has zero gradient"


# ---------------------------------------------------------------------------
# Tests: Output shape consistency
# ---------------------------------------------------------------------------

class TestOutputShape:
    """Verify output shapes match input shapes in all modes."""

    @pytest.mark.parametrize("T,tpf", [(1, 258), (7, 258), (4, 10), (8, 1)])
    def test_output_shape_matches_input(self, multi_level_cms: CMS, dim: int, T: int, tpf: int):
        B = 2
        x = torch.randn(B, T * tpf, dim)
        out = multi_level_cms(x, T=T, tokens_per_frame=tpf)
        assert out.shape == x.shape
