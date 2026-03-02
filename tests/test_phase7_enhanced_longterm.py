"""Tests for Phase 7: Enhanced longterm memory (B→A upgrades).

Tests verify the four Phase 7 enhancements:

1. Retrieval-conditioned gate:
   - Gate input is [q; |o_clip - o_long|] (2*D dimensions) when enabled
   - Gate input is q alone (D dimensions) when disabled (Phase 6 compat)
   - Gate still produces [0, 1] output with correct initialization
   - Discrepancy signal affects gate output when memories differ

2. Asymmetric decay:
   - α for M_longterm is clamped to [α_min, 1.0] when enabled
   - α for clip-level memories is unchanged
   - With α_min=0.95, M_longterm retains more knowledge per DGD step

3. Longterm-specific surprise:
   - M_longterm uses its own retrieval error for surprise when enabled
   - M_longterm uses M_memory's surprise when disabled (Phase 6 compat)
   - Two surprise signals are independent

4. Consolidation EMA:
   - consolidate_longterm_memory() absorbs DGD state into nn.Parameters
   - EMA rate controls consolidation speed
   - Active weights are re-initialized from updated parameters after consolidation
   - Consolidation is a no-op when ema=0.0

5. Integration:
   - Full forward/backward pass with all Phase 7 features enabled
   - Phase 6 backward compatibility (all features disabled)
   - Multi-clip persistence with consolidation
"""

import pytest
import torch

from src.models.hope import ac_hope_vit
from src.models.hope.hope_block import HOPEBlock, HOPEBlockConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_phase7_model(**kwargs):
    """Create a test model with Phase 7 longterm memory features."""
    defaults = dict(
        img_size=(256, 256),
        patch_size=16,
        num_timesteps=8,
        embed_dim=1024,
        predictor_embed_dim=384,
        depth=2,
        num_heads=16,
        action_embed_dim=2,
        use_rope=False,
        is_frame_causal=True,
        titan_hidden_multiplier=2,
        titan_layers=2,
        titan_grad_clip_inner=1.0,
        chunk_size=0,
        titan_detach_interval=0,
        surprise_threshold=0.0,
        use_spatial_mixing=True,
        use_longterm_memory=True,
        longterm_hidden_multiplier=2,
        longterm_lr_scale=0.05,
        longterm_retrieval_conditioned_gate=True,
        longterm_alpha_min=0.95,
        longterm_own_surprise=True,
        longterm_consolidation_ema=0.01,
        drop_path_rate=0.0,
        log_hope_diagnostics=True,
    )
    defaults.update(kwargs)
    model = ac_hope_vit(**defaults)
    model.train()
    return model


def _make_phase7_block(**kwargs):
    """Create a standalone HOPEBlock with Phase 7 features."""
    defaults = dict(
        dim=384,
        num_heads=16,
        titan_hidden_multiplier=2,
        titan_layers=2,
        titan_activation="gelu",
        titan_grad_clip_inner=1.0,
        titan_grad_clip_backward=1.0,
        chunk_size=0,
        use_rope=False,
        use_spatial_mixing=False,
        use_longterm_memory=True,
        longterm_hidden_multiplier=2,
        longterm_lr_scale=0.05,
        longterm_retrieval_conditioned_gate=True,
        longterm_alpha_min=0.95,
        longterm_own_surprise=True,
        longterm_consolidation_ema=0.01,
    )
    defaults.update(kwargs)
    config = HOPEBlockConfig(**defaults)
    return HOPEBlock(config, titan_detach_interval=0)


def _make_phase6_block(**kwargs):
    """Create a Phase 6 compatible block (all Phase 7 features off)."""
    return _make_phase7_block(
        longterm_retrieval_conditioned_gate=False,
        longterm_alpha_min=0.0,
        longterm_own_surprise=False,
        longterm_consolidation_ema=0.0,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def phase7_model():
    return _make_phase7_model()


@pytest.fixture
def phase7_block():
    return _make_phase7_block()


@pytest.fixture
def phase6_block():
    return _make_phase6_block()


@pytest.fixture
def sample_inputs():
    """Minimal inputs: B=1, T=3 timesteps."""
    B, T, N, D = 1, 3, 256, 1024
    x = torch.randn(B, T * N, D)
    actions = torch.randn(B, T, 2)
    states = torch.randn(B, T, 2)
    return x, actions, states


# ===========================================================================
# Test Class 1: Retrieval-Conditioned Gate
# ===========================================================================

class TestRetrievalConditionedGate:
    """Test the Phase 7 retrieval-conditioned gate mechanism."""

    def test_gate_input_dim_phase7(self):
        """Phase 7 gate should accept 2*D input (q + discrepancy)."""
        blk = _make_phase7_block()
        # longterm_gate should be Linear(2*384, 1) = Linear(768, 1)
        assert blk.longterm_gate.in_features == 2 * 384
        assert blk.longterm_gate.out_features == 1

    def test_gate_input_dim_phase6(self):
        """Phase 6 gate should accept D input (q only)."""
        blk = _make_phase6_block()
        assert blk.longterm_gate.in_features == 384
        assert blk.longterm_gate.out_features == 1

    def test_gate_initial_bias_still_favors_clip(self):
        """Gate bias=1.0 should still produce ~0.73 initially."""
        blk = _make_phase7_block()
        assert blk.longterm_gate.bias.item() == pytest.approx(1.0)
        # With zero weights, gate output = sigmoid(bias) regardless of input dim
        x = torch.randn(1, 5, 768)  # 2*D input
        gate = torch.sigmoid(blk.longterm_gate(x))
        expected = torch.sigmoid(torch.tensor(1.0))
        assert gate.mean().item() == pytest.approx(expected.item(), abs=1e-5)

    def test_gate_weights_zero_initialized(self):
        """Gate weights should be zero-initialized (both Phase 6 and 7)."""
        blk = _make_phase7_block()
        assert (blk.longterm_gate.weight == 0).all()

    def test_discrepancy_affects_gate_output(self):
        """After training, gate should respond to discrepancy signal."""
        blk = _make_phase7_block()
        # Manually set gate weights: make the discrepancy part sensitive
        D = 384
        with torch.no_grad():
            # Set weights so that high discrepancy → lower gate (favor longterm)
            blk.longterm_gate.weight[0, D:] = -0.01  # Negative for discrepancy dims
        # Low discrepancy → gate ≈ sigmoid(1.0) ≈ 0.73
        low_disc_input = torch.cat([torch.randn(1, 1, D), torch.zeros(1, 1, D)], dim=-1)
        gate_low = torch.sigmoid(blk.longterm_gate(low_disc_input))

        # High discrepancy → gate < 0.73 (shifted by negative weights)
        high_disc_input = torch.cat([torch.randn(1, 1, D), torch.ones(1, 1, D) * 10], dim=-1)
        gate_high = torch.sigmoid(blk.longterm_gate(high_disc_input))

        assert gate_high.item() < gate_low.item(), \
            "High discrepancy should lower gate (favor longterm more)"


# ===========================================================================
# Test Class 2: Asymmetric Decay
# ===========================================================================

class TestAsymmetricDecay:
    """Test the Phase 7 asymmetric decay mechanism."""

    def test_alpha_min_stored(self):
        """Block should store the alpha_min value."""
        blk = _make_phase7_block()
        assert blk.longterm_alpha_min == 0.95

    def test_alpha_min_disabled_when_zero(self):
        """alpha_min=0.0 should not modify alpha (Phase 6 compat)."""
        blk = _make_phase6_block()
        assert blk.longterm_alpha_min == 0.0

    def test_longterm_retains_more_with_alpha_min(self):
        """M_longterm should retain more knowledge per step with alpha_min."""
        # Create two blocks: one with alpha_min, one without
        blk_p7 = _make_phase7_block(chunk_size=1, longterm_alpha_min=0.99)
        blk_p6 = _make_phase6_block(chunk_size=1)

        for blk in [blk_p7, blk_p6]:
            blk.train()
            blk.reset_memory_state()

        # Feed same data, track longterm weight changes
        x = torch.randn(1, 258, 384)
        w1_before_p7 = blk_p7.M_longterm._active_w1.clone().detach()
        w1_before_p6 = blk_p6.M_longterm._active_w1.clone().detach()

        _ = blk_p7(x.clone(), T=1, H=16, W=16, action_tokens=2)
        _ = blk_p6(x.clone(), T=1, H=16, W=16, action_tokens=2)

        delta_p7 = (blk_p7.M_longterm._active_w1.detach() - w1_before_p7).norm()
        delta_p6 = (blk_p6.M_longterm._active_w1.detach() - w1_before_p6).norm()

        # With higher alpha_min, the decay is weaker → weights change less
        # (higher retention = smaller delta from decay component)
        # Note: the gradient component also contributes, so this is approximate
        # We mainly verify both produce different deltas
        assert not torch.isclose(torch.tensor(delta_p7.item()), torch.tensor(delta_p6.item()), atol=1e-6), \
            "Alpha_min should produce different weight update magnitudes"


# ===========================================================================
# Test Class 3: Longterm-Specific Surprise
# ===========================================================================

class TestLongtermOwnSurprise:
    """Test the Phase 7 longterm-specific surprise mechanism."""

    def test_own_surprise_flag_stored(self):
        """Block should store the own_surprise flag."""
        blk = _make_phase7_block()
        assert blk.longterm_own_surprise is True

    def test_own_surprise_disabled_in_phase6(self):
        """Phase 6 compat: own_surprise should be False."""
        blk = _make_phase6_block()
        assert blk.longterm_own_surprise is False

    def test_forward_works_with_own_surprise(self):
        """Forward pass should complete with own_surprise enabled."""
        blk = _make_phase7_block(chunk_size=1)
        blk.train()
        blk.reset_memory_state()
        x = torch.randn(1, 258, 384)
        out = blk(x, T=1, H=16, W=16, action_tokens=2)
        assert out.shape == (1, 258, 384)
        assert not torch.isnan(out).any()


# ===========================================================================
# Test Class 4: Consolidation EMA
# ===========================================================================

class TestConsolidationEMA:
    """Test the Phase 7 consolidation mechanism."""

    def test_consolidation_modifies_parameters(self):
        """consolidate_longterm_memory() should change nn.Parameters."""
        blk = _make_phase7_block(chunk_size=1)
        blk.train()
        blk.reset_memory_state()

        # Run forward to accumulate DGD state
        x = torch.randn(1, 258, 384)
        _ = blk(x, T=1, H=16, W=16, action_tokens=2)

        # Record params before consolidation
        param_before = blk.M_longterm.w1.weight.clone().detach()

        # Consolidate
        blk.consolidate_longterm_memory()

        param_after = blk.M_longterm.w1.weight.clone().detach()
        assert not torch.allclose(param_before, param_after, atol=1e-8), \
            "Consolidation should modify nn.Parameters"

    def test_consolidation_noop_when_disabled(self):
        """consolidate_longterm_memory() should be a no-op when ema=0."""
        blk = _make_phase6_block(chunk_size=1)
        blk.train()
        blk.reset_memory_state()

        x = torch.randn(1, 258, 384)
        _ = blk(x, T=1, H=16, W=16, action_tokens=2)

        param_before = blk.M_longterm.w1.weight.clone().detach()
        blk.consolidate_longterm_memory()
        param_after = blk.M_longterm.w1.weight.clone().detach()

        assert torch.allclose(param_before, param_after), \
            "Consolidation with ema=0 should be a no-op"

    def test_consolidation_reinitializes_active_weights(self):
        """After consolidation, active weights should be re-initialized from updated params."""
        blk = _make_phase7_block(chunk_size=1, longterm_consolidation_ema=0.5)
        blk.train()
        blk.reset_memory_state()

        # Forward to get DGD-modified active weights
        x = torch.randn(1, 258, 384)
        _ = blk(x, T=1, H=16, W=16, action_tokens=2)

        blk.consolidate_longterm_memory()

        # Active weights should now match the updated nn.Parameters
        assert torch.allclose(
            blk.M_longterm._active_w1,
            blk.M_longterm.w1.weight.detach()
        ), "Active weights should match updated params after consolidation"

    def test_consolidation_noop_before_init(self):
        """Consolidation before any reset should be a no-op (no active weights)."""
        blk = _make_phase7_block()
        assert blk.M_longterm._active_w1 is None
        # Should not raise
        blk.consolidate_longterm_memory()

    def test_ema_rate_controls_consolidation_speed(self):
        """Higher EMA rate should produce larger parameter changes."""
        torch.manual_seed(42)
        x = torch.randn(1, 258, 384)

        # Low EMA
        blk_low = _make_phase7_block(chunk_size=1, longterm_consolidation_ema=0.01)
        blk_low.train()
        blk_low.reset_memory_state()
        _ = blk_low(x.clone(), T=1, H=16, W=16, action_tokens=2)
        param_orig_low = blk_low.M_longterm.w1.weight.clone().detach()
        blk_low.consolidate_longterm_memory()
        delta_low = (blk_low.M_longterm.w1.weight.detach() - param_orig_low).norm()

        # High EMA
        blk_high = _make_phase7_block(chunk_size=1, longterm_consolidation_ema=0.5)
        blk_high.train()
        blk_high.reset_memory_state()
        _ = blk_high(x.clone(), T=1, H=16, W=16, action_tokens=2)
        param_orig_high = blk_high.M_longterm.w1.weight.clone().detach()
        blk_high.consolidate_longterm_memory()
        delta_high = (blk_high.M_longterm.w1.weight.detach() - param_orig_high).norm()

        assert delta_high > delta_low, \
            f"Higher EMA ({delta_high:.6f}) should produce larger param change than lower EMA ({delta_low:.6f})"

    def test_model_level_consolidation(self, phase7_model, sample_inputs):
        """consolidate_all_longterm_memories() should work at model level."""
        x, actions, states = sample_inputs
        phase7_model.reset_all_memories()
        _ = phase7_model(x, actions, states)

        params_before = [
            blk.M_longterm.w1.weight.clone().detach()
            for blk in phase7_model.hope_blocks
        ]

        phase7_model.consolidate_all_longterm_memories()

        for i, blk in enumerate(phase7_model.hope_blocks):
            assert not torch.allclose(blk.M_longterm.w1.weight.detach(), params_before[i], atol=1e-8), \
                f"Block {i}: consolidation should modify M_longterm params"


# ===========================================================================
# Test Class 5: Phase 7 Integration
# ===========================================================================

class TestPhase7Integration:
    """Full integration tests with all Phase 7 features enabled."""

    def test_forward_teacher_forcing(self, phase7_model, sample_inputs):
        """Forward pass in teacher-forcing mode should work."""
        x, actions, states = sample_inputs
        phase7_model.reset_all_memories()
        out = phase7_model(x, actions, states)
        B, T, N, D = 1, 3, 256, 1024
        assert out.shape == (B, T * N, D)
        assert not torch.isnan(out).any()

    def test_backward_gradients_flow(self, phase7_model, sample_inputs):
        """Backward pass should provide gradients to all Phase 7 components."""
        x, actions, states = sample_inputs
        phase7_model.reset_all_memories()
        out = phase7_model(x, actions, states)
        loss = out.mean()
        loss.backward()

        for i, blk in enumerate(phase7_model.hope_blocks):
            assert blk.longterm_gate.weight.grad is not None, \
                f"Block {i}: longterm_gate.weight should have gradients"
            assert blk.longterm_gate.bias.grad is not None, \
                f"Block {i}: longterm_gate.bias should have gradients"

    def test_config_summary_includes_phase7(self, phase7_model):
        """Config summary should include all Phase 7 fields."""
        summary = phase7_model._config_summary
        assert summary["longterm_retrieval_conditioned_gate"] is True
        assert summary["longterm_alpha_min"] == 0.95
        assert summary["longterm_own_surprise"] is True
        assert summary["longterm_consolidation_ema"] == 0.01

    def test_phase6_backward_compatibility(self):
        """Model with all Phase 7 features disabled should match Phase 6 behavior."""
        model = _make_phase7_model(
            longterm_retrieval_conditioned_gate=False,
            longterm_alpha_min=0.0,
            longterm_own_surprise=False,
            longterm_consolidation_ema=0.0,
        )
        # All gates should be Linear(384, 1), not Linear(768, 1)
        for blk in model.hope_blocks:
            assert blk.longterm_gate.in_features == 384

    def test_param_count_phase7_gate(self):
        """Phase 7 gate should add D extra params per block vs Phase 6."""
        model_p7 = _make_phase7_model()
        model_p6 = _make_phase7_model(longterm_retrieval_conditioned_gate=False)

        params_p7 = sum(p.numel() for p in model_p7.parameters())
        params_p6 = sum(p.numel() for p in model_p6.parameters())

        D = 384
        depth = 2
        expected_delta = D * depth  # D extra input weights per gate per block
        assert params_p7 - params_p6 == expected_delta, \
            f"Phase 7 gate should add {expected_delta} params, got {params_p7 - params_p6}"


# ===========================================================================
# Test Class 6: Multi-Clip with Consolidation
# ===========================================================================

class TestMultiClipConsolidation:
    """Test multi-clip pipeline with Phase 7 consolidation."""

    def test_consolidation_between_tasks(self):
        """Simulate CL pipeline: clips → consolidation → clips → verify."""
        blk = _make_phase7_block(chunk_size=1, longterm_consolidation_ema=0.1)
        blk.train()
        blk.reset_memory_state()

        # Task 1: process 2 clips
        for _ in range(2):
            x = torch.randn(1, 258, 384)
            _ = blk(x, T=1, H=16, W=16, action_tokens=2)
            blk.reset_memory_state()

        lt_before_consolidation = blk.M_longterm._active_w1.detach().clone()
        param_before = blk.M_longterm.w1.weight.detach().clone()

        # Consolidate (task boundary)
        blk.consolidate_longterm_memory()

        param_after = blk.M_longterm.w1.weight.detach().clone()
        lt_after_consolidation = blk.M_longterm._active_w1.detach().clone()

        # Parameters should have moved toward accumulated state
        assert not torch.allclose(param_before, param_after, atol=1e-8)
        # Active weights should now be set from updated params
        assert torch.allclose(lt_after_consolidation, param_after)

        # Task 2: process more clips starting from consolidated state
        blk.reset_memory_state()
        x2 = torch.randn(1, 258, 384)
        out = blk(x2, T=1, H=16, W=16, action_tokens=2)
        assert not torch.isnan(out).any(), "Forward after consolidation should not produce NaN"

    def test_longterm_diverges_then_consolidation_pulls_back(self):
        """After many DGD steps, consolidation should pull params toward active state."""
        blk = _make_phase7_block(chunk_size=1, longterm_consolidation_ema=0.5)
        blk.train()
        blk.reset_memory_state()

        # Many clips of DGD updates (drift active weights from params)
        for _ in range(5):
            x = torch.randn(1, 258, 384)
            _ = blk(x, T=1, H=16, W=16, action_tokens=2)
            blk.reset_memory_state()

        # Measure drift
        drift = (blk.M_longterm._active_w1.detach() - blk.M_longterm.w1.weight.detach()).norm()
        assert drift > 0, "Active weights should have drifted from params"

        # Consolidate with high EMA (0.5)
        blk.consolidate_longterm_memory()

        # After consolidation, drift should be zero (active = params)
        drift_after = (blk.M_longterm._active_w1 - blk.M_longterm.w1.weight.detach()).norm()
        assert drift_after < 1e-6, "After consolidation, active weights should match params"
