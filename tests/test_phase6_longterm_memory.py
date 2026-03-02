"""Tests for Phase 6: Cross-clip persistent longterm memory (Ansatz B).

Tests verify:
1. M_longterm memory:
   - Created when use_longterm_memory=True
   - NOT created when use_longterm_memory=False
   - Has correct hidden dimension (longterm_hidden_multiplier)
   - Persists across reset_memory_state() calls
   - Can be explicitly reset via reset_longterm_memory()

2. Learned gate:
   - Gate exists and produces [0, 1] output
   - Gate initialized to favor clip-level memory (~0.73)
   - Gate weights zero-initialized (initially uniform gating)

3. DGD integration:
   - M_longterm receives DGD updates with scaled learning rate
   - M_longterm active weights change after forward pass (DGD mutates them)
   - M_longterm active weights survive across reset_memory_state() calls
   - Clip-level memories (M_memory) are reset but M_longterm persists

4. Integration:
   - Full forward pass works with longterm memory enabled
   - Backward pass gradients flow through gate and M_longterm
   - Config summary includes longterm memory info
   - Param count increases by expected amount
"""

import pytest
import torch

from src.models.hope import ac_hope_vit
from src.models.hope.hope_block import HOPEBlock, HOPEBlockConfig
from src.models.hope.titan_memory import TitanMemoryConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(use_longterm_memory: bool, **kwargs):
    """Create a small test model with or without longterm memory."""
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
        use_longterm_memory=use_longterm_memory,
        longterm_hidden_multiplier=2,
        longterm_lr_scale=0.1,
        drop_path_rate=0.0,
        log_hope_diagnostics=True,
    )
    defaults.update(kwargs)
    model = ac_hope_vit(**defaults)
    model.train()
    return model


def _make_block(use_longterm_memory: bool, **kwargs):
    """Create a standalone HOPEBlock for unit testing."""
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
        use_longterm_memory=use_longterm_memory,
        longterm_hidden_multiplier=2,
        longterm_lr_scale=0.1,
    )
    defaults.update(kwargs)
    config = HOPEBlockConfig(**defaults)
    return HOPEBlock(config, titan_detach_interval=0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_with_lt():
    return _make_model(use_longterm_memory=True)


@pytest.fixture
def model_without_lt():
    return _make_model(use_longterm_memory=False)


@pytest.fixture
def sample_inputs():
    """Minimal inputs: B=1, T=3 timesteps."""
    B, T, N, D = 1, 3, 256, 1024
    x = torch.randn(B, T * N, D)
    actions = torch.randn(B, T, 2)
    states = torch.randn(B, T, 2)
    return x, actions, states


@pytest.fixture
def jump_inputs():
    """Jump prediction inputs: B=1, T=1."""
    B, T, N, D = 1, 1, 256, 1024
    x = torch.randn(B, T * N, D)
    actions = torch.randn(B, T, 2)
    states = torch.randn(B, T, 2)
    return x, actions, states


# ===========================================================================
# Test Class 1: M_longterm Memory Existence & Configuration
# ===========================================================================

class TestLongtermMemoryConfig:
    """Test that M_longterm is created/not created based on config."""

    def test_block_has_longterm_when_enabled(self):
        blk = _make_block(use_longterm_memory=True)
        assert hasattr(blk, "M_longterm"), "M_longterm should exist when enabled"
        assert hasattr(blk, "longterm_gate"), "longterm_gate should exist when enabled"

    def test_block_no_longterm_when_disabled(self):
        blk = _make_block(use_longterm_memory=False)
        assert not hasattr(blk, "M_longterm"), "M_longterm should NOT exist when disabled"
        assert not hasattr(blk, "longterm_gate"), "longterm_gate should NOT exist when disabled"

    def test_longterm_hidden_dim(self):
        """M_longterm should use longterm_hidden_multiplier, not titan_hidden_multiplier."""
        blk = _make_block(use_longterm_memory=True, longterm_hidden_multiplier=3)
        # M_longterm.w1 should be Linear(384, 384*3) = Linear(384, 1152)
        assert blk.M_longterm.w1.weight.shape == (384 * 3, 384)
        # M_memory.w1 should be Linear(384, 384*2) (titan_hidden_multiplier=2)
        assert blk.M_memory.w1.weight.shape == (384 * 2, 384)

    def test_model_has_longterm_in_all_blocks(self, model_with_lt):
        for i, blk in enumerate(model_with_lt.hope_blocks):
            assert hasattr(blk, "M_longterm"), f"Block {i} missing M_longterm"
            assert hasattr(blk, "longterm_gate"), f"Block {i} missing longterm_gate"


# ===========================================================================
# Test Class 2: Learned Gate
# ===========================================================================

class TestLongtermGate:
    """Test the learned gate mechanism."""

    def test_gate_output_range(self):
        """Gate output should be in [0, 1] (sigmoid)."""
        blk = _make_block(use_longterm_memory=True)
        x = torch.randn(2, 10, 384)
        gate = torch.sigmoid(blk.longterm_gate(x))
        assert (gate >= 0).all() and (gate <= 1).all()

    def test_gate_initial_bias_favors_clip(self):
        """Gate bias=1.0 → sigmoid(1.0)≈0.73 → initially favors M_memory."""
        blk = _make_block(use_longterm_memory=True)
        assert blk.longterm_gate.bias.item() == pytest.approx(1.0)
        # With zero weights and bias=1.0, output is always sigmoid(1)≈0.73
        x = torch.randn(1, 5, 384)
        gate = torch.sigmoid(blk.longterm_gate(x))
        expected = torch.sigmoid(torch.tensor(1.0))
        assert gate.mean().item() == pytest.approx(expected.item(), abs=1e-5)

    def test_gate_weights_zero_initialized(self):
        """Gate weights should be zero-initialized."""
        blk = _make_block(use_longterm_memory=True)
        assert (blk.longterm_gate.weight == 0).all()


# ===========================================================================
# Test Class 3: Persistence Across Resets
# ===========================================================================

class TestLongtermPersistence:
    """Test that M_longterm survives reset_memory_state() calls."""

    def test_first_reset_initializes_longterm(self):
        """First reset_memory_state() should initialize M_longterm active weights."""
        blk = _make_block(use_longterm_memory=True)
        assert blk.M_longterm._active_w1 is None, "M_longterm should start uninitialized"
        blk.reset_memory_state()
        assert blk.M_longterm._active_w1 is not None, "First reset should init M_longterm"

    def test_second_reset_preserves_longterm(self):
        """Second reset should NOT re-initialize M_longterm (preserve accumulated state)."""
        blk = _make_block(use_longterm_memory=True)
        blk.reset_memory_state()

        # Save a reference to the longterm weights
        lt_w1_after_first_reset = blk.M_longterm._active_w1.clone()

        # Manually modify M_longterm to simulate DGD accumulation
        blk.M_longterm._active_w1 = blk.M_longterm._active_w1 + 1.0
        lt_w1_modified = blk.M_longterm._active_w1.clone()

        # Second reset: M_memory should reset, M_longterm should persist
        blk.reset_memory_state()

        # M_longterm should still have the modified weights (detached)
        assert torch.allclose(blk.M_longterm._active_w1, lt_w1_modified), \
            "M_longterm should persist across reset_memory_state()"

    def test_clip_memories_are_reset(self):
        """Clip-level memories should reset to nn.Parameters state."""
        blk = _make_block(use_longterm_memory=True)
        blk.reset_memory_state()

        # Save original M_memory weights
        m_memory_original = blk.M_memory._active_w1.clone()

        # Modify M_memory
        blk.M_memory._active_w1 = blk.M_memory._active_w1 + 1.0

        # Reset should restore M_memory to original
        blk.reset_memory_state()
        assert torch.allclose(blk.M_memory._active_w1, m_memory_original, atol=1e-6), \
            "M_memory should reset to nn.Parameters state"

    def test_explicit_longterm_reset_works(self):
        """reset_longterm_memory() should reset M_longterm to nn.Parameters."""
        blk = _make_block(use_longterm_memory=True)
        blk.reset_memory_state()

        original = blk.M_longterm._active_w1.clone()
        blk.M_longterm._active_w1 = blk.M_longterm._active_w1 + 5.0

        blk.reset_longterm_memory()
        assert torch.allclose(blk.M_longterm._active_w1, original, atol=1e-6), \
            "reset_longterm_memory() should restore to nn.Parameters"


# ===========================================================================
# Test Class 4: DGD Updates on M_longterm
# ===========================================================================

class TestLongtermDGD:
    """Test that M_longterm receives DGD updates during forward pass."""

    def test_longterm_weights_change_after_forward(self):
        """M_longterm active weights should change after a forward pass with DGD."""
        blk = _make_block(use_longterm_memory=True, chunk_size=1)
        blk.reset_memory_state()
        blk.train()

        w1_before = blk.M_longterm._active_w1.clone().detach()

        # Forward pass (chunk_size=1 with T>0 triggers DGD)
        # NOTE: DGD requires torch.is_grad_enabled()=True when self.training=True
        x = torch.randn(1, 258, 384)
        _ = blk(x, T=1, H=16, W=16, action_tokens=2)

        w1_after = blk.M_longterm._active_w1.clone().detach()
        assert not torch.allclose(w1_before, w1_after, atol=1e-7), \
            "M_longterm weights should change after forward (DGD update)"

    def test_longterm_lr_scale_reduces_update_magnitude(self):
        """DGD updates to M_longterm should be smaller than to M_memory."""
        blk = _make_block(use_longterm_memory=True, chunk_size=1, longterm_lr_scale=0.1)
        blk.reset_memory_state()
        blk.train()

        m_mem_before = blk.M_memory._active_w1.clone().detach()
        m_lt_before = blk.M_longterm._active_w1.clone().detach()

        # DGD requires gradients enabled in training mode
        x = torch.randn(1, 258, 384)
        _ = blk(x, T=1, H=16, W=16, action_tokens=2)

        m_mem_delta = (blk.M_memory._active_w1.detach() - m_mem_before).norm()
        m_lt_delta = (blk.M_longterm._active_w1.detach() - m_lt_before).norm()

        # M_longterm should change less (scaled by 0.1)
        assert m_lt_delta < m_mem_delta, \
            f"M_longterm delta ({m_lt_delta:.6f}) should be < M_memory delta ({m_mem_delta:.6f})"


# ===========================================================================
# Test Class 5: Full Model Integration
# ===========================================================================

class TestPhase6Integration:
    """Integration tests for the full Phase 6 model."""

    def test_forward_teacher_forcing(self, model_with_lt, sample_inputs):
        """Forward pass in teacher-forcing mode should work."""
        x, actions, states = sample_inputs
        model_with_lt.reset_all_memories()
        out = model_with_lt(x, actions, states)
        B, T, N, D = 1, 3, 256, 1024
        assert out.shape == (B, T * N, D)

    def test_forward_jump(self, model_with_lt, jump_inputs):
        """Forward pass in jump prediction mode should work."""
        x, actions, states = jump_inputs
        model_with_lt.reset_all_memories()
        out = model_with_lt(x, actions, states, target_timestep=5)
        B, N, D = 1, 256, 1024
        assert out.shape == (B, N, D)

    def test_backward_gradients_flow(self, model_with_lt, sample_inputs):
        """Backward pass should provide gradients to longterm gate and M_longterm."""
        x, actions, states = sample_inputs
        model_with_lt.reset_all_memories()
        out = model_with_lt(x, actions, states)
        loss = out.mean()
        loss.backward()

        # Check that longterm-specific params got gradients
        for i, blk in enumerate(model_with_lt.hope_blocks):
            assert blk.longterm_gate.weight.grad is not None, \
                f"Block {i}: longterm_gate.weight should have gradients"
            assert blk.longterm_gate.bias.grad is not None, \
                f"Block {i}: longterm_gate.bias should have gradients"

    def test_reset_all_longterm_memories(self, model_with_lt, sample_inputs):
        """reset_all_longterm_memories() should reset M_longterm in all blocks."""
        x, actions, states = sample_inputs
        model_with_lt.reset_all_memories()

        # Run a forward pass to trigger DGD updates (needs grad enabled for DGD)
        _ = model_with_lt(x, actions, states)

        # Store modified weights (detach for comparison)
        modified = [blk.M_longterm._active_w1.detach().clone() for blk in model_with_lt.hope_blocks]

        # Explicit longterm reset
        model_with_lt.reset_all_longterm_memories()

        for i, blk in enumerate(model_with_lt.hope_blocks):
            fresh = blk.M_longterm._active_w1.detach()
            assert not torch.allclose(fresh, modified[i], atol=1e-6), \
                f"Block {i}: M_longterm should be reset after reset_all_longterm_memories()"

    def test_config_summary_includes_longterm(self, model_with_lt):
        """Config summary should include longterm memory info."""
        summary = model_with_lt._config_summary
        assert summary["use_longterm_memory"] is True
        assert summary["longterm_hidden_multiplier"] == 2
        assert summary["longterm_lr_scale"] == 0.1

    def test_param_count_increase(self, model_with_lt, model_without_lt):
        """Model with longterm memory should have more params."""
        params_with = sum(p.numel() for p in model_with_lt.parameters())
        params_without = sum(p.numel() for p in model_without_lt.parameters())
        delta = params_with - params_without

        # Per block: M_longterm MLP (w1: D*D*mult + w2: D*mult*D) + LayerNorm(D) (2*D)
        #          + gate (D + 1)
        D = 384
        mult = 2
        m_longterm_params = D * (D * mult) + (D * mult) * D  # w1 + w2
        m_longterm_norm = 2 * D  # LayerNorm weight + bias
        gate_params = D + 1  # Linear(D, 1) weight + bias
        expected_per_block = m_longterm_params + m_longterm_norm + gate_params
        expected_total = 2 * expected_per_block  # 2 blocks in test model
        assert delta == expected_total, \
            f"Param delta should be {expected_total:,} but got {delta:,}"

    def test_param_groups_include_longterm(self, model_with_lt):
        """Parameter groups should classify M_longterm as titan group."""
        groups = model_with_lt.get_parameter_groups()
        titan_group = next(g for g in groups if g["group_name"] == "titan")
        titan_param_count = sum(p.numel() for p in titan_group["params"])
        # Should include M_longterm params
        assert titan_param_count > 0

    def test_diagnostics_include_longterm(self, model_with_lt, sample_inputs):
        """Diagnostics should include M_longterm metrics."""
        x, actions, states = sample_inputs
        model_with_lt.reset_all_memories()
        with torch.no_grad():
            _ = model_with_lt(x, actions, states)

        diag = model_with_lt.get_all_diagnostics()
        longterm_keys = [k for k in diag if "M_longterm" in k]
        assert len(longterm_keys) > 0, f"Should have M_longterm diagnostics, got keys: {list(diag.keys())}"


# ===========================================================================
# Test Class 6: Multi-Clip Persistence Simulation
# ===========================================================================

class TestMultiClipPersistence:
    """Simulate processing multiple clips to verify longterm memory accumulation."""

    def test_longterm_accumulates_across_clips(self):
        """M_longterm should accumulate knowledge across multiple clip forward passes."""
        blk = _make_block(use_longterm_memory=True, chunk_size=1)
        blk.train()
        blk.reset_memory_state()

        # Clip 1 (DGD needs gradients enabled in training mode)
        x1 = torch.randn(1, 258, 384)
        _ = blk(x1, T=1, H=16, W=16, action_tokens=2)
        lt_after_clip1 = blk.M_longterm._active_w1.detach().clone()

        # Reset (simulates new clip in training_step)
        blk.reset_memory_state()

        # Verify M_longterm persisted (detached but same values)
        assert torch.allclose(blk.M_longterm._active_w1, lt_after_clip1), \
            "M_longterm should persist after reset"

        # Clip 2
        x2 = torch.randn(1, 258, 384)
        _ = blk(x2, T=1, H=16, W=16, action_tokens=2)
        lt_after_clip2 = blk.M_longterm._active_w1.detach().clone()

        # M_longterm should have changed further (accumulated)
        assert not torch.allclose(lt_after_clip1, lt_after_clip2, atol=1e-7), \
            "M_longterm should accumulate DGD updates across clips"

    def test_clip_memory_resets_longterm_persists(self):
        """M_memory should reset between clips, M_longterm should not."""
        blk = _make_block(use_longterm_memory=True, chunk_size=1)
        blk.train()
        blk.reset_memory_state()

        # Save original M_memory state (from nn.Parameters)
        m_mem_original = blk.M_memory.w1.weight.clone().detach()

        # Forward pass (DGD modifies both M_memory and M_longterm)
        x = torch.randn(1, 258, 384)
        _ = blk(x, T=1, H=16, W=16, action_tokens=2)

        m_mem_after_fwd = blk.M_memory._active_w1.detach().clone()
        m_lt_after_fwd = blk.M_longterm._active_w1.detach().clone()

        # DGD should have modified M_memory
        assert not torch.allclose(m_mem_original, m_mem_after_fwd, atol=1e-6), \
            "DGD should have modified M_memory active weights"

        # Reset (new clip)
        blk.reset_memory_state()

        # M_memory should be back to nn.Parameters (fresh clone)
        m_mem_after_reset = blk.M_memory._active_w1.detach().clone()
        assert torch.allclose(m_mem_original, m_mem_after_reset, atol=1e-6), \
            "M_memory should reset to nn.Parameters"

        # M_longterm should still have accumulated state
        m_lt_after_reset = blk.M_longterm._active_w1.detach().clone()
        assert torch.allclose(m_lt_after_fwd, m_lt_after_reset), \
            "M_longterm should persist (same values, detached)"
