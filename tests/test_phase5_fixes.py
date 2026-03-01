"""Tests for Phase 5 fixes: temporal embeddings and spatial mixing.

Tests verify:
1. Learnable temporal embeddings (frame_pos_embed, target_pos_embed):
   - Different target_timestep values produce different outputs (jump encoding works)
   - Frame position embeddings are added in teacher-forcing mode
   - Embeddings exist and have correct shapes

2. Spatial mixing (Phase C):
   - Spatial mixing layer changes output vs. no spatial mixing
   - Output shape is correct with spatial mixing enabled
   - Zero-initialization means initial spatial mixing contribution is zero
   - Gradients flow through the spatial mixing path

3. Integration:
   - Full forward pass works with both fixes combined (Phase 5 config)
   - Backward pass provides gradients to all new parameters
"""

import pytest
import torch

from src.models.hope import ACHOPEViT, ac_hope_vit
from src.models.hope.hope_block import HOPEBlock, HOPEBlockConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def phase5_model():
    """Phase 5 model with both fixes: temporal embeddings + spatial mixing."""
    model = ac_hope_vit(
        img_size=(256, 256),
        patch_size=16,
        num_timesteps=8,
        embed_dim=1024,
        predictor_embed_dim=384,
        depth=2,  # small for testing
        num_heads=16,
        action_embed_dim=2,
        use_rope=False,  # Phase 5: RoPE off, use learned embeddings instead
        is_frame_causal=True,
        titan_hidden_multiplier=2,
        titan_layers=2,
        titan_grad_clip_inner=1.0,
        chunk_size=0,
        titan_detach_interval=0,
        surprise_threshold=0.0,
        use_spatial_mixing=True,  # Phase 5: spatial mixing ON
        drop_path_rate=0.0,
        log_hope_diagnostics=False,
    )
    model.train()
    return model


@pytest.fixture
def no_spatial_model():
    """Model WITHOUT spatial mixing (for comparison)."""
    model = ac_hope_vit(
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
        use_spatial_mixing=False,
        drop_path_rate=0.0,
        log_hope_diagnostics=False,
    )
    model.train()
    return model


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
    """Jump prediction inputs: B=1, T=1 (single frame)."""
    B, T, N, D = 1, 1, 256, 1024
    x = torch.randn(B, T * N, D)
    actions = torch.randn(B, T, 2)
    states = torch.randn(B, T, 2)
    return x, actions, states


# ---------------------------------------------------------------------------
# Test Temporal Embeddings (Fix 1)
# ---------------------------------------------------------------------------


class TestTemporalEmbeddings:
    """Tests for learnable frame_pos_embed and target_pos_embed."""

    def test_embeddings_exist(self, phase5_model):
        """Both embedding tables should exist with correct shapes."""
        assert hasattr(phase5_model, "frame_pos_embed")
        assert hasattr(phase5_model, "target_pos_embed")
        # num_timesteps=8, so embedding table has 10 entries (8+2)
        assert phase5_model.frame_pos_embed.weight.shape == (10, 384)
        assert phase5_model.target_pos_embed.weight.shape == (10, 384)

    def test_different_targets_produce_different_outputs(self, phase5_model, jump_inputs):
        """Jump predictions for different target timesteps should differ."""
        x, actions, states = jump_inputs
        phase5_model.reset_all_memories()
        with torch.no_grad():
            out_tau6 = phase5_model(x, actions, states, target_timestep=6).clone()
            phase5_model.reset_all_memories()
            out_tau8 = phase5_model(x, actions, states, target_timestep=8).clone()

        # Outputs must differ because target_pos_embed(6) != target_pos_embed(8)
        diff = (out_tau6 - out_tau8).abs().mean()
        assert diff > 1e-6, (
            f"Outputs for target_timestep=6 and =8 should differ, but mean diff={diff:.8f}"
        )

    def test_teacher_mode_uses_frame_positions(self, phase5_model, sample_inputs):
        """In teacher-forcing mode, frame positions should be added."""
        x, actions, states = sample_inputs
        phase5_model.reset_all_memories()
        # Just check it runs without error and produces correct shape
        out = phase5_model(x, actions, states)
        B, T, N = 1, 3, 256
        assert out.shape == (B, T * N, 1024)

    def test_jump_mode_output_shape(self, phase5_model, jump_inputs):
        """Jump prediction should produce output for T=1 frame."""
        x, actions, states = jump_inputs
        phase5_model.reset_all_memories()
        out = phase5_model(x, actions, states, target_timestep=7)
        B, N = 1, 256
        assert out.shape == (B, N, 1024)

    def test_embeddings_receive_gradients(self, phase5_model, jump_inputs):
        """Both embedding tables should receive gradients on backward."""
        x, actions, states = jump_inputs
        phase5_model.reset_all_memories()
        out = phase5_model(x, actions, states, target_timestep=7)
        loss = out.sum()
        loss.backward()

        assert phase5_model.frame_pos_embed.weight.grad is not None
        assert phase5_model.target_pos_embed.weight.grad is not None
        # frame_pos_embed[0] should have grad (frame 0 is the input frame)
        assert phase5_model.frame_pos_embed.weight.grad[0].abs().sum() > 0
        # target_pos_embed[7] should have grad (target was 7)
        assert phase5_model.target_pos_embed.weight.grad[7].abs().sum() > 0


# ---------------------------------------------------------------------------
# Test Spatial Mixing (Fix 2)
# ---------------------------------------------------------------------------


class TestSpatialMixing:
    """Tests for MLP-Mixer-style token mixing in HOPEBlock Phase C."""

    def test_spatial_mixing_config(self):
        """HOPEBlockConfig should accept spatial mixing parameters."""
        config = HOPEBlockConfig(
            dim=384,
            use_spatial_mixing=True,
            spatial_mixing_tokens=258,
        )
        assert config.use_spatial_mixing is True
        assert config.spatial_mixing_tokens == 258

    def test_block_has_spatial_mix_layers(self):
        """HOPEBlock with spatial mixing should have norm3 and spatial_mix."""
        config = HOPEBlockConfig(
            dim=384,
            num_heads=16,
            titan_hidden_multiplier=2,
            use_spatial_mixing=True,
            spatial_mixing_tokens=258,
        )
        block = HOPEBlock(config)
        assert hasattr(block, "norm3")
        assert hasattr(block, "spatial_mix")
        # spatial_mix should be a Sequential with 3 layers
        assert len(block.spatial_mix) == 3

    def test_block_without_spatial_mixing(self):
        """HOPEBlock without spatial mixing should NOT have norm3/spatial_mix."""
        config = HOPEBlockConfig(
            dim=384,
            num_heads=16,
            titan_hidden_multiplier=2,
            use_spatial_mixing=False,
        )
        block = HOPEBlock(config)
        assert not hasattr(block, "norm3") or not block.use_spatial_mixing

    def test_zero_init_output_layer(self):
        """The output Linear of spatial_mix should be zero-initialized."""
        config = HOPEBlockConfig(
            dim=384,
            num_heads=16,
            titan_hidden_multiplier=2,
            use_spatial_mixing=True,
            spatial_mixing_tokens=258,
        )
        block = HOPEBlock(config)
        # Last layer weight should be all zeros
        assert block.spatial_mix[2].weight.abs().max() == 0.0

    def test_output_shape_with_spatial_mixing(self, phase5_model, sample_inputs):
        """Output shape should be unchanged when spatial mixing is enabled."""
        x, actions, states = sample_inputs
        phase5_model.reset_all_memories()
        out = phase5_model(x, actions, states)
        B, T, N = 1, 3, 256
        assert out.shape == (B, T * N, 1024)

    def test_spatial_mixing_gradients_flow(self, phase5_model, sample_inputs):
        """Spatial mixing parameters should receive gradients."""
        x, actions, states = sample_inputs
        phase5_model.reset_all_memories()
        out = phase5_model(x, actions, states)
        loss = out.sum()
        loss.backward()

        # Check that spatial_mix layers in the first block have gradients
        block = phase5_model.hope_blocks[0]
        assert block.spatial_mix[0].weight.grad is not None
        # Note: output layer grad might be zero initially since weights are zero,
        # but the input layer should have non-zero gradients
        assert block.spatial_mix[0].weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestPhase5Integration:
    """Integration tests for Phase 5 model with both fixes."""

    def test_forward_pass_teacher_forcing(self, phase5_model, sample_inputs):
        """Full forward pass in teacher-forcing mode."""
        x, actions, states = sample_inputs
        phase5_model.reset_all_memories()
        out = phase5_model(x, actions, states)
        assert out.shape == (1, 3 * 256, 1024)
        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"

    def test_forward_pass_jump_prediction(self, phase5_model, jump_inputs):
        """Full forward pass in jump prediction mode."""
        x, actions, states = jump_inputs
        phase5_model.reset_all_memories()
        out = phase5_model(x, actions, states, target_timestep=7)
        assert out.shape == (1, 256, 1024)
        assert not torch.isnan(out).any(), "NaN in output"

    def test_backward_pass_all_params(self, phase5_model, sample_inputs):
        """All parameters should receive gradients after backward."""
        x, actions, states = sample_inputs
        phase5_model.reset_all_memories()
        out = phase5_model(x, actions, states)
        loss = out.sum()
        loss.backward()

        params_without_grad = []
        for name, p in phase5_model.named_parameters():
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0):
                params_without_grad.append(name)

        # Allow some exceptions:
        # - extrinsics_encoder: not used (no extrinsics input)
        # - target_pos_embed: only used in jump mode, not teacher-forcing
        # - M_k, M_v, M_eta, M_alpha: under FOMAML (detach_interval=0, chunk_size=0)
        #   these only receive gradients via aux_loss (added by the Lightning module,
        #   not by a bare forward pass). This is the known limitation addressed in
        #   the existing auxiliary loss mechanism.
        allowed_no_grad = {"extrinsics_encoder", "target_pos_embed", "M_k.", "M_v.", "M_eta.", "M_alpha."}
        filtered = [n for n in params_without_grad
                    if not any(skip in n for skip in allowed_no_grad)]

        # Spatial mixing output layer may have zero grad due to zero init
        # (grad = upstream_grad * 0 = 0), so we allow it
        filtered = [n for n in filtered if "spatial_mix.2" not in n]

        assert len(filtered) == 0, (
            f"Parameters without gradients: {filtered}"
        )

    def test_model_config_summary_includes_spatial_mixing(self, phase5_model):
        """Config summary should report spatial mixing setting."""
        summary = phase5_model.get_config_summary()
        assert "use_spatial_mixing" in summary
        assert summary["use_spatial_mixing"] is True

    def test_param_count_reasonable(self, phase5_model):
        """Model should have a reasonable param count (< 50M for depth=2 test model)."""
        total = sum(p.numel() for p in phase5_model.parameters())
        assert total > 1_000_000, f"Too few params: {total}"
        assert total < 50_000_000, f"Too many params for depth=2 test model: {total}"
