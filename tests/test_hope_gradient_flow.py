"""Unit tests for HOPE gradient flow — the core meta-learning mechanism.

These tests verify that the FOMAML-style meta-gradient chain is intact:
after a forward + backward pass, every nn.Parameter that should receive
gradients (via the outer-loop optimizer) actually has a non-zero .grad.

The critical invariant is:
    In compute_and_apply_update(), the update rule
        new_w = alpha * w_old - eta * grad
    must keep w_old differentiable w.r.t. the nn.Parameters (via .clone()
    in reset_active_weights), so that the outer-loop loss backpropagates
    through the DGD update chain.

If any test here fails, the Titan memories are NOT being trained.
"""

import pytest
import torch

from src.models.hope import ACHOPEViT, ac_hope_vit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_model():
    """A minimal 1-layer HOPE model for fast gradient checks."""
    model = ac_hope_vit(
        img_size=(256, 256),
        patch_size=16,
        num_timesteps=8,
        embed_dim=1024,
        predictor_embed_dim=384,
        depth=1,
        num_heads=16,
        action_embed_dim=2,
        use_rope=True,
        is_frame_causal=True,
        titan_hidden_multiplier=2,
        titan_layers=2,
        titan_grad_clip_inner=1.0,
        chunk_size=0,
        titan_detach_interval=0,
        surprise_threshold=0.0,
        drop_path_rate=0.0,
        log_hope_diagnostics=False,
    )
    model.train()
    return model


@pytest.fixture
def sample_inputs():
    """Minimal inputs: B=1, T=3 timesteps, N=256 patches, D=1024."""
    B, T, N, D = 1, 3, 256, 1024
    x = torch.randn(B, T * N, D)
    actions = torch.randn(B, T, 2)
    states = torch.randn(B, T, 2)
    return x, actions, states


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTitanMemoryGradientFlow:
    """Verify that Titan memory parameters receive gradients."""

    def test_memory_weight_params_have_grad(self, tiny_model, sample_inputs):
        """After forward+backward, M_memory (which produces the output)
        must have non-zero gradients. Auxiliary memories (M_k, M_v, M_eta,
        M_alpha) only get gradients when their updated state is used in a
        subsequent forward call (i.e., with chunking or multi-step).

        This is the PRIMARY regression test for the .detach() bug fix.
        """
        x, actions, states = sample_inputs
        tiny_model.reset_all_memories()

        out = tiny_model(x, actions, states)
        loss = out.mean()
        loss.backward()

        block = tiny_model.hope_blocks[0]

        # M_memory MUST have gradients — its output directly feeds the loss
        mem = block.M_memory
        for pname, p in mem.named_parameters():
            assert p.grad is not None, (
                f"M_memory.{pname}: grad is None — meta-gradient chain is broken!"
            )
            assert p.grad.norm().item() > 0.0, (
                f"M_memory.{pname}: grad norm is 0 — parameter is not being trained!"
            )

    def test_auxiliary_memory_grads_with_chunking(self):
        """With chunk_size > 0, M_eta and M_alpha should receive gradients
        because their outputs (η, α) are kept in the computation graph
        through the DGD update, and chunk 2 reads state modified in chunk 1.

        M_k and M_v do NOT get gradients in the current architecture because
        their outputs are only used in the inner loss (where weights are
        detached for first-order gradient computation). This is a known
        limitation — training M_k/M_v would require second-order gradients
        or an architectural change (e.g., using k,v in attention output).
        """
        # Need enough timesteps for at least 2 chunks
        B, T, N, D = 1, 4, 256, 1024
        x = torch.randn(B, T * N, D)
        actions = torch.randn(B, T, 2)
        states = torch.randn(B, T, 2)

        model = ac_hope_vit(
            img_size=(256, 256),
            patch_size=16,
            num_timesteps=8,
            embed_dim=1024,
            predictor_embed_dim=384,
            depth=1,
            num_heads=16,
            action_embed_dim=2,
            use_rope=True,
            is_frame_causal=True,
            titan_hidden_multiplier=2,
            titan_layers=2,
            titan_grad_clip_inner=1.0,
            chunk_size=2,  # 2 timesteps per chunk → 2 chunks
            titan_detach_interval=0,
            surprise_threshold=0.0,
            drop_path_rate=0.0,
            log_hope_diagnostics=False,
        )
        model.train()
        model.reset_all_memories()

        out = model(x, actions, states)
        loss = out.mean()
        loss.backward()

        block = model.hope_blocks[0]

        # M_memory, M_eta, M_alpha should all have gradients with chunking
        for name in ["M_memory", "M_eta", "M_alpha"]:
            mem = getattr(block, name)
            for pname, p in mem.named_parameters():
                assert p.grad is not None, (
                    f"{name}.{pname}: grad is None — "
                    "meta-gradient not flowing through chunk boundary!"
                )

        # M_k and M_v have NO gradients (known limitation of first-order)
        for name in ["M_k", "M_v"]:
            mem = getattr(block, name)
            for pname, p in mem.named_parameters():
                assert p.grad is None, (
                    f"{name}.{pname}: unexpectedly has gradients — "
                    "if this changed, update the test expectations!"
                )

    def test_self_generated_targets_differ_per_memory(self, tiny_model, sample_inputs):
        """Each memory's generate_self_target() should produce a different
        target because each memory has different weights."""
        x, actions, states = sample_inputs
        tiny_model.reset_all_memories()

        block = tiny_model.hope_blocks[0]
        # Use a random v as input
        v = torch.randn(1, 256, 384)

        targets = {}
        for name in ["M_k", "M_v", "M_eta", "M_alpha", "M_memory"]:
            mem = getattr(block, name)
            targets[name] = mem.generate_self_target(v)

        # Pairwise check: at least some targets should differ
        names = list(targets.keys())
        any_differ = False
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if not torch.allclose(targets[names[i]], targets[names[j]], atol=1e-5):
                    any_differ = True
                    break
            if any_differ:
                break

        assert any_differ, (
            "All 5 memories produce identical self-targets — they are not specializing!"
        )


class TestMetaLearningChain:
    """Verify the outer-loop meta-learning properties."""

    def test_reset_preserves_gradient_chain(self, tiny_model):
        """reset_active_weights() must use .clone() (not .detach()) so
        that active weights remain connected to nn.Parameters."""
        block = tiny_model.hope_blocks[0]
        for name in ["M_k", "M_v", "M_eta", "M_alpha", "M_memory"]:
            mem = getattr(block, name)
            mem.reset_active_weights()

            # After .clone(), active weights should require grad
            assert mem._active_w1 is not None, (
                f"{name}._active_w1 is None after reset!"
            )
            assert mem._active_w2 is not None, (
                f"{name}._active_w2 is None after reset!"
            )
            assert mem._active_w1.requires_grad, (
                f"{name}._active_w1 does not require grad after reset!"
            )
            assert mem._active_w2.requires_grad, (
                f"{name}._active_w2 does not require grad after reset!"
            )

    def test_outer_loss_decreases_titan_params(self, tiny_model, sample_inputs):
        """A single optimizer step should change Titan memory parameters,
        confirming that gradients flow and the optimizer can update them."""
        x, actions, states = sample_inputs

        # Snapshot initial params
        block = tiny_model.hope_blocks[0]
        mem = block.M_memory
        w_before = {n: p.data.clone() for n, p in mem.named_parameters()
                    if "w1" in n or "w2" in n}

        # Forward + backward + step
        tiny_model.reset_all_memories()
        out = tiny_model(x, actions, states)
        loss = out.mean()
        loss.backward()

        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        optimizer.step()

        # Check that at least some weights changed
        any_changed = False
        for n, p in mem.named_parameters():
            if n in w_before:
                if not torch.allclose(p.data, w_before[n], atol=1e-8):
                    any_changed = True
                    break

        assert any_changed, (
            "M_memory weights did not change after optimizer.step() — "
            "gradients are not flowing to Titan nn.Parameters!"
        )


class TestChunking:
    """Test that chunk_size correctly splits temporal processing."""

    def test_chunk_size_zero_equals_full(self, sample_inputs):
        """chunk_size=0 (full sequence) should produce the same output as
        chunk_size equal to the number of timesteps (single chunk = full seq)."""
        kwargs = dict(
            img_size=(256, 256),
            patch_size=16,
            num_timesteps=8,
            embed_dim=1024,
            predictor_embed_dim=384,
            depth=1,
            num_heads=16,
            action_embed_dim=2,
            use_rope=True,
            is_frame_causal=True,
            titan_hidden_multiplier=2,
            titan_layers=2,
            titan_grad_clip_inner=1.0,
            surprise_threshold=0.0,
            drop_path_rate=0.0,
            log_hope_diagnostics=False,
            titan_detach_interval=0,
        )

        x, actions, states = sample_inputs
        T = 3  # from sample_inputs (timesteps)

        # Model A: chunk_size=0 (full sequence)
        torch.manual_seed(42)
        model_a = ac_hope_vit(chunk_size=0, **kwargs)
        model_a.train()
        model_a.reset_all_memories()

        # Model B: chunk_size=T (one chunk covering all timesteps = full sequence)
        torch.manual_seed(42)
        model_b = ac_hope_vit(chunk_size=T, **kwargs)
        model_b.train()
        model_b.reset_all_memories()

        with torch.no_grad():
            out_a = model_a(x, actions, states)
            out_b = model_b(x, actions, states)

        assert torch.allclose(out_a, out_b, atol=1e-5), (
            f"chunk_size=0 and chunk_size={T} produce different outputs! "
            f"Max diff: {(out_a - out_b).abs().max():.6e}"
        )


class TestDetachInterval:
    """Verify that detach_interval bounds VRAM without breaking training."""

    def test_detach_interval_still_trains(self):
        """With detach_interval=1 (aggressive detach), M_memory should
        still receive gradients (just shorter chains)."""
        # Use enough timesteps for chunking so memories get multi-step updates
        B, T, N, D = 1, 4, 256, 1024
        x = torch.randn(B, T * N, D)
        actions = torch.randn(B, T, 2)
        states = torch.randn(B, T, 2)

        model = ac_hope_vit(
            img_size=(256, 256),
            patch_size=16,
            num_timesteps=8,
            embed_dim=1024,
            predictor_embed_dim=384,
            depth=1,
            num_heads=16,
            action_embed_dim=2,
            use_rope=True,
            is_frame_causal=True,
            titan_hidden_multiplier=2,
            titan_layers=2,
            titan_grad_clip_inner=1.0,
            chunk_size=0,
            titan_detach_interval=1,  # Detach every single step
            surprise_threshold=0.0,
            drop_path_rate=0.0,
            log_hope_diagnostics=False,
        )
        model.train()
        model.reset_all_memories()

        out = model(x, actions, states)
        loss = out.mean()
        loss.backward()

        # M_memory should still get gradients even with aggressive detach
        # because the output directly feeds the loss
        block = model.hope_blocks[0]
        mem = block.M_memory
        for pname, p in mem.named_parameters():
            assert p.grad is not None, (
                f"M_memory.{pname}: grad is None with detach_interval=1"
            )
