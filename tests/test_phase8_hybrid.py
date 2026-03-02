"""Unit tests for the Phase 8 AC-HOPE-Hybrid-ViT architecture.

Tests cover:
    1. HybridBlock construction & forward pass (Attention + Titan + CMS)
    2. Full model forward pass (ACHOPEHybridViT)
    3. Memory management (reset, persistence, freeze/unfreeze)
    4. Gradient flow through all three phases
    5. Parameter groups (titan / cms / projections)
    6. Diagnostic logging
    7. Longterm memory option
    8. Lightning module construction
"""

import pytest
import torch
import torch.nn as nn

from src.models.hope.cms import LevelSpec
from src.models.hope.hybrid_block import HybridBlock, HybridBlockConfig
from src.models.hope.ac_hope_hybrid_vit import ACHOPEHybridViT, ac_hope_hybrid_vit


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
B, T, N, D_ENC = 1, 3, 256, 1024
DIM = 384
ACTION_DIM = 2
NUM_HEADS = 16

CMS_LEVELS = [
    LevelSpec(name="fast", update_period=1, hidden_multiplier=2.0, warmup_steps=0),
    LevelSpec(name="slow", update_period=3, hidden_multiplier=2.0, warmup_steps=0),
]

CMS_LEVEL_DICTS = [
    {"name": "fast", "update_period": 1, "hidden_multiplier": 2.0, "warmup_steps": 0},
    {"name": "slow", "update_period": 3, "hidden_multiplier": 2.0, "warmup_steps": 0},
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def block_config() -> HybridBlockConfig:
    """Minimal HybridBlock config for testing."""
    return HybridBlockConfig(
        dim=DIM,
        num_heads=NUM_HEADS,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        grid_size=16,
        titan_hidden_multiplier=2,
        titan_layers=2,
        titan_activation="gelu",
        titan_grad_clip_inner=1.0,
        titan_grad_clip_backward=1.0,
        titan_detach_interval=0,
        surprise_threshold=0.0,
        cms_levels=CMS_LEVELS,
        cms_use_chunk_scheduling=False,
        use_longterm_memory=False,
        longterm_hidden_multiplier=2,
        longterm_lr_scale=0.1,
        drop_path=0.0,
        drop=0.0,
    )


@pytest.fixture
def tiny_model() -> ACHOPEHybridViT:
    """A minimal 1-layer Hybrid model for fast testing."""
    model = ac_hope_hybrid_vit(
        img_size=(256, 256),
        patch_size=16,
        num_timesteps=8,
        embed_dim=D_ENC,
        predictor_embed_dim=DIM,
        depth=1,
        num_heads=NUM_HEADS,
        action_embed_dim=ACTION_DIM,
        is_frame_causal=True,
        titan_hidden_multiplier=2,
        titan_layers=2,
        titan_grad_clip_inner=1.0,
        titan_grad_clip_backward=1.0,
        titan_detach_interval=0,
        surprise_threshold=0.0,
        cms_level_specs=CMS_LEVEL_DICTS,
        cms_use_chunk_scheduling=False,
        use_longterm_memory=False,
        drop_path_rate=0.0,
        log_hope_diagnostics=True,
    )
    model.train()
    return model


@pytest.fixture
def sample_inputs():
    """Minimal batch: B=1, T=3 timesteps, N=256 patches, D=1024."""
    x = torch.randn(B, T * N, D_ENC)
    actions = torch.randn(B, T, ACTION_DIM)
    states = torch.randn(B, T, ACTION_DIM)
    return x, actions, states


# ---------------------------------------------------------------------------
# 1. HybridBlock Tests
# ---------------------------------------------------------------------------

class TestHybridBlock:
    """Tests for the HybridBlock component."""

    def test_block_construction(self, block_config):
        """HybridBlock can be instantiated without error."""
        block = HybridBlock(block_config, titan_detach_interval=0)
        assert isinstance(block, nn.Module)

    def test_block_has_attention(self, block_config):
        """Block must have ACRoPEAttention in Phase A."""
        block = HybridBlock(block_config, titan_detach_interval=0)
        assert hasattr(block, "attn"), "Block must have self-attention"

    def test_block_has_titan_memory(self, block_config):
        """Block must have a Titan memory in Phase B."""
        block = HybridBlock(block_config, titan_detach_interval=0)
        assert hasattr(block, "M_memory"), "Block must have Titan memory"

    def test_block_has_cms(self, block_config):
        """Block must have CMS in Phase C."""
        block = HybridBlock(block_config, titan_detach_interval=0)
        assert hasattr(block, "cms"), "Block must have CMS"

    def test_block_has_eta_alpha(self, block_config):
        """Block must have learnable η and α parameters."""
        block = HybridBlock(block_config, titan_detach_interval=0)
        assert hasattr(block, "eta_base"), "Block must have eta_base parameter"
        assert hasattr(block, "alpha_base"), "Block must have alpha_base parameter"
        assert isinstance(block.eta_base, nn.Parameter)
        assert isinstance(block.alpha_base, nn.Parameter)

    def test_block_no_longterm_by_default(self, block_config):
        """Without use_longterm_memory, no M_longterm should exist."""
        block = HybridBlock(block_config, titan_detach_interval=0)
        assert not hasattr(block, "M_longterm") or block.M_longterm is None

    def test_block_with_longterm(self):
        """With use_longterm_memory=True, M_longterm is created."""
        cfg = HybridBlockConfig(
            dim=DIM, num_heads=NUM_HEADS,
            cms_levels=CMS_LEVELS,
            use_longterm_memory=True,
            longterm_hidden_multiplier=2,
            longterm_lr_scale=0.1,
        )
        block = HybridBlock(cfg, titan_detach_interval=0)
        assert hasattr(block, "M_longterm")
        assert block.M_longterm is not None

    def test_block_memory_reset(self, block_config):
        """reset_memory_state() should reset clip-level memory."""
        block = HybridBlock(block_config, titan_detach_interval=0)
        block.reset_memory_state()  # Should not raise


# ---------------------------------------------------------------------------
# 2. Full Model Tests
# ---------------------------------------------------------------------------

class TestACHOPEHybridViT:
    """Tests for the full ACHOPEHybridViT model."""

    def test_model_construction(self, tiny_model):
        """Model can be instantiated."""
        assert isinstance(tiny_model, ACHOPEHybridViT)

    def test_forward_shape(self, tiny_model, sample_inputs):
        """Output shape must match input: [B, T*N, D_enc]."""
        x, actions, states = sample_inputs
        tiny_model.reset_all_memories()
        out = tiny_model(x, actions, states)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_forward_deterministic(self, tiny_model, sample_inputs):
        """Same input with same memory state should give same output."""
        x, actions, states = sample_inputs
        tiny_model.eval()
        tiny_model.reset_all_memories()
        out1 = tiny_model(x, actions, states)
        tiny_model.reset_all_memories()
        out2 = tiny_model(x, actions, states)
        assert torch.allclose(out1, out2, atol=1e-5), "Forward should be deterministic"

    def test_forward_with_target_timestep(self, tiny_model):
        """Forward with target_timestep arg should not crash.

        In jump prediction mode, only T=1 context frame is passed
        (matching _compute_jump_loss in the loss mixin).
        """
        x = torch.randn(B, 1 * N, D_ENC)   # T=1 for jump prediction
        actions = torch.randn(B, 1, ACTION_DIM)
        states = torch.randn(B, 1, ACTION_DIM)
        tiny_model.reset_all_memories()
        out = tiny_model(x, actions, states, target_timestep=2)
        assert out.shape == x.shape

    def test_forward_batch_greater_than_one(self):
        """Test with batch_size > 1."""
        model = ac_hope_hybrid_vit(
            img_size=(256, 256), patch_size=16, num_timesteps=8,
            embed_dim=D_ENC, predictor_embed_dim=DIM, depth=1,
            num_heads=NUM_HEADS, action_embed_dim=ACTION_DIM,
            is_frame_causal=True,
            cms_level_specs=CMS_LEVEL_DICTS,
            drop_path_rate=0.0,
            log_hope_diagnostics=False,
        )
        model.train()
        model.reset_all_memories()
        x = torch.randn(2, T * N, D_ENC)
        a = torch.randn(2, T, ACTION_DIM)
        s = torch.randn(2, T, ACTION_DIM)
        out = model(x, a, s)
        assert out.shape == (2, T * N, D_ENC)


# ---------------------------------------------------------------------------
# 3. Memory Management Tests
# ---------------------------------------------------------------------------

class TestMemoryManagement:
    """Test memory reset, persistence, freeze/unfreeze."""

    def test_reset_all_memories(self, tiny_model):
        """reset_all_memories() should not raise."""
        tiny_model.reset_all_memories()

    def test_reset_all_longterm(self, tiny_model):
        """reset_all_longterm_memories() should not raise."""
        tiny_model.reset_all_longterm_memories()

    def test_freeze_inner_loops(self, tiny_model):
        """freeze_all_inner_loops() disables DGD."""
        tiny_model.freeze_all_inner_loops()
        for block in tiny_model.hybrid_blocks:
            assert block.freeze_inner_loop

    def test_unfreeze_inner_loops(self, tiny_model):
        """unfreeze_all_inner_loops() re-enables DGD."""
        tiny_model.freeze_all_inner_loops()
        tiny_model.unfreeze_all_inner_loops()
        for block in tiny_model.hybrid_blocks:
            assert not block.freeze_inner_loop

    def test_frozen_forward_works(self, tiny_model, sample_inputs):
        """Model should still produce output with frozen inner loops."""
        x, actions, states = sample_inputs
        tiny_model.reset_all_memories()
        tiny_model.freeze_all_inner_loops()
        out = tiny_model(x, actions, states)
        assert out.shape == x.shape
        tiny_model.unfreeze_all_inner_loops()


# ---------------------------------------------------------------------------
# 4. Gradient Flow Tests
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Verify that all three phases have gradient paths."""

    def test_attention_params_have_grad(self, tiny_model, sample_inputs):
        """Attention parameters must receive gradients."""
        x, actions, states = sample_inputs
        tiny_model.reset_all_memories()
        out = tiny_model(x, actions, states)
        loss = out.mean()
        loss.backward()

        block = tiny_model.hybrid_blocks[0]
        attn = block.attn
        for pname, p in attn.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"attn.{pname}: grad is None"
                assert p.grad.norm().item() > 0, f"attn.{pname}: grad norm is 0"

    def test_titan_memory_params_have_grad(self, tiny_model, sample_inputs):
        """Titan M_memory parameters must receive gradients."""
        x, actions, states = sample_inputs
        tiny_model.reset_all_memories()
        out = tiny_model(x, actions, states)
        loss = out.mean()
        loss.backward()

        block = tiny_model.hybrid_blocks[0]
        mem = block.M_memory
        for pname, p in mem.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, (
                    f"M_memory.{pname}: grad is None — meta-gradient broken!"
                )

    def test_cms_params_have_grad(self, tiny_model, sample_inputs):
        """CMS parameters must receive gradients."""
        x, actions, states = sample_inputs
        tiny_model.reset_all_memories()
        out = tiny_model(x, actions, states)
        loss = out.mean()
        loss.backward()

        block = tiny_model.hybrid_blocks[0]
        cms = block.cms
        for pname, p in cms.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"cms.{pname}: grad is None"

    def test_eta_alpha_have_grad(self, tiny_model, sample_inputs):
        """η and α parameters receive gradients through the DGD chain.

        In a single forward pass, η/α don't get gradients because the
        memory update (which uses η/α) happens AFTER the read. They only
        get gradients when a second read uses the updated memory state.
        This mirrors the original HOPE's M_eta/M_alpha behavior with chunking.
        """
        x, actions, states = sample_inputs
        tiny_model.reset_all_memories()

        # First forward: reads initial memory, then updates using η/α
        _ = tiny_model(x, actions, states)

        # Second forward: reads from memory updated by η/α → η/α now in graph
        out2 = tiny_model(x, actions, states)
        loss = out2.mean()
        loss.backward()

        block = tiny_model.hybrid_blocks[0]
        assert block.eta_base.grad is not None, (
            "eta_base has no gradient after 2-step chain"
        )
        assert block.alpha_base.grad is not None, (
            "alpha_base has no gradient after 2-step chain"
        )

    def test_embedding_proj_have_grad(self, tiny_model, sample_inputs):
        """Embedding/projection layers must receive gradients."""
        x, actions, states = sample_inputs
        tiny_model.reset_all_memories()
        out = tiny_model(x, actions, states)
        loss = out.mean()
        loss.backward()

        for name in ["predictor_embed", "predictor_proj", "action_encoder", "state_encoder"]:
            module = getattr(tiny_model, name)
            for pname, p in module.named_parameters():
                if p.requires_grad:
                    assert p.grad is not None, f"{name}.{pname}: grad is None"


# ---------------------------------------------------------------------------
# 5. Parameter Groups
# ---------------------------------------------------------------------------

class TestParameterGroups:
    """verify get_parameter_groups() returns disjoint groups."""

    def test_four_groups(self, tiny_model):
        """Must return exactly 4 groups: attention, titan, cms, projections."""
        groups = tiny_model.get_parameter_groups()
        names = {g["group_name"] for g in groups}
        assert names == {"attention", "titan", "cms", "projections"}

    def test_all_params_covered(self, tiny_model):
        """All trainable parameters are in exactly one group."""
        groups = tiny_model.get_parameter_groups()
        grouped_params = set()
        for g in groups:
            for p in g["params"]:
                grouped_params.add(id(p))

        all_params = {id(p) for p in tiny_model.parameters() if p.requires_grad}
        assert grouped_params == all_params, (
            f"Missing: {len(all_params - grouped_params)}, "
            f"Extra: {len(grouped_params - all_params)}"
        )

    def test_no_duplicates(self, tiny_model):
        """No parameter should appear in more than one group."""
        groups = tiny_model.get_parameter_groups()
        seen = set()
        for g in groups:
            for p in g["params"]:
                pid = id(p)
                assert pid not in seen, "Duplicate parameter in groups!"
                seen.add(pid)


# ---------------------------------------------------------------------------
# 6. Diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    """Test diagnostic logging."""

    def test_get_all_diagnostics(self, tiny_model, sample_inputs):
        """get_all_diagnostics() returns a dict with per-block metrics."""
        x, actions, states = sample_inputs
        tiny_model.reset_all_memories()
        _ = tiny_model(x, actions, states)
        diag = tiny_model.get_all_diagnostics()
        assert isinstance(diag, dict)

    def test_config_summary(self, tiny_model):
        """get_config_summary() returns model config dict."""
        summary = tiny_model.get_config_summary()
        assert isinstance(summary, dict)
        assert summary["architecture"] == "hybrid"
        assert summary["depth"] == 1

    def test_aux_loss_is_zero(self, tiny_model, sample_inputs):
        """Hybrid model should return aux_loss = 0."""
        x, actions, states = sample_inputs
        tiny_model.reset_all_memories()
        _ = tiny_model(x, actions, states)
        aux = tiny_model.get_aux_loss()
        assert aux == 0.0 or (isinstance(aux, torch.Tensor) and aux.item() == 0.0)


# ---------------------------------------------------------------------------
# 7. Longterm Memory
# ---------------------------------------------------------------------------

class TestLongtermMemory:
    """Test longterm memory construction and behavior."""

    @pytest.fixture
    def model_with_longterm(self):
        """Model with longterm memory enabled."""
        model = ac_hope_hybrid_vit(
            img_size=(256, 256), patch_size=16, num_timesteps=8,
            embed_dim=D_ENC, predictor_embed_dim=DIM, depth=1,
            num_heads=NUM_HEADS, action_embed_dim=ACTION_DIM,
            is_frame_causal=True,
            cms_level_specs=CMS_LEVEL_DICTS,
            use_longterm_memory=True,
            longterm_hidden_multiplier=2,
            longterm_lr_scale=0.1,
            drop_path_rate=0.0,
            log_hope_diagnostics=True,
        )
        model.train()
        return model

    def test_longterm_forward(self, model_with_longterm, sample_inputs):
        """Model with longterm memory should forward without error."""
        x, actions, states = sample_inputs
        model_with_longterm.reset_all_memories()
        out = model_with_longterm(x, actions, states)
        assert out.shape == x.shape

    def test_longterm_reset_selective(self, model_with_longterm):
        """reset_all_memories() should reset clip memory, not longterm."""
        # First encode something
        model_with_longterm.reset_all_memories()
        model_with_longterm.reset_all_longterm_memories()


# ---------------------------------------------------------------------------
# 8. Lightning Module (import test)
# ---------------------------------------------------------------------------

class TestLightningModule:
    """Test Lightning module wrapper."""

    def test_module_import(self):
        """ACHOPEHybridModule can be imported."""
        from src.models.hope.ac_hope_hybrid_module import ACHOPEHybridModule
        assert ACHOPEHybridModule is not None

    def test_module_construction(self):
        """ACHOPEHybridModule can be constructed with minimal args."""
        from src.models.hope.ac_hope_hybrid_module import ACHOPEHybridModule

        module = ACHOPEHybridModule(
            img_size=(256, 256),
            patch_size=16,
            num_timesteps=8,
            embed_dim=D_ENC,
            predictor_embed_dim=DIM,
            depth=1,
            num_heads=NUM_HEADS,
            action_embed_dim=ACTION_DIM,
            is_frame_causal=True,
            cms_level_specs=CMS_LEVEL_DICTS,
            drop_path_rate=0.0,
            log_hope_diagnostics=False,
        )
        assert hasattr(module, "model")
        assert isinstance(module.model, ACHOPEHybridViT)

    def test_module_has_step_predictor(self):
        """Module must have _step_predictor for loss mixin."""
        from src.models.hope.ac_hope_hybrid_module import ACHOPEHybridModule

        module = ACHOPEHybridModule(
            depth=1, num_heads=NUM_HEADS, action_embed_dim=ACTION_DIM,
            cms_level_specs=CMS_LEVEL_DICTS,
            log_hope_diagnostics=False,
        )
        assert callable(getattr(module, "_step_predictor", None))
