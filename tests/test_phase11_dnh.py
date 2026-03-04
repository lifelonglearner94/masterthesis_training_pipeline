"""Unit tests for the Phase 11 DNH-HOPE-Hybrid architecture.

Tests cover:
    1. SMM (SelfModifyingMemory) construction & forward
    2. DynamicNestedHierarchy construction, forward, add/remove levels
    3. StructuralEvolutionController step, add, prune, frequency modulation
    4. DynamicCMS construction, forward, add/remove levels
    5. Meta-loss computation
    6. DNHHybridBlock construction & forward
    7. Full model (ACDNHHOPEHybridViT) construction, forward, memory management
    8. Parameter groups (5 groups)
    9. Gradient flow through all phases
    10. Structural evolution integration
"""

import pytest
import torch
import torch.nn as nn

from src.models.hope.cms import LevelSpec
from src.models.hope.dnh.smm import MetaNetwork, SelfModifyingMemory, SMMConfig
from src.models.hope.dnh.dynamic_hierarchy import (
    DNHConfig,
    DynamicNestedHierarchy,
)
from src.models.hope.dnh.structural_evolution import (
    EvolutionConfig,
    StructuralEvolutionController,
)
from src.models.hope.dnh.dynamic_cms import DynamicCMS
from src.models.hope.dnh.meta_loss import compute_meta_loss, compute_level_meta_losses
from src.models.hope.dnh_hybrid_block import DNHHybridBlock, DNHHybridBlockConfig
from src.models.hope.ac_dnh_hope_hybrid_vit import (
    ACDNHHOPEHybridViT,
    ac_dnh_hope_hybrid_vit,
)


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
def smm_config() -> SMMConfig:
    """Minimal SMM config."""
    return SMMConfig(
        dim=DIM,
        titan_hidden_multiplier=2,
        titan_layers=2,
        titan_activation="gelu",
        titan_grad_clip_inner=1.0,
        titan_grad_clip_backward=1.0,
        titan_detach_interval=0,
        meta_hidden_dim=DIM,
        meta_activation="gelu",
        level_index=0,
    )


@pytest.fixture
def dnh_config() -> DNHConfig:
    """Minimal DNH config."""
    return DNHConfig(
        dim=DIM,
        L_init=2,
        L_max=4,
        L_min=2,
        titan_hidden_multiplier=2,
        titan_layers=2,
        titan_activation="gelu",
        titan_grad_clip_inner=1.0,
        titan_grad_clip_backward=1.0,
        titan_detach_interval=0,
        meta_hidden_dim=DIM,
        surprise_threshold=0.0,
        use_longterm_memory=False,
    )


@pytest.fixture
def block_config() -> DNHHybridBlockConfig:
    """Minimal DNHHybridBlock config."""
    return DNHHybridBlockConfig(
        dim=DIM,
        num_heads=NUM_HEADS,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        grid_size=16,
        dnh_L_init=2,
        dnh_L_max=4,
        dnh_L_min=2,
        titan_hidden_multiplier=2,
        titan_layers=2,
        titan_activation="gelu",
        titan_grad_clip_inner=1.0,
        titan_grad_clip_backward=1.0,
        titan_detach_interval=0,
        surprise_threshold=0.0,
        meta_hidden_dim=DIM,
        cms_levels=CMS_LEVELS,
        cms_use_chunk_scheduling=False,
        cms_L_max=4,
        cms_L_min=2,
        use_longterm_memory=False,
        drop_path=0.0,
        drop=0.0,
    )


@pytest.fixture
def tiny_model() -> ACDNHHOPEHybridViT:
    """A minimal 1-layer DNH Hybrid model for fast testing."""
    model = ac_dnh_hope_hybrid_vit(
        img_size=(256, 256),
        patch_size=16,
        num_timesteps=8,
        embed_dim=D_ENC,
        predictor_embed_dim=DIM,
        depth=1,
        num_heads=NUM_HEADS,
        action_embed_dim=ACTION_DIM,
        is_frame_causal=True,
        dnh_L_init=2,
        dnh_L_max=4,
        dnh_L_min=2,
        titan_hidden_multiplier=2,
        titan_layers=2,
        titan_grad_clip_inner=1.0,
        titan_grad_clip_backward=1.0,
        titan_detach_interval=0,
        surprise_threshold=0.0,
        meta_hidden_dim=DIM,
        cms_level_specs=CMS_LEVEL_DICTS,
        cms_use_chunk_scheduling=False,
        cms_L_max=4,
        cms_L_min=2,
        use_longterm_memory=False,
        drop_path_rate=0.0,
        log_hope_diagnostics=True,
    )
    model.train()
    return model


@pytest.fixture
def sample_inputs():
    """Minimal batch: B=1, T=3, N=256, D=1024."""
    x = torch.randn(B, T * N, D_ENC)
    actions = torch.randn(B, T, ACTION_DIM)
    states = torch.randn(B, T, ACTION_DIM)
    return x, actions, states


# ===========================================================================
# 1. MetaNetwork Tests
# ===========================================================================

class TestMetaNetwork:
    """Tests for the MetaNetwork component."""

    def test_construction(self):
        """MetaNetwork can be created."""
        net = MetaNetwork(dim=DIM, hidden_dim=DIM, activation="gelu")
        assert isinstance(net, nn.Module)

    def test_forward_shape(self):
        """Output shape matches [B, N, D]."""
        net = MetaNetwork(dim=DIM)
        k = torch.randn(B, 10, DIM)
        v = torch.randn(B, 10, DIM)
        context = torch.randn(DIM)
        out = net(k, v, context)
        assert out.shape == (B, 10, DIM)

    def test_output_bounded(self):
        """Tanh activation bounds output to [-1, 1]."""
        net = MetaNetwork(dim=DIM)
        k = torch.randn(B, 10, DIM) * 10
        v = torch.randn(B, 10, DIM) * 10
        context = torch.randn(DIM)
        out = net(k, v, context)
        assert out.abs().max().item() <= 1.0 + 1e-6

    def test_near_zero_init(self):
        """Initial output should be near zero due to small init."""
        net = MetaNetwork(dim=DIM)
        k = torch.randn(B, 10, DIM)
        v = torch.randn(B, 10, DIM)
        context = torch.randn(DIM) * 0.02
        out = net(k, v, context)
        assert out.abs().mean().item() < 0.5, "Initial output should be small"


# ===========================================================================
# 2. SelfModifyingMemory Tests
# ===========================================================================

class TestSelfModifyingMemory:
    """Tests for the SelfModifyingMemory module."""

    def test_construction(self, smm_config):
        """SMM can be instantiated."""
        smm = SelfModifyingMemory(smm_config)
        assert isinstance(smm, nn.Module)
        assert hasattr(smm, "memory")
        assert hasattr(smm, "meta_net")
        assert hasattr(smm, "level_context")
        assert hasattr(smm, "eta_base")
        assert hasattr(smm, "alpha_base")
        assert hasattr(smm, "modulation_gate")

    def test_forward_shape(self, smm_config):
        """Forward produces correct output shape."""
        smm = SelfModifyingMemory(smm_config)
        smm.memory.reset_active_weights()
        q = torch.randn(B, 10, DIM)
        k = torch.randn(B, 10, DIM)
        v = torch.randn(B, 10, DIM)
        out = smm(q, k, v)
        assert out.shape == (B, 10, DIM)

    def test_modulation_gate_starts_small(self, smm_config):
        """Gate starts near 0 (sigmoid(-2) ≈ 0.12)."""
        smm = SelfModifyingMemory(smm_config)
        gate_val = torch.sigmoid(smm.modulation_gate).item()
        assert gate_val < 0.2, f"Gate should start small, got {gate_val}"

    def test_update_memory(self, smm_config):
        """update_memory() should not raise."""
        smm = SelfModifyingMemory(smm_config)
        smm.memory.reset_active_weights()
        q = torch.randn(B, 10, DIM)
        k = torch.randn(B, 10, DIM)
        v = torch.randn(B, 10, DIM)
        out = smm(q, k, v)
        smm.update_memory(k, v, out, surprise_threshold=0.0, freeze_inner=False)

    def test_update_memory_frozen(self, smm_config):
        """Frozen update should be a no-op."""
        smm = SelfModifyingMemory(smm_config)
        smm.memory.reset_active_weights()
        q = torch.randn(B, 10, DIM)
        k = torch.randn(B, 10, DIM)
        v = torch.randn(B, 10, DIM)
        out = smm(q, k, v)
        # Frozen — should not raise
        smm.update_memory(k, v, out, surprise_threshold=0.0, freeze_inner=True)

    def test_cl_lifecycle_hooks(self, smm_config):
        """CL hooks should not raise."""
        smm = SelfModifyingMemory(smm_config)
        smm.memory.reset_active_weights()
        smm.reset_active_weights()
        smm.detach_active_weights()
        smm.clear_active_weights()
        smm.reset_diagnostics()

    def test_diagnostics(self, smm_config):
        """Diagnostics should return a non-empty dict."""
        smm = SelfModifyingMemory(smm_config)
        smm.memory.reset_active_weights()
        diag = smm.get_diagnostics()
        assert isinstance(diag, dict)
        assert "smm/modulation_gate" in diag

    def test_gradient_norm(self, smm_config):
        """get_gradient_norm() should return 0.0 without gradients."""
        smm = SelfModifyingMemory(smm_config)
        norm = smm.get_gradient_norm()
        assert norm == 0.0


# ===========================================================================
# 3. DynamicNestedHierarchy Tests
# ===========================================================================

class TestDynamicNestedHierarchy:
    """Tests for the DynamicNestedHierarchy module."""

    def test_construction(self, dnh_config):
        """DNH can be instantiated with L_init levels."""
        dnh = DynamicNestedHierarchy(dnh_config)
        assert dnh.num_levels == 2
        assert len(dnh.freq_raw) == 2

    def test_forward_shape(self, dnh_config):
        """Forward produces [B, N, D]."""
        dnh = DynamicNestedHierarchy(dnh_config)
        for level in dnh.levels:
            level.memory.reset_active_weights()
        x = torch.randn(B, 10, DIM)
        out = dnh(x)
        assert out.shape == (B, 10, DIM)

    def test_frequencies_property(self, dnh_config):
        """Frequencies should be positive reals."""
        dnh = DynamicNestedHierarchy(dnh_config)
        freqs = dnh.frequencies
        assert len(freqs) == 2
        assert all(f > 0 for f in freqs)

    def test_periods_property(self, dnh_config):
        """Periods should be positive integers."""
        dnh = DynamicNestedHierarchy(dnh_config)
        periods = dnh.periods
        assert len(periods) == 2
        assert all(isinstance(p, int) and p >= 1 for p in periods)

    def test_add_level(self, dnh_config):
        """Adding a level should increase num_levels."""
        dnh = DynamicNestedHierarchy(dnh_config)
        for level in dnh.levels:
            level.memory.reset_active_weights()
        initial = dnh.num_levels
        success = dnh.add_level()
        assert success
        assert dnh.num_levels == initial + 1
        assert len(dnh.freq_raw) == initial + 1

    def test_add_level_at_max(self, dnh_config):
        """Cannot add beyond L_max."""
        dnh_config.L_max = 2
        dnh_config.L_init = 2
        dnh = DynamicNestedHierarchy(dnh_config)
        for level in dnh.levels:
            level.memory.reset_active_weights()
        success = dnh.add_level()
        assert not success
        assert dnh.num_levels == 2

    def test_remove_level(self, dnh_config):
        """Removing a level should decrease num_levels."""
        dnh_config.L_init = 3
        dnh = DynamicNestedHierarchy(dnh_config)
        initial = dnh.num_levels
        success = dnh.remove_level(1)
        assert success
        assert dnh.num_levels == initial - 1

    def test_remove_level_at_min(self, dnh_config):
        """Cannot remove below L_min."""
        dnh = DynamicNestedHierarchy(dnh_config)
        assert dnh.num_levels == dnh_config.L_min
        success = dnh.remove_level(0)
        assert not success

    def test_cl_lifecycle(self, dnh_config):
        """All CL lifecycle methods should not raise."""
        dnh = DynamicNestedHierarchy(dnh_config)
        for level in dnh.levels:
            level.memory.reset_active_weights()
        dnh.reset_all_levels()
        dnh.clear_all_levels()
        dnh.freeze_inner_loops()
        assert dnh.freeze_inner_loop
        dnh.unfreeze_inner_loops()
        assert not dnh.freeze_inner_loop

    def test_diagnostics(self, dnh_config):
        """Diagnostics should return dict with num_levels."""
        dnh = DynamicNestedHierarchy(dnh_config)
        for level in dnh.levels:
            level.memory.reset_active_weights()
        diag = dnh.get_diagnostics()
        assert isinstance(diag, dict)
        assert "dnh/num_levels" in diag
        assert diag["dnh/num_levels"] == 2.0

    def test_get_level_gradient_norms(self, dnh_config):
        """Should return gradient norms per level."""
        dnh = DynamicNestedHierarchy(dnh_config)
        norms = dnh.get_level_gradient_norms()
        assert len(norms) == dnh.num_levels
        assert all(n == 0.0 for n in norms)  # no grads yet

    def test_with_longterm_memory(self):
        """DNH with longterm memory should work."""
        cfg = DNHConfig(
            dim=DIM,
            L_init=2,
            L_max=4,
            L_min=2,
            titan_hidden_multiplier=2,
            titan_layers=2,
            use_longterm_memory=True,
            longterm_hidden_multiplier=2,
            longterm_lr_scale=0.1,
        )
        dnh = DynamicNestedHierarchy(cfg)
        assert hasattr(dnh, "M_longterm")
        for level in dnh.levels:
            level.memory.reset_active_weights()
        dnh.M_longterm.reset_active_weights()
        x = torch.randn(B, 10, DIM)
        out = dnh(x)
        assert out.shape == (B, 10, DIM)


# ===========================================================================
# 4. StructuralEvolutionController Tests
# ===========================================================================

class TestStructuralEvolutionController:
    """Tests for the evolution controller."""

    def test_construction(self):
        """Controller can be instantiated."""
        cfg = EvolutionConfig()
        ctrl = StructuralEvolutionController(cfg, max_levels=5)
        assert ctrl._global_step == 0

    def test_initialize_tracking(self):
        """Tracking arrays sized correctly."""
        cfg = EvolutionConfig()
        ctrl = StructuralEvolutionController(cfg)
        ctrl.initialize_tracking(3)
        assert len(ctrl._grad_norm_ema) == 3
        assert len(ctrl._freq_momentum) == 3

    def test_step_warmup(self, dnh_config):
        """During warmup, no evolution happens."""
        cfg = EvolutionConfig(warmup_steps=100, evolution_interval=1)
        ctrl = StructuralEvolutionController(cfg)
        dnh = DynamicNestedHierarchy(dnh_config)
        ctrl.initialize_tracking(dnh.num_levels)

        events = ctrl.step(dnh, meta_loss=10.0)
        assert "evolution/added_level" not in events
        assert dnh.num_levels == 2

    def test_step_addition_when_loss_high(self, dnh_config):
        """Level addition triggers when meta_loss > tau_add after warmup."""
        cfg = EvolutionConfig(
            tau_add=0.1,
            warmup_steps=0,
            evolution_interval=1,
            enable_addition=True,
            enable_pruning=False,
            enable_freq_modulation=False,
        )
        ctrl = StructuralEvolutionController(cfg, max_levels=dnh_config.L_max)
        dnh = DynamicNestedHierarchy(dnh_config)
        for level in dnh.levels:
            level.memory.reset_active_weights()
        ctrl.initialize_tracking(dnh.num_levels)

        events = ctrl.step(dnh, meta_loss=1.0)
        assert dnh.num_levels == 3
        assert events.get("evolution/added_level") == 1

    def test_step_no_addition_when_loss_low(self, dnh_config):
        """No addition when meta_loss < tau_add."""
        cfg = EvolutionConfig(
            tau_add=10.0,
            warmup_steps=0,
            evolution_interval=1,
            enable_addition=True,
            enable_pruning=False,
            enable_freq_modulation=False,
        )
        ctrl = StructuralEvolutionController(cfg)
        dnh = DynamicNestedHierarchy(dnh_config)
        for level in dnh.levels:
            level.memory.reset_active_weights()
        ctrl.initialize_tracking(dnh.num_levels)

        events = ctrl.step(dnh, meta_loss=0.01)
        assert dnh.num_levels == 2
        assert "evolution/added_level" not in events

    def test_pruning_with_patience(self, dnh_config):
        """Pruning happens after sustained low gradient norm."""
        dnh_config.L_init = 3
        cfg = EvolutionConfig(
            warmup_steps=0,
            evolution_interval=1,
            enable_addition=False,
            enable_pruning=True,
            enable_freq_modulation=False,
            epsilon_prune=0.1,
            prune_patience=3,
        )
        ctrl = StructuralEvolutionController(cfg, max_levels=dnh_config.L_max)
        dnh = DynamicNestedHierarchy(dnh_config)
        for level in dnh.levels:
            level.memory.reset_active_weights()
        ctrl.initialize_tracking(dnh.num_levels)

        # Run enough steps with zero grads to trigger pruning
        for _ in range(5):
            ctrl.step(dnh, meta_loss=0.01)

        # After enough patience, one level should be pruned
        assert dnh.num_levels == 2, "Should have pruned to L_min"

    def test_diagnostics(self):
        """Diagnostics should return counts."""
        cfg = EvolutionConfig()
        ctrl = StructuralEvolutionController(cfg)
        ctrl.initialize_tracking(2)
        diag = ctrl.get_diagnostics()
        assert "evolution/global_step" in diag
        assert "evolution/total_additions" in diag
        assert "evolution/total_prunings" in diag


# ===========================================================================
# 5. DynamicCMS Tests
# ===========================================================================

class TestDynamicCMS:
    """Tests for the DynamicCMS module."""

    def test_construction(self):
        """DynamicCMS can be created."""
        cms = DynamicCMS(dim=DIM, levels=CMS_LEVELS)
        assert cms.num_levels == 2

    def test_forward_shape(self):
        """Forward produces [B, N, D]."""
        cms = DynamicCMS(dim=DIM, levels=CMS_LEVELS)
        x = torch.randn(B, 10, DIM)
        out = cms(x)
        assert out.shape == (B, 10, DIM)

    def test_add_level(self):
        """Adding a CMS level should increase count."""
        cms = DynamicCMS(dim=DIM, levels=CMS_LEVELS, cms_L_max=4)
        success = cms.add_level(name="extra", update_period=5, hidden_multiplier=2.0)
        assert success
        assert cms.num_levels == 3

    def test_add_level_auto_params(self):
        """Adding level without specifying params should succeed."""
        cms = DynamicCMS(dim=DIM, levels=CMS_LEVELS, cms_L_max=4)
        success = cms.add_level()
        assert success
        assert cms.num_levels == 3

    def test_add_level_at_max(self):
        """Cannot add beyond max."""
        cms = DynamicCMS(dim=DIM, levels=CMS_LEVELS, cms_L_max=2)
        success = cms.add_level()
        assert not success

    def test_remove_level(self):
        """Removing a level should decrease count."""
        three_levels = CMS_LEVELS + [
            LevelSpec(name="ultra", update_period=5, hidden_multiplier=2.5)
        ]
        cms = DynamicCMS(dim=DIM, levels=three_levels, cms_L_min=2)
        success = cms.remove_level(2)
        assert success
        assert cms.num_levels == 2

    def test_remove_level_at_min(self):
        """Cannot remove below min."""
        cms = DynamicCMS(dim=DIM, levels=CMS_LEVELS, cms_L_min=2)
        success = cms.remove_level(0)
        assert not success

    def test_gradient_norms(self):
        """Gradient norms should be returned per level."""
        cms = DynamicCMS(dim=DIM, levels=CMS_LEVELS)
        norms = cms.get_level_gradient_norms()
        assert len(norms) == 2

    def test_diagnostics(self):
        """Diagnostics should include level count."""
        cms = DynamicCMS(dim=DIM, levels=CMS_LEVELS)
        diag = cms.get_diagnostics()
        assert "dynamic_cms/num_levels" in diag

    def test_forward_with_chunk_scheduling(self):
        """Forward with chunk scheduling should work."""
        cms = DynamicCMS(dim=DIM, levels=CMS_LEVELS, use_chunk_scheduling=True)
        x = torch.randn(B, T * 258, DIM)  # T frames × (256 patches + 2 cond)
        out = cms(x, T=T, tokens_per_frame=258)
        assert out.shape == x.shape


# ===========================================================================
# 6. Meta-Loss Tests
# ===========================================================================

class TestMetaLoss:
    """Tests for meta-loss computation."""

    def test_compute_meta_loss_basic(self):
        """Basic meta-loss should equal task loss when no extras."""
        task_loss = torch.tensor(1.5)
        meta = compute_meta_loss(task_loss)
        assert torch.allclose(meta, task_loss)

    def test_compute_meta_loss_structural_penalty(self):
        """Structural changes should increase meta-loss."""
        task_loss = torch.tensor(1.0)
        meta = compute_meta_loss(task_loss, structural_changes=2, lambda_structural=0.1)
        assert meta.item() > task_loss.item()
        assert abs(meta.item() - 1.2) < 1e-5

    def test_compute_meta_loss_drift_penalty(self):
        """Drift penalty should increase meta-loss when hidden changes."""
        task_loss = torch.tensor(1.0)
        current = torch.randn(B, 10, DIM)
        previous = current + torch.randn_like(current) * 0.5  # different
        meta = compute_meta_loss(
            task_loss, current_hidden=current,
            previous_hidden=previous, mu_drift=1.0,
        )
        assert meta.item() > task_loss.item()

    def test_compute_meta_loss_no_drift_when_same(self):
        """No drift penalty when hidden representations are identical."""
        task_loss = torch.tensor(1.0)
        hidden = torch.randn(B, 10, DIM)
        meta = compute_meta_loss(
            task_loss, current_hidden=hidden,
            previous_hidden=hidden.clone(), mu_drift=1.0,
        )
        assert abs(meta.item() - task_loss.item()) < 1e-5

    def test_compute_level_meta_losses(self):
        """Per-level losses should be computed."""
        target = torch.randn(B, 10, DIM)
        outputs = [torch.randn(B, 10, DIM) for _ in range(3)]
        losses = compute_level_meta_losses(outputs, target)
        assert len(losses) == 3
        assert all(l.ndim == 0 for l in losses)  # scalar losses


# ===========================================================================
# 7. DNHHybridBlock Tests
# ===========================================================================

class TestDNHHybridBlock:
    """Tests for the DNHHybridBlock component."""

    def test_construction(self, block_config):
        """Block can be instantiated."""
        block = DNHHybridBlock(block_config)
        assert isinstance(block, nn.Module)

    def test_has_attention(self, block_config):
        """Block has self-attention (Phase A)."""
        block = DNHHybridBlock(block_config)
        assert hasattr(block, "attn")

    def test_has_dnh(self, block_config):
        """Block has DynamicNestedHierarchy (Phase B)."""
        block = DNHHybridBlock(block_config)
        assert hasattr(block, "dnh")
        assert isinstance(block.dnh, DynamicNestedHierarchy)

    def test_has_dynamic_cms(self, block_config):
        """Block has DynamicCMS (Phase C)."""
        block = DNHHybridBlock(block_config)
        assert hasattr(block, "dynamic_cms")
        assert isinstance(block.dynamic_cms, DynamicCMS)

    def test_forward_shape(self, block_config):
        """Forward produces same shape as input."""
        block = DNHHybridBlock(block_config)
        # Reset all memory states
        block.reset_memory_state()
        tokens_per_frame = 2 + 16 * 16  # cond + patches
        N_total = T * tokens_per_frame
        x = torch.randn(B, N_total, DIM)
        out = block(x, T=T, H=16, W=16, action_tokens=2)
        assert out.shape == x.shape

    def test_reset_memory_state(self, block_config):
        """reset_memory_state() should not raise."""
        block = DNHHybridBlock(block_config)
        block.reset_memory_state()

    def test_diagnostics(self, block_config):
        """get_diagnostics() should return dict."""
        block = DNHHybridBlock(block_config)
        block.reset_memory_state()
        diag = block.get_diagnostics()
        assert isinstance(diag, dict)
        assert "dnh/num_levels" in diag


# ===========================================================================
# 8. Full Model Tests
# ===========================================================================

class TestACDNHHOPEHybridViT:
    """Tests for the full ACDNHHOPEHybridViT model."""

    def test_model_construction(self, tiny_model):
        """Model can be instantiated."""
        assert isinstance(tiny_model, ACDNHHOPEHybridViT)

    def test_forward_shape(self, tiny_model, sample_inputs):
        """Output shape matches input: [B, T*N, D_enc]."""
        x, actions, states = sample_inputs
        tiny_model.reset_all_memories()
        out = tiny_model(x, actions, states)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_forward_deterministic(self, tiny_model, sample_inputs):
        """Same input + same memory state → same output."""
        x, actions, states = sample_inputs
        tiny_model.eval()
        tiny_model.reset_all_memories()
        out1 = tiny_model(x, actions, states)
        tiny_model.reset_all_memories()
        out2 = tiny_model(x, actions, states)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_forward_with_target_timestep(self, tiny_model):
        """Jump prediction mode (T=1, target_timestep) should work."""
        x = torch.randn(B, 1 * N, D_ENC)
        actions = torch.randn(B, 1, ACTION_DIM)
        states = torch.randn(B, 1, ACTION_DIM)
        tiny_model.reset_all_memories()
        out = tiny_model(x, actions, states, target_timestep=2)
        assert out.shape == x.shape

    def test_forward_batch_greater_than_one(self):
        """Batch > 1 should work."""
        model = ac_dnh_hope_hybrid_vit(
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

    def test_config_summary(self, tiny_model):
        """Config summary should include DNH-specific keys."""
        summary = tiny_model.get_config_summary()
        assert summary["architecture"] == "dnh_hybrid"
        assert summary["has_dnh"]
        assert summary["has_attention"]
        assert summary["has_rope"]

    def test_structural_summary(self, tiny_model):
        """Structural summary should report per-block info."""
        tiny_model.reset_all_memories()
        summary = tiny_model.get_structural_summary()
        assert "num_memory_levels" in summary
        assert "num_cms_levels" in summary
        assert len(summary["num_memory_levels"]) == 1  # depth=1
        assert summary["num_memory_levels"][0] == 2

    def test_aux_loss_is_zero(self, tiny_model):
        """DNH hybrid has no aux loss."""
        aux = tiny_model.get_aux_loss()
        assert aux.item() == 0.0


# ===========================================================================
# 9. Memory Management Tests
# ===========================================================================

class TestDNHMemoryManagement:
    """Test memory reset, persistence, freeze/unfreeze."""

    def test_reset_all_memories(self, tiny_model):
        """reset_all_memories() should not raise."""
        tiny_model.reset_all_memories()

    def test_reset_all_longterm(self, tiny_model):
        """reset_all_longterm_memories() should not raise."""
        tiny_model.reset_all_longterm_memories()

    def test_freeze_inner_loops(self, tiny_model):
        """freeze_all_inner_loops() disables DGD for all blocks."""
        tiny_model.freeze_all_inner_loops()
        for blk in tiny_model.dnh_blocks:
            assert blk.dnh.freeze_inner_loop

    def test_unfreeze_inner_loops(self, tiny_model):
        """unfreeze_all_inner_loops() re-enables DGD."""
        tiny_model.freeze_all_inner_loops()
        tiny_model.unfreeze_all_inner_loops()
        for blk in tiny_model.dnh_blocks:
            assert not blk.dnh.freeze_inner_loop

    def test_frozen_forward_works(self, tiny_model, sample_inputs):
        """Model should produce output with frozen inner loops."""
        x, actions, states = sample_inputs
        tiny_model.reset_all_memories()
        tiny_model.freeze_all_inner_loops()
        out = tiny_model(x, actions, states)
        assert out.shape == x.shape
        tiny_model.unfreeze_all_inner_loops()

    def test_freeze_attention(self, tiny_model):
        """freeze_attention() should freeze attn params."""
        tiny_model.freeze_attention()
        for blk in tiny_model.dnh_blocks:
            for name, param in blk.named_parameters():
                if "attn." in name:
                    assert not param.requires_grad
        tiny_model.unfreeze_attention()

    def test_reset_titan_for_new_task(self, tiny_model):
        """reset_titan_for_new_task() should not raise."""
        tiny_model.reset_all_memories()
        tiny_model.reset_titan_for_new_task()


# ===========================================================================
# 10. Gradient Flow Tests
# ===========================================================================

class TestDNHGradientFlow:
    """Verify gradient paths through all three phases."""

    def test_attention_params_have_grad(self, tiny_model, sample_inputs):
        """Attention parameters must receive gradients."""
        x, actions, states = sample_inputs
        tiny_model.reset_all_memories()
        out = tiny_model(x, actions, states)
        loss = out.mean()
        loss.backward()

        blk = tiny_model.dnh_blocks[0]
        for pname, p in blk.attn.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"attn.{pname}: no gradient"
                assert p.grad.norm().item() > 0, f"attn.{pname}: zero gradient"

    def test_dnh_memory_params_have_grad(self, tiny_model, sample_inputs):
        """DNH memory parameters must receive gradients."""
        x, actions, states = sample_inputs
        tiny_model.reset_all_memories()
        out = tiny_model(x, actions, states)
        loss = out.mean()
        loss.backward()

        blk = tiny_model.dnh_blocks[0]
        # Check meta-network has gradients
        for pname, p in blk.dnh.named_parameters():
            if "meta_net" in pname and p.requires_grad:
                assert p.grad is not None, f"dnh.{pname}: no gradient"

    def test_cms_params_have_grad(self, tiny_model, sample_inputs):
        """Dynamic CMS parameters must receive gradients."""
        x, actions, states = sample_inputs
        tiny_model.reset_all_memories()
        out = tiny_model(x, actions, states)
        loss = out.mean()
        loss.backward()

        blk = tiny_model.dnh_blocks[0]
        for pname, p in blk.dynamic_cms.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"cms.{pname}: no gradient"

    def test_backward_does_not_crash(self, tiny_model, sample_inputs):
        """Full forward + backward should complete without error."""
        x, actions, states = sample_inputs
        tiny_model.reset_all_memories()
        out = tiny_model(x, actions, states)
        loss = out.sum()
        loss.backward()
        # Verify at least some grads exist
        grads_found = sum(
            1 for p in tiny_model.parameters()
            if p.grad is not None and p.grad.norm().item() > 0
        )
        assert grads_found > 0, "No parameters received gradients"


# ===========================================================================
# 11. Parameter Groups Tests
# ===========================================================================

class TestParameterGroups:
    """Verify the 5-group parameter classification."""

    def test_five_groups(self, tiny_model):
        """Model should return exactly 5 parameter groups."""
        groups = tiny_model.get_parameter_groups()
        assert len(groups) == 5

    def test_group_names(self, tiny_model):
        """Groups should have correct names."""
        groups = tiny_model.get_parameter_groups()
        names = [g["group_name"] for g in groups]
        assert names == ["attention", "dnh_memory", "dnh_controller", "cms", "projections"]

    def test_all_params_classified(self, tiny_model):
        """Every trainable parameter should be in exactly one group."""
        groups = tiny_model.get_parameter_groups()
        group_params = set()
        for g in groups:
            for p in g["params"]:
                param_id = id(p)
                assert param_id not in group_params, "Parameter in multiple groups"
                group_params.add(param_id)

        trainable = set(
            id(p) for p in tiny_model.parameters() if p.requires_grad
        )
        assert group_params == trainable, (
            f"Mismatch: {len(trainable)} trainable params, "
            f"{len(group_params)} classified"
        )

    def test_dnh_controller_group_has_freq_params(self, tiny_model):
        """DNH controller group should contain frequency parameters."""
        groups = tiny_model.get_parameter_groups()
        ctrl_group = [g for g in groups if g["group_name"] == "dnh_controller"][0]
        assert len(ctrl_group["params"]) > 0, "Controller group should not be empty"


# ===========================================================================
# 12. Structural Evolution Integration Tests
# ===========================================================================

class TestStructuralEvolutionIntegration:
    """Test that structural evolution works with the full model."""

    def test_step_evolution(self, tiny_model):
        """step_structural_evolution() should run without error."""
        tiny_model.reset_all_memories()
        cfg = EvolutionConfig(
            warmup_steps=0,
            evolution_interval=1,
            enable_addition=False,
            enable_pruning=False,
            enable_freq_modulation=False,
        )
        controllers = [
            StructuralEvolutionController(cfg, max_levels=4)
            for _ in tiny_model.dnh_blocks
        ]
        for ctrl in controllers:
            ctrl.initialize_tracking(2)

        events = tiny_model.step_structural_evolution(controllers, meta_loss=0.5)
        assert isinstance(events, dict)

    def test_evolution_adds_level(self, tiny_model):
        """Evolution should add a level when meta_loss > tau_add."""
        tiny_model.reset_all_memories()
        cfg = EvolutionConfig(
            tau_add=0.1,
            warmup_steps=0,
            evolution_interval=1,
            enable_addition=True,
            enable_pruning=False,
            enable_freq_modulation=False,
        )
        controllers = [
            StructuralEvolutionController(cfg, max_levels=4)
            for _ in tiny_model.dnh_blocks
        ]
        for ctrl in controllers:
            ctrl.initialize_tracking(2)

        events = tiny_model.step_structural_evolution(controllers, meta_loss=1.0)
        # Check that a level was added
        summary = tiny_model.get_structural_summary()
        assert summary["num_memory_levels"][0] == 3

    def test_model_forward_after_evolution(self, tiny_model, sample_inputs):
        """Model should still produce valid output after structural change."""
        x, actions, states = sample_inputs
        tiny_model.reset_all_memories()

        # Add a level
        cfg = EvolutionConfig(
            tau_add=0.01, warmup_steps=0, evolution_interval=1,
            enable_addition=True, enable_pruning=False,
            enable_freq_modulation=False,
        )
        controllers = [
            StructuralEvolutionController(cfg, max_levels=4)
            for _ in tiny_model.dnh_blocks
        ]
        for ctrl in controllers:
            ctrl.initialize_tracking(2)
        tiny_model.step_structural_evolution(controllers, meta_loss=1.0)

        # Reset memories for the new structure
        tiny_model.reset_all_memories()

        # Forward should work with expanded structure
        out = tiny_model(x, actions, states)
        assert out.shape == x.shape


# ===========================================================================
# 13. Param Count Test
# ===========================================================================

class TestParamCount:
    """Verify approximate parameter budget."""

    def test_single_block_param_count(self, block_config):
        """Single DNHHybridBlock should have reasonable param count."""
        block = DNHHybridBlock(block_config)
        n_params = sum(p.numel() for p in block.parameters())
        # Should be larger than Phase 8 block due to meta-networks
        # Phase 8 block had ~3-4M; DNH block with 2 SMM levels should have ~4-6M
        assert n_params > 1_000_000, f"Too few params: {n_params}"
        assert n_params < 20_000_000, f"Too many params: {n_params}"

    def test_full_model_param_count(self, tiny_model):
        """Full model with depth=1 should have reasonable params."""
        n_params = sum(p.numel() for p in tiny_model.parameters())
        assert n_params > 2_000_000
        assert n_params < 30_000_000  # Only 1 layer
        print(f"Phase 11 model (depth=1): {n_params:,} params")

    def test_twelve_layer_param_count(self):
        """Full 12-layer model should be ~60M."""
        model = ac_dnh_hope_hybrid_vit(
            img_size=(256, 256), patch_size=16, num_timesteps=8,
            embed_dim=D_ENC, predictor_embed_dim=DIM, depth=12,
            num_heads=NUM_HEADS, action_embed_dim=ACTION_DIM,
            is_frame_causal=True,
            dnh_L_init=2,
            titan_hidden_multiplier=2,
            titan_layers=2,
            cms_level_specs=CMS_LEVEL_DICTS,
            drop_path_rate=0.1,
        )
        n_params = sum(p.numel() for p in model.parameters())
        # Target: ~60M (±20M is acceptable)
        assert n_params > 30_000_000, f"Too few: {n_params:,}"
        assert n_params < 100_000_000, f"Too many: {n_params:,}"
        print(f"Phase 11 model (depth=12): {n_params:,} params")
