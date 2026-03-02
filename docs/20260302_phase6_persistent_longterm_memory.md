# Phase 6: Persistent Cross-Clip Longterm Memory (Ansatz B)

**Date:** 2026-03-02
**Config:** `configs/experiment/cl_ac_hope_phase6.yaml`
**Param count:** ~46.5M (43.5M Phase 5 base + 3.0M longterm overhead)
**Builds on:** Phase 5 (temporal embeddings + spatial mixing)

---

## 1. Motivation

### 1.1 The Cross-Clip Amnesia Problem

In Phase 4, the AC-HOPE-ViT architecture achieved its first result marginally beating the naive ViT lower bound (Avg Error 0.3711 vs 0.3720, Gap Closed = 0.8%). However, forgetting *increased* (0.0646 vs 0.0481–0.0554 in Phases 1–2), revealing a fundamental architectural limitation:

> **All five Titan memories ($M_k$, $M_v$, $M_\eta$, $M_\alpha$, $M_\text{memory}$) are fully reset between clips.**

The `reset_memory_state()` function calls `reset_active_weights()` on every memory, which clones the `nn.Parameter` weights into fresh `_active_w1` / `_active_w2` tensors. Any knowledge accumulated by DGD during the previous clip is **discarded**. The model's only mechanism for cross-clip knowledge transfer is standard backpropagation updating the `nn.Parameters` — functionally equivalent to ordinary SGD-based finetuning, negating HOPE's central advantage of in-context adaptation.

This is analogous to a brain with only working memory (hippocampus) but no long-term consolidation (neocortex): it can adapt fluidly within a single experience but retains nothing between experiences.

### 1.2 Phase 4 CL Results (Status Quo Ante)

| Experiment | Avg Error ↓ | Forgetting ↓ | Plasticity ↓ | Gap Closed |
|------------|----------:|----------:|----------:|----------:|
| ViT Lower Bound (Naive) | 0.3720 | 0.0856 | 0.3007 | 0.0% |
| ViT Upper Bound (Joint) | 0.2502 | 0.0000 | 0.2502 | 100.0% |
| HOPE Phase 4 | 0.3711 | 0.0646 | 0.3105 | +0.8% |

Phase 4 slightly lowered Avg Error but destabilized forgetting. The memory reset strategy contributed to this instability: every new physics-shift task destroys clip-level DGD state, forcing the model to re-derive task-specific adaptations from scratch using only the slowly-evolving `nn.Parameters`.

### 1.3 Design Objectives

Phase 6 introduces a **persistent longterm memory** to address both CL metrics simultaneously:

1. **Reduce Plasticity**: Clip-level memories start clean → no inertia from old tasks
2. **Reduce Forgetting**: Persistent memory preserves shared physics patterns across tasks, providing a stable reference even as `nn.Parameters` drift under finetuning

---

## 2. Approach: Separation of Concerns (Ansatz B)

### 2.1 Alternative Approaches Considered

Three candidate approaches were analyzed for the persistent memory problem:

| Approach | Mechanism | Plasticity Effect | Forgetting Effect |
|----------|-----------|:-:|:-:|
| **A**: EMA Consolidation | Exponential moving average of post-DGD weights → next clip initialization | Neutral (warm start helps) | Worsens (warm start = old-task bias) |
| **B**: Persistent Slow-Titan + Gate | Separate M_longterm with slow DGD, never reset, gated combination | **Improves** (M_memory always clean) | **Improves** (M_longterm preserves common patterns) |
| **C**: KV Memory Bank | External key-value store across clips | Improves (clean memories) | Improves (but computationally expensive, complex retrieval) |

**Ansatz B was selected** as the only approach that simultaneously improves both plasticity and forgetting without significant computational overhead. Ansatz A was rejected because warm-starting clip memories from an EMA of old-task weights introduces bias toward previous distributions — exactly the catastrophic forgetting mechanism we aim to counter. Ansatz C was rejected for engineering complexity (nearest-neighbor retrieval, memory management) disproportionate to the expected benefit over B.

### 2.2 Theoretical Foundation: Complementary Learning Systems

The design draws on Complementary Learning Systems theory (McClelland et al., 1995; Kumaran et al., 2016), which proposes that biological brains use two complementary memory systems for learning:

| Property | Hippocampus | Neocortex |
|----------|:-----------:|:---------:|
| Learning rate | Fast (one-shot) | Slow (gradual) |
| Memory duration | Short-term / episodic | Long-term / semantic |
| Specialization | Current context | Shared structure |
| Analogous component | **$M_\text{memory}$** (clip-level) | **$M_\text{longterm}$** (persistent) |

The core hypothesis:

> By separating fast episodic memory (adapted per clip via rapid DGD) from slow consolidated memory (accumulated across clips via scaled-down DGD), the architecture can simultaneously achieve high plasticity (fast system adapts freely) and low forgetting (slow system preserves invariant physics).

### 2.3 Formal Architecture

#### Memory Retrieval (Gated Combination)

At inference time within `_titan_forward_chunk()`, after the standard retrieval step, we compute a gated combination of clip-level and longterm memory outputs:

$$o_t^\text{clip} = M_{\text{memory},t-1}(q_t)$$

$$o_t^\text{long} = M_{\text{longterm},t-1}(q_t)$$

$$g_t = \sigma(W_g \cdot q_t + b_g) \in [0, 1]$$

$$o_t = g_t \cdot o_t^\text{clip} + (1 - g_t) \cdot o_t^\text{long}$$

where $\sigma$ is the sigmoid function, $W_g \in \mathbb{R}^{D \times 1}$ is a learned projection, $b_g$ is a learned bias, and $g_t$ is broadcast across the feature dimension.

#### Gate Initialization Strategy

The gate is initialized to **favor clip-level memory**:

- $W_g = \mathbf{0}$ (zero-initialized weights)
- $b_g = 1.0$ → $\sigma(1.0) \approx 0.73$

This means at initialization, $g_t \approx 0.73$ for all tokens: approximately 73% of the output comes from $M_\text{memory}$ (clip-level) and 27% from $M_\text{longterm}$. This ensures:

1. **Backward compatibility**: With zero-init weights, the gate output is constant across all tokens and queries. The initial architecture behaves like a weighted average of Phase 5 (clip-level) and a secondary memory, rather than a completely different architecture.
2. **Conservative start**: The model defaults to trusting the existing clip-level system (which already works at Phase 4 level) and must actively *learn* to rely on longterm memory where it helps.
3. **Smooth optimization**: Gradients through $W_g$ start from an information-rich operating point ($\sigma'(1.0) \approx 0.20$), avoiding saturation at $\sigma(0) = 0.5$ where the gate would be maximally uncertain.

During training, the gate learns to modulate per-token: for tokens where longterm memory provides better predictions (e.g., shared physics dynamics), $g_t \to 0$; for tokens where current-clip adaptation is needed (e.g., task-specific patterns), $g_t \to 1$.

#### Memory Update (Scaled DGD)

$M_\text{longterm}$ receives the **same DGD update** as the five clip-level memories, but with a reduced learning rate:

$$\hat{v}_{\text{long},t} = M_{\text{longterm},t-1}(v_t) \quad \text{(self-generated target, Eq. 83)}$$

$$\eta_{\text{long}} = \eta_t \cdot \lambda_\text{scale}$$

$$M_{\text{longterm},t} = M_{\text{longterm},t-1}(\alpha_t I - \eta_{\text{long}} \, k_t k_t^\top) - \eta_{\text{long}} (M_{\text{longterm},t-1} k_t - \hat{v}_{\text{long},t}) k_t^\top$$

where $\lambda_\text{scale} = 0.1$ is the `longterm_lr_scale` hyperparameter. The 10× reduction in DGD learning rate ensures:

1. **Stability**: Longterm weights change slowly, accumulating the statistical mean of DGD updates across many clips rather than overfitting to any single clip
2. **Signal averaging**: High-surprise clips contribute more (surprise-gated updates), providing a natural importance weighting
3. **Gradient compatibility**: The scaled learning rate is still large enough for meaningful DGD updates but small enough to prevent the longterm memory from tracking clip-level transients

#### Cross-Clip Memory Lifecycle

The `reset_memory_state()` function implements asymmetric reset semantics:

```
reset_memory_state()   — called at the start of each new clip
├── Clip-level memories (M_k, M_v, M_eta, M_alpha, M_memory):
│   └── reset_active_weights()  → clone nn.Parameters → _active_w1/_active_w2
│                                  (fresh start, full gradient chain to Parameters)
├── CMS:
│   └── reset_step_counter()    → restart frame-aware scheduling
├── Auxiliary loss:
│   └── torch.tensor(0.0)       → reset accumulator
└── M_longterm:
    ├── First call (_active_w1 is None):
    │   └── reset_active_weights()  → initialize from nn.Parameters (like other memories)
    └── Subsequent calls:
        └── .detach()              → cut cross-clip computation graph only
                                     (accumulated DGD knowledge PRESERVED)
```

The `.detach()` on subsequent calls is critical: without it, the computation graph would grow unboundedly across clips (each clip's forward pass chains onto the previous one), eventually causing OOM. With detach, the accumulated weight values are preserved but treated as constants for the purpose of the next clip's gradient computation. The meta-learned `nn.Parameters` of $M_\text{longterm}$ still receive outer-loop gradients from the current clip's loss.

---

## 3. Architecture Details

### 3.1 M_longterm: Compact Persistent Titan Memory

$M_\text{longterm}$ is a standard `TitanMemory` instance with a smaller hidden dimension than $M_\text{memory}$:

| Property | M_memory (clip-level) | M_longterm (persistent) |
|----------|:---------------------:|:-----------------------:|
| Type | `TitanMemory` | `TitanMemory` |
| Hidden multiplier | 4 (= 1536) | 2 (= 768) |
| Architecture | Linear(384→1536) → GELU → Linear(1536→384) + LN + skip | Linear(384→768) → GELU → Linear(768→384) + LN + skip |
| DGD learning rate | $\eta_t$ (full) | $\eta_t \times 0.1$ (scaled) |
| Reset between clips | **Yes** (full reset to nn.Parameters) | **No** (detach only) |
| Parameters per block | $D \cdot H + H \cdot D + D + D = 2DH + 2D$ | Same formula, smaller $H$ |
| Parameters (D=384, H=1536 or 768) | 1,180,416 | 590,592 |

The smaller hidden dimension (multiplier=2 vs 4) is deliberate:

1. **Parameter efficiency**: +3M total vs +6M if full-size, keeping overall budget at 46.5M (vs 43M ViT baseline)
2. **Regularization**: Smaller capacity forces $M_\text{longterm}$ to learn compressed, general representations rather than memorizing clip-specific details
3. **Complementary capacity**: The fast $M_\text{memory}$ has high capacity for task-specific adaptation; the slow $M_\text{longterm}$ has lower capacity for shared invariants

### 3.2 Gate: Learned Per-Token Interpolation

The gate is implemented as a single `nn.Linear(D, 1)`:

| Property | Value | Rationale |
|----------|-------|-----------|
| Input | $q_t \in \mathbb{R}^D$ (query vector) | Query encodes *what* the model is looking for — natural conditioning signal |
| Output | Scalar $g_t \in (0, 1)$ via sigmoid | Broadcast across D features, controlling clip vs longterm mix |
| Weight init | $W_g = \mathbf{0}_{D \times 1}$ | Ensures constant gate at init (no input dependence) |
| Bias init | $b_g = 1.0$ | $\sigma(1.0) \approx 0.73$ favoring clip-level memory |
| Parameters | $D + 1 = 385$ per block, 1925 total (5 blocks) | Negligible (<0.005% of total params) |

The gate operates per-token (each of the $C$ tokens in a chunk gets its own $g_t$), enabling the model to selectively use longterm memory for some tokens/patches and clip-level memory for others. This is important for physics-shift CL: spatial patches encoding common dynamics (e.g., general fluid behavior) should draw more from longterm memory, while patches at physics discontinuities should rely on current-clip adaptation.

### 3.3 Parameter Budget

| Component | Per Block | × Blocks | Total |
|-----------|----------:|---------:|------:|
| Phase 5 base (Titan + CMS + Spatial Mixing) | ~8.7M | 5 | ~43.5M |
| M_longterm (D=384, H=768) | 590,592 | 5 | 2,952,960 |
| Longterm gate (D+1) | 385 | 5 | 1,925 |
| Temporal embeddings (Phase 5) | — | — | 6,912 |
| **Total** | | | **~46.5M** |

The +3M overhead (7% over Phase 5, 8% over ViT baseline at 43M) buys a qualitatively new capability: persistent cross-clip memory. This has no direct ViT equivalent — ViT baseline relies solely on weight updates for cross-task transfer.

---

## 4. Implementation

### 4.1 Modified Files

| File | Changes | Purpose |
|------|---------|---------|
| [hope_block.py](../src/models/hope/hope_block.py) | `HOPEBlockConfig` +3 fields; `__init__` +M_longterm +gate; `_titan_forward_chunk` +gated retrieval; `_update_memories` +longterm DGD; `reset_memory_state` +asymmetric reset; `+reset_longterm_memory()`; `get_diagnostics` +M_longterm | Core memory architecture |
| [ac_hope_vit.py](../src/models/hope/ac_hope_vit.py) | Constructor wires 3 params; `HOPEBlockConfig` gets longterm fields; `get_parameter_groups` includes M_longterm in titan patterns; `+reset_all_longterm_memories()`; `_config_summary` +longterm | Model-level integration |
| [ac_hope_module.py](../src/models/hope/ac_hope_module.py) | Constructor wires 3 params; M3 optimizer `titan_patterns` updated | Lightning module integration |
| [ac_hope_vit.yaml](../configs/model/ac_hope_vit.yaml) | +3 defaults (`use_longterm_memory: false`, `longterm_hidden_multiplier: 2`, `longterm_lr_scale: 0.1`) | Default config (disabled) |
| [cl_ac_hope_phase6.yaml](../configs/experiment/cl_ac_hope_phase6.yaml) | NEW — full Phase 6 experiment config | Experiment configuration |
| [test_phase6_longterm_memory.py](../tests/test_phase6_longterm_memory.py) | NEW — 23 tests across 6 test classes | Verification |

### 4.2 Code Integration Points

#### 4.2.1 Gated Retrieval in `_titan_forward_chunk()`

```python
# Standard clip-level retrieval
output = self.M_memory(q)  # o_t = M_{memory,t-1}(q_t)

# Gated longterm memory retrieval (Ansatz B)
if self.use_longterm_memory:
    output_longterm = self.M_longterm(q)
    gate = torch.sigmoid(self.longterm_gate(q))  # [B, C, 1]
    output = gate * output + (1.0 - gate) * output_longterm
```

The gate computes a single scalar per token, broadcast across all $D$ features. Both `M_memory` and `M_longterm` compute full forward passes; the gate merely controls the interpolation.

#### 4.2.2 Scaled DGD Update in `_update_memories()`

```python
# After updating the 5 clip-level memories...
if self.use_longterm_memory:
    self_target_lt = self.M_longterm.generate_self_target(v)
    eta_longterm = eta * self.longterm_lr_scale  # 10× slower
    self.M_longterm.compute_and_apply_update(
        key=k, value=self_target_lt,
        error_signal=surprise, lr=eta_longterm, alpha=alpha,
    )
```

$M_\text{longterm}$ uses its own self-generated targets (Eq. 83 from Behrouz 2025), ensuring self-referential learning: the memory decides *what* to learn based on its current state, not the clip-level memories' targets.

#### 4.2.3 Asymmetric Reset in `reset_memory_state()`

```python
# Clip-level: full reset from nn.Parameters
for mem in [self.M_k, self.M_v, self.M_eta, self.M_alpha, self.M_memory]:
    mem.reset_active_weights()

# M_longterm: initialize once, then preserve
if self.use_longterm_memory:
    if self.M_longterm._active_w1 is None:
        self.M_longterm.reset_active_weights()   # First clip
    else:
        self.M_longterm._active_w1 = self.M_longterm._active_w1.detach()
        self.M_longterm._active_w2 = self.M_longterm._active_w2.detach()
```

#### 4.2.4 Optimizer Parameter Groups

M_longterm parameters are classified as Titan parameters and receive the same outer-loop LR scaling (`titan_lr_scale=0.3`) and reduced weight decay (`titan_weight_decay=0.005`):

```python
titan_patterns = {"M_k.", "M_v.", "M_eta.", "M_alpha.", "M_memory.", "M_longterm."}
```

This is correct because the outer-loop optimizer updates the `nn.Parameters` (meta-learned initial state) of all Titan memories identically. The DGD inner-loop learning rate distinction (`longterm_lr_scale`) operates independently in the forward pass.

### 4.3 Backward Compatibility

The feature is entirely gated by `use_longterm_memory: bool` (default `false`). When disabled:

- No `M_longterm` or `longterm_gate` modules are created
- `_titan_forward_chunk()` skips the gated retrieval branch
- `_update_memories()` skips the longterm DGD update
- `reset_memory_state()` has no longterm logic
- All existing Phase 5 tests pass without modification (verified: 37/37)

---

## 5. Experimental Configuration

### 5.1 Phase 6 Config: `cl_ac_hope_phase6.yaml`

| Parameter | Value | Note |
|-----------|-------|------|
| `depth` | 5 | Same as Phase 5 |
| `titan_hidden_multiplier` | 4 | Standard clip-level capacity |
| `use_rope` | false | Temporal info from learnable embeddings |
| `use_spatial_mixing` | true | Phase C cross-token interaction (Phase 5) |
| **`use_longterm_memory`** | **true** | **NEW: enables persistent M_longterm** |
| **`longterm_hidden_multiplier`** | **2** | **Compact slow memory (768 hidden)** |
| **`longterm_lr_scale`** | **0.1** | **10× slower DGD than clip-level** |
| CMS levels | 2.0 / 2.5 / 3.0 | Heterogeneous, same as Phase 5 |
| `chunk_size` | 1 | Per-token Titan updates |
| `optimizer_type` | adamw | Outer-loop optimizer |
| `learning_rate` | 2.5e-4 | Base LR |
| `titan_lr_scale` | 0.3 | Outer-loop LR scaling for all Titan params |
| `max_epochs` (base) | 65 | Standard training length |

### 5.2 CL Pipeline

The CL pipeline remains unchanged from Phase 5:

1. **Base training**: 5000 clips, 65 epochs, 90/10 train/val split
2. **5 sequential physics-shift tasks** (10 epochs each, 1000 clips):
   - Scaling shift
   - Dissipation shift
   - Discretization shift
   - Kinematics shift
   - Compositional OOD
3. **Evaluation**: 100 clips per task, metrics: Avg Error, Plasticity, Forgetting

The key behavioral difference: during CL finetuning (steps 2a–2e), `reset_memory_state()` is called between clips within each task as before, but $M_\text{longterm}$ now **accumulates** DGD updates across all clips and tasks, building a persistent representation of observed physics dynamics.

---

## 6. Hypothesized Effects on CL Metrics

### 6.1 Plasticity (Error on Current Task)

**Expected: Improvement (lower plasticity error)**

$M_\text{memory}$ is reset clean for every clip, ensuring no old-task inertia in the fast adaptation pathway. The learned gate provides an additional adaptation channel: during physics shifts, the gate can increase $g_t$ (favor fresh $M_\text{memory}$) for tokens in regions where the physics changed, while keeping $g_t$ low (favor $M_\text{longterm}$) for tokens in regions where the physics is similar to the long-term average.

### 6.2 Forgetting (Degradation on Previous Tasks)

**Expected: Improvement (lower forgetting)**

$M_\text{longterm}$ preserves the DGD-accumulated representation of all previously seen physics. When re-evaluated on old tasks, even if `nn.Parameters` have drifted under finetuning, the persistent $M_\text{longterm}$ weights still encode accurate representations for those tasks. The gate learns to route toward $M_\text{longterm}$ for familiar patterns, stabilizing predictions.

### 6.3 Key Metrics to Monitor

| Metric | What to Watch | Success Criterion |
|--------|---------------|-------------------|
| `val/loss_jump` | Jump prediction loss | < 0.5 (Phase 5 temporal embeddings break plateau) |
| `val/loss_teacher` | Teacher-forcing loss | Smooth convergence |
| Plasticity | Avg error on current task | ≤ 0.30 (approaching ViT baseline) |
| Forgetting | Degradation on old tasks | ≤ 0.05 (improvement over Phase 4's 0.0646) |
| Avg Error | Overall CL error | < 0.37 (beating naive ViT lower bound convincingly) |
| Gap Closed | (Naive − HOPE) / (Naive − Joint) | > 5% (meaningful improvement) |
| `M_longterm/param_norm` | Weight magnitude of M_longterm | Should grow slowly across tasks |
| `hope/mean_surprise` | Average surprise signal | Should spike at physics shifts, then decrease |

---

## 7. Testing

### 7.1 Test Suite: `tests/test_phase6_longterm_memory.py`

23 tests across 6 test classes, all passing:

| Test Class | # Tests | What It Verifies |
|------------|:-------:|------------------|
| `TestLongtermMemoryConfig` | 4 | Block creates M_longterm when enabled; no M_longterm when disabled; correct hidden dimension from multiplier; all model blocks have M_longterm |
| `TestLongtermGate` | 3 | Gate output in (0,1); initial bias=1.0 produces ~0.73 output; weights zero-initialized |
| `TestLongtermPersistence` | 4 | First `reset_memory_state()` initializes M_longterm; second reset preserves values (detach only); clip-level memories are properly reset; explicit `reset_longterm_memory()` works |
| `TestLongtermDGD` | 2 | M_longterm active weights change after forward (DGD mutates); longterm update magnitude < clip-level (lr_scale=0.1 verified) |
| `TestPhase6Integration` | 8 | Forward pass (teacher-forcing + jump); backward gradients flow through gate and M_longterm; `reset_all_longterm_memories()` works; config summary includes longterm fields; param count increase matches expectation; parameter groups include M_longterm; diagnostics include M_longterm metrics |
| `TestMultiClipPersistence` | 2 | M_longterm accumulates DGD updates across 2 clips (weights differ after clip 2 vs clip 1); clip-level M_memory resets to nn.Parameters while M_longterm persists |

### 7.2 Regression Testing

All 37 existing tests pass without modification:

- `test_hope_gradient_flow.py` — 7 tests (DGD gradient chain, meta-learning, chunking)
- `test_cms_frame_scheduling.py` — 14 tests (frame-aware CMS scheduling)
- `test_phase5_fixes.py` — 16 tests (temporal embeddings, spatial mixing, integration)

**Total: 60/60 tests passing, 0 regressions.**

---

## 8. Relation to Prior Work

### 8.1 Titans: Learning to Memorize at Test Time (Behrouz et al., 2025)

The original Titans paper introduces the Neural Long-Term Memory module as a separate deeper network (3+ layers) with its own surprise-gated DGD and persistent state across segments. Our $M_\text{longterm}$ adapts this concept to the CL setting:

- **Same**: Persistent memory that survives across processing segments (our "clips" ≈ their "segments"), surprise-gated DGD updates, operates in parallel with a fast/short-term memory
- **Different**: Our gating mechanism is a learned linear projection (theirs uses a fixed coarse gate based on surprise magnitude); our DGD scaling factor ($\lambda=0.1$) is a hyperparameter (theirs uses a separate momentum-based update rule); our memory is a 2-layer MLP matching the `TitanMemory` architecture (theirs is a deeper 3-layer network)

### 8.2 Complementary Learning Systems (McClelland et al., 1995)

The dual-memory architecture directly implements the CLS framework: $M_\text{memory}$ serves as the hippocampal fast-learning system (high plasticity, rapid adaptation, short retention), while $M_\text{longterm}$ serves as the neocortical slow-learning system (low plasticity, gradual consolidation, long retention). The learned gate implements the "interplay" mechanism that CLS theory posits as essential for balancing stability and plasticity.

### 8.3 Relation to EWC and SI

Elastic Weight Consolidation (EWC, Kirkpatrick et al., 2017) and Synaptic Intelligence (SI, Zenke et al., 2017) protect important weights via regularization. Our approach is complementary: rather than preventing weight changes, we maintain a **separate memory** that accumulates task-invariant knowledge. These approaches are not mutually exclusive — EWC/SI could be applied to the outer-loop `nn.Parameters` of both $M_\text{memory}$ and $M_\text{longterm}$ for additional protection.

---

## 9. Limitations and Risks

| Risk | Description | Mitigation |
|------|-------------|------------|
| **M_longterm saturation** | Persistent DGD updates may cause weight divergence over many tasks | `longterm_lr_scale=0.1` limits update magnitude; DGD decay factor $\alpha$ provides natural weight regulation |
| **Gate collapse** | Gate may learn a constant value (always clip or always longterm) | Zero-init weights + nonzero bias provide a smooth starting point; per-token conditioning enables spatial differentiation |
| **Computation overhead** | Extra forward pass through M_longterm + gate per chunk | ~15% FLOPs increase (smaller network: mult=2 vs mult=4); negligible wall-clock impact given DGD dominates |
| **Limited memory capacity** | Hidden multiplier 2 may be too small for complex physics | Hyperparameter — can increase to 3 or 4 in future phases |
| **Cross-clip OOM** | Unbounded graph growth if detach fails | Explicit `.detach()` on every `reset_memory_state()` call; tested in `TestLongtermPersistence` |
| **Root Cause 3 unaddressed** | Jump prediction still uses empty clip-level memories (1 frame → 1 chunk) | M_longterm partially mitigates this: even with empty M_memory, longterm provides accumulated context. Full fix requires Hybrid architecture (see Phase 5 docs, Fix 3). |

---

## 10. Running Phase 6

```bash
# Full CL pipeline (base training + 5 CL tasks)
uv run src/cl_train.py experiment=cl_ac_hope_phase6 paths.data_dir=/path/to/clips

# With specific W&B project
uv run src/cl_train.py experiment=cl_ac_hope_phase6 \
    paths.data_dir=/path/to/clips \
    logger.wandb.project=masterthesis_phase6
```

---

## 11. Summary

Phase 6 adds a persistent cross-clip longterm memory ($M_\text{longterm}$) to the AC-HOPE-ViT architecture, implementing the CLS-inspired separation between fast episodic memory (clip-level, reset every clip) and slow consolidated memory (persistent, accumulated via scaled DGD). A learned per-token gate controls the interpolation between these two memory systems. The architectural change adds ~3M parameters (7% overhead) while introducing a qualitatively new capability: in-context knowledge transfer across clips and tasks without relying solely on outer-loop backpropagation.

| Aspect | Phase 5 | Phase 6 |
|--------|---------|---------|
| Temporal signal | ✓ Learnable embeddings | ✓ (inherited) |
| Spatial interaction | ✓ Phase C mixing | ✓ (inherited) |
| Cross-clip memory | ✗ All memories reset | **✓ M_longterm persists** |
| Param count | ~43.5M | ~46.5M |
| Tests | 37 passing | 60 passing (+23 new) |
