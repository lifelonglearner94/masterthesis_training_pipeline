# Phase 8: AC-HOPE-Hybrid-ViT — Attention + Titan Memory + CMS

**Date:** 2026-03-02
**Config:** `configs/experiment/cl_ac_hope_phase8.yaml`
**Param count:** ~48.7M base / ~55.8M with longterm memory
**Builds on:** Root cause analysis from Phase 5, CL pipeline from Phase 6
**Supersedes:** `20260302_phase8_plan_NOT_IMPLEMENTED.md` (Subspace-Aware DGD — abandoned)

---

## 1. Motivation

### 1.1 Why All Previous HOPE Phases Failed

Phases 1–7 of AC-HOPE-ViT **replaced** the ViT baseline's self-attention with Titan memory. After extensive experimentation, three fundamental architectural deficits were identified that no amount of hyperparameter tuning, optimizer changes, or memory enhancements could overcome:

| Root Cause | Severity | Description |
|------------|:--------:|-------------|
| **Token Blindness** | Critical | Titan memory is a per-token MLP: $q \to W_2 \cdot \text{act}(W_1 \cdot q)$. Each token is processed independently — there is **no pairwise token interaction**. Self-attention computes $\text{softmax}(QK^\top/\sqrt{d})V$, giving every token access to every other token. Without this, the model cannot learn spatial relationships between patches. |
| **Shallow depth** | Severe | The original HOPE uses depth=5 (vs ViT's depth=24). The parameter budget was consumed by 5 Titan memories per block ($M_k$, $M_v$, $M_\eta$, $M_\alpha$, $M_\text{memory}$), each containing a 2-layer MLP. With ~43M params spread across 5 blocks × 5 memories, each block has massive width but the network lacks the depth for hierarchical feature extraction. |
| **No RoPE** | Significant | 3D Rotary Position Embeddings (RoPE) require Q·K dot products to inject positional information. Titan memory feeds Q directly into an MLP (no dot product), so RoPE is fundamentally incompatible. Phase 5 added learnable temporal embeddings as a workaround, but these provide only global frame identity — not the fine-grained relative positional encoding that RoPE gives to attention. |

### 1.2 CL Results of Phases 1–7 (All Below ViT Baseline)

| Experiment | Avg Error ↓ | Forgetting ↓ | Plasticity ↓ | Gap Closed |
|------------|----------:|----------:|----------:|----------:|
| ViT Lower Bound (Naive) | 0.3720 | 0.0856 | 0.3007 | 0.0% |
| ViT Upper Bound (Joint) | 0.2502 | 0.0000 | 0.2502 | 100.0% |
| HOPE Phase 1 | 0.3958 | 0.0554 | 0.3496 | -19.5% |
| HOPE Phase 2 | 0.3987 | 0.0481 | 0.3586 | -21.9% |
| HOPE Phase 4 | 0.3711 | 0.0646 | 0.3105 | +0.8% |
| HOPE Phase 6 | 0.3706 | 0.0646 | 0.3174 | +1.1% |

The consistent message: HOPE **cannot beat the ViT baseline** on plasticity (learning quality). The forgetting numbers are comparable or worse. The core problem is not the CL mechanism — it's that removing attention crippled the model's fundamental representational capacity.

### 1.3 The Key Insight

> **Titan memory should AUGMENT attention, not REPLACE it.**

The ViT baseline's strength is token interaction via self-attention. HOPE's strength is in-context DGD adaptation via Titan memory. Phase 8 combines both: attention handles what it does best (spatial/temporal reasoning), Titan handles what it does best (fast in-context adaptation for CL).

---

## 2. Architecture: Hybrid Block Design

### 2.1 Three Approaches Considered

| Approach | Mechanism | Assessment |
|----------|-----------|------------|
| **A**: Hybrid — Interspersed attention and Titan blocks (alternating) | Even blocks: ViT attention; Odd blocks: HOPE Titan | Loses context between alternating blocks |
| **B**: Adapter — Pretrained ViT backbone + lightweight Titan adapter heads | Frozen ViT + trainable projection to Titan | Effective but limited: Titan cannot influence ViT representations |
| **C**: **Unified Hybrid Block** — All three in every block | Attention → Titan → CMS in every block | Best of both worlds: every token gets interaction + adaptation + multi-freq |

**Approach C was selected** because it gives every layer the full capability set while maintaining a clean residual stream.

### 2.2 HybridBlock Architecture

Each of the 12 Hybrid blocks contains three sequential phases applied as residual additions:

```
Input x ─────────────────────┐
                             │
    Phase A: Attention       │
    y = ACRoPEAttention(     │
         LayerNorm(x))       │
    x = x + DropPath(y)   ◄─┘
    ┌────────────────────────┘
    │
    Phase B: Titan Memory    │
    y = TitanForward(        │
         LayerNorm(x))       │
    x = x + DropPath(y)   ◄─┘
    ┌────────────────────────┘
    │
    Phase C: CMS             │
    y = CMS(LayerNorm(x))    │
    x = x + DropPath(y)   ◄─┘
                             │
Output x ◄───────────────────┘
```

#### Phase A: ACRoPEAttention (from ViT baseline)

Standard multi-head self-attention with 3D RoPE (temporal + height + width), frame-causal masking, and action/state token processing. Identical to the ViT baseline's `ACBlock.attn`.

**What this restores vs original HOPE:**
- Pairwise token interaction (fixes Token Blindness)
- 3D Rotary Position Embeddings (fixes No RoPE)
- Content-based retrieval across the full sequence

#### Phase B: Simplified Titan Memory (from HOPE, heavily simplified)

Replaces the original HOPE's 5 Titan memories with a single $M_\text{memory}$ plus simplified learned parameters for $\eta$ and $\alpha$:

**Original HOPE (per block, 5 memories):**
$$k = M_k(x), \quad v = M_v(x), \quad \eta = M_\eta(x), \quad \alpha = M_\alpha(x), \quad o = M_\text{memory}(q)$$

Each $M_\star$ is a full `TitanMemory` (2-layer MLP with DGD). This consumed ~8M params per block × 5 blocks = ~40M params just for Titan.

**Phase 8 Hybrid (per block, 1 memory + 3 projections + 2 scalars):**
$$q = W_q \cdot x, \quad k = W_k \cdot x, \quad v = W_v \cdot x$$
$$\eta = \text{softplus}(\boldsymbol{\eta}_\text{base}), \quad \alpha = \sigma(\boldsymbol{\alpha}_\text{base})$$
$$o = M_\text{memory}(q)$$

Where $W_q, W_k, W_v \in \mathbb{R}^{D \times D}$ are standard linear projections, and $\boldsymbol{\eta}_\text{base}, \boldsymbol{\alpha}_\text{base} \in \mathbb{R}^D$ are learned per-feature parameters.

**Why this simplification is safe:**
- $K, V$ projections get **direct outer-loop gradients** (they're standard `nn.Linear` in the forward path). The original HOPE used Titan memories for K/V only because it replaced all attention projections. With attention retained, there's no need for adaptive K/V — standard projections suffice.
- $\eta, \alpha$ as learned parameters are tuned by the outer optimizer (AdamW) across all training data. The original M_eta/M_alpha provided input-dependent adaptation, but in practice they consumed ~2×D×D params per block with minimal benefit.
- **No auxiliary loss needed.** The original HOPE required `aux_loss` to provide a training signal to $M_k$ and $M_v$ (their gradients were indirect via the DGD chain). With standard projections, the gradient path is direct.

**DGD update rule (same as original HOPE, applied to single $M_\text{memory}$):**

$$\hat{v} = M_\text{memory}.\text{generate\_self\_target}(v)$$
$$s = \|v - M_\text{memory}(q)\|_2 \quad \text{(surprise)}$$
$$M_\text{memory} \leftarrow \alpha \cdot M_\text{memory} - \eta \cdot \nabla_M \text{MSE}(M_\text{memory}(k), \hat{v})$$

#### Phase C: CMS (unchanged from original HOPE)

Multi-frequency MLPs with heterogeneous update periods. Same implementation as `src/models/hope/cms.py`. Default levels:

| Level | Update Period | Hidden Multiplier |
|-------|:---:|:---:|
| fast | 1 (every token) | 2.0 |
| medium | 3 | 2.5 |
| slow | 7 | 3.0 |

### 2.3 Positional Encoding: RoPE + Temporal Embeddings

Both coexist in the hybrid architecture:

| Encoding | Applied where | What it provides |
|----------|:---:|-------------|
| 3D RoPE | Phase A (Attention) | Fine-grained relative positional encoding via Q·K rotation |
| Learnable temporal embeddings | Model level (before blocks) | Global frame identity for all three phases |

The temporal embeddings are added once to the full sequence before the hybrid blocks. RoPE is applied inside each attention layer. This means:
- **Phase A** benefits from both (embeddings in the residual + RoPE in attention)
- **Phase B and C** benefit from the temporal embeddings in the residual stream

### 2.4 Longterm Memory (for CL)

Same design principle as Phase 6, but applied to the simplified single-memory architecture:

- $M_\text{longterm}$: A second `TitanMemory` per block, **never reset** between clips
- Learned gate: $g = \sigma(W_g \cdot q + b_g)$ interpolates between clip-level and longterm output
- DGD update rate: $\eta_\text{lt} = \eta \times \lambda$ where $\lambda = 0.1$ (10× slower)

Reset semantics (same as Phase 6):
- `reset_all_memories()`: Resets $M_\text{memory}$ (clip-level), **preserves** $M_\text{longterm}$
- `reset_all_longterm_memories()`: Resets $M_\text{longterm}$ (full reset)

---

## 3. Parameter Budget

### 3.1 Without Longterm Memory

| Component | Parameters |
|-----------|----------:|
| Titan (M_memory + projections + η/α per block × 12) | 14,178,816 |
| CMS (3 levels per block × 12) | 26,618,112 |
| Projections (embed, decode, encoders, norms, embeddings) | 7,923,328 |
| **Total** | **48,720,256** |

### 3.2 With Longterm Memory (CL config)

| Component | Parameters |
|-----------|----------:|
| Titan (M_memory + M_longterm + projections + gates + η/α × 12) | 21,270,540 |
| CMS (3 levels per block × 12) | 26,618,112 |
| Projections (embed, decode, encoders, norms, embeddings) | 7,923,328 |
| **Total** | **55,811,980** |

### 3.3 Comparison

| Model | Depth | Titan Memories/Block | Total Params |
|-------|:---:|:---:|----------:|
| ViT Baseline | 24 | 0 | ~43M |
| HOPE Phase 6 | 5 | 5 (+longterm) | ~46.5M |
| **Hybrid Phase 8** | 12 | 1 (+longterm) | **~55.8M** |
| Hybrid Phase 8 (no longterm) | 12 | 1 | ~48.7M |

The hybrid is ~12.8M larger than the ViT baseline. This is because each block now contains attention **plus** Titan **plus** CMS (instead of only attention+MLP or only Titan+CMS). The extra capacity is allocated to CL-specific mechanisms (Titan memory, CMS multi-frequency) that the baseline lacks entirely.

---

## 4. Key Differences: Hybrid vs Original HOPE

| Aspect | Original HOPE (Phase 1–7) | Hybrid (Phase 8) |
|--------|:---:|:---:|
| Self-attention | ✗ Replaced by Titan | ✓ Retained (ACRoPEAttention) |
| 3D RoPE | ✗ Incompatible | ✓ Works in attention |
| Depth | 5 blocks | 12 blocks |
| Titan memories per block | 5 ($M_k, M_v, M_\eta, M_\alpha, M_\text{memory}$) | 1 ($M_\text{memory}$) |
| η (DGD learning rate) | Full TitanMemory MLP | Learned `nn.Parameter` vector |
| α (DGD decay) | Full TitanMemory MLP | Learned `nn.Parameter` vector |
| K/V for DGD | From Titan memories ($M_k, M_v$) | From `nn.Linear` projections |
| Auxiliary loss | Required (train $M_k, M_v$) | Not needed (projections have direct gradients) |
| Token interaction | ✗ Per-token only | ✓ Full pairwise via attention |
| CMS | ✓ Same | ✓ Same |
| Longterm memory | ✓ 6th Titan memory | ✓ 2nd Titan memory per block |

---

## 5. Implementation

### 5.1 New Files

| File | Class/Function | Role |
|------|---------------|------|
| `src/models/hope/hybrid_block.py` | `HybridBlockConfig`, `HybridBlock` | Single hybrid block (Attn + Titan + CMS) |
| `src/models/hope/ac_hope_hybrid_vit.py` | `ACHOPEHybridViT`, `ac_hope_hybrid_vit()` | Full 3-stage model |
| `src/models/hope/ac_hope_hybrid_module.py` | `ACHOPEHybridModule` | Lightning module (train/val/test/TTA/curriculum) |
| `configs/model/ac_hope_hybrid_vit.yaml` | — | Model config defaults |
| `configs/experiment/cl_ac_hope_phase8.yaml` | — | CL experiment config |
| `tests/test_phase8_hybrid.py` | 34 tests | Full test suite |

### 5.2 Modified Files

| File | Change |
|------|--------|
| `src/models/hope/__init__.py` | Added exports for hybrid classes |

### 5.3 Dependency Graph

```
ACHOPEHybridModule (Lightning)
    └── ACHOPEHybridViT (nn.Module)
            ├── HybridBlock × 12
            │       ├── ACRoPEAttention  (from ac_predictor.utils.modules)
            │       ├── TitanMemory      (from hope.titan_memory — reused)
            │       ├── CMS              (from hope.cms — reused)
            │       ├── η_base, α_base   (nn.Parameter — new)
            │       └── M_longterm       (TitanMemory — optional)
            ├── predictor_embed / predictor_proj  (nn.Linear)
            ├── action_encoder / state_encoder    (nn.Linear)
            └── frame_pos_embed / target_pos_embed (nn.Embedding)
```

### 5.4 CL Pipeline Compatibility

The hybrid model is fully compatible with `src/cl_train.py`:

- **Detection:** `cl_train.py` detects HOPE models via `"hope" in cfg.get("task_name", "")`. The Phase 8 task name is `cl_ac_hope_phase8` → detected as HOPE model.
- **Memory management:** `reset_all_memories()`, `freeze_all_inner_loops()`, `unfreeze_all_inner_loops()` are all implemented with identical signatures.
- **Optimizer groups:** `get_parameter_groups()` returns 3 groups (titan, cms, projections) with per-group LR scaling, matching the existing optimizer setup.

---

## 6. Gradient Flow Analysis

### 6.1 η/α Gradient Chain

The learned $\eta$ and $\alpha$ parameters only receive gradients through a **2-step DGD chain** (same principle as the original HOPE's $M_\eta$/$M_\alpha$ with chunking):

1. **Forward 1:** Read from $M_\text{memory}$ → compute surprise → apply DGD update using $\eta, \alpha$
2. **Forward 2:** Read from updated $M_\text{memory}$ → the output now depends on $\eta, \alpha$ through the update rule → backward propagates gradients to $\eta_\text{base}, \alpha_\text{base}$

In single-forward training (standard teacher forcing), $\eta/\alpha$ get gradients because within one forward pass, the DGD update modifies the memory state which is immediately read again in subsequent steps of the same sequence.

### 6.2 Verified Gradient Paths (from test suite)

| Component | Gradient in 1 forward | Why |
|-----------|:---:|-------------|
| Attention (QKV, proj) | ✓ | Direct path: input → attention → output → loss |
| $M_\text{memory}$ (w1, w2) | ✓ | Direct path: query → memory read → output → loss |
| CMS (fc1, fc2) | ✓ | Direct path: normed input → CMS → output → loss |
| $\eta_\text{base}$ | ✓ (2-step) | Through DGD update chain: update → next read → loss |
| $\alpha_\text{base}$ | ✓ (2-step) | Through DGD update chain: update → next read → loss |
| Embedding/Projection layers | ✓ | Direct path through model |

---

## 7. Configuration

### 7.1 Key Hyperparameters

```yaml
# Architecture
depth: 12                        # 12 hybrid blocks (vs 5 HOPE / 24 ViT)
num_heads: 16                    # Same as ViT baseline
predictor_embed_dim: 384         # Same as ViT baseline

# Titan memory (compact)
titan_hidden_multiplier: 2       # Halved from HOPE's 4 (attention handles the rest)
titan_layers: 2                  # Same as HOPE
titan_detach_interval: 1         # Detach graph every step (VRAM safety)
surprise_threshold: 0.1          # Only update on surprising inputs

# Longterm memory (for CL)
use_longterm_memory: true
longterm_hidden_multiplier: 2
longterm_lr_scale: 0.1           # 10× slower DGD updates

# Optimizer
learning_rate: 2.5e-4
titan_lr_scale: 0.3              # Titan params at 30% of base LR
cms_lr_scale: 1.0
aux_loss_weight: 0.0             # Not needed for hybrid
```

### 7.2 Usage

```bash
# Run CL experiment
uv run src/cl_train.py experiment=cl_ac_hope_phase8 paths.data_dir=/path/to/clips

# Override settings
uv run src/cl_train.py experiment=cl_ac_hope_phase8 \
    model.depth=8 \
    model.use_longterm_memory=false \
    model.titan_hidden_multiplier=4
```

---

## 8. Test Coverage

34 tests in `tests/test_phase8_hybrid.py`:

| Test Class | Count | What it verifies |
|------------|:---:|-------------|
| `TestHybridBlock` | 8 | Block construction, attention/titan/CMS presence, η/α parameters, longterm option, memory reset |
| `TestACHOPEHybridViT` | 5 | Model construction, forward shape, determinism, target_timestep, batch>1 |
| `TestMemoryManagement` | 5 | reset_all, reset_longterm, freeze/unfreeze, frozen forward |
| `TestGradientFlow` | 5 | Attention grads, Titan grads, CMS grads, η/α grads (2-step), embedding grads |
| `TestParameterGroups` | 3 | 3 groups, all params covered, no duplicates |
| `TestDiagnostics` | 3 | Diagnostics dict, config summary, aux_loss=0 |
| `TestLongtermMemory` | 2 | Forward with longterm, selective reset |
| `TestLightningModule` | 3 | Import, construction, _step_predictor method |

---

## 9. Design Rationale: Why Not the Planned Phase 8 (Subspace-Aware DGD)?

The original Phase 8 plan (`20260302_phase8_plan_NOT_IMPLEMENTED.md`) proposed Soft Orthogonal Projection + EWC to address forgetting on highly correlated tasks. This plan was **abandoned** in favour of the hybrid architecture because:

1. **Symptom vs root cause:** Subspace-Aware DGD would improve forgetting on a model that already can't learn well (plasticity ≈ 0.32 vs ViT's 0.30). Improving CL properties on a fundamentally weaker learner is futile — the model first needs to match the baseline's learning capacity.

2. **Prioritization:** The immediate bottleneck is plasticity (learning quality), not forgetting. The hybrid architecture addresses the plasticity bottleneck by restoring attention. Once the model can learn as well as the ViT baseline, CL improvements (including Subspace-Aware DGD) become meaningful and can be added as a future Phase 9.

3. **Orthogonality risk:** As the original plan noted, with >95% feature overlap between tasks, orthogonal projection removes gradient components that may be necessary for new-task learning. Adding orthogonal constraints to an already weak model would further degrade plasticity.

---

## 10. Expected Outcomes

### 10.1 What Should Improve

| Metric | Why |
|--------|-----|
| **Plasticity** | Attention restores token interaction → richer representations → better per-task learning. This is the primary improvement target. |
| **Base training loss** | Deeper network (12 vs 5) + attention + RoPE → should match or approach ViT baseline's training loss. |
| **Forward transfer** | Richer representations transfer better between similar tasks. |

### 10.2 What May Not Improve (Yet)

| Metric | Why |
|--------|-----|
| **Forgetting** | Longterm memory is present but simplified (1 memory vs 6). If forgetting remains high, Phase 7 enhancements (retrieval-conditioned gate, asymmetric decay, own surprise, EMA consolidation) can be ported to the hybrid architecture. |
| **VRAM usage** | 12 blocks × (attention + titan + CMS) will use more VRAM than 5 blocks × (titan + CMS). The `titan_detach_interval: 1` and `use_activation_checkpointing: false` settings can be adjusted. |

### 10.3 Success Criteria

Phase 8 is successful if:
1. **Plasticity ≤ 0.30** (matching ViT baseline's learning quality)
2. **Avg Error < 0.37** (beating the ViT lower bound)
3. **Gap Closed > 5%** (meaningful CL improvement over naive finetuning)
