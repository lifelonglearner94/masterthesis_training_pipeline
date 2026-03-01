# Phase 5: Architectural Root Cause Analysis & Fixes

**Date:** 2026-03-01
**Config:** `configs/experiment/cl_ac_hope_phase5.yaml`
**Param count:** ~43.6M (target: 43M ViT baseline, Δ=+0.6M)

---

## Background: Why All 4 Previous HOPE Phases Failed

After Phase 4 showed jump loss plateauing at ~0.5 ("SEHR ENTTÄUSCHEND"), a deep architectural analysis of the entire HOPE codebase was performed. The diagnosis revealed that Phases 1–4 were fighting **symptoms** (hyperparameters, optimizer, CMS scheduling) while three **fundamental architectural deficits** remained unaddressed:

| Phase | What it tried | Why it failed |
|-------|--------------|---------------|
| Phase 1 | Initial HOPE architecture | All 3 root causes present |
| Phase 2 | Gradient flow fixes (NaN defense, aux loss detach) | Fixed training stability but not representational capacity |
| Phase 3 | Disabled RoPE, param matching | Correctly removed RoPE but created Root Cause 1 (no temporal signal at all) |
| Phase 4 | Frame-aware CMS + M3 Muon optimizer | Fixed CMS scheduling bug but both root causes 1 & 2 still present |

**Current results (all worse than the naive ViT lower bound):**

| Experiment | Avg Error ↓ | Forgetting ↓ | Plasticity ↓ | Gap Closed |
|------------|----------:|----------:|----------:|----------:|
| ViT Lower Bound (Naive) | 0.3720 | 0.0856 | 0.3007 | 0.0% |
| ViT Upper Bound (Joint) | 0.2502 | 0.0000 | 0.2502 | 100.0% |
| HOPE Phase 1 | 0.3958 | 0.0554 | 0.3496 | -19.5% |
| HOPE Phase 2 | 0.3987 | 0.0481 | 0.3586 | -21.9% |

---

## Root Cause Analysis

### Root Cause 1 (CRITICAL): Jump Prediction Has No Target Signal

**The Problem:**
Phase 3 correctly disabled RoPE (`use_rope: false`) because RoPE requires Q·K dot products, but HOPE feeds Q→MLP independently (no dot product). However, this removed **ALL** temporal position information from the model. The `target_timestep` parameter was only used inside `_apply_rope()`, which was disabled.

**Consequence:**
In jump prediction mode, the model receives frame 0 and must predict frame τ ∈ {1, 2, ..., T}. But with no temporal signal, it cannot distinguish *which* future frame to predict. The optimal strategy becomes predicting the **average of all possible future frames** — explaining the loss plateau at exactly ~0.5.

**Why previous phases didn't catch this:**
The `target_timestep` parameter was still *passed* through the full call chain (`forward() → _process_hope_blocks() → _apply_rope()`), so the code *looked* correct. Only when you trace the data flow with `use_rope=false` do you discover that `target_timestep` is completely ignored.

### Root Cause 2 (FUNDAMENTAL): Zero Spatial Interaction Between Tokens

**The Problem:**
HOPE processes every token **independently** through per-token MLPs:
- Titan memories: Q→MLP, K→MLP, V→MLP (all per-token)
- CMS: per-token multi-frequency MLPs

Meanwhile, the ViT baseline has **24 layers of self-attention**, where every token attends to every other token. HOPE has literally zero cross-token spatial interaction.

**Consequence:**
Poor plasticity (0.35–0.39 vs ViT's 0.30). The model cannot learn spatial relationships between patches, severely limiting representational capacity despite having ~43M parameters.

### Root Cause 3 (STRUCTURAL): Jump Prediction Uses Empty Memories

**The Problem:**
Jump prediction feeds only **1 frame** (frame 0) through the model. With `chunk_size=1` (258 tokens), the Titan memory sees exactly one chunk before producing output. The DGD memory update hasn't accumulated any experience yet — the memories are essentially empty.

This means HOPE's entire value proposition (in-context memory that adapts across the sequence) is wasted for the jump prediction task.

**Status:** NOT fixed in Phase 5. See [Section: Suggested Fix 3](#fix-3-hybrid-hope--transformer-architecture-future-work) below.

---

## Fix 1: Learnable Temporal Embeddings

### Approach

Two additive learned embeddings that inject temporal position information **before** the HOPE blocks:

1. **`frame_pos_embed`** — `nn.Embedding(T+2, D=384)`: Tells each token which input frame it belongs to
2. **`target_pos_embed`** — `nn.Embedding(T+2, D=384)`: Tells the model which future frame to predict

### How It Works

**Jump prediction mode** (`target_timestep is not None`):
```python
# All input tokens are from frame 0
x = x + frame_pos_embed(0)           # broadcast [1, 1, D] → [B, N, D]
# Tell model which future frame to predict
x = x + target_pos_embed(τ)          # broadcast [1, 1, D] → [B, N, D]
```

**Teacher-forcing mode** (`target_timestep is None`):
```python
# Each frame's tokens get a per-frame embedding
frame_indices = torch.arange(T)       # [0, 1, 2, ..., T-1]
frame_embs = frame_pos_embed(frame_indices)  # [T, D]
# Expand to all tokens in each frame and add
x = x + expand_per_frame(frame_embs)  # [B, T*tpf, D]
# (target_pos_embed not used — no specific target frame)
```

### Why This Fixes the Jump Loss Plateau

With the target embedding, predicting frame τ=1 vs τ=5 produces **different** input representations, allowing the model to learn frame-specific predictions instead of an average.

### Files Modified

| File | Change |
|------|--------|
| `src/models/hope/ac_hope_vit.py` | Added `self.frame_pos_embed` and `self.target_pos_embed` in `__init__()`. Temporal embedding injection in `_process_hope_blocks()` before the block loop. |

### Parameter Cost

- `frame_pos_embed`: 9 × 384 = 3,456 params
- `target_pos_embed`: 9 × 384 = 3,456 params
- **Total: 6,912 params** (negligible, <0.02% of total)

---

## Fix 2: Spatial Mixing (Phase C)

### Approach

MLP-Mixer-style per-frame token mixing, added as **Phase C** in every `HOPEBlock`:

```
Phase A: Self-Modifying Titan (per-token memory read/write)
Phase B: CMS multi-frequency MLPs (per-token frequency decomposition)
Phase C: Spatial Mixing (NEW — cross-token interaction within each frame)
```

### Architecture

For each frame independently:
```
input: [B, T, 258, D]   (258 = 256 patches + 2 conditioning tokens)

y = LayerNorm(x)
y = y.permute(0,1,3,2)   # transpose: [B, T, D, 258]
y = Linear(258→258)       # mix tokens (no bias)
y = GELU()
y = Linear(258→258)       # project back (no bias, ZERO-INITIALIZED)
y = y.permute(0,1,3,2)   # back to [B, T, 258, D]
x = x + DropPath(y)       # residual
```

### Key Design Decisions

1. **Per-frame mixing** (not global): Each frame's 258 tokens are mixed independently. This preserves the temporal structure that HOPE's memory system needs — tokens from different frames should NOT be mixed here (that's the memory's job).

2. **Zero-initialized output layer**: `nn.init.zeros_(spatial_mix[2].weight)` ensures the spatial mixing residual starts at exactly zero. At initialization, the block behaves identically to Phase 4 — the mixing gradually learns to contribute during training.

3. **No bias terms**: Both linear layers use `bias=False` to reduce parameters and prevent offset artifacts.

4. **Configurable**: Controlled by `use_spatial_mixing: bool` in config. Default is `false` for backward compatibility.

### Files Modified

| File | Change |
|------|--------|
| `src/models/hope/hope_block.py` | `HOPEBlockConfig`: added `use_spatial_mixing`, `spatial_mixing_tokens`. `HOPEBlock.__init__()`: added `norm3` + `spatial_mix` layers. `HOPEBlock.forward()`: added Phase C after Phase B. |
| `src/models/hope/ac_hope_vit.py` | Computes `spatial_mixing_tokens` from architecture params, passes to `HOPEBlockConfig`. |
| `src/models/hope/ac_hope_module.py` | Wires `use_spatial_mixing` parameter through to model constructor. |
| `configs/model/ac_hope_vit.yaml` | Added `use_spatial_mixing: false` default. |

### Parameter Cost

Per block: 2 × (258 × 258) = 133,128 params
5 blocks × 133,128 = **665,640 params** (+1.5% of total)

---

## Phase 5 Config Summary

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `depth` | 5 | Memory-efficient: fewer larger blocks |
| `titan_hidden_multiplier` | 4 | Standard 4× expansion |
| `use_rope` | false | No Q·K dot product in HOPE |
| `use_spatial_mixing` | **true** | **Fix 2** — cross-token interaction |
| CMS levels | 2.0 / 2.5 / 3.0 | Heterogeneous capacity across frequencies |
| `optimizer_type` | adamw | Simpler than M3 Muon for debugging new architecture |
| `learning_rate` | 2.5e-4 | Conservative base LR |
| `titan_lr_scale` | 0.3 | Slow inner-loop updates |
| `max_epochs` (base) | 65 | Standard training length |
| `batch_size` | 16 | Memory-compatible |
| `chunk_size` | 1 | Per-token Titan updates |
| Total params | ~43.6M | Matched to ViT baseline (43M) |

### What to Watch For

- **`val/loss_jump`**: Should drop well below 0.5 (the temporal embedding fix breaks the plateau)
- **`val/loss_teacher`**: Should show smoother convergence (spatial mixing adds expressivity)
- **Plasticity**: Target is ≤0.30 (matching ViT baseline, down from 0.35–0.39)
- **Forgetting**: Should remain low (HOPE memory mechanisms preserved)

---

## Tests

16 new tests in `tests/test_phase5_fixes.py`:

| Test Class | # Tests | What it verifies |
|------------|---------|------------------|
| `TestTemporalEmbeddings` | 5 | Embeddings exist with correct shapes; different target timesteps produce different outputs; teacher mode uses frame positions; jump mode output shape; embeddings receive gradients |
| `TestSpatialMixing` | 6 | Config creates spatial mixing layers; block has norm3 + spatial_mix; block without mixing has no layers; zero-init verified; output shape preserved; gradients flow through mixing |
| `TestPhase5Integration` | 5 | Full forward pass (teacher-forcing); full forward pass (jump); backward pass verifies all params get gradients (with known exclusions); config summary includes spatial_mixing; param count is reasonable |

All 37 tests pass (16 new + 21 existing). Zero regressions.

---

## Fix 3: Hybrid HOPE + Transformer Architecture (Future Work)

> **This fix was NOT implemented in Phase 5** but addresses the remaining Root Cause 3 and offers the most promising path to significantly exceeding the ViT baseline.

### The Core Insight

HOPE's Self-Modifying Titan memories are fundamentally designed for **sequential** processing — they accumulate experience across a sequence and adapt their behavior based on what they've seen. But in our CL setup:

- **Teacher-forcing** (T=7 frames, 1806 tokens): Titan sees a meaningful sequence and can build up memory. This is where HOPE should shine.
- **Jump prediction** (T=1 frame, 258 tokens): Titan sees exactly **one** chunk. The memories haven't learned anything yet — they're empty. The entire adaptive capability of HOPE is wasted.

Meanwhile, standard self-attention is **instantly** powerful: it computes pairwise token interactions without needing sequential buildup. A single attention layer on 258 tokens immediately captures spatial relationships that HOPE's per-token MLPs cannot.

### Proposed Architecture: Interleaved HOPE + Transformer

Replace the homogeneous stack of HOPE blocks with a **hybrid** architecture that plays to each component's strengths:

```
Layer 1:  Transformer Block (self-attention + FFN)    ← instant spatial reasoning
Layer 2:  HOPE Block (Titan + CMS + Spatial Mixing)   ← temporal memory
Layer 3:  Transformer Block                           ← spatial refinement
Layer 4:  HOPE Block                                  ← multi-frequency memory
Layer 5:  Transformer Block                           ← final spatial integration
```

**Alternating pattern: Transformer → HOPE → Transformer → HOPE → Transformer**

### Why This Should Work

1. **Jump prediction becomes viable**: The Transformer layers provide immediate spatial reasoning even with just 1 frame, compensating for HOPE's empty memories.

2. **Teacher-forcing gets the best of both worlds**: Transformer layers handle spatial interaction while HOPE layers handle temporal adaptation and memory.

3. **Gradient flow improves**: Standard attention has well-understood gradient properties. Alternating with HOPE blocks provides "gradient highways" through the network, reducing the risk of vanishing/exploding gradients in the Titan inner loop.

4. **Parameter efficiency**: Standard attention ($4D^2$ per layer for Q/K/V/O projections) is cheaper per-layer than Titan (5 memory networks), allowing more layers within the same param budget.

### Architecture Sizing

Rough parameter calculation for a 7-layer hybrid (4 Transformer + 3 HOPE):

| Component | Per-Layer Params | Count | Total |
|-----------|----------------:|------:|------:|
| Transformer (D=384, 6 heads) | ~1.2M | 4 | ~4.8M |
| HOPE (D=384, titan_mult=3, CMS 2/2.5/3) | ~8.5M | 3 | ~25.5M |
| Input/Output projections | — | — | ~2.5M |
| Temporal embeddings | — | — | ~0.007M |
| Spatial mixing (HOPE layers only) | ~0.13M | 3 | ~0.4M |
| **Total** | | | **~33.2M** |

This is **under** the 43M budget, leaving room to either:
- Increase HOPE depth (4 HOPE + 4 Transformer = 8 layers, ~43M)
- Increase `titan_hidden_multiplier` for richer memories
- Add more Transformer layers for better spatial reasoning

### Implementation Plan

1. **Create `HybridBlock` wrapper** that accepts a `block_type: "transformer" | "hope"` config
2. **Reuse existing `HOPEBlock`** for HOPE layers (with spatial mixing)
3. **Implement lightweight `TransformerBlock`** using standard multi-head self-attention (can reuse PyTorch's `nn.MultiheadAttention` or port from the original AC-ViT predictor)
4. **New config parameter**: `block_pattern: list[str]` e.g. `["T", "H", "T", "H", "T"]`
5. **Update `_process_hope_blocks()`** to iterate over mixed blocks, resetting only HOPE memories

### Key Design Considerations

- **Memory reset**: Only HOPE blocks have memory state. Transformer blocks are stateless. The `reset_memories()` call should only target HOPE blocks.
- **Temporal embeddings**: Applied once at the input (before all blocks), same as Phase 5. Both block types benefit.
- **Spatial mixing**: Only needed in HOPE blocks. Transformer blocks already have self-attention for spatial interaction.
- **CMS scheduling**: Only applies to HOPE blocks. Transformer FFN layers process all tokens uniformly.
- **Causal masking**: Apply to Transformer attention layers to maintain temporal causality. HOPE blocks are inherently causal (memory only reads past).

### Expected Benefits

| Metric | Phase 5 Expected | Hybrid Expected | Reasoning |
|--------|:---------:|:--------:|-----------|
| Jump Loss | <0.5 (fixed plateau) | <0.35 | Attention layers provide instant spatial reasoning |
| Plasticity | ~0.32 | ~0.28 | Better representational capacity from attention |
| Forgetting | ~0.05 | ~0.04 | HOPE memories still provide stability |
| Gap Closed | >0% | >30% | First time potentially beating naive baseline significantly |

### Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Attention layers increase memory usage | Use Flash Attention (native in PyTorch 2.x) |
| Two different param groups complicate optimization | Already solved: existing LR group system (titan_lr_scale) easily extends to transformer groups |
| Hybrid architecture harder to debug | Implement incrementally: first pure Transformer baseline to verify, then add HOPE layers one by one |
| CL forgetting increases (attention has no memory protection) | HOPE blocks still provide the stability mechanism; Transformer blocks may need small weight regularization (EWC-lite) |

### Why Not Just Use a Pure Transformer?

The pure ViT baseline already exists (24 layers, 43M params, Avg Error 0.3720 naive / 0.2502 joint). The whole point of this thesis is to test whether **HOPE's adaptive memory** can improve continual learning. The hybrid approach tests a more nuanced hypothesis:

> *HOPE's temporal memory provides CL stability benefits, but needs the "instant spatial reasoning" of attention to match the ViT's representational capacity.*

If the hybrid works, it demonstrates that:
1. Self-modifying memories genuinely help with forgetting
2. The right architecture needs both instant (attention) and adaptive (memory) processing
3. The key is knowing **where** in the network to place each type of computation

This is a much more interesting thesis result than "HOPE doesn't work" or "just use a Transformer."

---

## Summary of All Changes (Phase 5)

| File | Lines Changed | What |
|------|:------------:|------|
| `src/models/hope/ac_hope_vit.py` | ~50 | Temporal embeddings + spatial mixing wiring |
| `src/models/hope/hope_block.py` | ~30 | Phase C spatial mixing layers + config |
| `src/models/hope/ac_hope_module.py` | ~5 | `use_spatial_mixing` param passthrough |
| `configs/model/ac_hope_vit.yaml` | 1 | Default `use_spatial_mixing: false` |
| `configs/experiment/cl_ac_hope_phase5.yaml` | 200 | NEW — full experiment config |
| `tests/test_phase5_fixes.py` | ~350 | NEW — 16 unit/integration tests |
| `tmp/param_calc.py` | ~50 | NEW — architecture search calculator |
