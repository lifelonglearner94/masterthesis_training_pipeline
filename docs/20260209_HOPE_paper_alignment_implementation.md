# HOPE Paper Alignment — Implementation Protocol

**Date:** 2026-02-09
**Scope:** Aligning the AC-HOPE-ViT implementation with the Titans/TTT paper (Behrouz 2025, Section 8)
**Affected experiments:** `ac_hope_vit_param_matched`, `ac_hope_vit_depth_matched`

---

## Motivation

The existing HOPE implementation had solid foundations (memory MLP structure, self-targets, chunked processing, meta-learned inits) but diverged from the paper in six concrete ways — from critical (dead M_k/M_v parameters) to medium (missing DGD preconditioner, scalar η/α, disabled test-time memory updates). This document records each change, its rationale from the paper, the code locations affected, and how it was validated.

---

## Change 1 — Fix M_k / M_v Gradient Flow (CRITICAL)

### Problem

Under first-order meta-learning (FOMAML), `M_k` and `M_v` outputs are only consumed inside the inner-loop gradient computation where leaf weights are detached. No gradient from the outer loss flows back to M_k/M_v nn.Parameters. The gradient flow tests explicitly confirmed `p.grad is None` for both memories — meaning **~22% of Titan parameters were dead weight** (random initialization, never updated by the optimizer).

### Paper Reference

All six memories $\mathcal{M}_\Box$ for $\Box \in \{k, v, q, \eta, \alpha, \text{memory}\}$ should have meta-learned initial states updated by the outer optimizer (Section 8.1, nested learning Pillar 3).

### Solution — Auxiliary Retrieval-Quality Loss

Added a small auxiliary loss that creates differentiable paths through M_k and M_v:

$$\mathcal{L}_\text{aux} = \text{MSE}(\mathcal{M}_\text{memory}(k), \text{sg}(v)) + \text{MSE}(\text{sg}(\mathcal{M}_\text{memory}(k)), v)$$

where $k = \mathcal{M}_k(x)$ and $v = \mathcal{M}_v(x)$ are live tensors in the computation graph, and $\text{sg}(\cdot)$ is stop-gradient. The first term trains M_k (how well its keys retrieve from memory), the second trains M_v (how well its values serve as retrieval targets).

This is weighted by `aux_loss_weight` (default 0.1) and added to the outer loss in `training_step()`.

### Files Changed

| File | What |
|---|---|
| `src/models/hope/hope_block.py` | `_update_memories()` computes aux loss; `_aux_loss` field added; `reset_memory_state()` resets it |
| `src/models/hope/ac_hope_vit.py` | `get_aux_loss()` aggregates across all blocks |
| `src/models/hope/ac_hope_module.py` | `training_step()` adds `aux_loss_weight * aux_loss` to outer loss; new `aux_loss_weight` hyperparameter |
| `configs/model/ac_hope_vit.yaml` | Added `aux_loss_weight: 0.1` |
| `configs/experiment/ac_hope_vit_param_matched.yaml` | Added `aux_loss_weight: 0.1` |
| `configs/experiment/ac_hope_vit_depth_matched.yaml` | Added `aux_loss_weight: 0.1` |
| `tests/test_hope_gradient_flow.py` | Updated `test_auxiliary_memory_grads_with_chunking`: M_k/M_v now expected to have gradients |

### Validation

Test `test_auxiliary_memory_grads_with_chunking` now asserts `p.grad is not None` for all five memories (M_memory, M_eta, M_alpha, M_k, M_v). Passes.

---

## Change 2 — Implement DGD Preconditioner (Eq. 93)

### Problem

The paper's core DGD update rule is:

$$\mathcal{M}_t = \mathcal{M}_{t-1}(\alpha_t I - \eta_t k_t k_t^\top) - \eta_t \nabla\mathcal{L}$$

The implementation used a simplified form:

```python
new_w = alpha * w_old - eta * grad  # Missing k_t k_t^T preconditioner
```

The $k_t k_t^\top$ term is the key DGD innovation — it decorrelates correlated token updates by scaling the weight decay proportionally to key magnitude per feature dimension.

### Paper Reference

Eq. 93, Section 8.1 (Behrouz 2025). The preconditioner is derived from the DGD objective for linear associative memories, adapted here as a diagonal approximation for nonlinear MLPs.

### Solution — Diagonal Approximation

For the nonlinear 2-layer MLP, the full $k_t k_t^\top$ outer product doesn't have a clean matrix form. We use a diagonal approximation:

$$k_\text{sq} = \text{mean}_{B,N}(k^2) \in \mathbb{R}^D$$

Then for each weight matrix:
- `w1 [hidden, D]`: preconditioner $(\alpha - \eta \cdot k_\text{sq})$ scales columns (input dimension)
- `w2 [D, hidden]`: preconditioner scales rows (output dimension)

```python
precond = alpha_feat - lr_feat * k_sq  # [D] per-feature
new_w = w_old * precond.unsqueeze(0) - lr_feat.unsqueeze(0) * grad  # w1
new_w = w_old * precond.unsqueeze(1) - lr_feat.unsqueeze(1) * grad  # w2
```

### Files Changed

| File | What |
|---|---|
| `src/models/hope/titan_memory.py` | `compute_and_apply_update()` — full DGD update with diagonal k²  preconditioner |

---

## Change 3 — Promote η and α from Scalar to Per-Feature Vectors (Eq. 88)

### Problem

The paper specifies per-token, per-feature adaptation:

$$\eta_t, \alpha_t \in \mathbb{R}^d$$

The implementation collapsed both to **single scalars** via `.mean(dim=-1, keepdim=True)` followed by `.mean()` in the update. This eliminated per-feature adaptation granularity.

### Paper Reference

Eq. 88, Section 8.1: "per-token η and α" with full dimensionality $d$.

### Solution

- In `hope_block.py`: Removed `.mean(dim=-1, keepdim=True)` — η and α stay as `[B, C, D]` tensors.
- In `titan_memory.py`: Average over batch and tokens to get `[D]` vectors (not scalars), then broadcast appropriately against weight matrices.

```python
# Before (scalar):
eta = F.softplus(eta_raw.mean(dim=-1, keepdim=True)) * 0.01  # [B, C, 1]
lr_mean = lr.mean()  # scalar

# After (per-feature vector):
eta = F.softplus(eta_raw) * 0.01  # [B, C, D]
lr_feat = lr.mean(dim=(0, 1))    # [D]
```

### Files Changed

| File | What |
|---|---|
| `src/models/hope/hope_block.py` | `_titan_forward_chunk()` — η/α kept as `[B, C, D]` |
| `src/models/hope/titan_memory.py` | `compute_and_apply_update()` — accepts `[B, N, D]`, averages to `[D]` per-feature vectors |

---

## Change 4 — Enable Memory Updates During Inference

### Problem

Memory updates were gated by `if self.training and torch.is_grad_enabled()`, completely disabling DGD self-modification at test time. The Titans paper explicitly states:

> "There is no distinction between training and test time."

This meant the core HOPE advantage — online in-context memory adaptation — was lost during inference. Memories used fixed initial states with no adaptation.

### Paper Reference

Section 8.1, paragraph on nested learning: the inner loop runs continuously. The memory adapts to each new sequence regardless of train/test mode.

### Solution

Changed the guard to:

```python
if torch.is_grad_enabled() or not self.training:
```

This enables DGD memory updates in three scenarios:
1. **Training with grad**: normal training (computes aux loss too)
2. **Eval without grad** (`torch.no_grad()`): inference — memories still self-modify, just no gradient tracking
3. **Eval with grad**: TTA mode — both memory adaptation and LayerNorm tuning

### Files Changed

| File | What |
|---|---|
| `src/models/hope/hope_block.py` | `_titan_forward_chunk()` — guard condition updated |

---

## Change 5 — Fix CMS Chunk-Scheduling Granularity

### Problem

The CMS step counter incremented by 1 per `forward()` call (i.e., per sequence/batch), not per token. With `chunk_scheduling=True`, the "slow" level (period=16) would skip **entire forward passes**, not individual tokens within a sequence. The paper intends CMS levels to operate at different *temporal frequencies within a sequence*.

### Paper Reference

Section 8.3, Eq. 96: CMS levels process at frequencies $f_1 > f_2 > \ldots > f_K$, meaning tokens within the same sequence are filtered by temporal period.

### Solution

Rewrote `CMS.forward()` for chunk-scheduling mode:

1. Step counter increments by `N` (number of tokens) per call, not by 1
2. Each token's temporal index is `base_step + token_position`
3. Slower levels build a boolean mask of qualifying tokens and only process those
4. Non-qualifying tokens pass through unchanged (identity)

```python
token_steps = torch.arange(N, device=x.device) + base_step
update_mask = (token_steps >= spec.warmup_steps) & (token_steps % spec.update_period == 0)
```

### Files Changed

| File | What |
|---|---|
| `src/models/hope/cms.py` | `forward()` — per-token frequency gating with proper step counting |

---

## Change 6 — Normalize Inner Loss by Element Count

### Problem

The inner loss was summed over all `B × N × D` elements without normalization:

```python
inner_loss = (inner_loss * gate).sum()  # ~1.8M-scale gradient magnitudes
```

This coupled gradient magnitude to sequence length and feature dimensions, making `grad_clip_inner` the only thing preventing explosion. Different batch sizes or sequence lengths would change the effective inner learning rate.

### Paper Reference

Standard practice for MSE-based optimization objectives. The paper doesn't specify reduction but the normalized form is standard for gradient-based updates.

### Solution

Changed `.sum()` to `.mean()`:

```python
inner_loss = (inner_loss * gate).mean()
```

This decouples gradient magnitude from data dimensions. The `grad_clip_inner=1.0` setting now operates on properly-scaled gradients.

### Files Changed

| File | What |
|---|---|
| `src/models/hope/titan_memory.py` | `compute_and_apply_update()` — `.sum()` → `.mean()` |

---

## Test Results

```
tests/test_hope_gradient_flow.py::TestTitanMemoryGradientFlow::test_memory_weight_params_have_grad          PASSED
tests/test_hope_gradient_flow.py::TestTitanMemoryGradientFlow::test_auxiliary_memory_grads_with_chunking     PASSED
tests/test_hope_gradient_flow.py::TestTitanMemoryGradientFlow::test_self_generated_targets_differ_per_memory PASSED
tests/test_hope_gradient_flow.py::TestMetaLearningChain::test_reset_preserves_gradient_chain                 PASSED
tests/test_hope_gradient_flow.py::TestMetaLearningChain::test_outer_loss_decreases_titan_params              PASSED
tests/test_hope_gradient_flow.py::TestChunking::test_chunk_size_zero_equals_full                             PASSED
tests/test_hope_gradient_flow.py::TestDetachInterval::test_detach_interval_still_trains                      PASSED

7 passed, 0 failed
```

Full test suite: **44 passed**, 1 pre-existing failure (unrelated `ACPredictorModule` trainer-detached issue in `test_iteration_based_scheduler`).

---

## Files Modified (Complete List)

| File | Changes |
|---|---|
| `src/models/hope/titan_memory.py` | DGD preconditioner (2), per-feature η/α (3), normalized inner loss (6) |
| `src/models/hope/hope_block.py` | Per-feature η/α (3), inference memory updates (4), aux loss for M_k/M_v (1) |
| `src/models/hope/cms.py` | Per-token chunk scheduling (5) |
| `src/models/hope/ac_hope_vit.py` | `get_aux_loss()` aggregator (1) |
| `src/models/hope/ac_hope_module.py` | `aux_loss_weight` param + wired into training_step (1) |
| `configs/model/ac_hope_vit.yaml` | Added `aux_loss_weight: 0.1` (1) |
| `configs/experiment/ac_hope_vit_param_matched.yaml` | Added `aux_loss_weight: 0.1` (1) |
| `configs/experiment/ac_hope_vit_depth_matched.yaml` | Added `aux_loss_weight: 0.1` (1) |
| `tests/test_hope_gradient_flow.py` | Updated M_k/M_v gradient assertions (1) |

---

## Remaining Considerations

1. **M_q memory:** The full paper (Eq. 80, 85) lists 6 memories including M_q; current code uses 5 with a static W_q. Adding it would increase params ~15%. Consider for future work.

2. **Short sequences:** Only 8 temporal timesteps per V-JEPA 2 clip (vs. 2K-8K in the paper's text experiments). With `chunk_size=1`, memories get at most 8 DGD updates per sequence. Consider concatenating clips for longer sequences.

3. **aux_loss_weight tuning:** Default 0.1 is conservative. Monitor `train/aux_loss_mk_mv` during training — if M_k/M_v gradients are still very small relative to M_memory, increase to 0.2-0.5.

4. **Inner loss scale after normalization:** The switch from `.sum()` to `.mean()` reduces inner-loop gradient magnitudes by a factor of ~B×N×D. The existing `grad_clip_inner=1.0` may now be too aggressive. Monitor `titan/mean_inner_grad_norm` — if consistently at the clip threshold, consider increasing to 5.0-10.0.
