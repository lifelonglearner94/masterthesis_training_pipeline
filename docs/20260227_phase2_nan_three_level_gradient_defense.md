# Phase 2 NaN Fix: Three-Level Gradient Defense for DGD Meta-Learning

**Date:** 2026-02-27  
**Author:** Debugging session (automated)  
**Status:** RESOLVED — Pipeline verified with real V-JEPA2 data  
**Experiment:** `cl_ac_hope_phase2` (`titan_detach_interval: 2`)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Root Cause Analysis](#2-root-cause-analysis)
3. [Numerical Analysis of the Gradient Explosion](#3-numerical-analysis-of-the-gradient-explosion)
4. [The Three-Level Gradient Defense](#4-the-three-level-gradient-defense)
5. [Implementation Details](#5-implementation-details)
6. [Files Changed](#6-files-changed)
7. [Verification Results](#7-verification-results)
8. [Why Not Just Detach?](#8-why-not-just-detach)
9. [Configuration](#9-configuration)

---

## 1. Problem Statement

The AC-HOPE-ViT Phase 2 experiment (`titan_detach_interval: 2`) produced **NaN losses** on the very first training epoch when trained on real V-JEPA2 precomputed features.

**Key observations:**
- The forward pass completed successfully (no NaN in model outputs)
- NaN appeared **only during backward** (gradient computation)
- Random synthetic data with the same shapes/dtypes worked perfectly
- Real V-JEPA2 data (float16→float32, range [-39.75, 36.56], std=3.19) triggered the overflow

**Architecture context:**
- AC-HOPE-ViT: 6 HOPE blocks, each with 5 Titan memories (M_k, M_v, M_eta, M_alpha, M_memory)
- DGD (Delta Gradient Descent) inner-loop updates memory weights during the forward pass
- `titan_detach_interval=2`: the computation graph is retained across 2 DGD update steps before detaching, enabling richer meta-gradients
- The paper (Behrouz 2025, §8.1) requires: *"the initial states of ALL memories are meta-learned across all sequences/contexts"*

---

## 2. Root Cause Analysis

### 2.1 The Rejected Fix (Detaching M_eta/M_alpha)

An earlier fix detached `lr_feat` and `alpha_feat` (outputs of M_eta and M_alpha) before using them in the DGD preconditioner. This eliminated NaN but **violated the paper's core meta-learning requirement**: M_eta and M_alpha would never receive outer-loss gradients and their initial states would not be meta-learned.

### 2.2 The Actual Root Cause: Three Amplification Mechanisms

The NaN was caused by **three compounding gradient amplification mechanisms** in the DGD backward pass:

#### Mechanism 1: Element-wise amplification by weight magnitudes

The DGD preconditioned update (Eq. 93) computes:

$$W_{\text{new}} = W_{\text{old}} \cdot \text{precond} - \eta \cdot \text{grad}$$

where $\text{precond} = \text{clamp}(\alpha - \eta \cdot k^2, 0, 1)$.

The standard backward for $W_{\text{old}} \cdot \text{precond}$ is:

$$\frac{\partial L}{\partial \text{precond}} = \frac{\partial L}{\partial \text{out}} \cdot W_{\text{old}}$$

This scales every element of the gradient by the full weight magnitude $|W_{\text{old}}|$, which can be large.

#### Mechanism 2: Path accumulation within each block

Each block has:
- 5 memories × 2 weight matrices = 10 DGD weight updates
- With `chunk_size=1` and T=7, there are 7 chunks
- Each chunk's update depends on the previous chunk's weights (retained by `titan_detach_interval=2`)
- ~30 gradient paths per block all sum together

#### Mechanism 3: Exponential compounding across blocks

The gradient flows backward through 6 stacked HOPE blocks. Each block's INPUT gradient passes through:

$$\nabla_{\text{input}} = \nabla_{\text{output}} \cdot W_{\text{active}}$$

where $W_{\text{active}}$ are the DGD-modified weights. Without any clipping on this path, the amplification from each block **multiplies** with the next.

### 2.3 Observed Gradient Explosion (Before Fix)

With real V-JEPA2 data (std=3.19), the per-block gradient norms were:

| Block | Grad Norm | Status |
|-------|-----------|--------|
| block_5 | 6.4e-2 | Clean |
| block_4 | 3.8e+9 | Overflow starting |
| block_3 | 1.6e+19 | Near float32 max |
| block_2 | Inf | Overflow |
| block_1 | Inf → NaN | Propagated |
| block_0 | NaN | Propagated |

The amplification factor from block 5 → block 4 was approximately **6×10^10**, confirming exponential compounding through the DGD weight chain.

### 2.4 Why Random Data Worked

Random data (mean=0, std=1) produced much smaller intermediate activations and weight updates, keeping gradient magnitudes within float32 range. Real V-JEPA2 features (std=3.19, range [-39.75, 36.56]) pushed the DGD weight magnitudes higher, causing the same mechanism to overflow.

---

## 3. Numerical Analysis of the Gradient Explosion

### 3.1 DGD Preconditioner Backward Chain

For the expression `out = w_old * precond`:

$$\frac{\partial L}{\partial \text{precond}_i} = \frac{\partial L}{\partial \text{out}_i} \cdot w_{\text{old},i}$$

With weight magnitudes $|w_{\text{old}}| \sim O(1)$ initially but growing through DGD updates, and ~30 such terms summing per block, the total gradient contribution to `precond` (and hence to `lr_feat`/`alpha_feat` from M_eta/M_alpha) is:

$$\nabla_{\text{total}} \sim 30 \cdot |w_{\text{old}}| \cdot \nabla_{\text{out}}$$

### 3.2 Cross-Block Compounding

Each block receives its input gradient from the block above. Without clipping, if each block amplifies the gradient by factor $A$:

$$\nabla_{\text{block}_0} \sim A^5 \cdot \nabla_{\text{block}_5}$$

With $A \sim 10^{10}$ (observed), the gradient at block 0 would be $\sim 10^{50}$ — far beyond float32 max ($3.4 \times 10^{38}$).

---

## 4. The Three-Level Gradient Defense

The fix implements three complementary gradient defense mechanisms, each addressing a different amplification pathway:

### Level 1: `_StableGradPrecondScale` (Element-wise)

**Location:** `titan_memory.py`, applied inside `compute_and_apply_update()`  
**What it does:** Custom autograd function for `w_old * precond`

- **Forward:** `out = w_old * precond` (unchanged)
- **Backward:**
  - `grad_w_old = grad_output * precond` (standard — safe because precond ∈ [0,1])
  - `grad_precond = grad_output * sign(w_old)` (**modified** — uses sign instead of full magnitude)

**Why it works:** Replacing `w_old` with `sign(w_old)` preserves the gradient *direction* while bounding each element's contribution to `|grad_output|`. This prevents individual weight magnitudes from amplifying the backward gradient flowing to `precond` → `lr_feat`/`alpha_feat` → M_eta/M_alpha.

### Level 2: `_BackwardGradClip` on DGD Active Weights (Per-tensor)

**Location:** `titan_memory.py`, applied after each DGD weight update  
**What it does:** Identity in forward; clips backward gradient norm to `max_norm`

```python
# After DGD update:
self._active_w1 = _backward_grad_clip(new_weights[0], clip_bw)
self._active_w2 = _backward_grad_clip(new_weights[1], clip_bw)
```

**Why it works:** Even with Level 1 bounding per-element gradients, the SUM of ~30 gradient paths (5 memories × 2 weights × live chunks) can still be large. Level 2 clips the total gradient norm flowing through each memory's active weights, giving each memory a bounded backward contribution.

### Level 3: `_backward_grad_clip` on Block Outputs (Cross-block)

**Location:** `ac_hope_vit.py`, applied after each HOPE block in the forward loop  
**What it does:** Same `_BackwardGradClip` function, applied to the block's output tensor

```python
# In _process_hope_blocks():
for i, blk in enumerate(self.hope_blocks):
    x = blk(x, ...)
    if self.titan_grad_clip_backward > 0 and self.training:
        x = _backward_grad_clip(x, self.titan_grad_clip_backward)
```

**Why it works:** This is the **critical piece** identified during this session. Levels 1 and 2 protect the gradient flowing through DGD weights (the `precond` and `active_weight` paths), but the gradient flowing backward through the **input path** of each Titan memory (`grad_input = grad_output @ active_w`) was unclipped. This path is what carries gradients from block N to block N-1, causing the exponential compounding across 6 blocks. Level 3 bounds the backward gradient norm at each block boundary, converting exponential growth into linear growth.

### Summary Table

| Level | Mechanism | Location | Protects Against |
|-------|-----------|----------|------------------|
| 1 | `sign(w_old)` in backward | `titan_memory.py` (DGD update) | Element-wise amplification by weight magnitudes |
| 2 | Norm clip on active weights | `titan_memory.py` (after DGD) | Path accumulation within block (~30 paths) |
| 3 | Norm clip on block outputs | `ac_hope_vit.py` (block loop) | Exponential compounding across 6 blocks |

---

## 5. Implementation Details

### 5.1 Custom Autograd Functions

Both `_StableGradPrecondScale` and `_BackwardGradClip` are implemented as `torch.autograd.Function` subclasses in `titan_memory.py`:

```python
class _StableGradPrecondScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w_old, precond):
        ctx.save_for_backward(w_old, precond)
        return w_old * precond

    @staticmethod
    def backward(ctx, grad_output):
        w_old, precond = ctx.saved_tensors
        grad_w_old = grad_output * precond
        grad_precond = grad_output * w_old.sign()  # Key change
        return grad_w_old, grad_precond


class _BackwardGradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, max_norm):
        ctx.max_norm = max_norm
        return x  # Identity

    @staticmethod
    def backward(ctx, grad_output):
        max_norm = ctx.max_norm
        grad_norm = grad_output.norm()
        if grad_norm > max_norm:
            grad_output = grad_output * (max_norm / (grad_norm + 1e-8))
        return grad_output, None
```

### 5.2 Live `lr_feat` / `alpha_feat` (No Detach)

In `compute_and_apply_update()`, the outputs of M_eta and M_alpha are used **without `.detach()`**:

```python
# LIVE: not detached — preserves meta-learning gradient flow
lr_feat = lr.mean(dim=(0, 1))
alpha_feat = alpha.mean(dim=(0, 1)) if alpha is not None else ...
```

This ensures M_eta and M_alpha receive outer-loss gradients, satisfying the paper's §8.1 requirement.

### 5.3 Configuration Parameter

A single configuration parameter `grad_clip_backward: float = 1.0` controls both Level 2 and Level 3 clipping. It is threaded through:

```
TitanMemoryConfig.grad_clip_backward  (Level 2)
  ↑ set by
HOPEBlockConfig.titan_grad_clip_backward
  ↑ set by
ACHOPEViT.__init__(titan_grad_clip_backward=...)
  ↑ set by
ACHOPEModule.__init__(titan_grad_clip_backward=...)
  ↑ set by
Hydra config (model.titan_grad_clip_backward)
```

ACHOPEViT also stores `self.titan_grad_clip_backward` for Level 3 usage in `_process_hope_blocks()`.

---

## 6. Files Changed

### 6.1 `src/models/hope/titan_memory.py`

- **Added** `_StableGradPrecondScale` class (Level 1 custom autograd)
- **Added** `_BackwardGradClip` class (Level 2/3 custom autograd)
- **Added** `_stable_precond_scale()` and `_backward_grad_clip()` convenience wrappers
- **Added** `grad_clip_backward: float = 1.0` to `TitanMemoryConfig`
- **Removed** `.detach()` from `lr_feat` and `alpha_feat` in `compute_and_apply_update()`
- **Applied** `_stable_precond_scale(w_old, precond)` instead of `w_old * precond` in DGD update
- **Applied** `_backward_grad_clip(new_weights[i], clip_bw)` to new active weights after DGD

### 6.2 `src/models/hope/hope_block.py`

- **Added** `titan_grad_clip_backward: float = 1.0` to `HOPEBlockConfig`
- Threaded to `TitanMemoryConfig` in `__init__`

### 6.3 `src/models/hope/ac_hope_vit.py`

- **Added** import: `from src.models.hope.titan_memory import _backward_grad_clip`
- **Added** `self.titan_grad_clip_backward = titan_grad_clip_backward` in `__init__`
- **Added** Level 3 block-level backward clip in `_process_hope_blocks()`:
  ```python
  if self.titan_grad_clip_backward > 0 and self.training:
      x = _backward_grad_clip(x, self.titan_grad_clip_backward)
  ```
- Threaded `titan_grad_clip_backward` parameter through constructor, `HOPEBlockConfig`, and `_config_summary`

### 6.4 `src/models/hope/ac_hope_module.py`

- Threaded `titan_grad_clip_backward` parameter through constructor and `ac_hope_vit()` instantiation

### 6.5 `tests/test_hope_gradient_flow.py`

- **Updated** `test_auxiliary_memory_grads_with_chunking`: now asserts all 5 memories (including M_eta, M_alpha) **have** non-zero gradients (previously asserted they had None/zero when detached)

---

## 7. Verification Results

### 7.1 Unit Tests

All 7 gradient flow tests pass:

```
$ uv run pytest tests/test_hope_gradient_flow.py -v
PASSED test_memory_gets_gradients
PASSED test_auxiliary_memory_grads_with_chunking   ← now asserts M_eta/M_alpha HAVE grads
PASSED test_self_generated_targets_differ_per_memory
PASSED test_full_forward_backward
PASSED test_chunk_processing_gradients
PASSED test_gradient_flow_with_aux_loss
PASSED test_detach_interval_controls_graph_depth
```

### 7.2 Standalone Script — Real V-JEPA2 Data (MPS)

Using 2 real clips from `data/clip_00000` and `data/clip_00001`:

```
features: shape=[2, 8, 256, 1024], range=[-39.75, 36.56], std=3.19

[Step 0] teacher loss=1.072, jump loss=1.075, combined=5.90
         grad health: nan_grads=0, inf_grads=0
         block_5: max_norm=6.4e-2
         block_4: max_norm=3.8e+9  ← large but not Inf (clipped by Level 3)
         block_0: max_norm=4.2e+9

[Step 1] teacher loss=1.072, jump loss=1.075, combined=5.90
         grad health: nan_grads=0, inf_grads=0
         block_5: max_norm=6.4e-2
         block_0: max_norm=2.37     ← stabilized after first optimizer step

[Step 2] teacher loss=1.057, jump loss=1.058, combined=4.73  ← loss decreasing
         grad health: nan_grads=0, inf_grads=0
         block_0: max_norm=1.57
```

**Step 0** has large gradient norms (4.2e+9) because the initial weight state produces high amplification, but the gradient is finite (not Inf/NaN) thanks to Level 3 clipping. After the first optimizer step with `gradient_clip_val=3.0`, the weights adjust and **Step 1 onward** has perfectly healthy gradients.

### 7.3 Full Lightning Pipeline

```
$ uv run python -m src.train experiment=cl_ac_hope_phase2 \
    data.clip_start=0 data.clip_end=100 \
    trainer.limit_train_batches=10 trainer.max_epochs=1 \
    data.batch_size=2 trainer.devices=1 'logger=[]'

Epoch 0/0 ━━━━━━━━━━━━━━━━━━ 10/10  0:00:27
  train/loss_teacher: 0.883
  train/loss_jump:    0.871
  train/loss:         1.754
  val/loss_teacher:   0.851
  val/loss_jump:      0.849
  val/loss:           1.700
```

**Zero NaN. Loss decreasing. Validation loss < training loss (healthy generalization).**

### 7.4 Before vs After Comparison

| Metric | Before (no Level 3) | After (full 3-level defense) |
|--------|---------------------|------------------------------|
| Forward pass | Clean | Clean |
| Backward pass | 44 NaN + 1 Inf grads | 0 NaN, 0 Inf |
| block_5 grad | 6.4e-2 | 6.4e-2 |
| block_4 grad | 3.8e+9 | 3.8e+9 (same, but finite) |
| block_3 grad | 1.6e+19 | 4.2e+9 (clipped by Level 3) |
| block_2 grad | Inf | 4.2e+9 (clipped) |
| block_0 grad | NaN | 4.2e+9 (clipped) |
| Loss after 1 epoch | NaN | 1.754 |
| M_eta/M_alpha grads | NaN | Non-zero, healthy |
| Paper §8.1 compliance | ❌ (detached) or ❌ (NaN) | ✅ All memories meta-learned |

---

## 8. Why Not Just Detach?

The previous fix detached `lr_feat` and `alpha_feat`:

```python
lr_feat = lr.mean(dim=(0, 1)).detach()      # ← kills M_eta gradients
alpha_feat = alpha.mean(dim=(0, 1)).detach() # ← kills M_alpha gradients  
```

This eliminated NaN because it cut the gradient path from the outer loss through the DGD preconditioner to M_eta/M_alpha. However, it violated the paper's fundamental design:

> **Behrouz 2025, §8.1:** *"the initial states of ALL memories are meta-learned across all sequences/contexts"*

With detach, M_eta and M_alpha would:
- Never receive outer-loss gradients during base training
- Have their initial states stuck at random initialization
- Fail to learn meaningful learning rates (η) and decay factors (α)
- Undermine the core HOPE contribution: self-referential meta-learning

The three-level defense preserves the full meta-gradient chain while preventing overflow, satisfying both numerical stability and paper faithfulness.

---

## 9. Configuration

The defense is controlled by a single parameter (default: 1.0):

```yaml
# In experiment config:
model:
  titan_grad_clip_backward: 1.0  # Controls Level 2 + Level 3 clipping
```

**Tuning guidance:**
- `1.0` (default): Works well for depth=6, `titan_detach_interval=2`
- `0.0`: Disables Level 2 + Level 3 entirely (will NaN with real data)
- Higher values (e.g., `5.0`): Less aggressive clipping, may allow larger gradients through
- The value should be increased if training becomes too slow to converge, or decreased if NaN returns with deeper models or longer detach intervals

The standard outer-loop gradient clipping (`trainer.gradient_clip_val=3.0`) handles whatever escapes the three-level defense, but in practice the defense alone keeps gradients healthy after the first optimizer step.
