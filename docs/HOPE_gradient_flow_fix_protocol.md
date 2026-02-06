# HOPE Gradient-Flow Fix — Scientific Protocol

**Date:** 2026-02-06
**Scope:** `src/models/hope/` — Titan memory meta-learning mechanism
**Reference paper:** Behrouz 2025, *Nested Learning: Self-Referential Titans with CMS* (Section 8)
**Codebase revision:** post-fix (current `master`)

---

## 1. Problem Statement

An academic code review ([HOPE_Implementation_Review.md](HOPE_Implementation_Review.md)) identified that the HOPE implementation's Titan memories were **not being trained**. The meta-learning mechanism (FOMAML-style first-order) was broken at multiple levels, rendering 60–100% of the Titan parameters dead weight depending on configuration.

### 1.1 Bugs Identified

| # | Severity | Description |
|---|----------|-------------|
| B1 | **Critical** | `.detach()` in `reset_active_weights()` severed the gradient chain from `nn.Parameters` to the outer loss for all 5 memories. |
| B2 | **Critical** | Shared `SelfModifier` MLP produced one target for all 5 memories — not self-referential (paper Eq. 83 specifies per-memory targets $\hat{v}_{\Box,t} = \mathcal{M}_{\Box,t-1}(v_t)$). |
| B3 | **High** | $\eta$ and $\alpha$ from `M_eta`/`M_alpha` were converted to Python floats via `.mean().item()`, disconnecting them from the computation graph. |
| B4 | **High** | Even after fixing B1/B3, `M_eta` and `M_alpha` only receive outer-loop gradients with temporal chunking (multi-step memory updates), not with single-pass processing. |
| B5 | **Structural** | `M_k` and `M_v` receive **zero gradients** regardless of configuration — a fundamental consequence of first-order meta-learning in this architecture. |

---

## 2. Root-Cause Analysis

### 2.1 The Meta-Learning Gradient Chain

The Titan memory uses a functional forward pattern. Each memory $\mathcal{M}_\Box$ has:
- **nn.Parameters** ($W_1, W_2$): the meta-learned initial state, updated by the outer optimizer.
- **Active weights** ($\tilde{W}_1, \tilde{W}_2$): derived from the parameters, modified in-place during the forward pass via DGD.

The intended gradient chain is:

$$
W_\text{param} \xrightarrow{\text{clone}} \tilde{W}_0 \xrightarrow{\text{DGD step 1}} \tilde{W}_1 \xrightarrow{\text{DGD step 2}} \cdots \xrightarrow{\text{forward}} o_t \xrightarrow{} \mathcal{L}_\text{outer}
$$

The outer loss $\mathcal{L}_\text{outer}$ backpropagates through the chain, computing $\frac{\partial \mathcal{L}_\text{outer}}{\partial W_\text{param}}$, which the outer optimizer (Adam) uses to update the meta-learned initial state.

### 2.2 Bug B1: Severed Gradient Chain

The original `reset_active_weights()` used:

```python
# BEFORE (broken):
self._active_w1 = self.w1.weight.detach().clone()
```

`.detach()` creates a new tensor disconnected from the computation graph. No gradient can flow from $\tilde{W}$ back to $W_\text{param}$. Result: **all 240 memory weight matrices** (5 memories × 24 blocks × 2 layers) received zero gradients from the outer optimizer. They were never trained.

### 2.3 Bug B5: Why M_k and M_v Cannot Be Trained (First-Order Limitation)

This is the most subtle finding and warrants detailed analysis.

#### 2.3.1 How Each Memory's Output Is Consumed

In `_titan_forward_chunk()`, the five memories produce:

```python
q = self.q_proj(x)          # Static projection (nn.Linear, NOT a Titan memory)
k = self.M_k(x)             # → used as key in _update_memories()
v = self.M_v(x)             # → used as value base in _update_memories()
eta_raw = self.M_eta(x)     # → used as learning rate in DGD update
alpha_raw = self.M_alpha(x) # → used as decay factor in DGD update
output = self.M_memory(q)   # → DIRECTLY feeds the loss ← only loss-connected output
```

The loss is computed from `output = self.M_memory(q)`. Only `M_memory`'s active weights participate in computing the loss-connected output.

#### 2.3.2 Gradient Paths Through the DGD Update

In `compute_and_apply_update()`, the DGD inner loop computes:

$$
\tilde{W}_\text{new} = \alpha \cdot \tilde{W}_\text{old} - \eta \cdot g_\text{inner}
$$

where $g_\text{inner} = \nabla_{\tilde{W}} \mathcal{L}_\text{inner}(k, \hat{v}_\Box)$ is computed via `torch.autograd.grad(..., create_graph=False)`.

The inputs to the DGD update and their gradient status:

| Input | Source | Graph-connected? | Reason |
|-------|--------|-------------------|--------|
| $\tilde{W}_\text{old}$ | Previous active weights | ✅ **Yes** | Connected to `nn.Parameters` via `.clone()` |
| $g_\text{inner}$ | `autograd.grad()` | ❌ No | `create_graph=False` + explicit `.detach()` (first-order) |
| $\eta$ (`lr_mean`) | `M_eta(x).mean()` | ✅ **Yes** (after fix) | Kept in graph; no `.detach()` |
| $\alpha$ (`alpha_mean`) | `M_alpha(x).mean()` | ✅ **Yes** (after fix) | Kept in graph; no `.detach()` |
| $k$ (key) | `M_k(x)` | ❌ **Dead end** | Only used inside inner loss via detached leaf weights |
| $v$ (value) | `mem.generate_self_target(v)` | ❌ **Dead end** | Explicitly `.detach()`ed in inner loss: `value.detach()` |

#### 2.3.3 Tracing M_k's Gradient Path

`M_k(x)` produces $k$, which enters `compute_and_apply_update()` as the `key` argument. Inside:

```python
w1_leaf = self._active_w1.detach().requires_grad_(True)   # ← detached leaf
h = F.linear(key, w1_leaf)                                 # key IS used here
retrieved = F.linear(h, w2_leaf) + key                      # key IS used here (residual)
inner_loss = F.mse_loss(retrieved, value.detach(), ...)     # inner loss
grads = torch.autograd.grad(inner_loss, [w1_leaf, w2_leaf], create_graph=False)
```

Although $k$ appears in the inner loss computation, `autograd.grad` only computes gradients w.r.t. `[w1_leaf, w2_leaf]` — not w.r.t. $k$. And `create_graph=False` means no second-order graph is built. The resulting `grads` are then detached:

```python
grad = grad.detach()
```

So $k$ enters a computational dead end: it contributes to `inner_loss` which only produces detached gradient values. No gradient from $\mathcal{L}_\text{outer}$ flows back through this path to `M_k`.

#### 2.3.4 Could M_k Get Gradients via Multi-Step (Chunking)?

With temporal chunking, chunk 2 reads memory state updated by chunk 1:

$$
\tilde{W}^{(2)} = \alpha \cdot \tilde{W}^{(1)} - \eta \cdot g^{(1)}
$$

The output of chunk 2 depends on $\tilde{W}^{(2)}$, which depends on $\tilde{W}^{(1)}$ (through $\tilde{W}_\text{old}$), which depends on `nn.Parameters`. This path gives `M_memory` its gradient.

But for `M_k`: the updated $\tilde{W}_k^{(2)}$ depends on $\tilde{W}_{k,\text{old}}^{(1)}$ through $w_\text{old}$ — this path IS connected. However, $\tilde{W}_k^{(2)}$ is only ever used to produce $k_2 = M_k^{(2)}(x_2)$, which again enters `compute_and_apply_update()` as a dead-end input. It is never on the path to $\mathcal{L}_\text{outer}$.

**Conclusion:** Under first-order meta-learning (FOMAML), `M_k` and `M_v` cannot receive outer-loop gradients because their outputs are consumed exclusively inside the inner-loop gradient computation, where all paths are detached.

#### 2.3.5 What Would Fix M_k and M_v?

Three possible approaches (not implemented, documented for future work):

1. **Second-order meta-learning (MAML):** Use `create_graph=True` in the inner loop. Then $g_\text{inner}$ remains in the graph, and the outer loss can differentiate through it. This enables gradients to flow through $k$ and $v$ because the inner loss (which uses them) remains graph-connected. **Cost:** 2–3× more VRAM, Hessian computation.

2. **Architectural change — use k,v in the output path:** Add an explicit attention mechanism where the output is computed as $o_t = \text{softmax}(q k^T / \sqrt{d}) \cdot v$ instead of (or in addition to) $o_t = M_\text{memory}(q)$. This puts $k$ and $v$ directly on the loss path. **Cost:** architectural redesign, additional parameters.

3. **Auxiliary loss on k,v quality:** Add a regularization term $\mathcal{L}_\text{aux} = \text{MSE}(M_\text{memory}(M_k(x)), M_v(x))$ that directly measures whether the keys produced by `M_k` lead to good retrievals of `M_v`'s values. **Cost:** moderate, hyperparameter tuning needed.

---

## 3. Changes Implemented

### 3.1 Fix B1 — Gradient Chain Restoration

**File:** `src/models/hope/titan_memory.py`, method `reset_active_weights()`

```python
# BEFORE (broken):
self._active_w1 = self.w1.weight.detach().clone()
self._active_w2 = self.w2.weight.detach().clone()

# AFTER (fixed):
self._active_w1 = self.w1.weight.clone()
self._active_w2 = self.w2.weight.clone()
```

`.clone()` without `.detach()` creates a new tensor that shares the same position in the computation graph. Gradients flow from $\tilde{W}$ back to $W_\text{param}$.

**Effect:** `M_memory` parameters now receive non-zero gradients from the outer loss in all configurations.

### 3.2 Fix B2 — Per-Memory Self-Generated Targets

**File:** `src/models/hope/hope_block.py`, method `_update_memories()` + `src/models/hope/titan_memory.py`, method `generate_self_target()`

The shared `SelfModifier` class was **deleted entirely**. Each memory now generates its own target via the new `generate_self_target()` method:

```python
# In _update_memories():
for mem in [self.M_k, self.M_v, self.M_eta, self.M_alpha, self.M_memory]:
    self_target = mem.generate_self_target(v)   # Eq. 83: v̂_□ = M_□(v)
    mem.compute_and_apply_update(key=k, value=self_target, ...)
```

```python
# In TitanMemory:
def generate_self_target(self, v: Tensor) -> Tensor:
    """v̂_□,t = M_{□,t-1}(v_t)"""
    return self.forward(v)
```

**Effect:** Each memory specializes toward its own function. The self-referential property (the memory decides what to learn based on its own current state) is restored per Eq. 83.

### 3.3 Fix B3/B4 — η and α Gradient Flow

**File:** `src/models/hope/titan_memory.py`, method `compute_and_apply_update()`

```python
# BEFORE (broken):
lr_mean = lr.detach().mean()     # or lr.mean().item() → Python float
alpha_mean = alpha.detach().mean()

# AFTER (fixed):
lr_mean = lr.mean()              # stays in computation graph
alpha_mean = alpha.mean()        # stays in computation graph
```

**Effect:** With temporal chunking (`chunk_size ≥ 1`), the DGD update $\tilde{W}_\text{new} = \alpha \cdot \tilde{W}_\text{old} - \eta \cdot g$ keeps $\alpha$ and $\eta$ graph-connected. When chunk 2 reads the updated memory state, the outer loss backpropagates through $\alpha_\text{mean}$ and $\text{lr}_\text{mean}$ to `M_alpha` and `M_eta` respectively.

### 3.4 Temporal Chunking

**File:** `src/models/hope/hope_block.py`, methods `_titan_forward()` + `_titan_forward_chunk()`

The forward pass now supports splitting the token sequence into temporal chunks. With `chunk_size=c`, the sequence of $T$ timesteps is processed in groups of $c$:

- Chunk 1: process timesteps $[0, c)$, update all 5 memories via DGD
- Chunk 2: process timesteps $[c, 2c)$ using updated memory state, update again
- ...

This creates a multi-step memory trajectory where $\tilde{W}^{(i+1)}$ depends on $\tilde{W}^{(i)}$, enabling meta-gradient flow across chunks for memories whose outputs modulate the DGD update ($\eta$, $\alpha$).

**Default:** `chunk_size=1` (per-timestep updates). Set to `0` for the original single-pass behavior.

### 3.5 Periodic Graph Detachment

**File:** `src/models/hope/titan_memory.py`

With chunking, the computation graph grows linearly with the number of chunks. To bound VRAM:

```python
if self.config.detach_interval > 0 and step % self.config.detach_interval == 0:
    self._active_w1 = self._active_w1.detach().clone()
    self._active_w2 = self._active_w2.detach().clone()
```

This truncates the meta-gradient chain every `detach_interval` steps, bounding graph depth while still allowing gradients for the most recent steps.

**Defaults:** `titan_detach_interval=0` (param-matched, 6 layers), `titan_detach_interval=4` (depth-matched, 12 layers).

### 3.6 Complete Diagnostics

**File:** `src/models/hope/hope_block.py`, method `get_diagnostics()`

All 5 memories now report diagnostics (previously only M_k, M_v, M_memory). Each reports:
- `titan/mean_inner_grad_norm`: average DGD gradient norm
- `titan/param_norm_w1`, `titan/param_norm_w2`: parameter norms
- `titan/num_updates`: DGD step count

Plus aggregate: `hope/mean_surprise`, `hope/max_surprise`.

---

## 4. Empirical Gradient Flow Verification

### 4.1 Test Setup

A 1-layer ACHOPEViT with `predictor_embed_dim=384`, `titan_hidden_multiplier=2`, random input `[B=1, T×N, D=1024]`. Forward + backward with `loss = output.mean()`.

### 4.2 Results: chunk_size=0 (No Chunking)

| Parameter Group | `.grad is not None` | `.grad.norm() > 0` |
|-----------------|---------------------|---------------------|
| `M_memory.w1.weight` | ✅ | ✅ |
| `M_memory.w2.weight` | ✅ | ✅ |
| `M_memory.norm.*` | ✅ | ✅ |
| `M_k.*` | ❌ | — |
| `M_v.*` | ❌ | — |
| `M_eta.*` | ❌ | — |
| `M_alpha.*` | ❌ | — |
| `q_proj.*` | ✅ | ✅ |
| `out_proj.*` | ✅ | ✅ |
| `cms.*` | ✅ | ✅ |

Without chunking, only `M_memory` and the non-Titan modules get gradients. This is expected: `M_eta`/`M_alpha` need multi-step updates for their outputs to influence a subsequent forward, and `M_k`/`M_v` are structurally disconnected (Section 2.3).

### 4.3 Results: chunk_size=2, T=4 (Temporal Chunking)

| Parameter Group | `.grad is not None` | `.grad.norm() > 0` |
|-----------------|---------------------|---------------------|
| `M_memory.*` | ✅ | ✅ |
| `M_eta.*` | ✅ | ✅ |
| `M_alpha.*` | ✅ | ✅ |
| `M_k.*` | ❌ | — |
| `M_v.*` | ❌ | — |
| `q_proj.*`, `out_proj.*`, `cms.*` | ✅ | ✅ |

With chunking, `M_eta` and `M_alpha` receive gradients because their outputs ($\eta$, $\alpha$) modulate the DGD update rule, and the updated memory state is read by the next chunk's forward pass. `M_k` and `M_v` remain untrained (see Section 2.3).

### 4.4 Gradient Flow Summary

```
                    chunk_size=0        chunk_size≥1
 M_memory           ✅ trained          ✅ trained
 M_eta              ❌ dead             ✅ trained
 M_alpha            ❌ dead             ✅ trained
 M_k                ❌ dead             ❌ dead (structural)
 M_v                ❌ dead             ❌ dead (structural)
 q_proj, out_proj   ✅ trained          ✅ trained
 CMS                ✅ trained          ✅ trained
```

**Recommendation:** Always use `chunk_size ≥ 1` to train M_eta and M_alpha. Default is `chunk_size=1`.

---

## 5. Parameter Budget Impact

The M_k and M_v issue means a fraction of Titan parameters are untrained:

| Config | Total Params | Titan Params | Untrained (M_k + M_v) | Fraction Dead |
|--------|-------------|--------------|------------------------|---------------|
| Param-matched (d=6) | ~42M | ~24M | ~9.4M | ~22% |
| Depth-matched (d=12) | ~83M | ~47M | ~18.8M | ~23% |

Each `TitanMemory` has $D \times (D \cdot m) + (D \cdot m) \times D = 2 D^2 m$ weight parameters (where $m$ is `titan_hidden_multiplier`), plus layer norm. With $D=384$ and $m=2$: ~590K per memory, ~1.18M for M_k + M_v per block.

These parameters still participate in the **inner loop** (DGD updates modify their active weights during the forward pass). They produce adaptive keys and values that influence memory updates. They are just not optimized by the outer optimizer — their `nn.Parameters` remain at their random initialization.

Whether this matters empirically is an open question. In MAML-style meta-learning, the initial state's quality is crucial. Random M_k/M_v initial weights may still produce functional keys/values after DGD adaptation, but they cannot improve across training episodes.

---

## 6. Test Coverage

**File:** `tests/test_hope_gradient_flow.py` — 7 tests, all passing.

| Test | What It Verifies |
|------|-----------------|
| `test_memory_weight_params_have_grad` | M_memory receives non-zero gradients (regression test for B1) |
| `test_auxiliary_memory_grads_with_chunking` | M_eta/M_alpha get gradients with chunking; M_k/M_v confirmed dead |
| `test_self_generated_targets_differ_per_memory` | 5 memories produce distinct self-targets (regression test for B2) |
| `test_reset_preserves_gradient_chain` | `.clone()` without `.detach()` keeps `requires_grad=True` |
| `test_outer_loss_decreases_titan_params` | `optimizer.step()` actually changes M_memory weights |
| `test_chunk_size_zero_equals_full` | `chunk_size=0` ≡ `chunk_size=T` (no behavioral change) |
| `test_detach_interval_still_trains` | Aggressive `detach_interval=1` still allows M_memory gradients |

---

## 7. Files Modified

| File | Changes |
|------|---------|
| `src/models/hope/titan_memory.py` | Removed `.detach()` from `reset_active_weights()`; added `generate_self_target()`; removed `.detach()` from `lr_mean`/`alpha_mean`; wired up `detach_interval`; updated docstrings |
| `src/models/hope/hope_block.py` | Deleted `SelfModifier` class; removed `self_mod_dim` from `HOPEBlockConfig`; added `chunk_size` config; split `_titan_forward` into chunking dispatcher + per-chunk processor; rewrote `_update_memories()` with per-memory self-targets; added M_eta/M_alpha to diagnostics |
| `src/models/hope/ac_hope_vit.py` | Replaced `self_mod_dim` with `chunk_size` + `titan_detach_interval` in constructor + config summary |
| `src/models/hope/ac_hope_module.py` | Same parameter replacement in Lightning module wrapper |
| `src/models/hope/__init__.py` | Removed `SelfModifier` from exports |
| `configs/model/ac_hope_vit.yaml` | `chunk_size: 1`, `titan_detach_interval: 0` (replaces `self_mod_dim`) |
| `configs/experiment/ac_hope_vit_param_matched.yaml` | Same config update |
| `configs/experiment/ac_hope_vit_depth_matched.yaml` | Same config update; `titan_detach_interval: 4` |
| `tests/test_hope_gradient_flow.py` | **Created.** 7 unit tests for gradient flow verification |
| `tmp/test_hope.py` | Updated smoke test for new parameter names |

---

## 8. Open Issues

1. **M_k and M_v are untrained** (Section 2.3.5). Three mitigation paths documented; none implemented. Empirical impact unknown — requires ablation study.

2. **DGD preconditioner missing.** The paper's full update (Eq. 93) includes the data-dependent preconditioner $(\alpha_t I - \eta_t k_t k_t^T)$. The implementation uses the simplified form $\alpha \cdot W_\text{old} - \eta \cdot g$. The preconditioner is only derived for linear memories in the paper; extending it to a 2-layer MLP with activation is non-trivial.

3. **Inner-loop gradient norms are large** (~1.8M in smoke tests). The `grad_clip_inner=1.0` clipping is active and essential. The root cause may be the scale mismatch between the inner MSE loss (computed over `[B, N, D]` tokens) and the weight matrices. Consider normalizing by token count.
