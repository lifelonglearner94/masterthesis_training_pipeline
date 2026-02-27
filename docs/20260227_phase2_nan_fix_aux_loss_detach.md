# Phase 2 NaN Fix: Detach M_memory Weights in Auxiliary Loss

**Date:** 2026-02-27  
**File changed:** `src/models/hope/hope_block.py` (`_update_memories()`)  
**Configs affected:** Any config with `titan_detach_interval > 1` (i.e. `cl_ac_hope_phase2.yaml`)

---

## Symptom

Running `cl_ac_hope_phase2` (which sets `titan_detach_interval: 2`) produced **NaN losses from the very first training step**. The original `cl_ac_hope` config (`detach_interval: 1`) trained normally.

## Root Cause

The auxiliary loss for M_k/M_v gradient flow was calling `self.M_memory(k)` — which uses M_memory's **post-DGD-update active weights** — to create a differentiable retrieval path.

With `detach_interval=1`, this was harmless: the active weights are detached immediately after every DGD step, so the aux_loss gradient flows only through `k` (→ M_k params) and `v` (→ M_v params). This is the intended design.

With `detach_interval=2`, at odd-numbered DGD steps the active weights are still **live** — connected to `nn.Parameters` through the full DGD update chain:

```
aux_loss
  → M_memory(k)                        # forward through post-update weights
    → _active_w1, _active_w2           # still live (not yet detached)
      → w_old * precond - lr * grad    # DGD update rule (Eq. 93)
        → precond = alpha_feat - lr_feat * k_sq
          → alpha_feat ← M_alpha params
          → lr_feat ← M_eta params
        → w_old                        # connected to nn.Parameter via .clone()
```

This creates an **unintended gradient amplification path**. The Jacobian `d(new_w)/d(precond) = w_old`, which has Frobenius norm ~40. Across 5 memories × 6 blocks × 7 temporal chunks, these intermediate backward values compound and overflow to `inf`. Once any `inf × 0 = NaN` appears in the backward pass, it propagates to all parameter gradients — and this happens **before** `gradient_clip_val` can intervene (clipping runs after `.backward()` completes, not during intermediate gradient computation).

## Fix

Detach M_memory's active weights when computing the auxiliary loss, so the gradient only flows through `k` and `v` (the intended paths for M_k and M_v gradient flow):

```python
# BEFORE (broken with detach_interval > 1):
retrieved_via_k = self.M_memory(k)  # gradient flows through M_memory's DGD chain

# AFTER (safe for any detach_interval):
_w1 = self.M_memory._active_w1.detach()
_w2 = self.M_memory._active_w2.detach()
h = F.linear(k, _w1)
h = self.M_memory.act(h)
retrieved_via_k = self.M_memory.norm(F.linear(h, _w2) + k)
```

This makes the aux_loss behave identically regardless of `detach_interval`:
- Gradient through `k` → M_k params ✓ (intended)
- Gradient through `v` → M_v params ✓ (intended)
- Gradient through M_memory's DGD update chain ✗ (blocked — was never intended)

## Verification

All 7 existing gradient flow tests pass after the fix:

```
tests/test_hope_gradient_flow.py::test_memory_weight_params_have_grad           PASSED
tests/test_hope_gradient_flow.py::test_auxiliary_memory_grads_with_chunking     PASSED
tests/test_hope_gradient_flow.py::test_self_generated_targets_differ_per_memory PASSED
tests/test_hope_gradient_flow.py::test_reset_preserves_gradient_chain           PASSED
tests/test_hope_gradient_flow.py::test_outer_loss_decreases_titan_params        PASSED
tests/test_hope_gradient_flow.py::test_chunk_size_zero_equals_full              PASSED
tests/test_hope_gradient_flow.py::test_detach_interval_still_trains             PASSED
```

## Why This Was Invisible With detach_interval=1

When `detach_interval=1`, the detach happens at step 1, step 2, step 3, etc. — i.e. **every** step. After detach, `_active_w1` and `_active_w2` are plain tensors with no grad history. So `self.M_memory(k)` was already equivalent to the fixed version: gradient could only flow through `k`, not through the weights. The bug was latent and only triggered when `detach_interval > 1`.

## Paper Alignment

The fix is consistent with the HOPE paper (Behrouz 2025, Section 8.3). The auxiliary loss is an implementation-level addition to provide gradient flow to M_k/M_v under FOMAML — it is not part of the theoretical HOPE formulation. The paper's DGD update rule (Eq. 93) does not involve backpropagating through the memory weights for auxiliary objectives. Detaching M_memory's weights in this context is therefore the correct interpretation.
