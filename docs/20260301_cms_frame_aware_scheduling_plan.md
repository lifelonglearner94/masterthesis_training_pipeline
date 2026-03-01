# CMS Frame-Aware Scheduling — Implementation Plan (Option B)

**Date:** 2026-03-01
**Goal:** Make CMS `update_period` operate on **frames** (temporal), not flat token indices (spatial+temporal mix).

---

## 1. Problem Summary

Currently, when `cms_use_chunk_scheduling: true`, the CMS forward pass in `src/models/hope/cms.py` builds an update mask using flat token indices:

```python
token_steps = torch.arange(N, device=x.device) + base_step
update_mask = (token_steps % spec.update_period == 0)
```

With 7 frames × 258 tokens/frame = 1806 total tokens, `update_period: 4` skips every 4th **patch**, creating a spatially random subsampling pattern with no temporal meaning. `update_period: 16` is similarly meaningless — it processes 1 out of every 16 patches regardless of which frame they belong to.

The paper's intent (Section 8.3) is for CMS levels to operate at different **temporal frequencies**: fast levels react to every frame, slow levels capture multi-frame or clip-level context.

---

## 2. Design

### Core Change: Frame-Level Masking

Instead of masking individual tokens, the CMS should:
1. Know the number of frames `T` and tokens per frame `tokens_per_frame`
2. Assign each token a **frame index** (0 to T-1)
3. Apply `update_period` to **frame indices**, not token indices
4. When a frame is "active" for a given level, **all tokens in that frame** are processed by that level's MLP

This means:
- `update_period: 1` → processes all frames (every token)
- `update_period: 2` → processes frames 0, 2, 4, 6 (all 258 tokens in each)
- `update_period: 7` → processes only frame 0 (once per clip)

### What Happens to Non-Processed Tokens?

Tokens in frames that are skipped by a CMS level get an **identity pass-through** (the residual connection in CMSBlock already handles this — we just don't call the block for those tokens). The information from active frames propagates downstream through the Titan memory and subsequent HOPE blocks.

### `_global_step` Semantics Change

The `_global_step` counter should now count **frames** processed, not tokens. Since it's reset per clip, it simply starts at 0 and is not strictly needed — we can compute frame indices directly from `T` and `tokens_per_frame`. We keep it for potential future use (e.g., multi-clip streaming) but the primary logic uses the explicitly passed `T`.

---

## 3. Files to Modify

### 3.1 `src/models/hope/cms.py` — CMS forward pass

**Changes:**

1. **`CMS.forward()` signature**: Add `T: int | None = None` and `tokens_per_frame: int | None = None` parameters.

2. **New scheduling logic** (replaces the current token-index masking):
   ```python
   def forward(self, x: Tensor, T: int | None = None, tokens_per_frame: int | None = None) -> Tensor:
       if not self.use_chunk_scheduling or T is None or tokens_per_frame is None:
           # Standard mode: all levels process all tokens
           for block in self.blocks:
               x = block(x)
           return x

       B, N, D = x.shape

       for block, spec in zip(self.blocks, self.levels_spec):
           if spec.update_period <= 1:
               # Fast level: process everything
               x = block(x)
               continue

           # Build frame-level mask: which frames are active?
           # Frame f is active if f % update_period == 0
           frame_indices = torch.arange(T, device=x.device)
           active_frames = (frame_indices % spec.update_period == 0)

           if not active_frames.any():
               continue

           if active_frames.all():
               x = block(x)
           else:
               # Expand frame mask to token mask:
               # each frame owns `tokens_per_frame` contiguous tokens
               token_mask = active_frames.repeat_interleave(tokens_per_frame)

               # Handle edge case: N might not equal T * tokens_per_frame exactly
               token_mask = token_mask[:N]

               indices = token_mask.nonzero(as_tuple=True)[0]
               x_sub = x[:, indices, :]
               x_sub = block(x_sub)
               x = x.clone()
               x[:, indices, :] = x_sub

       return x
   ```

3. **`LevelSpec` docstring**: Update `update_period` docstring from "Tokens between updates" to "Frames between updates (1=every frame, 2=every other frame, etc.)".

4. **`_global_step` advancement**: Change from `self._global_step += N` to `self._global_step += T` (counting frames, not tokens), or remove the advancement entirely since frame indices are computed from explicit `T`.

### 3.2 `src/models/hope/hope_block.py` — Pass T and tokens_per_frame to CMS

**Changes in `HOPEBlock.forward()`** (around line 228):

Replace:
```python
y = self.cms(y)
```

With:
```python
tokens_per_frame = (action_tokens + H * W) if H is not None and W is not None else None
y = self.cms(y, T=T, tokens_per_frame=tokens_per_frame)
```

This is a minimal change — the `T`, `H`, `W`, and `action_tokens` values are already available as parameters to `HOPEBlock.forward()`.

### 3.3 `configs/experiment/cl_ac_hope_phase3.yaml` — Update periods

**Change `cms_level_specs` to frame-aligned values:**

```yaml
cms_level_specs:
  - name: "fast"
    update_period: 1          # Every frame — immediate dynamics
    hidden_multiplier: 4.0
    warmup_steps: 0
  - name: "medium"
    update_period: 3          # Every 3rd frame — aligns with jump_k=3
    hidden_multiplier: 4.0
    warmup_steps: 0
  - name: "slow"
    update_period: 7          # Once per clip — clip-level physics context
    hidden_multiplier: 4.0
    warmup_steps: 0
```

**Rationale for values:**
- **fast=1**: Processes all 7 frames. Captures frame-to-frame velocity/acceleration.
- **medium=3**: Processes frames 0, 3, 6 (3 out of 7). Aligns with `jump_k: 3` — the medium memory "sees" the world at the same temporal resolution as the jump prediction objective.
- **slow=7**: Processes only frame 0 (1 out of 7). Encodes the initial state / clip-level physics constants. Acts as a "what kind of world is this?" prior that gets applied once and persists through the residual stream.

### 3.4 `tests/test_cms_frame_scheduling.py` — New test file

Add a focused test to verify:
1. **Frame-aware masking correctness**: With T=4, tokens_per_frame=3, update_period=2, verify only frames 0 and 2 (tokens 0-2 and 6-8) are processed.
2. **Fast level processes all tokens**: update_period=1 always processes everything.
3. **Fallback to standard mode**: When T or tokens_per_frame is None, all levels process all tokens (backward compatibility).
4. **Gradient flow**: Ensure gradients flow through both processed and skipped tokens (skipped tokens still get gradients from the residual path).

---

## 4. Implementation Order

| Step | File | Change | Risk |
|------|------|--------|------|
| 1 | `cms.py` | Add `T`, `tokens_per_frame` params to `forward()`, implement frame-level masking | Medium — logic change |
| 2 | `cms.py` | Update `LevelSpec` docstring | Trivial |
| 3 | `hope_block.py` | Pass `T`, `tokens_per_frame` through to `cms()` | Low — 2-line change |
| 4 | `tests/` | Add `test_cms_frame_scheduling.py` | None |
| 5 | Config | Update `cl_ac_hope_phase3.yaml` periods | Low |
| 6 | Smoke test | Run one training epoch, verify loss is finite and CMS diagnostics look sane | Validation |

---

## 5. Backward Compatibility

- When `cms_use_chunk_scheduling: false` (default for Phase 1/2 configs), behavior is **unchanged** — all levels process all tokens.
- When `T` or `tokens_per_frame` is `None` (e.g., called from a non-video context), falls back to standard all-tokens mode.
- Existing configs with `cms_use_chunk_scheduling: false` are not affected.

---

## 6. Future Extensions (Not in Scope)

- **Per-level learning rate scaling**: Apply different `cms_lr_scale` per CMS level (e.g., slower LR for the slow level to preserve clip-level knowledge during CL task transitions).
- **Adaptive update periods**: Learn the update periods via a gating mechanism rather than fixing them as hyperparameters.
- **Diagnostic logging**: Log per-level activation statistics to W&B to verify the temporal hierarchy is capturing different abstraction levels.
