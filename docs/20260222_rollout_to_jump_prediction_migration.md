# Rollout → Jump Prediction Migration

**Date:** 2026-02-22
**Based on:** `docs/20260221_Rollout zu Jump Prediction_ Scientific Plan.md`

## Summary

Replaced autoregressive rollout prediction with stochastic jump prediction across the entire codebase. Instead of predicting frames sequentially (z₁→z₂→...→z_T), the model now predicts a randomly sampled future frame z_τ directly from the initial state z₀ and action a₀, using RoPE positional encoding to condition on the target timestep τ.

## Core Concept

**Before (Rollout):** Given `context_frames` ground-truth frames, autoregressively predict `T_rollout` future frames, feeding each prediction back as input for the next step.

**After (Jump):** From the initial frame z₀ and action a₀, directly predict z_τ for a randomly sampled target τ ∈ {T+1−k, ..., T}, where k = `jump_k`. The model uses RoPE temporal position override (setting all token positions to τ−1) so the transformer's output corresponds to the prediction for frame τ.

## Parameters Removed

| Old Parameter | Description |
|---|---|
| `T_rollout` | Number of autoregressive rollout steps |
| `context_frames` | Number of ground-truth context frames |
| `loss_weight_rollout` | Weight for rollout loss term |

## Parameters Added/Renamed

| New Parameter | Default | Description |
|---|---|---|
| `jump_k` | 3 | Number of candidate jump targets (τ sampled from last k frames) |
| `loss_weight_jump` | 1.0 | Weight for jump prediction loss term |

## Loss Formula

```
L(φ, e) = (1 − λ(e)) · L_teacher_forcing + λ(e) · L_jump
```

Where λ(e) is controlled by `loss_weight_jump` / curriculum schedule, and L_jump is computed by:
1. Sampling τ uniformly from {T+1−k, ..., T}
2. Single forward pass: model(z₀, a₀, target_timestep=τ)
3. Loss between prediction and ground-truth z_τ

## Files Modified

### Architecture (RoPE injection)

| File | Change |
|---|---|
| `src/models/ac_predictor/utils/modules.py` | `ACRoPEAttention.forward()` and `ACBlock.forward()` accept `target_timestep: int \| None`. When set, overrides RoPE frame positions (d_mask) to `target_timestep - 1` for all tokens |
| `src/models/ac_predictor/ac_predictor.py` | `VisionTransformerPredictorAC.forward()` propagates `target_timestep` through all blocks |
| `src/models/hope/hope_block.py` | `HOPEBlock.forward()`, `_titan_forward()`, `_titan_forward_chunk()`, `_apply_rope()` all accept and use `target_timestep` |
| `src/models/hope/ac_hope_vit.py` | `ACHOPEViT.forward()` and `_process_hope_blocks()` propagate `target_timestep` |

### Loss Computation

| File | Change |
|---|---|
| `src/models/mixins/loss_mixin.py` | Removed `_compute_rollout_loss()` and `_compute_rollout_loss_per_timestep()`. Added `_compute_jump_loss()` (samples τ, single forward pass with `target_timestep`) and `_compute_jump_loss_per_timestep()` (iterates all τ for test analysis). Updated `_shared_step()` to use `loss_jump` / `loss_weight_jump` |

### Lightning Modules

| File | Change |
|---|---|
| `src/models/ac_predictor/lightning_module.py` | Constructor: removed `T_rollout`/`context_frames`, added `jump_k`. `_step_predictor()` accepts `target_timestep`. Curriculum validates `jump_k` vs `num_timesteps`. Test step uses `_compute_jump_loss_per_timestep`. TTA dispatches to `_tta_process_clip_jump`. All logging keys use `tau_` prefix for per-timestep metrics |
| `src/models/hope/ac_hope_module.py` | Same pattern as `lightning_module.py` |

### Baseline Models

| File | Change |
|---|---|
| `src/models/baseline_convlstm.py` | Accepts `jump_k` / `loss_weight_jump`. `_step_predictor()` accepts `target_timestep` (ignored — no RoPE). Removed `context_frames` / `T_rollout` |
| `src/models/baseline_identity.py` | Same pattern as ConvLSTM baseline |

### TTA (Test Time Adaptation)

| File | Change |
|---|---|
| `src/models/mixins/tta_mixin.py` | `_tta_full_rollout()` → `_tta_jump_prediction()`: for each τ in range, jump-predict from z₀. `_tta_process_clip_full_rollout()` → `_tta_process_clip_jump()`: adapt-then-evaluate with jump predictions. `_tta_process_clip_sequential()` rewritten with `_run_sequential_jump_adaptation()`: sequential adapt on each τ. Default `tta_mode` changed from `"full_rollout"` to `"jump"` |
| `src/models/ac_predictor/tta_wrapper.py` | `RolloutTTAAgent` → `JumpTTAAgent`. `step()` accepts `target_timestep` and passes to model. `SequentialTTAProcessor` rewritten for jump targets (iterates τ instead of autoregressive steps) |
| `src/models/ac_predictor/__init__.py` | Exports `JumpTTAAgent` instead of `RolloutTTAAgent` |

### Other Source

| File | Change |
|---|---|
| `src/eval.py` | Print statements updated: `context_frames` → removed, `T_rollout` → `jump_k` |

### Config Files (all YAML under `configs/`)

All 14+ YAML files updated:
- `T_rollout: N` → `jump_k: 3` (default)
- `context_frames: 1` → removed
- `loss_weight_rollout: 1.0` → `loss_weight_jump: 1.0`
- `tta_mode: "full_rollout"` → `tta_mode: "jump"`
- Comments updated to reflect jump prediction terminology

Affected configs:
- `configs/model/ac_predictor.yaml`
- `configs/model/ac_hope_vit.yaml`
- `configs/model/ac_predictor_tta.yaml`
- `configs/model/baseline_convlstm.yaml`
- `configs/model/baseline_identity.yaml`
- `configs/experiment/vjepa2_ac.yaml`
- `configs/experiment/ac_hope_vit_depth_matched.yaml`
- `configs/experiment/ac_hope_vit_param_matched.yaml`
- `configs/experiment/test_ac_predictor.yaml`
- `configs/experiment/test_ac_predictor_tta.yaml`
- `configs/experiment/test_ac_predictor_tta_extended.yaml`
- `configs/experiment/test_baseline_identity.yaml`
- `configs/experiment/baseline_convlstm.yaml`

### Tests

| File | Change |
|---|---|
| `tests/test_ac_predictor.py` | `TestRolloutLogic` → `TestJumpPredictionLogic`. All fixtures use `jump_k` instead of `T_rollout`/`context_frames`. Calls to `_compute_rollout_loss` → `_compute_jump_loss`. Curriculum schedule dicts use `jump_k` keys |

## RoPE Override Mechanism

The key technical detail: when `target_timestep=τ` is passed, the model overrides the temporal dimension of RoPE positional encoding. Normally, each frame gets its sequential position (0, 1, 2, ...). With jump prediction, all input tokens get position `τ−1`, so the model's output at that position corresponds to prediction of frame τ.

```python
# In ACRoPEAttention.forward():
if target_timestep is not None:
    d_mask = torch.full_like(d_mask, target_timestep - 1)
```

This works because:
- Teacher forcing: output at position k predicts frame k+1
- Jump: set all positions to τ−1, so output predicts frame τ
- Causal mask is sliced to [N+2, N+2] for single-frame input — full self-attention within the frame

## Verification

Final grep confirms zero remaining references to old terminology:
```bash
grep -rn 'T_rollout|context_frames|loss_weight_rollout|full_rollout|RolloutTTA|_compute_rollout' \
  src/ configs/ tests/ --include='*.py' --include='*.yaml'
# → No matches
```
