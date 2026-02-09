# Scientific Protocol: Fixed 3-Context Frame Autoregressive Rollout

**Date:** January 21, 2026
**Version:** 1.0
**Status:** Implemented

---

## 1. Objective

Modify the rollout loss computation to use a **fixed number of ground-truth context frames** (3) to predict the **remaining frames autoregressively** (5), rather than the previous approach of seeding with 1 ground-truth frame plus a teacher-forcing prediction.

### Rationale

- **Improved stability**: Using 3 ground-truth frames as context provides a more robust initialization for autoregressive rollout
- **Consistent evaluation**: Fixed context/prediction split enables fair comparison across experiments
- **Real-world alignment**: Mirrors deployment scenarios where initial observations are available before prediction is needed

---

## 2. Temporal Structure

### 2.1 Data Dimensions

| Dimension | Value | Description |
|-----------|-------|-------------|
| Original video frames | 16 | Raw video input |
| Encoded timesteps | 8 | After V-JEPA2 tubelet encoding (16 ÷ 2) |
| Context frames (C) | **3** | Ground-truth frames z₀, z₁, z₂ |
| Prediction steps (T_rollout) | **5** | Autoregressive predictions z₃, z₄, z₅, z₆, z₇ |

### 2.2 Frame Indexing

```
Timestep:     0     1     2  │  3     4     5     6     7
             ─────────────────┼──────────────────────────────
Context:     z₀    z₁    z₂  │
             (ground-truth)   │
                              │
Predictions:                  │  ẑ₃    ẑ₄    ẑ₅    ẑ₆    ẑ₇
                              │  (autoregressive)
```

---

## 3. Algorithm

### 3.1 Rollout Loss Computation

**Input:**
- Features: `[B, 8, N, D]` — 8 encoded frames, N patches, D dimensions
- Actions: `[B, 7, action_dim]` — 7 action steps
- States: `[B, 7, action_dim]` — 7 state steps

**Procedure:**

1. **Initialize context** with C=3 ground-truth frames:
   ```
   z_ar = [z₀, z₁, z₂]  # Shape: [B, 3×N, D]
   ```

2. **Autoregressive loop** for T_rollout=5 steps:
   ```
   for step in range(5):
       target_frame = 3 + step  # Predicting frame 3, 4, 5, 6, 7
       num_actions = 3 + step   # Use actions 0..(2+step)

       ẑ_next = Predictor(z_ar, actions[:, :num_actions], states[:, :num_actions])
       ẑ_next = ẑ_next[:, -N:]  # Extract last frame prediction

       z_ar = concat(z_ar, ẑ_next)  # Append prediction to context
   ```

3. **Compute loss** between predictions and ground-truth:
   ```
   predictions = z_ar[:, 3×N:]           # Frames 3-7 (predicted)
   targets = features[:, 3:8]            # Frames 3-7 (ground-truth)
   loss = mean(|predictions - targets|)  # L1 loss
   ```

### 3.2 Action/State Alignment

| Prediction Step | Target Frame | Actions Used | States Used |
|-----------------|--------------|--------------|-------------|
| 0 | z₃ | a₀, a₁, a₂ | s₀, s₁, s₂ |
| 1 | z₄ | a₀, a₁, a₂, a₃ | s₀, s₁, s₂, s₃ |
| 2 | z₅ | a₀, a₁, a₂, a₃, a₄ | s₀, s₁, s₂, s₃, s₄ |
| 3 | z₆ | a₀, a₁, a₂, a₃, a₄, a₅ | s₀, s₁, s₂, s₃, s₄, s₅ |
| 4 | z₇ | a₀, a₁, a₂, a₃, a₄, a₅, a₆ | s₀, s₁, s₂, s₃, s₄, s₅, s₆ |

---

## 4. Configuration

### 4.1 Key Parameters

```yaml
model:
  context_frames: 3    # Fixed: number of ground-truth context frames
  T_rollout: 5         # Fixed: number of autoregressive prediction steps
  T_teacher: 7         # Teacher-forcing still predicts all 7 steps
```

### 4.2 Curriculum Schedule (Loss Weights Only)

The curriculum no longer modifies `T_rollout` or `context_frames`. Only `loss_weight_teacher` is adjusted:

| Epoch | loss_weight_teacher | loss_weight_rollout | Effect |
|-------|---------------------|---------------------|--------|
| 0-4 | 1.0 | 1.0 | Balanced training |
| 5-6 | 0.7 | 1.0 | Shift toward rollout |
| 7+ | 0.3 | 1.0 | Rollout-dominant |

---

## 5. Comparison with Previous Approach

### 5.1 Previous: Dynamic Seeding (V-JEPA2 Paper)

```
Context: z₀ (ground-truth) + ẑ₁ (teacher-forcing prediction)
Predict: ẑ₂, ẑ₃, ... (T_rollout steps, dynamically adjusted)
```

**Characteristics:**
- Single ground-truth frame
- First prediction via teacher-forcing
- T_rollout increased via curriculum (2 → 4 → 6)

### 5.2 Current: Fixed 3-Frame Context

```
Context: z₀, z₁, z₂ (all ground-truth)
Predict: ẑ₃, ẑ₄, ẑ₅, ẑ₆, ẑ₇ (fixed 5 steps)
```

**Characteristics:**
- Three ground-truth frames
- No teacher-forcing in rollout seeding
- Fixed prediction horizon

### 5.3 Key Differences

| Aspect | Previous | Current |
|--------|----------|---------|
| Ground-truth context | 1 frame | 3 frames |
| Seeding method | GT + TF prediction | GT only |
| T_rollout | Dynamic (2→6) | Fixed (5) |
| First predicted frame | z₁ | z₃ |
| Curriculum adjusts | T_rollout + loss weights | Loss weights only |

---

## 6. Implementation Details

### 6.1 Modified Files

1. **`src/models/ac_predictor/lightning_module.py`**
   - Added `context_frames` parameter to `__init__`
   - Rewrote `_compute_rollout_loss` method

2. **`configs/experiment/vjepa2_ac.yaml`**
   - Set `context_frames: 3`
   - Set `T_rollout: 5`
   - Removed `T_rollout` from curriculum schedule

3. **`configs/model/ac_predictor.yaml`**
   - Added `context_frames: 1` as default (backward compatible)

### 6.2 Backward Compatibility

Setting `context_frames: 1` with `T_rollout: 7` reproduces behavior similar to the original approach (though without the teacher-forcing seed step).

---

## 7. Expected Outcomes

### 7.1 Hypotheses

1. **More stable training**: 3-frame context reduces early rollout errors
2. **Better long-horizon predictions**: Model learns to predict further into the future
3. **Cleaner loss curves**: Fixed horizon eliminates curriculum-induced discontinuities

### 7.2 Metrics to Monitor

- `train/loss_rollout` — Should decrease smoothly
- `val/loss_rollout` — Generalization of long-horizon predictions
- `train/loss_teacher` — Anchor loss, should remain stable

---

## 8. References

- V-JEPA2 Paper: Action-Conditioned Predictor architecture
- `docs/T_rollout_explanation.md` — Original rollout mechanics
- `docs/meine_eigene_Loss_Logik.md` — Curriculum learning rationale
