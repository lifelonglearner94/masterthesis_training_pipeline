# Understanding `T_rollout` in the AC Predictor

## What is `T_rollout`?

`T_rollout` controls **how many autoregressive prediction steps** the rollout loss computes. It determines how far into the future the model must predict using its own predictions (not ground-truth).

---

## Example: Training Iteration with `T_rollout=2` vs `T_rollout=4`

Assume you have 8 encoded timesteps (frames 0–7), so `features.shape = [B, 8, N, D]`.

### With `T_rollout=2` (current default):

| Step | Input Context | Action/State Context | Predicts | Source |
|------|---------------|---------------------|----------|--------|
| Seed | `z_0` (GT) | `a[0:1]`, `s[0:1]` | `z̃_1` | TF prediction |
| AR-1 | `[z_0, z̃_1]` | `a[0:2]`, `s[0:2]` | `z̃_2` | Model's own prediction |

**Loss:** Compare `[z̃_1, z̃_2]` against ground-truth `[z_1, z_2]`

→ Model only needs to maintain accuracy for **2 steps** of self-feeding.

---

### With `T_rollout=4`:

| Step | Input Context | Action/State Context | Predicts | Source |
|------|---------------|---------------------|----------|--------|
| Seed | `z_0` (GT) | `a[0:1]`, `s[0:1]` | `z̃_1` | TF prediction |
| AR-1 | `[z_0, z̃_1]` | `a[0:2]`, `s[0:2]` | `z̃_2` | Own prediction |
| AR-2 | `[z_0, z̃_1, z̃_2]` | `a[0:3]`, `s[0:3]` | `z̃_3` | Own prediction |
| AR-3 | `[z_0, z̃_1, z̃_2, z̃_3]` | `a[0:4]`, `s[0:4]` | `z̃_4` | Own prediction |

**Loss:** Compare `[z̃_1, z̃_2, z̃_3, z̃_4]` against ground-truth `[z_1, z_2, z_3, z_4]`

→ Model must maintain accuracy for **4 steps** of self-feeding — errors compound!

---

## Key Differences

| Aspect | `T_rollout=2` | `T_rollout=4` |
|--------|---------------|---------------|
| **Error accumulation** | 2 steps to compound | 4 steps to compound |
| **Training difficulty** | Easier | Harder |
| **Long-horizon planning** | Weaker | Stronger |
| **Compute cost** | 2 forward passes | 4 forward passes |
| **Risk of collapse** | Lower | Higher (needs stable TF anchor) |

---

## Practical Recommendation

From `meine_eigene_Loss_Logik.md`:

> Start with `T_rollout=2`, then increase (4 → 6 → 8) while reducing `loss_weight_teacher`

This curriculum prevents model collapse — the teacher-forcing loss provides a stable "anchor to reality" while the rollout loss gradually teaches long-horizon consistency.
