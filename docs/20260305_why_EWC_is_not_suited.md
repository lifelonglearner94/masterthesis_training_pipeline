# Why Elastic Weight Consolidation (EWC) Is Not Suited for Our Scenario

**Date:** 2026-03-05
**Context:** AC-HOPE Hybrid ViT, V-JEPA 2 feature map future frame prediction, continual learning pipeline
**Reference:** `docs/20260305_Elastic_Weight_Consolidation_Pros_and_Cons.md`

---

## 1 · The Core Argument in One Sentence

Our bottleneck is **plasticity** (learning new tasks), not **stability** (retaining old ones) — and EWC is a stability method that would make plasticity worse.

---

## 2 · Evidence from Our Results

| Experiment | Forgetting ↓ | Plasticity Error ↓ | Avg Error ↓ | Gap Closed |
|---|---|---|---|---|
| Eighth AC HOPE Run HYBRID | 0.0538 | 0.3145 | 0.3594 | 10.4% |
| Tenth AC HOPE Run (Hybrid + Replay) | 0.0188 | 0.3187 | 0.3344 | 30.9% |
| Lower Bound (Naive) | 0.0856 | 0.3007 | 0.3720 | 0.0% |
| Upper Bound (Joint) | 0.0000 | 0.2699 | 0.2502 | 100.0% |

Key observations:

- **Forgetting is already low.** The hybrid architecture achieves 0.0538 forgetting — the 2nd-lowest among all CL methods (only replay at 0.0188 is lower). There is very little stability to gain.
- **Plasticity error dominates.** Across all methods, plasticity error clusters between 0.30–0.36. The gap to the upper bound (0.2699) is far larger than the gap in forgetting. This is where meaningful improvement must come from.
- **EWC would rigidify parameters further.** Its quadratic penalty constrains weight updates — exactly the mechanism that would increase plasticity error. Expected outcome: forgetting drops marginally (0.054 → ~0.03), but plasticity error rises (0.31 → ~0.33+), yielding a **worse** overall average error.

---

## 3 · Technical Reasons EWC Is a Poor Fit

### 3.1 Regression with L1 Loss

EWC was derived for classification likelihoods. Our pipeline uses L1 loss (`loss_type: "l1"` in config), which has a **discontinuous gradient at zero**. The empirical Fisher — computed as the average of squared gradients $(\nabla_\theta \mathcal{L})^2$ — does not cleanly approximate curvature for L1. The resulting importance estimates would be noisy and theoretically unsound.

Switching to L2 only for the Fisher computation pass is possible but introduces a mismatch between what the Fisher protects (L2 surface) and what the model actually optimizes (L1 surface).

### 3.2 Diagonal Approximation Fails for Transformers

EWC relies on the **diagonal Fisher Information Matrix**, which assumes all parameters are statistically independent. In our 12-layer hybrid with self-attention, Q/K/V parameters are densely correlated — perturbing $W_Q$ deeply affects the required values of $W_K$ and $W_V$. The diagonal approximation captures none of these cross-parameter interactions.

As described in the EWC analysis doc (Section 5.1): the true low-error region in parameter space forms an oblique ellipse, but the diagonal Fisher forces it to align with coordinate axes. This geometric misalignment pushes the optimizer out of the true intersection of low-loss regions for old and new tasks.

### 3.3 The Architecture Already Has Built-In CL Mechanisms

The AC-HOPE Hybrid contains four layers of continual-learning-specific machinery:

1. **Attention**
2. **Titan memory** — per-token adaptation via Descent-Gradient-Descent (DGD), providing fast in-context learning
3. **Longterm memory** (`use_longterm_memory: true`) — cross-clip consolidation with slow EMA absorption
4. **CMS** — multi-frequency MLPs with heterogeneous update periods (1/3/7 steps)

These are *architecture-level* solutions to the same stability–plasticity dilemma that EWC addresses at the *loss-function level*. Adding EWC on top would be redundant — and likely interfering, since the Titan memory already needs parameter freedom to adapt its internal state.

### 3.4 Undersampled Fisher in High-Dimensional Output Space

We predict 256 patches × V-JEPA feature dimensions per frame. The Fisher must be estimated over this very large output space, but each CL task contains only ~1000 clips. The empirical Fisher becomes severely undersampled relative to the output dimensionality, leading to unreliable importance estimates that don't reflect true parameter sensitivity.

### 3.5 λ Tuning Is Unforgiving with Narrow Useful Range

The EWC penalty strength $\lambda$ is notoriously difficult to tune:

- **Too high** → network freezes, plasticity collapses, new tasks cannot be learned
- **Too low** → no stability benefit over naive finetuning

With our current moderate forgetting (0.054) and moderate plasticity error (0.31), the window of useful $\lambda$ values is extremely narrow. The expected gain from finding the sweet spot is marginal — at best a few percentage points of gap closed — while the GPU hours for the hyperparameter search are substantial.

---

## 4 · What EWC *Would* Get Right (Acknowledged but Insufficient)

For completeness, there are aspects of our setup where EWC's requirements are met:

- **Clean task boundaries exist.** Our pipeline has discrete task switches (base → scaling_shift → dissipation_shift → ...), so EWC's need for explicit boundaries (Section 5.5 of the analysis doc) is not an issue.
- **Only 6 phases.** Capacity saturation (Section 5.4) would not occur with so few tasks.
- **AdamW already tracks squared gradients.** The optimizer's second-moment accumulator is a running approximation of the diagonal Fisher ("Squisher" method, Section 6.4), so implementation cost would be near-zero.
- **Per-group application is possible.** Our config already has per-group LR scaling, so EWC could selectively target only attention/projection weights.

However, none of these mitigate the fundamental problem: EWC constrains the wrong axis of the stability–plasticity tradeoff for our scenario.

---

## 5 · Conclusion

EWC is designed for scenarios where forgetting is the dominant failure mode. In our V-JEPA feature prediction pipeline, forgetting is already well-controlled by the hybrid architecture's built-in memory mechanisms. The dominant failure mode is insufficient plasticity — the model's difficulty in learning new distribution shifts quickly enough. Adding EWC would further constrain weight updates, worsening the very metric that needs improvement.

**Decision: Do not implement EWC for the AC-HOPE Hybrid pipeline.**
