# Continual Learning in V-JEPA World Models: Balancing Stability and Plasticity through Self-Modifying Memory — Thesis Overview

**Research question:** Can self-modifying memory (HOPE) reduce catastrophic forgetting in Action-Conditioned world models under continual distribution shift?

We train Action-Conditioned (AC) predictors on pre-encoded **V-JEPA 2** latent features ($T{=}8$ timesteps, $N{=}256$ patches, $D{=}1024$) from a physics simulation dataset where objects receive force impulses and slide across surfaces. Models must adapt to **five progressive dynamic shifts** without forgetting previously learned physics.

---

## Continual Learning Protocol

```
Base Training (5000 clips, standard physics) → Full Eval
  → Task 1: Scaling Shift          (1000 clips) → Full Eval
  → Task 2: Dissipation / Ice      (1000 clips) → Full Eval
  → Task 3: Discretization / Walls (1000 clips) → Full Eval
  → Task 4: Kinematics / Rotation  (1000 clips) → Full Eval
  → Task 5: Compositional OOD      (1000 clips) → Full Eval
```

After each phase, evaluation runs on fixed validation clips from **every** partition (no weight updates). The resulting R-matrix $R[i,j]$ (loss on task $j$ after training on experience $i$) yields all CL metrics.

**Loss:** Stochastic Jump Prediction (L1 in V-JEPA 2 feature space). A random future timestep $\tau$ is predicted directly from the initial state, eliminating autoregressive error compounding:

$$\mathcal{L}_{\text{jump}}(\phi) = \| P_\phi(a_{1:\tau},\, s_1,\, z_1) - z_\tau \|_1$$

---

## Experiments

### Bounding Baselines

| Experiment | Description | ~Params |
|---|---|---|
| **Upper Bound** (Joint) | i.i.d. training on all data simultaneously — theoretical ceiling | 43M |
| **Lower Bound** (Naive) | Sequential fine-tuning, no CL mechanisms — maximum forgetting | 43M |

### HOPE Variants (Novel Architectures)

| Experiment | Architecture | Key Idea |
|---|---|---|
| **AC-HOPE Phase 6** | Titan Memory + CMS (depth 5) | Persistent `M_longterm` surviving across clips; clip-level `M_memory` resets for plasticity; learned gate interpolates both |
| **AC-HOPE Phase 8 Hybrid** | Attention + Titan + CMS (depth 12) | Full self-attention with 3D RoPE *plus* Titan memory as augmentation per block |
| **AC-HOPE Phase 10 Replay** | Phase 8 Hybrid + soft-freeze + replay | Attention near-frozen (0.02× LR), Titan reset at task boundaries, 30% reservoir replay |

### Standard Comparison Architectures (~43M, naive fine-tuning)

- **Transformer++** — Pre-norm Transformer with RoPE, GatedMLP
- **RetNet** — Retentive Network with multi-scale retention
- **GatedDeltaNet** — Gated delta-rule linear attention (Triton-fused)

### Additional Baselines

- **TTA (LayerNorm)** — Test-time adaptation of only LayerNorm params on the frozen AC-ViT
- **AC Titans** — Standalone Titans memory (no CMS, no attention hybrid)

---

## Results

> **Regression task.** All metrics use **Jump L1 Error** (lower is better).
> Feature space $\sigma = 3.18$; an error of 0.37 ≈ 11.6% of σ.

### At a Glance

| Experiment | Avg Error ↓ | Forgetting ↓ | Plasticity ↓ | Gap Closed |
|---|---:|---:|---:|---:|
| **Upper Bound (Joint)** | **0.2502** | 0.0000 | 0.2699 | 100.0% |
| **AC-HOPE Ph.10 Replay** | **0.3344** | **0.0188** | 0.3187 | **30.9%** |
| AC-HOPE Ph.8 Hybrid | 0.3594 | 0.0538 | 0.3145 | 10.4% |
| AC-HOPE Ph.6 (DNH Hybrid) | 0.3637 | 0.0534 | 0.3192 | 6.9% |
| Transformer++ | 0.3660 | 0.0798 | **0.2995** | 5.0% |
| RetNet | 0.3683 | 0.0808 | 0.3009 | 3.1% |
| AC-HOPE Ph.6 (Sixth Run) | 0.3696 | 0.0626 | 0.3174 | 2.0% |
| GatedDeltaNet | 0.3717 | 0.0820 | 0.3034 | 0.2% |
| **Lower Bound (Naive)** | 0.3720 | 0.0856 | 0.3007 | 0.0% |
| AC Titans | 0.3794 | 0.0738 | 0.3179 | −6.1% |
| TTA LayerNorm | 0.3946 | 0.0699 | 0.3363 | −18.5% |
| AC-HOPE Ph.1 (First Run) | 0.3958 | 0.0554 | 0.3496 | −19.5% |

### Key Observations

1. **Phase 10 Replay is the clear winner** among CL methods, closing **30.9%** of the gap to the upper bound — driven almost entirely by drastically reduced forgetting (0.019 vs. 0.086 for naive).
2. **Forgetting reduction is consistent across HOPE variants.** Even without replay, Phase 8 Hybrid and Phase 6 cut forgetting by 37–48% relative to the lower bound (0.053 vs. 0.086), confirming that Titan memory provides meaningful stability.
3. **Standard architectures cluster near the lower bound.** Transformer++ (5.0%), RetNet (3.1%), and GatedDeltaNet (0.2%) show marginal improvements, indicating that architectural inductive biases alone (linear attention, retention) are insufficient for CL without explicit memory mechanisms.
4. **Plasticity is not the bottleneck.** The best plasticity belongs to Transformer++ (0.2995) and the lower bound (0.3007) — models that forget the most. The stability–plasticity tradeoff is real: HOPE variants trade slightly worse plasticity for substantially better stability.
5. **TTA and early HOPE phases underperform the naive baseline**, showing that limited adaptation (LayerNorm-only) or immature architectures (Phase 1/2) can hurt more than help.
6. **Experience replay is the single most impactful mechanism.** Comparing Phase 10 vs. Phase 8 (same backbone), adding soft-freeze + replay drops forgetting from 0.054 → 0.019 and improves gap-closed from 10.4% → 30.9%.

### Stability–Plasticity Tradeoff

The fundamental CL tension: ideal methods sit in the **bottom-left** (low forgetting, low plasticity error).

- **Phase 10 Replay** achieves the best stability (forgetting = 0.019) with competitive plasticity (0.319), placing it closest to the ideal corner among all CL methods.
- **Standard architectures** (Transformer++, RetNet, GDN) cluster in the high-forgetting / good-plasticity region — they learn new tasks well but overwrite old knowledge.
- **HOPE variants without replay** occupy the middle ground — significantly better stability than standard models, with only modestly higher plasticity error.

---

## Metric Definitions

| Metric | Description |
|---|---|
| **Avg Error** | Mean L1 across all tasks after final training phase (`Top1_L1_Stream`) |
| **Forgetting** | Avg increase in old-task error over training (`StreamForgetting`; 0 = perfect stability) |
| **Plasticity** | Mean R-matrix diagonal — immediate error when first learning each task |
| **Gap Closed** | $\frac{\text{LB} - \text{Method}}{\text{LB} - \text{UB}} \times 100\%$ — how much of the lower→upper bound gap is closed |
