# Scientific Evaluation: Are the V-JEPA 2-AC Training Settings Appropriate for the HOPE Architecture?

**Date:** 2026-02-09
**Scope:** `configs/experiment/ac_hope_vit_depth_matched.yaml`, `configs/experiment/ac_hope_vit_param_matched.yaml`
**Baseline reference:** `configs/experiment/vjepa2_ac.yaml`
**Reference papers:** Behrouz 2025 (HOPE/Titans), Assran et al. (V-JEPA 2)

---

## 1. Executive Summary

The HOPE experiment configs (`ac_hope_vit_depth_matched`, `ac_hope_vit_param_matched`) carry over nearly **all** optimizer and training settings verbatim from the baseline V-JEPA 2-AC predictor (`vjepa2_ac.yaml`). While some of these transfers are scientifically justifiable for the sake of controlled comparison, **several settings are problematic** because the HOPE architecture fundamentally differs from a standard Transformer — it performs nested optimization with inner-loop gradient descent during the forward pass, has 5 self-modifying MLP memories per block, and uses a CMS multi-frequency hierarchy.

Below is a setting-by-setting evaluation with a verdict (✅ Appropriate, ⚠️ Questionable, ❌ Problematic) and concrete suggestions.

---

## 2. Setting-by-Setting Analysis

### 2.1 Learning Rate: `learning_rate: 4.25e-4`

| Aspect | Assessment |
|--------|------------|
| **Origin** | V-JEPA 2 paper (Appendix B.1) — tuned for a 300M-parameter standard Transformer predictor on 256-batch training with ~62h of robot video |
| **Verdict** | ⚠️ **Likely too high for HOPE** |

**Reasoning:**

The HOPE architecture has a fundamentally different gradient landscape than a standard Transformer:

1. **Bi-level optimization:** The outer optimizer updates meta-learned initial states of Titan memories, which then undergo inner-loop DGD updates during the forward pass. This creates a bi-level optimization surface with **higher curvature** than a standard Transformer's loss landscape. Meta-learning literature (Finn et al., MAML 2017; Nichol et al., Reptile 2018) consistently finds that outer-loop learning rates should be **lower** than single-level LR by 2×–10×.

2. **Gradient amplification through the memory chain:** With `chunk_size=1`, the DGD update creates a computation chain where the loss flows through `new_w = alpha * w_old - eta * grad`. The gradient magnitudes for `M_memory` are amplified relative to a direct attention layer because the loss signal passes through the functional MLP *and* the DGD update equation. The fix protocol documents inner-loop gradient norms of ~1.8M even with clipping — a sign of high sensitivity.

3. **Parameter asymmetry:** The depth-matched config has 83M params (vs. 43M baseline), and the param-matched has 42M but with only 6 layers. The optimal LR scales with model size and effective depth.

4. **Titan `lr_scale=0.5` partially compensates** — reducing Titan LR to `2.125e-4` — but this may still be too aggressive for meta-learned initial states.

**Recommendation:**

- **Reduce base LR to `1.0e-4` – `2.0e-4`** for HOPE experiments. This is closer to the default in `configs/model/ac_hope_vit.yaml` (`learning_rate: 1e-4`), which was presumably chosen with HOPE in mind before being overridden by the experiment configs.
- **Reduce `titan_lr_scale` to `0.1` – `0.3`** (i.e., Titan LR = `1e-5` to `6e-5`). Meta-learned initial states need conservative outer LR to avoid oscillation.
- **Keep `cms_lr_scale` at `1.0`** — CMS MLPs are standard feedforward blocks without inner-loop complications.

---

### 2.2 Weight Decay: `weight_decay: 0.04`

| Aspect | Assessment |
|--------|------------|
| **Origin** | V-JEPA 2 paper (Appendix B.1) |
| **Verdict** | ⚠️ **Problematic for Titan memories, fine for CMS** |

**Reasoning:**

Weight decay (L2 regularization) in AdamW shrinks parameters toward zero each step: $\theta \leftarrow (1 - \lambda) \theta - \eta \nabla L$. For Titan memories, this creates a **conflict**:

1. **Titan `nn.Parameters` are meta-learned initial states** ($\mathcal{M}_{\Box,0}$). They are the starting point for in-context adaptation. Shrinking them toward zero means the memories start each sequence from an increasingly identity-like state. While this *could* be beneficial (bias toward minimal modification), it fights against the meta-learning objective of learning *optimal* initial states.

2. **The DGD update rule already has its own weight decay** via $\alpha_t$ (the adaptive forget gate): $\mathcal{M}_t = \mathcal{M}_{t-1}(\alpha_t I - \eta_t k_t k_t^\top) - \eta_t \nabla L$. The $\alpha_t$ term controls in-context forgetting. Applying *additional* outer-loop weight decay on top means **double regularization** — one from AdamW and one from the inner-loop $\alpha$.

3. CMS blocks are standard MLPs where `0.04` weight decay is well-established and appropriate.

**Recommendation:**

- **Titan memories:** Reduce weight decay to `0.0` – `0.01` for the Titan parameter group. This requires extending `configure_optimizers()` to set per-group weight decay (currently all groups share `weight_decay: 0.04`).
- **CMS and projections:** Keep `0.04`.

---

### 2.3 Betas: `betas: [0.9, 0.999]`

| Aspect | Assessment |
|--------|------------|
| **Origin** | Standard AdamW defaults, also used in V-JEPA 2 |
| **Verdict** | ✅ **Acceptable, but consider adjustment** |

**Reasoning:**

- $\beta_1 = 0.9$ (momentum) and $\beta_2 = 0.999$ (squared gradient EMA) are the default AdamW settings and broadly appropriate.
- For meta-learning settings, some works (e.g., Meta-SGD, MAML++) use **lower $\beta_1$** (0.5–0.9) to reduce momentum's smoothing effect, which can mask the noisy but informative bi-level gradients.
- $\beta_2 = 0.999$ means the adaptive learning rate normalizer has very long memory — this is fine and can actually help stabilize the noisy meta-gradients.

**Recommendation:**

- Keep `betas: [0.9, 0.999]` as default. Optionally try `[0.5, 0.999]` for the Titan parameter group as an ablation.

---

### 2.4 Iteration-Based LR Schedule: `use_iteration_scheduler: true` with Warmup–Constant–Decay

| Aspect | Assessment |
|--------|------------|
| **Origin** | V-JEPA 2 paper: 4,500 warmup → 85,500 constant → 4,500 decay (at 256 batch, ~94,500 total iters) |
| **Copied as** | `warmup_pct: 0.085`, `constant_pct: 0.83`, `decay_pct: 0.085` |
| **Verdict** | ⚠️ **Schedule shape is fine, but warmup may be too short for HOPE** |

**Reasoning:**

1. **Schedule shape:** Warmup → Constant → Cosine Decay is a proven schedule for vision transformers. The shape itself transfers well.

2. **Warmup duration matters more for HOPE:** The bi-level optimization surface has higher curvature early in training, when memories are randomly initialized and the inner-loop DGD updates produce noisy meta-gradients. The V-JEPA 2 baseline only has standard Transformer parameters to warm up; HOPE additionally needs the meta-learned initial states to stabilize. With ~8.5% warmup, at batch_size=32 and 30 epochs, the warmup period may be only a few hundred steps — potentially too short.

3. **`warmup_start_lr: 7.5e-5`** — this is reasonable as a starting LR and actually in the range we'd recommend for peak LR, suggesting the warmup range itself might be fine but the target peak LR is the issue (see §2.1).

**Recommendation:**

- **Increase `warmup_pct` to `0.15` – `0.20`** for HOPE experiments, giving the meta-learning more time to stabilize before hitting peak LR.
- Consider a **warmup → constant → cosine** schedule (the current implementation linearly decays to 0 in the decay phase rather than using cosine; verify if this is intentional).

---

### 2.5 Curriculum Schedule

```yaml
curriculum_schedule:
  - epoch: 0
    loss_weight_teacher: 1.0
  - epoch: 7
    loss_weight_teacher: 0.7
  - epoch: 11
    loss_weight_teacher: 0.3
```

| Aspect | Assessment |
|--------|------------|
| **Origin** | Custom schedule for V-JEPA 2-AC baseline |
| **Verdict** | ⚠️ **Concept is valid, but timing needs adjustment for HOPE** |

**Reasoning:**

The curriculum reduces teacher-forcing weight over training, shifting emphasis toward rollout loss. This is pedagogically sound: learn single-step prediction first, then autoregressive multi-step prediction.

**However, HOPE needs longer stabilization:**

1. **Epoch 7 is premature:** In a 30-epoch run, phase 2 starts at 23%. HOPE's Titan memories are still learning to produce meaningful adaptive projections at this point. The meta-learned initial states need the stable teacher-forcing signal for longer. For the baseline Transformer, epoch 7 is fine because attention parameters converge faster.

2. **Epoch 11 is very aggressive:** Reducing teacher weight to 0.3 at 37% of training means the model is heavily rollout-dominant while Titan memories may not have converged. Rollout loss has **compounding errors** — for HOPE, where each prediction step involves DGD memory updates, errors compound *through* the memory state as well, not just through the feature predictions.

3. **No `loss_weight_rollout` schedule:** The rollout weight stays at 1.0 throughout. For HOPE, it might be beneficial to *increase* rollout weight gradually (e.g., 0.5 → 1.0 → 1.5) rather than *decrease* teacher weight, keeping a stable teacher signal while adding rollout pressure.

**Recommendation:**

- **Delay phase transitions:**
  ```yaml
  curriculum_schedule:
    - epoch: 0
      loss_weight_teacher: 1.0
    - epoch: 12
      loss_weight_teacher: 0.7
    - epoch: 20
      loss_weight_teacher: 0.5
  ```
- **Never go below 0.5 for `loss_weight_teacher`** in the first HOPE experiments. Completely disabling teacher forcing removes the most stable gradient signal the meta-learning relies on.
- **Consider coupling the curriculum with HOPE diagnostics:** Only advance to the next phase when `hope/mean_surprise` stabilizes (i.e., memories are producing useful predictions).

---

### 2.6 Teacher-Forcing and Rollout Loss Design

```yaml
T_teacher: 7
T_rollout: 7
context_frames: 1
loss_weight_teacher: 1.0
loss_weight_rollout: 1.0
```

| Aspect | Assessment |
|--------|------------|
| **Origin** | V-JEPA 2 paper uses T=2 for rollout in their 300M predictor; the config maximizes both to 7 |
| **Verdict** | ❌ **T_rollout=7 is too aggressive for HOPE** |

**Reasoning:**

1. **The original V-JEPA 2 paper uses T_rollout=2.** The baseline config already deviates from the paper by using T_rollout=7. For a standard Transformer, this is already aggressive but manageable because attention is a well-conditioned operation.

2. **HOPE rollout compounds through memory state evolution:** During rollout, each step $t$ feeds the prediction $\hat{z}_t$ back as input. In HOPE, this means:
   - The Titan memories receive $\hat{z}_t$ (potentially noisy) as input
   - DGD updates the memory based on this noisy input
   - The corrupted memory state then produces the next prediction $\hat{z}_{t+1}$

   This creates a **feedback loop between prediction errors and memory corruption** that doesn't exist in the standard Transformer baseline. Over 7 rollout steps, this can cause catastrophic error accumulation.

3. **Memory reset semantics during rollout:** The code calls `reset_all_memories()` once at the start of `training_step()`. During rollout, the memories evolve through 7 steps of DGD updates with increasingly unreliable inputs. The memory state diverges from what it would be with ground-truth inputs.

4. **Gradient paths through 7-step rollout with chunk_size=1:** With `chunk_size=1` and `T_rollout=7`, the computation graph becomes very deep: each of the 7 rollout steps × 12 or 6 HOPE blocks × DGD updates per block. This can cause vanishing/exploding gradients in the meta-learning chain.

**Recommendation:**

- **Start with `T_rollout: 2`** (matching the original V-JEPA 2 paper). This is more stable for initial HOPE training.
- **Optionally schedule T_rollout via curriculum:** Start at 2, increase to 4 at epoch 15, then to 7 at epoch 25 (only after teacher-forcing has stabilized).
- **Consider resetting memories between rollout steps** (or at least periodically) to prevent memory state corruption from feeding back into predictions.

---

### 2.7 Gradient Clipping: `gradient_clip_val: 1.01`

| Aspect | Assessment |
|--------|------------|
| **Origin** | V-JEPA 2 paper |
| **Verdict** | ⚠️ **Too tight for HOPE, but direction is correct** |

**Reasoning:**

1. HOPE has **two levels of gradient clipping**: the outer clip (`gradient_clip_val: 1.01` applied by PyTorch Lightning) and the inner clip (`titan_grad_clip_inner: 1.0` inside the DGD update).

2. The outer clip at 1.01 is very tight. In a standard Transformer, gradient norms are relatively predictable. In HOPE, the meta-gradients through the DGD chain can have higher variance, especially early in training. A clip at 1.01 may **suppress informative gradient signal**, effectively preventing the meta-learning from making meaningful parameter updates.

3. The gradient flow fix protocol documents that inner-loop gradient norms are ~1.8M before clipping — the gradient landscape is inherently more volatile.

**Recommendation:**

- **Increase `gradient_clip_val` to `2.0` – `5.0`** for HOPE experiments. The inner-loop clipping (`titan_grad_clip_inner: 1.0`) already provides local stability; the outer clip should be more permissive to allow meta-learning gradients through.
- Monitor gradient norm histograms via wandb to find the optimal threshold empirically.

---

### 2.8 Max Epochs: `max_epochs: 30`

| Aspect | Assessment |
|--------|------------|
| **Origin** | Same as baseline |
| **Verdict** | ⚠️ **Likely insufficient for HOPE** |

**Reasoning:**

1. **Meta-learning converges slower:** The bi-level structure means each outer step makes smaller effective progress (it needs to account for the inner-loop behavior). Meta-learning methods typically require 2–5× more epochs than single-level optimization.

2. **With reduced LR (per §2.1), longer training is needed** to compensate.

3. **Curriculum extends training:** If we delay curriculum phases (per §2.5), we need more total epochs for the rollout-dominant phase to have sufficient training time.

**Recommendation:**

- **Increase to `max_epochs: 50` – `80`** for HOPE experiments.
- Use early stopping based on validation loss with `patience=10` to avoid wasted compute if the model converges earlier.

---

### 2.9 `titan_detach_interval: 4` (Depth-Matched) / Not Set (Param-Matched)

| Aspect | Assessment |
|--------|------------|
| **Origin** | Novel setting for HOPE |
| **Verdict** | ✅ **Reasonable engineering tradeoff** |

**Reasoning:**

Periodic detachment truncates the meta-gradient chain to save VRAM. With `detach_interval=4` and `chunk_size=1`, the gradient chain through memory updates is at most 4 steps deep before detachment. The gradient flow tests confirm that `M_memory`, `M_eta`, and `M_alpha` still receive gradients even with aggressive detachment.

**The param-matched config uses `chunk_size=1` with `titan_detach_interval=4`** — this is reasonable. Memory gets 4 steps of gradient flow before being cut.

**Note:** The tests show that `M_k` and `M_v` are structurally untrained regardless of detach settings (22–23% of Titan params are dead weight). This is a separate architectural issue, not a settings issue.

---

### 2.10 `surprise_threshold: 0.0`

| Aspect | Assessment |
|--------|------------|
| **Origin** | Default: always update memories |
| **Verdict** | ✅ **Appropriate for initial experiments** |

Setting threshold to 0 means all memories update at every step. This maximizes learning signal. A non-zero threshold could be explored later to improve efficiency, but for initial experiments, always-update is the safe choice.

---

### 2.11 Loss Function: L1 (MAE) via `loss_exp: 1.0`

| Aspect | Assessment |
|--------|------------|
| **Origin** | V-JEPA 2 paper |
| **Verdict** | ✅ **Appropriate** |

L1 loss is robust to outliers in V-JEPA embeddings (which can spike to ±37). This is equally valid for HOPE since the input/output space is identical — both architectures predict the same V-JEPA 2 feature maps. The loss function is independent of the backbone architecture.

---

### 2.12 Batch Size: 32 (Depth-Matched) / 64 (Param-Matched)

| Aspect | Assessment |
|--------|------------|
| **Origin** | Adapted for VRAM constraints |
| **Verdict** | ⚠️ **Consider the meta-learning implications** |

**Reasoning:**

1. Smaller batch sizes introduce more gradient noise per step. For meta-learning, this can be **both beneficial** (implicit regularization) and **harmful** (noisy meta-gradient estimates).

2. The V-JEPA 2 paper uses batch_size=256. The baseline uses 64. The depth-matched HOPE uses 32 — an 8× reduction from the paper. This changes the effective learning dynamics significantly.

3. With iteration-based scheduling, the total number of iterations depends on `dataset_size / batch_size * max_epochs`. A smaller batch size means more iterations and effectively a different training trajectory for the same schedule percentages.

**Recommendation:**

- The batch sizes are constrained by VRAM and are acceptable. But **be aware** that the iteration-based schedule (warmup_pct, etc.) will produce different absolute warmup durations at different batch sizes. Document the effective number of warmup/constant/decay iterations for each config.

---

## 3. Settings That Are Correctly Transferred

These settings are architecture-independent and correctly shared:

| Setting | Value | Reason It Transfers |
|---------|-------|---------------------|
| `seed: 42` | ✅ | Reproducibility — architecture-independent |
| `action_embed_dim: 2` | ✅ | Dataset property, not architecture property |
| `context_frames: 1` | ✅ | Task design, shared evaluation protocol |
| `use_rope: true` | ✅ | Both architectures use same spatial token layout |
| `normalize_reps: true` | ✅ | V-JEPA feature normalization, input/output pipeline |
| `loss_exp: 1.0` (L1) | ✅ | Loss function on same feature space |
| `use_activation_checkpointing: true` | ✅ | Memory optimization, no training effect |

---

## 4. Summary of Recommended Changes

### Priority 1 — Critical (likely to cause training failure or wasted compute)

| Setting | Current | Recommended | Rationale |
|---------|---------|-------------|-----------|
| `learning_rate` | `4.25e-4` | `1.0e-4` – `2.0e-4` | Bi-level optimization needs lower outer LR |
| `titan_lr_scale` | `0.5` | `0.1` – `0.3` | Meta-learned initial states are highly sensitive |
| `T_rollout` | `7` | `2` (schedule to 4–7 later) | Memory corruption feedback loop over 7 steps |
| `gradient_clip_val` | `1.01` | `2.0` – `5.0` | Too tight for meta-learning gradient variance |

### Priority 2 — Important (affects convergence speed and quality)

| Setting | Current | Recommended | Rationale |
|---------|---------|-------------|-----------|
| `weight_decay` (Titan group) | `0.04` | `0.0` – `0.01` | Double regularization with inner-loop α |
| `warmup_pct` | `0.085` | `0.15` – `0.20` | HOPE needs longer warmup for memory stabilization |
| Curriculum epoch 7→0.7 | `epoch: 7` | `epoch: 12` | HOPE needs longer teacher-forcing stabilization |
| Curriculum epoch 11→0.3 | `epoch: 11` | `epoch: 20`, min 0.5 | Don't remove teacher signal too aggressively |
| `max_epochs` | `30` | `50` – `80` | Meta-learning converges slower |

### Priority 3 — Recommended (good practice, lower impact)

| Setting | Current | Recommended | Rationale |
|---------|---------|-------------|-----------|
| `betas` (Titan group) | `[0.9, 0.999]` | Try `[0.5, 0.999]` | Lower momentum may help noisy meta-gradients |
| T_rollout curriculum | None | Schedule 2→4→7 | Gradual rollout complexity increase |
| Early stopping | Not configured | `patience=10` on val loss | Avoid wasted compute |

---

## 5. Suggested Experiment Config (Depth-Matched)

Below is a concrete suggestion for a HOPE-adapted version of the depth-matched config. Only the training settings are changed; architecture remains identical.

```yaml
# --- Optimizer (adapted for HOPE bi-level optimization) ---
learning_rate: 1.5e-4             # ↓ from 4.25e-4 (bi-level needs lower outer LR)
weight_decay: 0.04                # Applied per-group, see below
betas: [0.9, 0.999]

# Per-group LR scaling
titan_lr_scale: 0.2               # ↓ from 0.5 (Titan LR = 1.5e-4 × 0.2 = 3e-5)
cms_lr_scale: 1.0

# --- LR schedule (longer warmup for meta-learning) ---
use_iteration_scheduler: true
warmup_pct: 0.15                  # ↑ from 0.085 (HOPE needs longer warmup)
constant_pct: 0.75                # Adjusted to sum to 1.0
decay_pct: 0.10
warmup_start_lr: 3.0e-5           # ↓ proportional to new peak LR

# --- Curriculum (delayed for HOPE) ---
curriculum_schedule:
  - epoch: 0
    loss_weight_teacher: 1.0
  - epoch: 15
    loss_weight_teacher: 0.7
  - epoch: 25
    loss_weight_teacher: 0.5

# --- Temporal settings (conservative rollout) ---
T_teacher: 7                      # Keep full teacher forcing
T_rollout: 2                      # ↓ from 7 (start conservative)
context_frames: 1

# --- Trainer ---
max_epochs: 60                    # ↑ from 30 (meta-learning needs more epochs)
gradient_clip_val: 3.0            # ↑ from 1.01 (allow meta-gradient signal through)
gradient_clip_algorithm: norm
```

---

## 6. Fundamental Question: Does the V-JEPA 2-AC Loss Design Make Sense for HOPE?

### 6.1 Teacher-Forcing Loss

**Verdict: ✅ Yes — this transfers well.**

Teacher-forcing loss measures single-step prediction accuracy: $\mathcal{L}_{tf} = \frac{1}{T}\sum_k ||\hat{z}_{k+1} - z_{k+1}||_1$. This is a clean, stable signal that measures whether the HOPE backbone can process interleaved action-state-feature tokens and predict the next feature map. The HOPE architecture replaces the Transformer backbone but the input/output contract is identical. Teacher-forcing provides the primary gradient signal for training the meta-learned initial states — it is essential.

### 6.2 Rollout Loss

**Verdict: ⚠️ Conceptually valid, but execution needs care.**

Rollout loss measures autoregressive prediction quality: $\mathcal{L}_{ro} = ||\hat{z}_{T+1} - z_{T+1}||_1$. For a standard Transformer, rollout simply feeds predictions back as inputs. For HOPE, rollout has an additional dimension:

- **Memory state evolution during rollout:** The Titan memories adapt to the (potentially noisy) rollout predictions. This means the memory state at rollout step 5 is trained on 4 steps of self-generated, possibly erroneous features. This is actually an **interesting property** — it forces the memories to be robust to distribution shift — but it also makes training much harder.

- **Recommendation:** Start with short rollouts (T=2) and use the teacher-forcing signal as the primary training objective. Increase rollout length only after the memories have stabilized (monitored via `hope/mean_surprise` diagnostics).

### 6.3 The Original HOPE Paper's Training Context

The original HOPE architecture (Behrouz 2025) was trained on **language modeling** — next-token prediction on text sequences with typical sequence lengths of 2K–8K tokens. The training objective was cross-entropy loss with standard autoregressive prediction. Key differences from the V-JEPA 2-AC setting:

| Aspect | Original HOPE (Text) | This Implementation (Video Features) |
|--------|----------------------|--------------------------------------|
| Input type | Discrete text tokens | Continuous V-JEPA 2 embeddings (D=1024) |
| Sequence length | 2K–8K tokens | ~2,064 tokens (8 timesteps × 258 tokens/frame) |
| Loss function | Cross-entropy (classification) | L1/MAE (regression) |
| Training paradigm | Single-level autoregressive | Bi-objective (teacher + rollout) |
| Memory update frequency | Many updates per sequence | Few updates (8 timesteps max) |
| Batch size | Large (4K–8K) | Small (32–64) |
| Training data scale | Billions of tokens | ~62 hours of robot video |

The short sequence length (2,064 tokens with only 8 temporal steps) means the Titan memories perform very few DGD updates per sequence. In the text setting, memories get hundreds or thousands of updates per sequence, allowing them to truly adapt. With only 8 timesteps, the memories barely have time to specialize — this raises the question of whether the HOPE architecture provides sufficient benefit over standard attention in this regime.

---

## 7. Additional Architectural Concerns (Beyond Training Settings)

### 7.1 The M_k / M_v Dead Weight Problem

Per the gradient flow protocol, `M_k` and `M_v` are **structurally untrained** under first-order meta-learning — their `nn.Parameters` never receive gradients. This represents 22–23% of Titan parameters as dead weight. The current training settings cannot fix this; it requires an architectural change (see Section 2.3.5 of the gradient flow protocol).

**Impact on training settings evaluation:** The effective model capacity is ~23% smaller than the parameter count suggests. The param-matched experiment (42M nominal) may actually only have ~32M trainable Titan params, making it significantly underpowered compared to the 43M baseline where 100% of parameters are trained.

### 7.2 CMS Update Periods vs. Sequence Length

The CMS levels have update periods of 1, 4, and 16 tokens. With only 8 timesteps per sequence (×258 tokens/frame = ~2,064 total tokens), the "slow" level (period=16) updates ~129 times per sequence, and the "medium" level (period=4) updates ~516 times. These update periods were designed for text with thousands of tokens. The periods likely need re-calibration for the video setting:

- **"Slow" update_period=16 tokens:** In the video context, 16 tokens = a fraction of one frame. This is actually "fast" in temporal terms.
- **Consider temporal-level CMS:** Instead of token-level periods, define periods in terms of timesteps: fast=every timestep, medium=every 2 timesteps, slow=every 4 timesteps.

---

## 8. Conclusion

**The blanket copy of V-JEPA 2-AC training settings is scientifically problematic.** While the loss design (teacher-forcing + rollout) conceptually transfers, the optimizer settings (LR, weight decay, schedule, gradient clipping) and curriculum timing do not account for the fundamentally different optimization landscape of nested/bi-level learning.

The most critical changes are:
1. **Lower learning rate** (especially for Titan memories)
2. **Conservative initial rollout length** (T=2, not T=7)
3. **More permissive gradient clipping** (3.0–5.0, not 1.01)
4. **Longer warmup and training** (more epochs, delayed curriculum)

These changes should be validated empirically with the HOPE diagnostic infrastructure already in place — specifically monitoring `hope/mean_surprise`, `titan/mean_inner_grad_norm`, and `titan/param_norm_w1/w2` across training.
