# AC-HOPE-ViT: Hyperparameter Review & Depth-Fairness Analysis

**Date:** 2026-02-26
**Context:** Review of `configs/experiment/cl_ac_hope.yaml` against CL results in `docs/20260226_ac_hope_vs_lower_bound_comparison.md`

---

## 1. Hyperparameters That Could Enhance AC-HOPE-ViT in This CL Setup

### A. `titan_detach_interval: 1` → Increase to 2–4

*(Likelihood of CL improvement: **High** — this directly governs meta-gradient quality, the core mechanism HOPE relies on for cross-task transfer. VRAM cost: **+100–200%** peak activation memory, since the retained computation graph scales linearly with the unroll horizon; each additional step keeps ~5 memories × 6 blocks of intermediate activations alive. Compute cost: **Negligible** — forward/backward FLOPs are unchanged; only memory allocation grows.)*

This is arguably the **single most impactful change**. Currently, the computation graph is detached at every DGD step. That means the outer optimizer only ever sees a *one-step* unrolled inner loop — essentially FOMAML with horizon 1. The meta-learned initial memory states ($M_{\square,0}$) therefore receive almost no signal about *how well they serve as starting points* for multi-step adaptation. Increasing to 2–4 lets the outer loss backpropagate through 2–4 inner-loop DGD updates, giving the meta-learned initialization a richer learning signal. The cost is VRAM (linear in the interval), but on a datacenter GPU this is feasible.

**Why it helps CL specifically:** Better meta-learned initial states mean the memories start closer to a universally useful configuration, improving transfer across tasks.

### B. `surprise_threshold: 0.0` → 0.05–0.2

*(Likelihood of CL improvement: **Medium-High** — directly reduces unnecessary memory overwrites during task shifts, which is the primary forgetting mechanism in Titan memories. VRAM cost: **Zero** — the surprise norm is already computed; only a scalar comparison is added. Compute cost: **Slightly negative (faster)** — tokens below threshold skip the DGD update entirely, saving inner-loop backward passes.)*

Currently, *every* token triggers a memory write, regardless of whether the retrieval was already accurate. In a CL setting, this is dangerous: tokens from the new task distribution will overwrite memory contents that were serving old-task retrieval well — even when the old knowledge was still correct. A non-zero surprise threshold acts as a **write gate**: only update memory when $\|M(q_t) - v_t\|$ exceeds the threshold. This directly reduces unnecessary memory churn and protects previously learned associations. It is analogous to an experience replay filter.

### C. `gradient_clip_val: 3.0` → Lower to 1.0–1.5

*(Likelihood of CL improvement: **Medium** — addresses the observed training instability but does not add representational capacity; the benefit is indirect (smoother optimization → less accidental forgetting). VRAM cost: **Zero** — clip norm is computed on existing gradients. Compute cost: **Zero** — a single norm computation + clamp per step.)*

The comparison document explicitly calls out "highly volatile, jagged training curves with massive loss spikes." The AC\_ViT uses `gradient_clip_val: 1.01`. HOPE uses 3.0 — three times looser. Given that bi-bilevel optimization already introduces conflicting gradients between memory preservation and new-distribution adaptation, tighter outer-loop clipping would dampen the spikes and smooth convergence. This won't add capacity, but it reduces the probability of catastrophic parameter jumps during distribution shifts.

### D. `task_training.learning_rate: 5e-5` → Consider 1e-5–3e-5

*(Likelihood of CL improvement: **Medium** — reduces outer-loop weight drift, the dominant forgetting channel, but may also reduce plasticity on new tasks; there is a sweet spot to find. VRAM cost: **Zero** — no architectural change. Compute cost: **Zero** — same number of optimizer steps.)*

The fine-tuning LR is already 1/3 of the base LR, but given that the analysis shows continued forgetting across all 5 phases, lowering this further would slow the rate at which the outer-loop weights drift from their base-trained configuration. The HOPE inner-loop (DGD) is designed to do the fast adaptation; the outer loop should be kept conservative in CL.

### E. `titan_hidden_multiplier: 2` → Increase to 3 (with depth=5)

*(Likelihood of CL improvement: **Medium-Low** — wider memories can store richer associations, but shrinking depth from 6 → 5 blocks loses one round of hierarchical refinement; net effect is uncertain without ablation. VRAM cost: **Roughly neutral** — wider memories per block but one fewer block; peak activation memory stays within ~±10%. Compute cost: **Roughly neutral** — per-block FLOPs increase ~50% but one fewer block yields ~17% FLOP reduction; net ~+25% FLOPs.)*

The multiplier was reduced from the default 4 to 2 for parameter matching. This halves each Titan memory's capacity. With 5 memories per block, the per-memory representation bottleneck may limit what each memory can encode. Increasing to 3 (at the cost of reducing from 6 → 5 blocks for param parity) gives each memory 50% more representational capacity, which matters because the key/value/eta/alpha memories must each learn qualitatively different functions.

### F. `aux_loss_weight: 0.1` → Try 0.3–0.5

*(Likelihood of CL improvement: **Medium** — better-trained key/value memories improve retrieval precision, which benefits knowledge retention; but over-weighting the aux loss may distort the primary prediction objective. VRAM cost: **Zero** — the auxiliary loss is already computed. Compute cost: **Zero** — only the scalar weight in the loss sum changes.)*

The auxiliary loss ensures $M_k$ and $M_v$ receive gradient signal for retrieval quality. At 0.1, this signal may be too weak relative to the main prediction loss, leaving these memories under-trained. Stronger auxiliary supervision means sharper keys and more informative values — directly improving the quality of what the Titan memory retrieves.

### G. CMS `warmup_steps: 0` → Set slow=200, medium=100

*(Likelihood of CL improvement: **Low-Medium** — primarily affects early base-training stability rather than the CL task sequence directly; the benefit is cleaner initialization for downstream fine-tuning. VRAM cost: **Zero** — warmup simply zeros out gradients for inactive levels. Compute cost: **Slightly negative (faster)** — inactive CMS levels skip forward/backward during warmup.)*

All three CMS levels activate from step 0. In early training, the slow level (period=16) sees very few updates and has near-random weights, but its output still feeds into the residual stream. Setting warmup delays for slower levels lets the fast level establish a reasonable baseline before multi-frequency effects kick in, reducing early training noise.

---

## 2. Fairness of Parameter Matching at Depth 6 vs. Depth 24

### Why It Is Defensible (and Standard Practice)

**Parameter matching is the most widely accepted comparison protocol** in the deep learning literature (see e.g., DeiT, Swin, Mamba vs. Transformer comparisons). When comparing architectures, controlling for total parameter count ensures that any performance difference is attributable to the **architectural inductive bias**, not simply to having more learnable weights. The ~42M vs ~43M matching achieves this.

### Why It Systematically Disadvantages HOPE

Each HOPE block is roughly **3.5–4× more parameter-dense** than a standard Transformer block:

| Component | Standard Transformer Block (dim=384) | HOPE Block (dim=384, mult=2) |
|---|---|---|
| Attention / Titan | ~0.6M (Q,K,V,O projections) | ~2.95M (5 Titan memories × 2-layer MLP) |
| FFN / CMS | ~1.2M (2-layer MLP, mult=4) | ~3.5M (3 CMS levels × 2-layer MLP) |
| Norms + misc | ~0.003M | ~0.3M |
| **Total per block** | **~1.8M** | **~6.8M** |

So parameter matching forces $24 / 4 \approx 6$ layers — exactly what was chosen. The problem: **depth and width are not interchangeable.**

1. **Sequential refinement:** A 24-layer Transformer performs 24 rounds of global-information-mixing → nonlinear-transformation. HOPE gets only 6 such rounds. In representation learning, depth provides *compositional hierarchy* — each layer can build on abstractions from the previous one. The function class expressible by 24 sequential compositions is strictly richer than 6.

2. **Residual gradient paths:** With 24 layers there are far more skip-connection pathways for gradient flow and feature refinement. The 6-layer model has a sparser gradient landscape.

3. **The "inner depth" argument partially mitigates this:** Each HOPE block internally performs (a) 5 sequential memory read/write operations, and (b) 3 sequential CMS levels. So the *effective computational depth* per HOPE block is arguably $5 + 3 = 8$ sequential nonlinear operations vs. $2$ (attention + MLP) in a standard block. By this logic, 6 HOPE blocks provide $6 \times 8 = 48$ effective sequential operations vs. $24 \times 2 = 48$ for the Transformer. **This is the strongest argument for fairness.**

### Honest Assessment

The comparison is **acceptably fair but not perfectly fair**, and this should be acknowledged:

- **The parameter-matched design is standard and defensible.** Reviewers will accept it.
- **The depth asymmetry creates a known confound.** The standard Transformer benefits from 4× more rounds of residual-stream refinement.
- **However, the FLOP asymmetry runs in the opposite direction.** HOPE blocks are significantly more expensive per forward pass (5 DGD-updated memory reads + writes + 3 CMS MLPs). If anything, HOPE uses *more* compute for fewer parameters.

### Recommended Thesis Discussion (Limitations Section)

> "Parameter matching constrains AC-HOPE-ViT to depth 6 versus depth 24 for AC\_ViT. While each HOPE block performs substantially more sequential computation per layer (5 memory operations + 3-level CMS ≈ 8 nonlinear stages vs. 2 in a standard Transformer block), the reduced number of residual-stream refinement rounds remains a confound. Future work should include a depth-matched comparison (both at depth 24, different parameter counts) and a FLOP-matched comparison to fully disentangle the contributions of architectural inductive bias from depth and compute."

This pre-empts reviewer criticism and demonstrates methodological awareness.


---

As a professor specialized in Continual Machine Learning (CL) and meta-learning architectures, I have reviewed the hyperparameter adjustments proposed for the AC-HOPE-ViT model.

The HOPE architecture, with its bi-level optimization (inner-loop DGD for fast adaptation, outer-loop for meta-initialization) and Continuum Memory System, is theoretically well-positioned for CL. However, its default hyperparameters are clearly tuned for stationary i.i.d. training rather than sequential task learning.

Here is my scientific evaluation of the proposed adjustments, categorized by their theoretical impact on the **stability-plasticity dilemma** and **catastrophic forgetting**.

---

### Tier 1: Fundamental CL Mechanisms (Highly Recommended)

These adjustments directly address the core theoretical challenges of continual learning in memory-augmented and meta-learned systems.

**1. `surprise_threshold: 0.0` → 0.05–0.2 (Rating: Excellent)**
*   **Scientific Rationale:** In standard associative memory, unconditional writes cause rapid overwriting of past knowledge (catastrophic forgetting). By introducing a surprise threshold, you are implementing a **novelty-driven update mechanism**. This directly preserves the *stability* of the memory for previously learned distributions while allowing *plasticity* only when the current retrieval error is high. This is theoretically analogous to surprise-based experience replay or sparse coding updates.
*   **Verdict:** Implement immediately. It is computationally free and directly mitigates memory churn.

**2. `task_training.learning_rate: 5e-5` → 1e-5–3e-5 (Rating: Very Strong)**
*   **Scientific Rationale:** HOPE relies on a "fast weights / slow weights" paradigm. The inner-loop DGD provides the fast weights (plasticity), while the outer-loop meta-weights provide the slow weights (stability). If the outer-loop learning rate is too high during the CL phase, the meta-learned initialization ($M_{\square,0}$) will overfit to the current task, destroying forward and backward transfer.
*   **Verdict:** Implement immediately. The outer loop must remain conservative during sequential fine-tuning.

**3. `titan_detach_interval: 1` → 2–4 (Rating: Strong, but computationally expensive)**
*   **Scientific Rationale:** Detaching the graph at every step reduces the outer-loop optimization to First-Order MAML (FOMAML) with a horizon of 1. This yields highly myopic meta-gradients that do not optimize for multi-step adaptation. Increasing the unroll horizon provides a much richer gradient signal, forcing the initial memory states to be universally useful starting points across tasks.
*   **Verdict:** Highly recommended if your VRAM budget allows. It fundamentally improves the quality of the meta-learned representations.

### Tier 2: Optimization & Stability (Recommended)

These adjustments address the optimization difficulties inherent in bi-level and memory-augmented architectures.

**4. `gradient_clip_val: 3.0` → 1.0–1.5 (Rating: Solid)**
*   **Scientific Rationale:** Bi-level optimization landscapes are notoriously rugged, often exhibiting exploding gradients due to the unrolled inner loop. In a CL setting, a massive gradient spike during a task transition can permanently destroy previously learned representations. Tighter clipping acts as a necessary regularization mechanism against distribution-shift shocks.
*   **Verdict:** Implement. It will smooth the loss curve and prevent catastrophic parameter jumps.

**5. `aux_loss_weight: 0.1` → 0.3–0.5 (Rating: Plausible)**
*   **Scientific Rationale:** The efficacy of a Titan memory is strictly bounded by the quality of its addressing mechanism (keys and values). If the auxiliary loss is too weak, the memory will suffer from "addressing collapse," where distinct concepts map to the same memory slots.
*   **Verdict:** Worth testing, but monitor the primary prediction loss. Over-weighting auxiliary losses can sometimes bottleneck the main objective.

### Tier 3: Architectural Trade-offs (Proceed with Caution)

These adjustments alter the fundamental inductive biases of the network and require careful empirical ablation.

**6. `titan_hidden_multiplier: 2` → 3 (with depth=5) (Rating: Skeptical)**
*   **Scientific Rationale:** Trading depth for width is a dangerous game in representation learning. While wider memories increase raw storage capacity (useful for CL), reducing depth from 6 to 5 removes an entire layer of compositional hierarchy. In visual tasks, hierarchical feature extraction is often more critical than raw memory width.
*   **Verdict:** Do not implement this simultaneously with the other changes. If you must test it, do so in a strictly isolated ablation study.

**7. CMS `warmup_steps: 0` → slow=200, medium=100 (Rating: Marginal for CL)**
*   **Scientific Rationale:** Staggered warmup prevents slow-updating components from being corrupted by early, noisy gradients.
*   **Verdict:** This is excellent for the *initial pre-training* phase, but largely irrelevant for the sequential CL fine-tuning phases where weights are already mature.

---

### Final Professor's Recommendation & Action Plan

Do not change all hyperparameters at once, as this will destroy your ability to attribute causality to your results. I recommend the following phased approach for your thesis:

**Phase 1: The "Free" CL Fixes (Implement First)**
1.  Set `surprise_threshold: 0.1` (Novelty gating).
2.  Lower `task_training.learning_rate: 2e-5` (Protect slow weights).
3.  Set `gradient_clip_val: 1.2` (Prevent distribution-shift shocks).
*Run your CL benchmark. I hypothesize this alone will significantly reduce catastrophic forgetting compared to your baseline.*

**Phase 2: The Meta-Learning Fix (If Phase 1 is insufficient)**
1.  Increase `titan_detach_interval: 3`.
*This will cost VRAM, but it is the most theoretically rigorous way to improve the cross-task generalization of the HOPE module.*

**Phase 3: Representation Tuning (Optional)**
1.  Increase `aux_loss_weight: 0.3`.

**Leave the depth/width trade-off (`titan_hidden_multiplier`) alone for now.** Your parameter-matching defense in Section 2 of the document is academically sound. Focus on fixing the optimization and memory-update dynamics first.
