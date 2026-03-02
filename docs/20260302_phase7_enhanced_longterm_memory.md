# Phase 7: Enhanced Longterm Memory — All B→A Upgrades

**Date:** 2026-03-02
**Config:** `configs/experiment/cl_ac_hope_phase7.yaml`
**Param count:** ~46.5M (same as Phase 6 + negligible gate input increase)
**Builds on:** Phase 6 (persistent cross-clip longterm memory)

---

## 1. Motivation

Phase 6 introduced the persistent longterm memory ($M_\text{longterm}$) with a separation-of-concerns design (Ansatz B). A rigorous scientific review identified four B-grade design decisions that, while functional, limited the architecture's theoretical alignment with Complementary Learning Systems (CLS) theory and its expected effectiveness on both plasticity and forgetting.

Phase 7 addresses **all four** to achieve A-level design:

| Issue | Phase 6 Grade | Root Cause | Phase 7 Fix | Phase 7 Grade |
|-------|:---:|------------|-------------|:---:|
| Gate design | B | Conditions on $q$ alone; cannot see memory discrepancy | Retrieval-conditioned gate: $[q; |o_\text{clip} - o_\text{long}|]$ | A |
| DGD scaling | B- | Same $\alpha$ for both memories; arbitrary $\lambda$ | Asymmetric decay: $\alpha_\text{lt} \geq 0.95$; tighter $\lambda=0.05$ | A |
| Surprise signal | B- | Shared M_memory surprise for both systems | M_longterm uses its own retrieval error | A |
| Reset semantics | B+ | No consolidation; meta-learning signal purely local | EMA consolidation between CL tasks | A |

---

## 2. Architectural Changes

### 2.1 Retrieval-Conditioned Gate

**Phase 6:** $g_t = \sigma(W_g \cdot q_t + b_g)$, where $W_g \in \mathbb{R}^{D \times 1}$

**Phase 7:** $g_t = \sigma(W_g \cdot [q_t; |o_t^\text{clip} - o_t^\text{long}|] + b_g)$, where $W_g \in \mathbb{R}^{2D \times 1}$

The discrepancy signal $|o_t^\text{clip} - o_t^\text{long}|$ provides critical information:

| Discrepancy | Interpretation | Desired Gate Behavior |
|:-----------:|----------------|----------------------|
| Low | Both memories agree → familiar physics | Either memory works; gate can be neutral |
| High + clip better | Novel physics that clip has adapted to | $g_t \to 1$ (favor clip-level) |
| High + longterm better | Previously seen physics, clip hasn't adapted yet | $g_t \to 0$ (favor longterm) |

Without the discrepancy signal, the gate must learn to infer this from $q$ alone — a much harder task that requires memorizing which query patterns correspond to familiar vs. novel physics.

**Parameters:** +$D$ per gate per block (384 extra weights per gate × 5 blocks = 1,920 total). Negligible.

### 2.2 Asymmetric Decay

**Phase 6:** $M_\text{longterm}$ receives the same $\alpha_t$ as clip-level memories.

**Phase 7:** $\alpha_t^\text{longterm} = \text{clamp}(\alpha_t, \alpha_\text{min}, 1.0)$, with $\alpha_\text{min} = 0.95$.

This aligns with CLS theory where the neocortical (slow) system has:
- **Higher retention**: $\alpha \geq 0.95$ means $\leq 5\%$ decay per step
- **Complementary to fast system**: clip-level memories can decay freely ($\alpha \in [0, 1]$) to specialize on current physics

The DGD update with asymmetric decay becomes:

$$M_{\text{longterm},t} = M_{\text{longterm},t-1}(\alpha_t^\text{lt} I - \eta_t^\text{lt} k_t k_t^\top) - \eta_t^\text{lt}(M_{\text{longterm},t-1} k_t - \hat{v}_{\text{lt},t}) k_t^\top$$

where $\alpha_t^\text{lt} = \max(\alpha_t, 0.95)$ and $\eta_t^\text{lt} = \eta_t \times 0.05$.

Combined with the tighter learning rate scale ($\lambda = 0.05$ vs 0.1 in Phase 6), this makes $M_\text{longterm}$ a genuinely slow, high-retention memory.

### 2.3 Longterm-Specific Surprise

**Phase 6:** $M_\text{longterm}$ uses $M_\text{memory}$'s surprise for update gating.

**Phase 7:** $M_\text{longterm}$ computes its own surprise:

$$s_t^\text{lt} = \|v_t - M_{\text{longterm},t-1}(q_t)\|_2$$

This decouples the two memory systems:
- $M_\text{memory}$ updates when **it** is surprised (clip-level adaptation)
- $M_\text{longterm}$ updates when **it** is surprised (longterm knowledge gaps)

A token pattern that $M_\text{memory}$ finds surprising (because it was just reset) but $M_\text{longterm}$ recognizes (because it saw similar patterns in previous clips) will trigger a strong $M_\text{memory}$ update but a weak $M_\text{longterm}$ update. This is exactly the desired behavior: the fast system adapts, the slow system stays stable.

### 2.4 Consolidation EMA

**Phase 6:** After `.detach()`, $M_\text{longterm}$'s accumulated state is a constant for the next clip's optimization. No feedback loop to `nn.Parameters`.

**Phase 7:** Between CL tasks (not individual clips), consolidation absorbs accumulated state:

$$\theta_\text{new} = (1 - \gamma) \cdot \theta_\text{old} + \gamma \cdot w_\text{active}$$

where $\gamma = 0.01$ (1% per task). After consolidation, active weights are re-initialized from the updated $\theta_\text{new}$ (detached).

**Why between tasks, not clips:**
- Between clips is too frequent — the DGD state after a single clip is noisy
- Between tasks accumulates $\sim$1000 clips of DGD state, providing a more stable signal for consolidation
- The CL pipeline already has a natural task boundary where this fits

**Analogy:** This is the architectural equivalent of sleep consolidation — the "slow" system periodically absorbs a compressed summary of recent experiences into its permanent structure.

---

## 3. Effect on CL Metrics

### 3.1 Plasticity (Error on Current Task) — Expected: Strong Improvement

All four changes contribute:
1. **Retrieval-conditioned gate** routes novel tokens (high discrepancy, clip better) to $M_\text{memory}$ with precision → no longterm interference
2. **Asymmetric decay** prevents $M_\text{longterm}$ from decaying useful initialization → better starting point
3. **Own surprise** prevents $M_\text{longterm}$ from updating on patterns it already knows → stability
4. **Consolidation** improves the meta-learned initial state → better clip-level adaptation starting point

### 3.2 Forgetting (Degradation on Previous Tasks) — Expected: Strong Improvement

1. **Retrieval-conditioned gate** routes familiar tokens (low discrepancy or longterm better) to $M_\text{longterm}$ → stable predictions on old tasks
2. **Asymmetric decay** ensures $M_\text{longterm}$ retains ≥95% of knowledge per step → slow forgetting
3. **Own surprise** prevents over-updating on patterns that $M_\text{longterm}$ already encodes well → preservation
4. **Consolidation** moves accumulated knowledge into `nn.Parameters` → survives even if active weights are perturbed

### 3.3 Success Criteria

| Metric | Phase 6 Expected | Phase 7 Target | Rationale |
|--------|:-:|:-:|-----------|
| Plasticity | ≤ 0.30 | ≤ 0.28 | Gate + consolidation improve adaptation |
| Forgetting | ≤ 0.05 | ≤ 0.035 | Asymmetric decay + own surprise + consolidation |
| Avg Error | < 0.37 | < 0.35 | Compound improvements |
| Gap Closed | > 5% | > 10% | Twice Phase 6 target |

---

## 4. Implementation

### 4.1 Modified Files

| File | Changes | Purpose |
|------|---------|---------|
| [hope_block.py](../src/models/hope/hope_block.py) | `HOPEBlockConfig` +4 fields; `__init__` retrieval-conditioned gate (2D input); `_titan_forward_chunk` discrepancy-gated retrieval + longterm surprise; `_update_memories` +longterm_surprise arg, asymmetric α, own surprise; `+consolidate_longterm_memory()` | Core Phase 7 mechanisms |
| [ac_hope_vit.py](../src/models/hope/ac_hope_vit.py) | Constructor +4 params; HOPEBlockConfig wiring; `_config_summary` +4 fields; `+consolidate_all_longterm_memories()` | Model-level integration |
| [ac_hope_module.py](../src/models/hope/ac_hope_module.py) | Constructor +4 params; wiring to ac_hope_vit | Lightning module integration |
| [ac_hope_vit.yaml](../configs/model/ac_hope_vit.yaml) | +4 defaults (all disabled by default) | Default config (backward compatible) |
| [cl_ac_hope_phase7.yaml](../configs/experiment/cl_ac_hope_phase7.yaml) | NEW — Phase 7 experiment config | Experiment configuration |
| [test_phase7_enhanced_longterm.py](../tests/test_phase7_enhanced_longterm.py) | NEW — comprehensive test suite | Verification |

### 4.2 Backward Compatibility

All four features are independently gated by boolean/float config fields (all default to `false`/`0.0`):
- `longterm_retrieval_conditioned_gate: false` → uses Phase 6 simple gate
- `longterm_alpha_min: 0.0` → no clamping (Phase 6 behavior)
- `longterm_own_surprise: false` → uses M_memory's surprise (Phase 6 behavior)
- `longterm_consolidation_ema: 0.0` → no consolidation (Phase 6 behavior)

All existing Phase 5/6 tests pass without modification.

### 4.3 Consolidation Integration Point

The `consolidate_all_longterm_memories()` method should be called by `cl_train.py` between CL tasks:

```python
# After completing task N's finetuning, before starting task N+1:
model.model.consolidate_all_longterm_memories()
```

This fits naturally into the existing CL pipeline's task boundary.

---

## 5. Hyperparameter Choices

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `longterm_lr_scale` | 0.05 | Tighter than Phase 6 (0.1): 20× slower than clip-level, ensuring genuine slow accumulation |
| `longterm_alpha_min` | 0.95 | ≤5% decay per step; balances retention with ability to forget truly obsolete patterns |
| `longterm_consolidation_ema` | 0.01 | 1% per task; after 5 CL tasks, ~5% of meta-learned state is from accumulated DGD |
| Gate architecture | Linear(2D, 1) | Minimal param increase; discrepancy signal provides most of the information gain |

### 5.1 Ablation Recommendations

To isolate each Phase 7 contribution:

| Ablation | Config Override | What It Tests |
|----------|----------------|---------------|
| Phase 6 baseline | All Phase 7 features = off | Baseline comparison |
| +Gate only | `longterm_retrieval_conditioned_gate: true` | Gate design contribution |
| +Decay only | `longterm_alpha_min: 0.95` | Asymmetric decay contribution |
| +Surprise only | `longterm_own_surprise: true` | Decoupled surprise contribution |
| +Consolidation only | `longterm_consolidation_ema: 0.01` | Consolidation contribution |
| Full Phase 7 | All enabled | Combined effect |

---

## 6. Relation to Prior Work

### 6.1 Complementary Learning Systems (McClelland et al., 1995; Kumaran et al., 2016)

Phase 7 brings the architecture significantly closer to CLS theory:

| CLS Property | Phase 6 | Phase 7 |
|--------------|:-------:|:-------:|
| Fast vs slow learning rates | ✓ (10×) | ✓ (20×, tighter) |
| Asymmetric retention | ✗ (same α) | ✓ (α ≥ 0.95) |
| Independent surprise/error signals | ✗ (shared) | ✓ (own surprise) |
| Offline consolidation | ✗ (none) | ✓ (EMA between tasks) |
| Context-aware routing | Partial (q only) | ✓ (discrepancy-conditioned) |

### 6.2 Titans: Learning to Memorize at Test Time (Behrouz et al., 2025)

The original Titans paper's Neural Long-Term Memory module uses a separate surprise signal and persistence semantics. Phase 7's `longterm_own_surprise` directly aligns with this design — each memory module's update is driven by its own error signal, not a shared global signal.
