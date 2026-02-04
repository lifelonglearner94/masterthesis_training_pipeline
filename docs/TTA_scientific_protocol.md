# Test Time Adaptation (TTA) Scientific Protocol

**Date:** February 4, 2026
**Experiment Context:** V-JEPA2 Action-Conditioned Predictor with TTA

---

## 1. Initial Experiment Results

### Configuration (Baseline TTA)
| Parameter | Value |
|-----------|-------|
| `tta_enabled` | `true` |
| `tta_mode` | `"full_rollout"` |
| `tta_num_adaptation_steps` | `1` |
| `tta_lr` | `1e-4` |
| `tta_grad_clip` | `1.0` |
| `tta_reset_per_clip` | `false` |
| Test clips | 22000-22999 (1000 clips) |
| Adapted parameters | LayerNorm only (~0.09%) |

### Results
- **Mean Rollout Loss:** 0.3639
- **Mean Pre-Adapt Loss:** 0.3642
- **Mean Post-Adapt Loss:** 0.3639
- **Mean Improvement:** 0.000246 (negligible)
- **Runtime:** ~4 hours for 1000 clips

### Key Observation
TTA showed almost no improvement (0.0002) over 1000 clips with the conservative baseline settings.

---

## 2. Diagnostic Analysis

### W&B Logging Issues Identified
1. **Aggregated logging:** `log_every_n_steps: 50` caused Lightning to aggregate metrics, showing only ~20 data points instead of 1000
2. **Unnecessary constant logs:** `test/tta_enabled`, `test/tta_mode` cluttered the dashboard

### Analysis of Rolling Average Plot (`test/tta_loss_rolling_50`)
| Phase | Clips | Behavior |
|-------|-------|----------|
| Initial descent | 0-250 | Sharp decrease from ~0.36 → ~0.34 (**~5.5% improvement**) |
| Oscillation | 250-750 | Loss oscillates between 0.34-0.36 |
| Spike | ~750 | Noticeable upward spike, then return to baseline |

### Interpretation
1. **TTA IS working** in the first 250 clips (demonstrable learning)
2. **Saturation after 250 clips:** The model reaches a local optimum
3. **Oscillation indicates heterogeneous data:** Clips in range 22000-23000 may have sub-distributions that pull parameters in conflicting directions
4. **Diminishing returns expected:** With `reset_per_clip=false`, early clips have large improvement potential; later clips are already close to adapted optimum

---

## 3. Root Cause Analysis

### Why Improvement Was So Small (0.000246)?

| Factor | Issue | Impact |
|--------|-------|--------|
| **Single adaptation step** | `tta_num_adaptation_steps: 1` | Only ONE gradient update per clip |
| **Conservative LR** | `tta_lr: 1e-4` | Small parameter changes |
| **Aggressive gradient clipping** | `tta_grad_clip: 1.0` | Truncates useful gradients |
| **Limited parameters** | LayerNorm only (~0.09%) | Can only adjust feature normalization, not representations |

### Mathematical Limitation of LayerNorm-Only Adaptation
LayerNorm adaptation can only perform affine transformation:
$$y = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$$

This **cannot** change:
- Attention patterns
- MLP mappings
- Feature representations

If the domain shift requires representational changes beyond re-centering/scaling, LayerNorm adaptation has fundamental limits.

---

## 4. Recommended Configuration Changes

### For Stronger Adaptation (Implemented in `test_ac_predictor_tta.yaml`)
| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `tta_num_adaptation_steps` | 1 | **5** | 5× more gradient updates per clip |
| `tta_lr` | 1e-4 | **5e-4** | 5× higher learning rate |
| `tta_grad_clip` | 1.0 | **5.0** | Allow larger gradient flow |
| `log_every_n_steps` | 50 | **1** | Per-clip logging |

### Expected Improvement
With these changes, expect approximately 5-10× larger per-clip improvements due to:
- More gradient steps
- Larger effective learning rate
- Less gradient truncation

---

## 5. Extended Layer Adaptation

### New Experiment: `test_ac_predictor_tta_extended.yaml`

To increase learning capacity beyond LayerNorm, we added support for adapting attention output projections.

### Layer Adaptation Options

| Option | Layers | Parameters | Risk | Capacity |
|--------|--------|------------|------|----------|
| `"layernorm"` | LayerNorm γ, β | ~0.09% | Very Low | Limited |
| **`"layernorm+attn_proj"`** | + Attention output projection | ~1-2% | Low-Medium | **Good** |
| `"layernorm+bias"` | + All bias terms | ~0.5% | Low | Medium |
| `"layernorm+mlp_out"` | + MLP fc2/fc3 | ~3-5% | Medium | High |

### Why Attention Output Projections (`attn.proj`)?

1. **Function:** Controls how multi-head attention outputs are combined back into embedding space
2. **Efficiency:** One linear layer per transformer block (vs. 2-3 for MLP)
3. **Scientific precedent:** Similar to LoRA/adapter approaches in fine-tuning literature
4. **Low catastrophic forgetting risk:** Doesn't change attention patterns, only how they're projected

### Extended TTA Configuration
```yaml
tta_adapt_layers: "layernorm+attn_proj"
tta_lr: 2e-4          # Slightly lower due to more parameters
tta_num_adaptation_steps: 5
tta_grad_clip: 5.0
tta_reset_per_clip: false
```

---

## 6. Experimental Design

### Ablation Study Matrix

| Experiment | Config File | `tta_adapt_layers` | `reset_per_clip` |
|------------|-------------|-------------------|------------------|
| Baseline (no TTA) | `test_ac_predictor.yaml` | N/A | N/A |
| TTA LayerNorm | `test_ac_predictor_tta.yaml` | `"layernorm"` | `false` |
| TTA Extended | `test_ac_predictor_tta_extended.yaml` | `"layernorm+attn_proj"` | `false` |

### Metrics to Compare

1. **Final mean rollout loss** (`test/final_mean_loss_rollout`)
2. **Rolling loss trend** (`test/tta_loss_rolling_50`) - should show downward trend
3. **Per-clip improvement** (`test/tta_improvement`) - should be positive
4. **Trainable parameter count** (logged at start)

### Run Commands
```bash
# Baseline (no TTA)
uv run src/eval.py experiment=test_ac_predictor ckpt_path=<checkpoint>

# TTA with LayerNorm only
uv run src/eval.py experiment=test_ac_predictor_tta ckpt_path=<checkpoint>

# TTA with extended layer adaptation
uv run src/eval.py experiment=test_ac_predictor_tta_extended ckpt_path=<checkpoint>
```

---

## 7. Implementation Details

### Files Modified

| File | Changes |
|------|---------|
| `src/models/mixins/tta_mixin.py` | Added `tta_adapt_layers` parameter; updated `_tta_configure_params()` to support multiple layer types |
| `src/models/ac_predictor/lightning_module.py` | Added `tta_adapt_layers` parameter to `__init__` |
| `configs/experiment/test_ac_predictor_tta_extended.yaml` | New config for extended adaptation |

### Key Code: Layer Selection Logic
```python
# In _tta_configure_params()
if "attn_proj" in self.tta_adapt_layers:
    for name, module in self.model.named_modules():
        if ".proj" in name and ".attn" in name and isinstance(module, nn.Linear):
            for param in module.parameters():
                param.requires_grad = True
```

---

## 8. Conclusions and Next Steps

### Key Findings
1. **TTA works but requires tuning:** The baseline was too conservative
2. **Cumulative adaptation shows learning curve:** First 250 clips show clear improvement
3. **Saturation is expected:** With heterogeneous data, the model converges to an "average" optimum
4. **LayerNorm has limited capacity:** ~0.09% of parameters may be insufficient for significant adaptation

### Recommended Next Experiments
1. Run `test_ac_predictor_tta_extended.yaml` to test ~1-2% parameter adaptation
2. Compare rolling loss curves between LayerNorm-only and extended adaptation
3. If extended adaptation helps, consider `"layernorm+mlp_out"` for even higher capacity

### Open Questions
- Does the oscillation pattern (clips 250-750) correlate with specific clip characteristics (e.g., scene changes)?
- Would sorting clips by similarity improve cumulative adaptation?
- Is there a "sweet spot" for number of clips before saturation?

---

## Appendix: Configuration Files

### A. Base TTA Config (`test_ac_predictor_tta.yaml`)
- Adapts: LayerNorm only
- Parameters: ~0.09%
- Settings: 5 steps, LR=5e-4, grad_clip=5.0

### B. Extended TTA Config (`test_ac_predictor_tta_extended.yaml`)
- Adapts: LayerNorm + Attention output projections
- Parameters: ~1-2%
- Settings: 5 steps, LR=2e-4, grad_clip=5.0
