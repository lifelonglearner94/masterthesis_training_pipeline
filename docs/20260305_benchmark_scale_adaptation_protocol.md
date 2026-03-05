# Benchmark Scale Adaptation Protocol

**Date:** 2026-03-05
**Context:** CL benchmarks (Split CIFAR-100, Permuted MNIST) use scaled-down versions of the real-data architectures. This document records every parameter difference and justifies the adaptation.

---

## 1. Design Principle

The benchmark models use the **exact same Python classes** (`ACHOPEHybridViT`, `ACDNHHOPEHybridViT`) as the real-data experiments.
All architectural *design choices* — attention + Titan memory + CMS composition, DGD inner-loop, CMS frequency hierarchy, longterm memory, RoPE — are preserved verbatim.

Only four categories of parameters change:

| Category | Reason |
|---|---|
| **Input geometry** | CIFAR-100 is 32×32×3; real data is V-JEPA 2 features [8, 256, 1024] |
| **Embedding dimension** | Scaled to match input complexity and avoid massive over-parameterisation |
| **Depth / heads** | Reduced proportionally to keep param budget sensible for 50K-sample datasets |
| **DNH capacity caps** | Slightly lower `L_max` and `meta_hidden_dim` to match reduced embedding width |

This is standard practice in the CL literature (e.g., Buzzega et al. 2020, Wang et al. 2024 DNH paper Table 1 uses a ResNet-18 backbone on CIFAR-100, not their full-scale model).

---

## 2. Phase 8 — AC-HOPE-Hybrid-ViT

### Parameters that differ

| Parameter | Real-Data Config | Benchmark Config | Factor |
|---|---|---|---|
| `embed_dim` | 1024 | 384 | 2.7× smaller |
| `predictor_embed_dim` | 384 | 192 | 2× smaller |
| `depth` | 12 | 8 | 1.5× fewer blocks |
| `num_heads` | 16 | 6 | 2.7× fewer heads |
| `img_size` | [256, 256] | [32, 32] | Different modality |
| `patch_size` | 16 | 4 | Yields 64 patches in both cases* |
| `num_timesteps` | 8 | 1 | Single image vs video |
| `action_embed_dim` | 7 (or 2) | 1 (dummy) | No actions in classification |
| `is_frame_causal` | true | false | No temporal causality needed |

\* Real data: 256/16 = 16 → 16×16 = 256 patches/frame × 8 frames = 2048 tokens.
  Benchmark: 32/4 = 8 → 8×8 = 64 patches × 1 frame = 64 tokens.

### Parameters that are identical

| Parameter | Value |
|---|---|
| `titan_hidden_multiplier` | 2 |
| `titan_layers` | 2 |
| `titan_activation` | gelu |
| `titan_grad_clip_inner` | 1.0 |
| `titan_grad_clip_backward` | 1.0 |
| `titan_detach_interval` | 2 |
| `surprise_threshold` | 0.1 |
| `use_longterm_memory` | true |
| `longterm_hidden_multiplier` | 2 |
| `longterm_lr_scale` | 0.1 |
| `cms_level_specs` | {fast: 2.0, medium: 2.5, slow: 3.0} |
| `cms_use_chunk_scheduling` | true |
| `drop_rate` | 0.0 |
| `drop_path_rate` | 0.1 |
| `qkv_bias` | true |
| `attn_drop_rate` | 0.0 |

### Source files

- Real-data model: `configs/model/ac_hope_hybrid_vit.yaml`
- Real-data experiment overrides: `configs/experiment/cl_ac_hope_phase8_hybrid.yaml`
- Benchmark model: `configs/model/benchmark_hybrid.yaml`

---

## 3. Phase 11 — AC-DNH-HOPE-Hybrid-ViT

### Parameters that differ

| Parameter | Real-Data Config | Benchmark Config | Factor |
|---|---|---|---|
| `embed_dim` | 1024 | 384 | 2.7× smaller |
| `predictor_embed_dim` | 384 | 192 | 2× smaller |
| `depth` | 12 | 8 | 1.5× fewer blocks |
| `num_heads` | 16 | 6 | 2.7× fewer heads |
| `dnh_L_max` | 5 | 4 | 1 fewer max level |
| `meta_hidden_dim` | 192 | 96 | 2× smaller |
| `img_size` | [256, 256] | [32, 32] | Different modality |
| `patch_size` | 16 | 4 | See note above |
| `num_timesteps` | 8 | 1 | Single image vs video |
| `action_embed_dim` | 7 (or 2) | 1 (dummy) | No actions in classification |
| `is_frame_causal` | true | false | No temporal causality needed |

### Parameters that are identical

| Parameter | Value |
|---|---|
| `dnh_L_init` | 2 |
| `dnh_L_min` | 2 |
| `titan_hidden_multiplier` | 1 |
| `titan_layers` | 2 |
| `titan_activation` | gelu |
| `titan_grad_clip_inner` | 1.0 |
| `titan_grad_clip_backward` | 1.0 |
| `titan_detach_interval` | 2 |
| `surprise_threshold` | 0.1 |
| `use_longterm_memory` | true |
| `longterm_hidden_multiplier` | 1 |
| `longterm_lr_scale` | 0.1 |
| `cms_level_specs` | {fast: 1.5, medium: 2.0, slow: 2.5} |
| `cms_use_chunk_scheduling` | true |
| `cms_L_max` | 5 |
| `cms_L_min` | 2 |
| `drop_rate` | 0.0 |
| `drop_path_rate` | 0.1 |
| `qkv_bias` | true |
| `attn_drop_rate` | 0.0 |

### Source files

- Real-data model: `configs/model/ac_dnh_hope_hybrid_vit.yaml`
- Real-data experiment overrides: `configs/experiment/cl_ac_hope_phase11_dnh.yaml`
- Benchmark model: `configs/model/benchmark_dnh.yaml`

---

## 4. Why the Benchmark is ~4000× Faster

The training speed difference between real-data and benchmark experiments is expected and arises from three compounding factors:

| Factor | Real Data | Benchmark | Ratio |
|---|---|---|---|
| Sequence length (tokens) | 8 × 256 = 2048 | 1 × 64 = 64 | 32× |
| Self-attention cost O(N²) | 2048² ≈ 4.2M | 64² ≈ 4K | ~1024× |
| Embedding dimension | 1024 | 384 | 2.7× |
| Depth (blocks) | 12 | 8 | 1.5× |
| **Approximate combined factor** | | | **~4,000×** |

---

## 5. What the Benchmark Validates (and What It Does Not)

### Does validate

- The **architectural design** — same class, same forward-pass composition (attention → Titan/DNH → CMS → longterm memory), same DGD inner-loop mechanics.
- All **relative design choices** — Titan multipliers, CMS frequency hierarchy, DNH evolution thresholds, surprise gating, longterm memory.
- That the architecture functions correctly under a standard CL protocol (Split CIFAR-100 task-incremental, Permuted MNIST domain-incremental) and produces meaningful AA/BWT metrics comparable to published baselines.

### Does not validate

- That the **specific scale** (1024-dim, 12-layer, 16-head) behaves identically to the benchmark scale (384-dim, 8-layer, 6-head) under catastrophic forgetting.
- Any claim that "the exact same model" was tested on both real data and benchmarks.

### Recommended thesis language

> "We validate the AC-HOPE-Hybrid and AC-DNH-HOPE-Hybrid architectural designs on standard CL benchmarks (Split CIFAR-100, Permuted MNIST) using the same model classes scaled to match the input dimensionality and complexity of these datasets. All structural choices — attention–memory composition, DGD inner-loop, CMS frequency hierarchy, and longterm memory — are preserved."
