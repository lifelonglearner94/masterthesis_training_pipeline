# Phase 9 & Phase 10 — Implementation Documentation

**Date:** 2026-03-03
**Context:** After 8 HOPE experiment phases, the best result (Phase 8 Hybrid) closes only ~10% of the Lower→Upper Bound gap.

---

## Motivation

Analysis of the Phase 8 Hybrid results revealed two key problems:

| Metric | Lower Bound (Naive) | Phase 8 Hybrid | Upper Bound (Joint) |
|--------|-------------------:|---------------:|--------------------:|
| Avg Error ↓ | 0.3720 | 0.3594 | 0.2502 |
| Forgetting ↓ | 0.0856 | 0.0538 | 0.0000 |
| Plasticity ↓ | 0.3007 | 0.3145 | 0.2699 |
| Gap Closed | 0% | 10.4% | 100% |

1. **Forgetting improved** (37% reduction vs naive) — Titan memory and longterm memory help.
2. **Plasticity DEGRADED** (0.3007 → 0.3145) — Titan's inner-loop DGD interferes with outer-loop gradient descent during task adaptation. The model learns new tasks *worse* than naive finetuning.

The gap closure is only 10% because the forgetting improvement gets cancelled by the plasticity degradation.

---

## Phase 9: Soft-Frozen Attention + Titan Reset + CMS Adaptation

**Config:** `configs/experiment/cl_ac_hope_phase9.yaml`
**Targets:** Both forgetting AND plasticity, while enabling lifelong co-adaptation

### Three interventions

#### 1. Soft-freeze attention during task training (very low LR)

Attention layers (~18M params) learned universal spatial-temporal token mixing (3D RoPE) during base training. Physics shifts mostly affect feature dynamics (CMS/Titan domain), not spatial structure. However, for lifelong learning, hard-freezing is too rigid — over many tasks, frozen attention becomes misaligned with the evolving CMS/Titan representations.

Instead, attention is trained with a very small LR scale (`attention_lr_scale: 0.02`, effective LR = 1.4e-4 × 0.02 = 2.8e-6), making it the **slowest timescale** in the multi-timescale hierarchy:
- Attention: 2.8e-6 (slowest — preserves structure, allows slow co-adaptation)
- CMS: 1.4e-4 (medium — primary task adaptation)
- Titan: 2.8e-4 (fastest — reset each task, aggressive learning)

This follows the same multi-timescale principle already built into CMS (fast/medium/slow levels) and matches biological learning where even "slow-learning" brain regions still adapt.

**Implementation:** The existing `attention_lr_scale` mechanism in optimizer parameter groups handles this — no `requires_grad=False` needed. Methods `freeze_attention()` / `unfreeze_attention()` remain available for ablation experiments.

#### 2. Reset Titan memories at each task boundary

Stale memory states from previous tasks create interference with outer-loop gradient descent on new tasks. Resetting to fresh meta-learned initial weights gives each task a clean Titan slate. The meta-learned initialisation IS the useful knowledge — it was learned during base training.

**Implementation:** New method `reset_titan_for_new_task()` on `ACHOPEHybridViT`:
- Resets M_memory active weights to meta-learned nn.Parameters
- Resets M_longterm active weights (unlike `reset_all_memories()` which preserves longterm)
- Resets all diagnostic counters and CMS step counters

#### 3. Keep CMS fully adaptable

CMS multi-frequency MLPs (fast/medium/slow) carry the task-specific adaptation burden with `cms_lr_scale: 1.0` (effective LR = 1.4e-4). Titan also gets an aggressive LR (`titan_lr_scale: 2.0`, effective = 2.8e-4) since it starts fresh each task.

### Integration in `cl_train.py`

The `run_task_training_finetune()` function reads config flags from `cl.task_training`:
- `freeze_attention: false` + `attention_lr_scale: 0.02` → soft freeze via tiny optimizer LR
- `reset_titan_memories: true` → calls `model.model.reset_titan_for_new_task()` before training

### Parameter budget during task training

| Component | Params | Effective LR | Role |
|-----------|-------:|:------------:|:----:|
| Attention | ~18M | 2.8e-6 | Slow co-adaptation (soft freeze) |
| Titan | ~8M | 2.8e-4 | Fast task learning (reset each task) |
| CMS | ~14M | 1.4e-4 | Primary task adaptation |
| Projections/Norms | ~4M | 1.4e-4 | Trainable |
| **Total** | **~44M** | multi-scale | **All trainable, 3-timescale hierarchy** |

---

## Phase 10: Experience Replay

**Config:** `configs/experiment/cl_ac_hope_phase10_replay.yaml`
**Targets:** Forgetting via data-level protection (complementary to Phase 9's parameter-level protection)

### Core insight

The upper bound trains on ALL data jointly. No amount of architectural cleverness in a pure sequential learner can match seeing old data again. Phase 10 adds a replay buffer that samples from past experiences and mixes them into each task training batch.

### Replay buffer design

**Implementation:** `src/utils/replay_buffer.py`

#### Reservoir sampling (Vitter 1985)

The `ReplayBuffer` class maintains a fixed-size random sample of all past clips:
- When buffer is not full: clip added directly
- When buffer is full: new clip replaces a random existing clip with probability `max_size / total_seen`
- This guarantees **uniform coverage** across all clips ever seen, regardless of insertion order

#### Population strategy

1. After base training: buffer populated from base training clips (0–4900, excluding eval clips)
2. After each task: completed task clips are added through reservoir sampling (`add_task_clips: true`)
3. Buffer diversity grows over time while maintaining uniform representation

#### Interleaved training batches

`_CombinedReplayLoader` merges task and replay DataLoaders:
- Each training step: draws one batch from task loader + one from replay loader
- Default split: 70% new task data + 30% replay data (`replay_ratio: 0.3`)
- Replay loader cycles when exhausted (buffer is small, ~500 clips)
- Random sampling each step prevents overfitting to buffer contents

#### Memory cost

500 clips × ~40KB per clip (features + actions as tensors) ≈ 20MB — negligible.

### Config structure

The presence of a `cl.replay` section in the config activates the replay buffer:

```yaml
cl:
  replay:
    buffer_size: 500        # Max clips in buffer
    replay_ratio: 0.3       # 30% replay per batch
    add_task_clips: true    # Grow buffer with completed tasks
```

The `_run_sequential_pipeline()` in `cl_train.py` checks for this section and:
1. Creates and populates a `ReplayBuffer` after base training
2. Routes task training through `run_task_training_finetune_with_replay()` instead of `run_task_training_finetune()`

---

## Files changed

| File | Change |
|------|--------|
| `configs/experiment/cl_ac_hope_phase8_selective_freeze.yaml` | Renamed to `cl_ac_hope_phase9.yaml`, rewritten |
| `configs/experiment/cl_ac_hope_phase9.yaml` | **NEW** — Phase 9 config |
| `configs/experiment/cl_ac_hope_phase10_replay.yaml` | **NEW** — Phase 10 config |
| `src/models/hope/ac_hope_hybrid_vit.py` | Added `freeze_attention()`, `unfreeze_attention()`, `reset_titan_for_new_task()` |
| `src/utils/replay_buffer.py` | **NEW** — `ReplayBuffer` with reservoir sampling |
| `src/cl_train.py` | Added Phase 9 hooks in `run_task_training_finetune()`, added `_create_replay_buffer()`, `_add_task_clips_to_buffer()`, `_ReplayInterleavedDataModule`, `_CombinedReplayLoader`, `run_task_training_finetune_with_replay()`, updated `_run_sequential_pipeline()` |

---

## Usage

Both experiments require a base checkpoint from the Phase 8 Hybrid run.

### Phase 9

```bash
uv run src/cl_train.py experiment=cl_ac_hope_phase9 \
  cl.resume_from_base_checkpoint=/path/to/base_training_final.ckpt \
  paths.data_dir=/path/to/clips
```

### Phase 10

```bash
uv run src/cl_train.py experiment=cl_ac_hope_phase10_replay \
  cl.resume_from_base_checkpoint=/path/to/base_training_final.ckpt \
  paths.data_dir=/path/to/clips
```

---

## Expected impact

| Experiment | Forgetting | Plasticity | Avg Error | Gap Closed |
|------------|----------:|----------:|----------:|----------:|
| Lower Bound (actual) | 0.0856 | 0.3007 | 0.3720 | 0% |
| Phase 8 Hybrid (actual) | 0.0538 | 0.3145 | 0.3594 | 10% |
| Phase 9 (estimated) | ~0.02 | ~0.29 | ~0.30 | ~15–25% |
| Phase 10 (estimated) | ~0.01 | ~0.28 | ~0.28 | ~40–60% |
| Upper Bound (actual) | 0.0000 | 0.2699 | 0.2502 | 100% |

Phase 9 addresses the plasticity regression by removing Titan interference. Phase 10 addresses the data access gap that no architectural trick can solve alone.
