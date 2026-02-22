# Adaptive World Models with HOPE

Master thesis research: Can self-modifying memory (HOPE) outperform standard Test-Time Adaptation for Action-Conditioned video prediction in the latent space of V-JEPA 2?

## What this is

We train **Action-Conditioned (AC) Predictors** on pre-encoded V-JEPA 2 features from a physics simulation dataset. Objects receive a force impulse and slide — we test adaptation to **progressive dynamic shifts** in a **Continual Learning** protocol.

Three models are compared under identical CL conditions:
- **ViT-AC** — 24-layer Vision Transformer baseline (from V-JEPA 2)
- **ViT-AC + TTA** — Same model, adapted at test-time via LayerNorm parameters
- **AC-HOPE-ViT** — Novel architecture using Titan Memory + Continuum Memory System (HOPE, Behrouz 2025), which self-modifies during the forward pass

Plus two bounding baselines:
- **Lower Bound** — Naive sequential finetuning (maximizes catastrophic forgetting)
- **Upper Bound** — Joint (i.i.d.) training on all data (theoretical performance ceiling)

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)

## Data Format

Pre-encoded V-JEPA 2 feature maps + action arrays as `.npy` files:

```
data_dir/
├── clip_00001/
│   ├── feature_maps/
│   │   └── vjepa2_vitl16.npy   # [T, N, D] e.g. [8, 256, 1024]
│   └── actions_states/
│       └── actions.npy         # [T_original, action_dim]
├── clip_00002/
└── ...
```

V-JEPA 2 uses tubelet_size=2, so 16 RGB frames → 8 encoded timesteps → 7 action steps.

## Continual Learning Pipeline

The main experiment is a **CL pipeline** with progressive dynamic shifts:

```
Base Training (5000 clips)
  → Eval
  → Task 1: Scaling Shift (1000 clips)       → Eval
  → Task 2: Dissipation Shift / Ice (1000)    → Eval
  → Task 3: Discretization Shift / Walls (1000) → Eval
  → Task 4: Kinematics Shift / Rotation (1000)  → Eval
  → Task 5: Compositional OOD (1000)           → Eval
```

After each phase, a **full evaluation** (no weight updates) runs on fixed validation clips from every partition. CL metrics (FWT, BWT, forgetting, R-matrix) are computed automatically.

### Running the full CL experiments

```bash
# AC-ViT + TTA (tasks via test-time adaptation)
uv run src/cl_train.py experiment=cl_ac_vit paths.data_dir=/path/to/clips

# AC-HOPE-ViT (tasks via finetuning, eval with frozen inner-loop)
uv run src/cl_train.py experiment=cl_ac_hope paths.data_dir=/path/to/clips

# Lower Bound — naive sequential finetuning (no CL mechanisms)
uv run src/cl_train.py experiment=cl_lower_bound paths.data_dir=/path/to/clips

# Upper Bound — joint training on all data (i.i.d., best-case ceiling)
uv run src/cl_train.py experiment=cl_upper_bound paths.data_dir=/path/to/clips
```

Each experiment creates **separate W&B runs per phase**, grouped under a single experiment group for easy comparison.

### CL Metrics

Computed via an R-matrix `R[i,j]` = loss on task `j` after training experience `i`:

| Metric | Description |
|---|---|
| **Top1_L1_Stream** | Average loss on all seen tasks |
| **StreamForgetting** | Average forgetting across previously learned tasks |
| **ExperienceForgetting** | Per-task forgetting relative to best past performance |
| **ForwardTransfer** | Zero-shot performance on unseen future tasks |
| **Top1_L1_Exp** | Per-task loss at each evaluation point |

## Baseline Identity
```bash
uv run src/eval.py experiment=test_baseline_identity paths.data_dir=/path/to/clips
```

## Single-Phase Training

### ViT-AC Predictor (baseline)

```bash
uv run src/train.py experiment=vjepa2_ac paths.data_dir=/path/to/your/clips
```

### AC-HOPE-ViT (novel)

```bash
uv run src/train.py experiment=ac_hope_vit_param_matched paths.data_dir=/path/to/clips
```

## Evaluation (TTA)

Test-Time Adaptation — adapts LayerNorm parameters online:

```bash
uv run src/eval.py experiment=test_ac_predictor_tta_extended \
    paths.data_dir=/path/to/your/clips \
    ckpt_path=/path/to/checkpoint.ckpt
```

## Project Structure

```
src/
├── cl_train.py              # CL pipeline orchestrator (main entry point)
├── train.py                 # Single-phase training
├── eval.py                  # Single-phase evaluation
├── models/
│   ├── ac_predictor/        # AC-ViT (standard transformer)
│   ├── hope/                # AC-HOPE-ViT (Titan + CMS)
│   └── mixins/              # Shared loss computation & TTA
├── datamodules/             # Pre-computed feature data pipeline
├── utils/
│   └── cl_metrics.py        # CL metrics tracker (R-matrix, FWT, BWT, forgetting)
└── callbacks/               # W&B logging & verbose diagnostics

configs/experiment/
├── cl_ac_vit.yaml           # CL pipeline: AC-ViT + TTA
├── cl_ac_hope.yaml          # CL pipeline: AC-HOPE-ViT
├── cl_lower_bound.yaml      # CL pipeline: naive finetuning (lower bound)
├── cl_upper_bound.yaml      # CL pipeline: joint training (upper bound)
├── vjepa2_ac.yaml           # Single-phase AC-ViT training
├── ac_hope_vit_param_matched.yaml  # Single-phase HOPE training
└── test_ac_predictor_tta.yaml      # TTA evaluation
```

## Configuration

[Hydra](https://hydra.cc/)-based. All configs live in `configs/`. Logging via [Weights & Biases](https://wandb.ai/).
