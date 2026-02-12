# Adaptive World Models with HOPE

Master thesis research: Can self-modifying memory (HOPE) outperform standard Test-Time Adaptation for Action-Conditioned video prediction in the latent space of V-JEPA 2?

## What this is

We train **Action-Conditioned (AC) Predictors** on pre-encoded V-JEPA 2 features from a physics simulation dataset. Objects receive a force impulse and slide — we test adaptation to domain shifts (changed friction/mass) in an **A→B→A** protocol.

Three models are compared:
- **ViT-AC** — 24-layer Vision Transformer baseline (from V-JEPA 2)
- **ViT-AC + TTA** — Same model, adapted at test-time via LayerNorm parameters
- **AC-HOPE-ViT** — Novel architecture using Titan Memory + Continuum Memory System (HOPE, Behrouz 2025), which self-modifies during the forward pass

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

## Training

### ViT-AC Predictor (baseline)

```bash
uv run src/train.py experiment=vjepa2_ac paths.data_dir=/path/to/your/clips
```

### AC-HOPE-ViT (novel)

```bash
uv run src/train.py experiment=ac_hope_vit_param_matched paths.data_dir=/path/to/clips
```

## Evaluation (TTA)

Test-Time Adaptation with rollout — adapts LayerNorm + Attention output projections online:

```bash
uv run src/eval.py experiment=test_ac_predictor_tta_extended \
    paths.data_dir=/path/to/your/clips \
    ckpt_path=/path/to/checkpoint.ckpt
```

## Configuration

[Hydra](https://hydra.cc/)-based. All configs live in `configs/`. Logging via [Weights & Biases](https://wandb.ai/).
