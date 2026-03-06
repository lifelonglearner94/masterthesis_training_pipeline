# Adaptive World Models with HOPE

Master thesis research: **Can self-modifying memory (HOPE) reduce catastrophic forgetting in Action-Conditioned world models under continual distribution shift?**

We train Action-Conditioned (AC) predictors on pre-encoded **V-JEPA 2** latent features from a physics simulation dataset. Objects receive force impulses and slide across surfaces — the model must adapt to **progressive dynamic shifts** in a **Continual Learning** (CL) protocol without forgetting previously learned physics.

---

## Models Under Comparison

### HOPE Variants (Novel Architectures)

| Experiment | Config | Architecture | Key Idea |
|---|---|---|---|
| **AC-HOPE Phase 6** | `cl_ac_hope_phase6` | Titan Memory + CMS (depth 5, ~46.5M) | Persistent longterm memory (`M_longterm`) that survives across clips. Clip-level `M_memory` resets each clip for plasticity; `M_longterm` accumulates slowly via DGD (10x lower LR) for forgetting resistance. A learned gate interpolates between the two. |
| **AC-HOPE Phase 8 Hybrid** | `cl_ac_hope_phase8_hybrid` | Attention + Titan + CMS (depth 12, ~44M) | Retains full self-attention with 3D RoPE and *adds* Titan memory as augmentation (not replacement). 1 compact Titan memory per block, learned eta/alpha, no aux loss. 12 blocks possible because per-block cost is lower. |
| **AC-HOPE Phase 10 Replay** | `cl_ac_hope_phase10_replay` | Phase 8 Hybrid + soft-frozen attention + Titan reset + experience replay (~44M) | Builds on Phase 8 with CL-specific mechanisms: attention gets a near-frozen LR (0.02x) to preserve token mixing, Titan memories reset at task boundaries, CMS carries task-specific adaptation, and a reservoir-sampled replay buffer mixes 30% past data into each training batch. |

### Bounding Baselines

| Experiment | Config | Description |
|---|---|---|
| **Upper Bound** | `cl_upper_bound` | Joint (i.i.d.) training on *all* data simultaneously. Defines the theoretical performance ceiling — no CL method should systematically exceed this. AC-ViT, ~43M params. |
| **Lower Bound** | `cl_lower_bound` | Naive sequential fine-tuning with no CL mechanisms. Maximises catastrophic forgetting and defines the performance floor. AC-ViT, ~43M params. |

### Standard Comparison Architectures

The CL pipeline is also tested with parameter-matched (~43M) standard architectures under identical naive fine-tuning conditions:

- **GatedDeltaNet** — Gated delta-rule linear attention with Triton-fused chunk operations (`cl_gated_delta_net`)
- **RetNet** — Retentive Network with multi-scale retention and chunkwise-recurrent computation (`cl_retnet`)
- **Transformer++** — Pre-norm Transformer with RoPE, GatedMLP, and optional Flash Attention (`cl_transformer_pp`)

### Standard CL Benchmarks

The repository is ready to evaluate any architecture on established CL benchmarks:

- **Permuted MNIST** — 10 domain-incremental tasks (pixel permutations), `cl_permuted_mnist`
- **Split CIFAR-100** — 10 class-incremental tasks (10 classes each), `cl_split_cifar100`

All benchmark models are parameter-matched (~9.3M backbone params). Supported architectures: `benchmark_hybrid`, `benchmark_dnh`, `benchmark_hope`, `benchmark_gdn`, `benchmark_titans`, `benchmark_retnet`, `benchmark_transformer_pp`.

---

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- GPU with >= 24 GB VRAM recommended for the main CL pipeline

```bash
# Core dependencies
uv sync

# For comparison architectures (GatedDeltaNet, RetNet, Transformer++)
uv sync --extra comparison-architectures
```

---

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

V-JEPA 2 uses `tubelet_size=2`, so 16 RGB frames -> 8 encoded timesteps -> 7 action steps.

---

## Continual Learning Pipeline

The main experiment is a sequential CL pipeline with progressive dynamic shifts:

```
Base Training (5000 clips, standard physics)
  → Full Evaluation
  → Task 1: Scaling Shift         (1000 clips)  → Full Evaluation
  → Task 2: Dissipation / Ice     (1000 clips)  → Full Evaluation
  → Task 3: Discretization / Walls (1000 clips) → Full Evaluation
  → Task 4: Kinematics / Rotation (1000 clips)  → Full Evaluation
  → Task 5: Compositional OOD     (1000 clips)  → Full Evaluation
```

After each phase, evaluation runs on fixed validation clips from **every** partition (no weight updates). CL metrics are computed automatically from the resulting R-matrix.

### Running the Main CL Experiments

```bash
# AC-HOPE Phase 6 — Persistent longterm memory
uv run src/cl_train.py experiment=cl_ac_hope_phase6 paths.data_dir=/path/to/clips

# AC-HOPE Phase 8 — Hybrid (Attention + Titan + CMS)
uv run src/cl_train.py experiment=cl_ac_hope_phase8_hybrid paths.data_dir=/path/to/clips

# AC-HOPE Phase 10 — Hybrid + soft-freeze + replay
uv run src/cl_train.py experiment=cl_ac_hope_phase10_replay \
    cl.resume_from_base_checkpoint=/path/to/base_training_final.ckpt \
    paths.data_dir=/path/to/clips

# Upper Bound — Joint training on all data
uv run src/cl_train.py experiment=cl_upper_bound paths.data_dir=/path/to/clips

# Lower Bound — Naive sequential fine-tuning
uv run src/cl_train.py experiment=cl_lower_bound paths.data_dir=/path/to/clips
```

### Running Comparison Architectures

```bash
# GatedDeltaNet
uv run src/cl_train.py experiment=cl_gated_delta_net paths.data_dir=/path/to/clips

# RetNet
uv run src/cl_train.py experiment=cl_retnet paths.data_dir=/path/to/clips

# Transformer++
uv run src/cl_train.py experiment=cl_transformer_pp paths.data_dir=/path/to/clips
```

Each experiment creates **separate W&B runs per phase**, grouped under a single experiment group.

### CL Metrics

Computed via an R-matrix where `R[i,j]` = loss on task `j` after training on experience `i`:

| Metric | Description |
|---|---|
| **Top1_L1_Stream** | Average loss across all seen tasks |
| **StreamForgetting** | Average forgetting across previously learned tasks |
| **ExperienceForgetting** | Per-task forgetting relative to best past performance |
| **ForwardTransfer** | Zero-shot performance on unseen future tasks |
| **Top1_L1_Exp** | Per-task loss at each evaluation point |

---

## Standard CL Benchmarks

Evaluate any architecture on Permuted MNIST or Split CIFAR-100:

```bash
# Permuted MNIST (10 tasks, domain-incremental)
uv run src/cl_benchmark_train.py experiment=cl_permuted_mnist model=<model>

# Split CIFAR-100 (10 tasks, class-incremental)
uv run src/cl_benchmark_train.py experiment=cl_split_cifar100 model=<model>
```

Replace `<model>` with any of: `benchmark_hybrid`, `benchmark_dnh`, `benchmark_hope`, `benchmark_gdn`, `benchmark_titans`, `benchmark_retnet`, `benchmark_transformer_pp`.

---

## Project Structure

```
src/
├── cl_train.py              # CL pipeline orchestrator (main entry point)
├── cl_benchmark_train.py    # Standard CL benchmarks (MNIST, CIFAR-100)
├── train.py                 # Single-phase training
├── eval.py                  # Single-phase evaluation
├── models/
│   ├── ac_predictor/        # AC-ViT baseline (standard transformer)
│   ├── hope/                # AC-HOPE-ViT, Hybrid, DNH variants (Titan + CMS)
│   ├── gated_delta_net/     # GatedDeltaNet (gated delta-rule linear attention)
│   ├── retnet/              # RetNet (multi-scale retention)
│   ├── transformer_pp/      # Transformer++ (pre-norm + RoPE + GatedMLP)
│   └── benchmark_classifier.py  # Shared classifier head for CL benchmarks
├── datamodules/             # Lightning DataModules for all datasets
├── callbacks/               # Custom callbacks (CL metrics, diagnostics)
└── utils/                   # Shared utilities

configs/
├── config.yaml              # Hydra root config
├── experiment/              # Per-experiment overrides
│   ├── cl_ac_hope_phase6.yaml
│   ├── cl_ac_hope_phase8_hybrid.yaml
│   ├── cl_ac_hope_phase10_replay.yaml
│   ├── cl_upper_bound.yaml
│   ├── cl_lower_bound.yaml
│   ├── cl_gated_delta_net.yaml
│   ├── cl_retnet.yaml
│   ├── cl_transformer_pp.yaml
│   ├── cl_permuted_mnist.yaml
│   └── cl_split_cifar100.yaml
└── model/                   # Architecture configs
```

---

## Tech Stack

- **[PyTorch](https://pytorch.org/)** + **[Lightning](https://lightning.ai/)** — training framework
- **[Hydra](https://hydra.cc/)** — hierarchical config management
- **[W&B](https://wandb.ai/)** — experiment tracking and logging
- **[uv](https://github.com/astral-sh/uv)** — Python package manager
- **[flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention)** — Triton kernels for RetNet and GatedDeltaNet
