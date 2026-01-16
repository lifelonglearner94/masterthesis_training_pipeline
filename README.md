# HOPE vs ViT Research

Gold standard research repository comparing HOPE and Vision Transformers (ViT).

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager

### Installation

```bash
# Install dependencies
make install
# or directly with uv
uv sync
```

### Training

```bash
# Train with default config (auto-detects best hardware)
make train

# Train with specific experiment
uv run src/train.py experiment=vjepa2_ac

# Train with custom overrides
uv run src/train.py model=hope trainer=gpu seed=123
```

## 💻 Hardware Support

This repository supports multiple hardware platforms with automatic detection:

| Platform | Accelerator | Precision | Notes |
|----------|-------------|-----------|-------|
| **NVIDIA CUDA** | `gpu` | `16-mixed` | Best performance, FlashAttention available |
| **Apple Silicon (MPS)** | `mps` | `32` | Good performance, limited fp16 support |
| **CPU** | `cpu` | `32` | Slowest, but always works |

### Platform-Specific Training

```bash
# Auto-detect best available hardware (recommended)
uv run src/train.py trainer=auto

# Force NVIDIA CUDA GPU (fails if not available)
uv run src/train.py trainer=gpu

# Use default (also auto-detects)
uv run src/train.py
```

### Hardware Detection

The pipeline automatically logs detected hardware at startup:
```
============================================================
Hardware Detection
============================================================
Device: Apple Silicon (MPS)
Precision: 32
PyTorch Lightning Accelerator: mps
============================================================
```

On non-CUDA systems, you'll see a warning about FlashAttention not being available. This is expected - the pipeline uses PyTorch's `scaled_dot_product_attention` with automatic backend selection.

### AC Predictor Model

The repository includes an Action-Conditioned Vision Transformer Predictor (from V-JEPA2) for training on pre-computed encoder features.

```bash
# Train AC Predictor with pre-computed features
uv run src/train.py model=ac_predictor data=precomputed_features

# With GPU and custom batch size
uv run src/train.py model=ac_predictor data=precomputed_features trainer=gpu data.batch_size=16
```

**Expected data format** (`.npy` files):
```
data/
├── train/
│   ├── episode_0000/
│   │   ├── features.npy   # [T+1, N, D] encoder features
│   │   ├── actions.npy    # [T, 7] end-effector changes
│   │   └── states.npy     # [T, 7] end-effector states
│   └── ...
└── val/
    └── ...
```

### Testing

```bash
make test
```

## 📁 Project Structure

```
.
├── configs/                 # Hydra configuration files
│   ├── config.yaml         # Main config entry point
│   ├── callbacks/          # Callback configurations
│   ├── data/               # DataModule configurations
│   ├── experiment/         # Full experiment configs
│   ├── hparams_search/     # Hyperparameter search configs
│   ├── logger/             # Logger configurations
│   ├── model/              # Model configurations
│   ├── paths/              # Path configurations
│   └── trainer/            # Trainer configurations
├── src/                    # Source code
│   ├── callbacks/          # Custom callbacks
│   ├── datamodules/        # PyTorch Lightning DataModules
│   ├── models/             # Model implementations
│   └── utils/              # Utility functions
├── tests/                  # Unit tests
├── pyproject.toml          # Project dependencies (PEP 621)
├── uv.lock                 # Locked dependencies
└── Makefile                # Workflow automation
```

## 🔧 Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management. The main config file is `configs/config.yaml`.

### Running Experiments

```bash
# Override single parameters
uv run src/train.py trainer.max_epochs=100

# Use predefined experiments
uv run src/train.py experiment=example

# Multi-run with different seeds
uv run src/train.py --multirun seed=42,123,456
```

### Hyperparameter Search

```bash
uv run src/train.py --multirun hparams_search=optuna
```

## 🐳 Docker

```bash
# Build image
docker build -t hope-vit-research .

# Run training
docker run --gpus all hope-vit-research python src/train.py
```

## 📊 Logging

Experiments are logged to [Weights & Biases](https://wandb.ai/). Set your API key:

```bash
cp .env.example .env
# Edit .env with your WANDB_API_KEY
```

## 📝 License

MIT
