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
# Train with default config
make train

# Train with specific experiment
uv run src/train.py experiment=example

# Train with custom overrides
uv run src/train.py model=hope trainer=gpu seed=123
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
├── Dockerfile              # Container definition
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
docker build -t titans-vit-research .

# Run training
docker run --gpus all titans-vit-research python src/train.py
```

## 📊 Logging

Experiments are logged to [Weights & Biases](https://wandb.ai/). Set your API key:

```bash
cp .env.example .env
# Edit .env with your WANDB_API_KEY
```

## 📝 License

MIT
