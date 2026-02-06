# HOPE vs ViT Research

Gold standard research repository comparing HOPE and Vision Transformers (ViT).

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager


## 💻 Hardware Support

This repository supports multiple hardware platforms with automatic detection:

| Platform | Accelerator | Precision | Notes |
|----------|-------------|-----------|-------|
| **NVIDIA CUDA** | `gpu` | `16-mixed` | Best performance, FlashAttention available |
| **Apple Silicon (MPS)** | `mps` | `32` | Good performance, limited fp16 support |
| **CPU** | `cpu` | `32` | Slowest, but always works |


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
# Train AC Predictor with V-JEPA2 settings (recommended)
uv run src/train.py experiment=vjepa2_ac paths.data_dir=/path/to/your/clips

# Train with default settings
uv run src/train.py model=ac_predictor data=precomputed_features paths.data_dir=/path/to/your/clips

# With GPU and custom batch size
uv run src/train.py model=ac_predictor data=precomputed_features trainer=gpu data.batch_size=16 paths.data_dir=/path/to/your/clips
```

### Baseline ConvLSTM Model, without TTA

A simpler ConvLSTM baseline for comparison with the ViT AC Predictor. Uses identical training protocol, loss functions, and curriculum schedule for scientific comparability.

```bash
# Train baseline ConvLSTM (scientifically comparable to vjepa2_ac)
uv run src/train.py experiment=baseline_convlstm paths.data_dir=/path/to/your/clips
```

| Model | Parameters | Architecture |
|-------|------------|--------------|
| **ViT AC Predictor** (`vjepa2_ac`) | ~43M | 24-layer Transformer with RoPE |
| **ConvLSTM Baseline** (`baseline_convlstm`) | ~5M | ConvLSTM with residual learning |



### Testing

#### TTA Full rollout mode (recommended)
* only LayerNorm
uv run src/eval.py experiment=test_ac_predictor_tta \
    paths.data_dir=/path/to/your/clips \
    ckpt_path=/path/to/checkpoint.ckpt

* LayerNorm + Attention output projections
uv run src/eval.py experiment=test_ac_predictor_tta_extended paths.data_dir=/path/to/your/clips ckpt_path=/path/to/checkpoint.ckpt


## Novel AC_HOPE_ViT

uv run src/train.py experiment=ac_hope_vit_param_matched paths.data_dir=/path/to/clips
uv run src/train.py experiment=ac_hope_vit_depth_matched paths.data_dir=/path/to/clips


**Expected data format** (`.npy` files):
```
data_dir/
├── clip_00001/
│   ├── feature_maps/
│   │   └── vjepa2_vitl16.npy   # [T*N, D] flattened or [T, N, D] (e.g., 2048x1024 for 8 timesteps, 256 patches)
│   └── actions_states/
│       └── actions.npy         # [T_original, action_dim] (value at index 1 is preserved -> index 0)
├── clip_00002/
└── ...
```

**Note on temporal alignment**: V-JEPA2 uses tublet encoding (tubelet_size=2), so feature maps have `num_timesteps = original_frames // 2`. Actions/states are automatically adjusted to `T_actions = num_timesteps - 1`. For 16 original frames → 8 encoded timesteps → 7 action timesteps.

**IMPORTANT**: The `num_timesteps` parameter in configs refers to ENCODED timesteps, NOT original video frames.


## 🔧 Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management. The main config file is `configs/config.yaml`.


## 📊 Logging

Experiments are logged to [Weights & Biases](https://wandb.ai/). Set your API key:

```bash
cp .env.example .env
# Edit .env with your WANDB_API_KEY
```
