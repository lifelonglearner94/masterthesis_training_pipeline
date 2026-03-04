# Titans MAC Architecture — Implementation Documentation

**Date:** 2026-03-04
**File:** `src/models/titans.py`
**Config:** `configs/model/titans.yaml`, `configs/experiment/cl_titans.yaml`
**Reference:** Behrouz et al., "Titans: Learning to Memorize at Test Time", 2024

---

## 1. Overview

The Titans model implements the **Memory as a Context (MAC)** variant from the Titans paper, adapted for **action-conditioned spatiotemporal prediction** over pre-computed V-JEPA2 encoder feature maps. The model predicts future frames from 8-timestep sequences of 16×16 patch grids with 1024-dimensional features, conditioned on 2D actions.

The key innovation is the **Neural Memory Module (NMM)** — a small MLP whose weights are updated at *test time* via surprise-gated gradient descent (a form of test-time training / TTT). This gives the model an implicit form of continual adaptation independent of the outer optimizer.

**Parameter count:** ~43M (param-matched with the AC-ViT predictor for fair comparison).

---

## 2. Architecture Diagram

```
Input: z ∈ [B, T×N, 1024]    actions ∈ [B, T, 2]
       (T=8 timesteps, N=256 spatial patches, D=1024)

For each timestep t = 0, 1, ..., T-1:
│
├─ 1×1 Conv Encoder: [B, 1024, 16, 16] → [B, 768, 16, 16]
│
├─ Action Tiling + Linear Projection:
│     concat([feat, action_tiled], dim=C)  →  [B, 770, 16, 16]
│     flatten + Linear(770 → 768)          →  [B, 256, 768]
│
├─ + Temporal Position Embedding:                          ← NEW
│     tokens += TemporalEmbed(pos_t)       →  [B, 256, 768]
│
├─ 4× MACTitanLayer (pre-norm residual):
│     tokens = tokens + MACTitanLayer(LayerNorm(tokens))
│
├─ Final LayerNorm
│
├─ 1×1 Conv Decoder: [B, 768, 16, 16] → [B, 1024, 16, 16]
│
└─ Residual: z_{t+1} = z_t + δ

Output: [B, T×N, 1024]
```

---

## 3. Components in Detail

### 3.1 Neural Memory Module (NMM)

The NMM is a small MLP (`n_layers_nmm=2`, hidden width `2 × hidden_dim = 1536`) whose weights serve as **long-term associative memory**. It learns a mapping from keys to values via an inner-loop gradient descent step at every forward pass.

**Architecture:**

```
Linear(768 → 1536) → SiLU → Linear(1536 → 768)
```

**Key/Value projections:**

```
K: Linear(768 → 768, no bias)    # Xavier init
V: Linear(768 → 768, no bias)    # Xavier init
```

**Associative Memory Loss (inner loop):**

$$l(\mathcal{M}_{t-1}; x_t) = \| \mathcal{M}_{t-1}(\text{SiLU}(\ell_2\text{-norm}(x_t W_K))) - \text{SiLU}(x_t W_V) \|_2^2$$

**Surprise-Gated Momentum Update:**

$$S_t = \eta \cdot S_{t-1} - \theta \cdot \text{clip}(\nabla l, -1, 1)$$

$$\mathcal{M}_t = \alpha \cdot \mathcal{M}_{t-1} + S_t$$

Where:
- $\alpha = 0.999$ — weight decay towards initial weights (forgetting)
- $\eta = 0.8$ — surprise momentum (how much past surprise carries over)
- $\theta = 0.3$ — surprise learning rate (step size for current gradient)

**Critical implementation detail:** The NMM maintains a separate `_running_params` dictionary (detached clones of the learned `nn.Parameter` tensors) that the TTT inner loop modifies. The actual `nn.Parameter` tensors are **never mutated in-place** during forward — only the outer optimizer touches them. This prevents the corruption of PyTorch's autograd graph, which would otherwise cause NaN gradients. The K and V projection weights are excluded from inner-loop updates (only the MLP body is adapted).

**Retrieval** is a simple forward pass through the MLP using the running parameters:

$$y_t = \mathcal{M}_t^*(\ell_2\text{-norm}(x_t W_Q))$$

**Reset:** Surprise accumulators and running params are reset between batches (`reset_surprise()`), so each sequence starts from the learned initial weights.

### 3.2 MAC (Memory as a Context) Layer

Each MACTitanLayer combines three memory systems:

| Memory Type | Scope | Mechanism |
|---|---|---|
| **Persistent Memory (PM)** | Task knowledge | 4 learnable tokens, data-independent |
| **Neural Memory (NMM)** | Long-term context | MLP weights adapted via TTT |
| **Self-Attention** | Short-term context | Standard transformer attention |

**Forward pass of a single MACTitanLayer:**

```
Input: x ∈ [B, N, hidden_dim]      (N = 256 spatial tokens)

1. Retrieve from NMM:
   queries = SiLU(ℓ₂-norm(Q(x)))
   nmm_vals = NMM.retrieve(queries)    ← uses _running_params
   nmm_vals = LayerNorm(nmm_vals)      ← keeps attention scores bounded

2. Prepend PM + NMM context:
   x_cat = [PM₄ ; nmm_vals₂₅₆ ; x₂₅₆]    → [B, 516, 768]

3. Self-Attention (pre-norm TransformerEncoderLayer):
   att_out = TransformerEncoderLayer(x_cat)
   x_out = att_out[:, 260:, :]         ← extract input-position tokens only

4. Per-token projection:
   x_flat = Linear(x_out)

5. Update NMM:
   NMM.update(x_flat)                  ← surprise-gated inner-loop step

6. Gated output:
   y = NMM.retrieve(ℓ₂-norm(Q(x_flat)))    ← post-update retrieval
   output = LayerNorm(x_flat) ⊙ σ(y)

Output: [B, N, hidden_dim]
```

**Stability features:**
- `norm_first=True` in `TransformerEncoderLayer` (pre-norm — prevents activation explosion)
- `LayerNorm` on NMM retrieved values before attention
- `LayerNorm` before sigmoid gating
- Gradient clipping (±1.0) inside the NMM inner loop
- Scaled persistent memory init (`0.02 × randn`)

### 3.3 Temporal Position Embedding

**Purpose:** The model needs to know *which timestep* it is predicting. This is critical for **jump prediction** where the model receives only frame 0 and action 0, but must predict a specific future frame τ (e.g., frame 6, 7, or 8).

**Implementation:** A learned `nn.Embedding(num_timesteps + 1, hidden_dim)` table (9 × 768 for T=8). At each timestep, the corresponding embedding vector is added to all spatial tokens before they enter the MAC layers.

**Position assignment:**

| Mode | Position used | Rationale |
|---|---|---|
| **Teacher forcing** | `pos = t` (loop index 0…T-1) | Natural temporal ordering |
| **Jump prediction** | `pos = target_timestep - 1` | Matches AC-ViT RoPE convention: output corresponds to prediction of frame `target_timestep` |

**Why learned embeddings instead of 3D RoPE?**

The AC-ViT predictor uses 3D Rotary Position Embeddings that encode (time, height, width) directly inside multi-head attention. RoPE requires modifying the attention computation to apply rotations to Q/K vectors — but our MAC layer uses PyTorch's `nn.TransformerEncoderLayer` as a black-box attention core, and also concatenates PM + NMM + input tokens with heterogeneous semantics.

A learned additive temporal embedding is:
1. **Simpler** — just `tokens += temporal_embed(t)` before the MAC layers
2. **Compatible** with the NMM retrieval/update pipeline (no attention internals to change)
3. **Sufficient** — the 16×16 spatial structure is already captured by the Conv encoder/decoder, so only the temporal axis needs a position signal
4. **Low overhead** — 9 × 768 = 6,912 parameters

### 3.4 TitansBackbone

The backbone wraps the MAC layers into an action-conditioned autoregressive predictor:

**Per-timestep pipeline:**
1. **Encode:** 1×1 Conv `[B, 1024, 16, 16] → [B, 768, 16, 16]`
2. **Action conditioning:** Tile action `[B, 2] → [B, 2, 16, 16]`, concat with features, linear projection `770 → 768`
3. **Temporal embedding:** Add `temporal_embed(t)` to all 256 tokens
4. **MAC stack:** 4 MACTitanLayers with pre-norm residual connections + final LayerNorm
5. **Decode:** 1×1 Conv `[B, 768, 16, 16] → [B, 1024, 16, 16]` (zero-initialized for stable residual start)
6. **Residual:** `z_{t+1} = z_t + δ`

NMM surprise state is reset at the start of each batch, so the TTT adaptation happens fresh within each 8-frame sequence.

---

## 4. Parameter Budget

Default configuration (`hidden_dim=768, n_layers=4, n_layers_nmm=2`):

| Component | Parameters | Notes |
|---|---|---|
| Conv Encoder | ~0.8M | 1×1 conv: 1024 → 768 |
| Action Projection | ~0.6M | Linear: 770 → 768 |
| Temporal Embedding | ~0.007M | 9 × 768 |
| 4× MACTitanLayer | ~41.0M | Attention + NMM + PM + norms |
| Conv Decoder | ~0.8M | 1×1 conv: 768 → 1024 |
| LayerNorms (backbone) | ~0.03M | 4× pre-norm + 1× final |
| **Total** | **~43.1M** | Param-matched with AC-ViT |

**Per MACTitanLayer (~10.25M):**
- TransformerEncoderLayer (2 heads): ~7.1M
- NMM MLP (768 → 1536 → 768): ~2.4M
- NMM K/V projections: ~1.2M
- Q projection: ~0.6M
- Final linear: ~0.6M
- Persistent memory: 4 × 768 = 3,072
- LayerNorms: ~6,144

---

## 5. Training Configuration

### 5.1 Loss Function

Identical to other CL models (via `ACPredictorLossMixin`):

$$L = \lambda_{\text{tf}} \cdot L_{\text{teacher}} + \lambda_{\text{jump}} \cdot L_{\text{jump}}$$

- **Teacher forcing** ($L_{\text{tf}}$): Model receives frames 0…T-1 in sequence, predicts frames 1…T. L1 loss against ground truth.
- **Jump prediction** ($L_{\text{jump}}$): Model receives frame 0 + action 0 with `target_timestep=τ`, predicts frame τ directly. τ sampled uniformly from the last `jump_k` frames.

### 5.2 Optimizer

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 4.25e-4 |
| Weight decay | 0.04 |
| Betas | (0.9, 0.999) |
| Gradient clip | 1.01 (norm) |
| Precision | fp32 |

### 5.3 LR Schedule (iteration-based)

| Phase | Fraction | Description |
|---|---|---|
| Warmup | 8.5% | Linear from 7.5e-5 → 4.25e-4 |
| Constant | 83% | Hold at 4.25e-4 |
| Decay | 8.5% | Linear decay to 0 |

---

## 6. CL Pipeline

Experiment config: `configs/experiment/cl_titans.yaml`

```
Base Training:    clips 0–5000,  40 epochs
Task 1 (scaling_shift):        clips 5000–6000,  10 epochs (finetune)
Task 2 (dissipation_shift):    clips 6000–7000,  10 epochs (finetune)
Task 3 (discretization_shift): clips 7000–8000,  10 epochs (finetune)
Task 4 (kinematics_shift):     clips 8000–9000,  10 epochs (finetune)
Task 5 (compositional_ood):    clips 9000–10000, 10 epochs (finetune)
```

Pipeline mode: `sequential` with naive `finetune` (no replay, no regularisation — the NMM's test-time adaptation is the implicit CL mechanism).

---

## 7. Key Implementation Decisions & Lessons

### 7.1 Separating NMM running params from nn.Parameters

The original Titans reference implementation (`temp/titans-lmm/`) mutates `param.data` in-place during the NMM update. This works in their setup because they use a **separate optimizer for outer params only** (excluding NMM parameters). In our Lightning module with a single AdamW optimizer, in-place mutation corrupts the autograd graph → **NaN gradients**.

**Solution:** The NMM maintains a `_running_params` dict (detached clones) that the TTT inner loop updates. The `nn.Parameter` tensors are only ever modified by the outer optimizer. `functional_call()` routes retrieval through whichever param set is appropriate.

### 7.2 Pre-norm vs Post-norm Transformer

Using `norm_first=False` (PyTorch default, post-norm) caused training to **collapse to NaN after ~1k steps**. The loss would drop initially, then flatline, then suddenly produce NaN. Switching to `norm_first=True` (pre-norm) resolved this — raw activations are normalised *before* entering attention/FFN, preventing the exponential growth that post-norm allows.

### 7.3 LayerNorm on NMM context

NMM retrieved values can grow unbounded as the running params drift from initialisation. Without normalisation, these large values cause attention score overflow. `LayerNorm` on NMM context before attention and before output gating keeps activations bounded.

### 7.4 Temporal Position Embedding for Jump Prediction

Without temporal embeddings, the model has no way to distinguish "predict frame 6" from "predict frame 8" during jump prediction — both receive the same frame 0 input. The learned `nn.Embedding` table provides this signal. During teacher forcing the natural loop index provides position; during jump prediction the `target_timestep` parameter overrides it.

---

## 8. Usage

```bash
# Base training only
uv run src/cl_train.py experiment=cl_titans paths.data_dir=/path/to/clips

# Full CL pipeline
uv run src/cl_train.py experiment=cl_titans paths.data_dir=/path/to/clips

# Resume from base training checkpoint
uv run src/cl_train.py experiment=cl_titans \
    paths.data_dir=/path/to/clips \
    cl.resume_from_base_checkpoint=/path/to/base_training_final.ckpt
```

---

## 9. Relation to Paper Variants

The Titans paper proposes three architectural variants:

| Variant | Description | Implemented? |
|---|---|---|
| **MAC** (Memory as a Context) | NMM output prepended as attention context | **Yes** ← this implementation |
| **MAG** (Gated Memory) | NMM and SWA as parallel branches with gating | No |
| **MAL** (Memory as a Layer) | NMM as sequential compression layer before attention | No |

Our implementation follows MAC because it most naturally fits the "process spatial tokens with additional context" paradigm of our action-conditioned predictor, and it is the variant emphasised in the paper for sequence modelling tasks.
