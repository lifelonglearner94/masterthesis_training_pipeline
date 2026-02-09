# Detailed Execution Trace: `test_ac_predictor` Experiment

This document details the exact step-by-step execution flow when running the `test_ac_predictor` experiment.

**Command:**
```bash
uv run src/eval.py experiment=test_ac_predictor ckpt_path=/path/to/checkpoint.ckpt
```

**Configuration Reference:** `configs/experiment/test_ac_predictor.yaml`

---

## 1. Initialization Phase

The `src/eval.py` script starts and initializes the environment.

1.  **Configuration Loading**:
    *   Hydra loads `configs/config.yaml`.
    *   Apply overrides from `experiment/test_ac_predictor.yaml`.
    *   **Key Overrides**:
        *   `train`: False, `test`: True.
        *   `data`: `precomputed_features` (test clips 15000-16000).
        *   `model`: `ac_predictor` (Teacher-Forcing + Autoregressive).
        *   `callbacks`: Only `RichProgressBar` (checkpointing disabled).
    *   **Model Parameters**:
        *   `num_timesteps`: 8 (Encoded timesteps).
        *   `context_frames`: 3 (Frames 0, 1, 2 used as seed).
        *   `T_rollout`: 4 (Predicting frames 3, 4, 5, 6).
        *   `T_teacher`: 7 (Max teacher forcing steps).

2.  **Dataset Instantiation (`PrecomputedFeaturesDataModule`)**:
    *   Target clip range: `15000` to `16000`.
    *   Batch size: `32`.

3.  **Model Instantiation (`ACPredictorModule`)**:
    *   Architecture: `vit_ac_predictor` initialized.
    *   **Weights Loading**: Manually loads `state_dict` from `ckpt_path`.

---

## 2. Data Loading Process (Per Batch)

Inside `src/datamodules/precomputed_features.py`:

1.  **Feature Loading**:
    *   Reads `.npy` file for a clip (e.g., `clip_15000/feature_maps/vjepa2_vitl16.npy`).
    *   **Shape Handling**:
        *   Input: `[16, 256, 1280]` (Frames, H, W, D) or `[4096, 1280]` (Flattened) or `[8, 256, 1280]` (Already encoded).
        *   Standardizes to: `[T_encoded, N, D]` (e.g., `[8, 256, 1280]`).
    *   **Truncation**: Caps length at `num_timesteps + 1` (usually 9, but actual data is likely 8 steps). Result length `T=8`.

2.  **Action & State Alignment**:
    *   **Actions**:
        *   Reads `actions.npy` (Length 16).
        *   **Critical Resampling**: Takes action at original index `1` and moves to index `0` in the output array. This aligns the "causing" action with the "effect" frame interval.
        *   Result shape: `[T-1, 7]` (e.g., `[7, 7]`).
    *   **States**:
        *   Initialized as zeros: `[7, 7]`.

3.  **Collation**:
    *   Batches are padded to max length (though usually uniform).
    *   Final Batch Tensors:
        *   `features`: `[B, 8, 256, 1280]`
        *   `actions`: `[B, 7, 7]`
        *   `states`: `[B, 7, 7]`

---

## 3. Test Step Execution (`ACPredictorModule.test_step`)

For each batch, the `test_step` method is executed.

### A. Teacher Forcing Loss (Comparison Baseline)
*   **Method**: `_compute_teacher_forcing_loss`
*   **Process**:
    *   Input: All available context frames and actions.
    *   Output: Predicts frames 1..7 in parallel based on ground truth context.
    *   Loss: Mean feature distance between predicted and actual frames.

### B. Autoregressive Rollout (The Core Test)

This is the primary metric for evaluation (`_compute_rollout_loss_per_timestep`).

**Setup:**
*   **Total Available Frames**: 8 (Indices 0..7).
*   **Context (`C`)**: 3 (Indices 0, 1, 2).
*   **Ref Target (`T_pred`)**: `min(8 - 3, 4) = 4` steps.
*   **Prediction Targets**: Indices 3, 4, 5, 6.

**Algorithm Step-by-Step:**

1.  **Context Initialization**:
    *   Seed `z_ar` with Ground Truth frames: `[z₀, z₁, z₂]`.
    *   Shape: `[B, 3*N, D]`.

2.  **Autoregressive Loop (4 Iterations)**:

    *   **Iteration 1 (Predicting Index 3)**:
        *   **Input Context**: `[z₀, z₁, z₂]` (Ground Truth).
        *   **Action Context**: `[a₀, a₁, a₂]`.
        *   **Model Forward**: Predicts entire sequence `[z₁', z₂', z₃']`.
        *   **Extraction**: Takes last frame `z₃'`.
        *   **Update**: Append `z₃'` to `z_ar`. New context: `[z₀, z₁, z₂, z₃']` (Mixed GT and Pred).

    *   **Iteration 2 (Predicting Index 4)**:
        *   **Input Context**: `[z₀, z₁, z₂, z₃']`.
        *   **Action Context**: `[a₀, a₁, a₂, a₃]`.
        *   **Model Forward**: Predicts `[z₁', z₂', z₃'', z₄']`.
        *   **Extraction**: Takes last frame `z₄'`.
        *   **Update**: Append `z₄'` to `z_ar`. New context: `[z₀, z₁, z₂, z₃', z₄']`.

    *   **Iteration 3 (Predicting Index 5)**:
        *   **Input Context**: `[z₀, ..., z₄']`.
        *   **Action Context**: Actions up to `a₄`.
        *   **Result**: Generates `z₅'`.

    *   **Iteration 4 (Predicting Index 6)**:
        *   **Input Context**: `[z₀, ..., z₅']`.
        *   **Action Context**: Actions up to `a₅`.
        *   **Result**: Generates `z₆'`.

3.  **Loss Calculation**:
    *   **Predictions**: `[z₃', z₄', z₅', z₆']`.
    *   **Ground Truth**: `[z₃, z₄, z₅, z₆]`.
    *   **Per-Timestep Loss**: L1/L2 distance calculated separately for each step (Step 1=Frame 3, Step 4=Frame 6).

---

## 4. Logging and Aggregation

1.  **Per-Step Metrics**:
    *   `test/loss_step_3`: Loss for first predicted frame.
    *   `test/loss_step_4`: Loss for second predicted frame.
    *   ...etc.

2.  **Clip-Level Storage**:
    *   The model stores a dictionary for *every single clip* in the test set containing:
        *   `clip_name`
        *   `loss_rollout` (Average over the 4 steps)
        *   `per_timestep_losses` (Detailed breakdown)

3.  **Epoch End**:
    *   Aggregates all clip results.
    *   Computes global averages.
    *   Prints standard deviation and error bars.

## Summary of Dimensions

| Dimension | Value | Description |
| :--- | :--- | :--- |
| **Input Frames** | 16 | Original video frames |
| **Encoded Timesteps** | 8 | V-JEPA2 Features (Tubelet size 2) |
| **Context Size** | 3 | Ground truth frames given to model (0, 1, 2) |
| **Rollout Steps** | 4 | Number of autoregressive predictions (3, 4, 5, 6) |
| **Action Sequence** | 2 | Actions aligned to time intervals (Overridden in config) |
