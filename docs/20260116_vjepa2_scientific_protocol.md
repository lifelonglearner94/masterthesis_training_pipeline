# Scientific Protocol: V-JEPA 2 Architecture & Data Flow Analysis

**Date:** January 16, 2026
**Subject:** Analysis of Spatio-Temporal Tokenization, Data Flow, and Action-Conditioning in V-JEPA 2

---

## 1. Concept: Spatio-Temporal Tokenization ("Tublets")

**Observation:**
The V-JEPA 2 architecture utilizes a 3D patching mechanism to convert video input into discrete tokens. These tokens are referred to as **tublets**.

**Technical Specifications:**
*   **Implementation**: `PatchEmbed3D` class using `nn.Conv3d` projection.
*   **Structure**: A tublet encapsulates a volume of pixel data defined by `(tubelet_size, patch_size, patch_size)`.
*   **Standard Configuration**:
    *   `patch_size`: 16 pixels
    *   `tubelet_size`: 2 frames
*   **Implication**: One token represents a $16 \times 16$ spatial region across 2 consecutive frames. This provides an inherent 2x temporal downsampling at the tokenization stage.

---

## 2. Component Analysis

### 2.1 Encoder (Vision Transformer)
The encoder processes the video as a flattened sequence of tublets.

*   **Input**: $(B, 3, T, H, W)$
*   **Tokenization Output**: $(B, N, D_{enc})$, where $N = (T / \text{tubelet\_size}) \times (H / \text{patch\_size}) \times (W / \text{patch\_size})$.
*   **Feature Dimensions ($D_{enc}$)**:
    *   ViT-Large: 1024
    *   ViT-Huge: 1280
    *   ViT-Giant: 1408

### 2.2 Action-Conditional Predictor (AC-Predictor)
The AC-Predictor is a secondary transformer designed to predict latent features conditioned on actions.

*   **Input Handling**: Contains a projection layer `predictor_embed` to map any $D_{enc}$ to its internal $D_{pred}$ (typically 384).
*   **Grid Dependency**: Uses a "Input Injection" mechanism where the flattened sequence is reshaped back into a 3D grid $(T', H', W')$ to insert Action ($a$) and State ($s$) tokens at corresponding timesteps.
*   **Constraint**: The feature map fed to the predictor must strictly match the spatial-temporal grid configuration ($T', H', W'$) the predictor was initialized with.

---

## 3. Data Flow Dynamics

The standard flow of a tensor through the pipeline is as follows:

1.  **Video Input**: $(B, 3, 16, 224, 224)$
2.  **Encoder (ViT-L, tubelet=2)**:
    *   Tokens are created. Temporal dim reduces $16 \to 8$.
    *   Output: $(B, 8 \times 14 \times 14, 1024)$
3.  **Projection**:
    *   Mapped to predictor dim: $(B, 1568, 384)$
4.  **Action Injection (AC-Predictor)**:
    *   Sequence reshaped to $(B, 8, 196, 384)$
    *   Actions/States injected. New shape: $(B, 8, 198, 384)$
    *   Flattened for processing.
5.  **Output**:
    *   Projected back to $D_{enc} = 1024$ for loss calculation.

---

## 4. Critical Findings: Temporal Alignment

**Problem Statement:**
When `tubelet_size > 1` (e.g., 2), the encoder output has lower temporal resolution ($T/2$) than the input video ($T$). However, action data is typically recorded per-frame ($T$).

### 4.1 Verified Implementation Strategy (DROID Training)
Analysis of `app/vjepa_droid/train.py` reveals that the repository handles this mismatch via **Dense Evaluation** rather than subsampling actions.

*   **Mechanism**: The encoder processing loop artificially constructs inputs.
*   **Process**:
    1.  Single frames are selected.
    2.  Frames are duplicated (`.repeat(1, 1, 2, 1, 1)`) to satisfy the `tubelet_size=2` convolution kernel.
    3.  Encoder runs on these "fake" tublets.
*   **Result**: The system generates $T$ visual tokens for $T$ frames, maintaining 1:1 alignment with the $T$ action tokens. This bypasses the downsampling effect of the tublet at the cost of compute efficiency.

### 4.2 Alternative Strategies
If standard video inference is required (processing full clips):

**Strategy A: Action Subsampling**
*   **Method**: Downsample action sequence to match encoder output.
*   **Ratio**: If `tubelet_size=2`, take every 2nd action.
*   **Pros**: Computationally efficient.
*   **Cons**: Loss of high-frequency control information.

**Strategy B: Unit Tublet Size**
*   **Method**: Set `tubelet_size=1`.
*   **Result**: 1 Frame = 1 Token. Perfect alignment.
*   **Pros**: Simplifies logic; retains full temporal resolution.
*   **Cons**: Doubles encoder sequence length ($N$), leading to $4\times$ computational cost in Attention layers (quadratic scaling).


# DECIDED FOR STRATEGY A ACTION SUBSAMPLING
