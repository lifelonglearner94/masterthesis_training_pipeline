# V-JEPA 2 Repository Analysis

## 1. Structure of "Tublets"

In this repository, **tublets** refer to the spatio-temporal tokens created from the input video. They are implemented in the `PatchEmbed3D` class.

*   **Implementation**: A "tublet" is created using a 3D Convolution (`nn.Conv3d`).
*   **Dimensions**:
    *   Instead of just spatial patches ($16 \times 16$), the model takes a chunk of time as well.
    *   The kernel size and stride are defined as `(tubelet_size, patch_size, patch_size)`.
    *   **Code Reference**: [src/models/utils/patch_embed.py](src/models/utils/patch_embed.py#L42-L47)
    *   **Typical Value**: A standard configuration in this repo uses a `tubelet_size` of **2**. This means one token represents a $16 \times 16$ spatial area across 2 consecutive frames.

## 2. Data Flow & Tensor Dimensions

Here is the flow of a video clip through the system, based on the `VisionTransformer` and `VisionTransformerPredictorAC` classes.

### A. Input Data
*   **Shape**: $(B, 3, T, H, W)$
*   Example: Batch of videos with 16 frames and $224 \times 224$ resolution.

### B. Encoder (ViT)
1.  **Tokenization**: The `PatchEmbed3D` layer converts the video into tokens.
    *   Start: $(B, 3, T, H, W)$
    *   After Conv3d: $(B, D_{enc}, T', H', W')$ where $T' = T / \text{tubelet\_size}$ and $H', W' = H,W / \text{patch\_size}$.
2.  **Flattening**: The spatial and temporal dimensions are flattened into a single sequence.
    *   **Shape**: $(B, N, D_{enc})$
    *   Where $N = T' \times H' \times W'$ is the total number of tokens.
3.  **Transformer Blocks**: The sequence length $N$ and dimension $D_{enc}$ remain constant through the encoder blocks.
    *   **Output**: $(B, N, D_{enc})$

### C. AC-Predictor (Action-Conditional)
1.  **Input Projection**: The encoder output is projected to the predictor's internal dimension.
    *   **Shape**: $(B, N_{ctxt}, D_{enc}) \rightarrow (B, N_{ctxt}, D_{pred})$
2.  **Action Injection**: This is the crucial step. The sequence is reshaped back into a grid to insert actions at each timestep.
    *   Reshape: $(B, T', H' \times W', D_{pred})$
    *   Concatenation: Action tokens ($a$) and State tokens ($s$) are inserted for each timestep.
    *   New Shape: $(B, T', (H' \times W' + 2), D_{pred})$
    *   *Note: "+2" accounts for the action and state token added to every time step.*
3.  **Processing**: The predictor transformer processes this sequence.
4.  **Output**: The extra action/state tokens are stripped, and the features are projected back to the encoder dimension for loss calculation.
    *   **Final Shape**: $(B, N, D_{enc})$

**Code Reference**: The injection logic is in [src/models/ac_predictor.py](src/models/ac_predictor.py#L145-L153).

## 3. Embedding Dimensions

Different encoders have different feature dimensions ($D_{enc}$). These are defined in the factory functions in [src/models/vision_transformer.py](src/models/vision_transformer.py):

*   **ViT-Large (`vit_large`)**: 1024
*   **ViT-Huge (`vit_huge`)**: 1280
*   **ViT-Giant (`vit_giant_xformers`)**: 1408

The **Predictor** often operates at a smaller dimension (e.g., $D_{pred} = 384$) as seen in config files like [configs/train/vitl16/pretrain-256px-16f.yaml](configs/train/vitl16/pretrain-256px-16f.yaml#L76), regardless of the encoder size.

## 4. Suitable Feature Maps for AC-Predictor

The AC-Predictor is flexible regarding the **embedding dimension** because it has an input projection layer `self.predictor_embed` that maps whatever input it gets to its internal dimension.

However, it is strict regarding the **Grid Structure**:
*   The predictor calculates positional embeddings and masks based on a specific `img_size`, `patch_size`, and `tubelet_size`.
*   Therefore, any feature map fed into the AC-Predictor must match the **spatial and temporal grid** ($T', H', W'$) that the predictor was initialized with.
*   You cannot, for example, feed features from a model with `patch_size=32` into a predictor expecting `patch_size=16`, because the sequence length $N$ (and the grid reshape logic) would be incorrect.
