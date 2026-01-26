# V-JEPA2 Patch Correspondence & Variability Analysis

## 1. Executive Summary

The investigation confirms that V-JEPA2 feature maps contain strong, fixed spatial positional encodings that dominate the signal. When analyzing raw features, this positional bias obscures actual visual content changes. By subtracting a "positional template," we successfully isolated the content residuals, allowing us to detect the object (the pink cube) seen in the source imagery.

## 2. Visual Evidence: Linking the Cube to the Data

The most significant validation of the hypothesis is found by comparing the input image (`grafik.png`) with the position-corrected heatmaps (`output.png`).

- **The Input (`grafik.png`)**: A pink cube is located in the center/center-left of the grid.
- **The Output (`output.png` - Bottom Row)**: The position-corrected variability maps show a distinct "blob" of high variability (dark area in L2 Norm, light area in Cosine Similarity) in the exact same center-left spatial location.

**Conclusion**: The raw V-JEPA2 features (Top Row) were too noisy to see the object due to positional embedding interference. Once corrected (Bottom Row), the feature variability accurately highlights the location of the pink cube, proving it is the source of the "content change" in the video clip.

## 3. Key Technical Findings

### A. The "Positional Dominance" Phenomenon

The study uncovered a counterintuitive behavior in V-JEPA2 features:

- **Observation**: A patch at $(x, y)$ in Video A is more similar to a patch at $(x, y)$ in Video B than it is to a patch at $(x, y)$ in a different frame of Video A.
- **Data**:
  - Cross-clip similarity (same pos): **0.711** (Very High)
  - Within-clip similarity (same pos): **0.434** (Lower)
- **Implication**: The model "bakes in" the $(x, y)$ coordinate into the feature vector so strongly that it overrides the actual visual pixel data (the content).

### B. Isolating Content (The Fix)

To see the actual visual dynamics (like the cube moving), the analysis applied a subtraction method:

$$
\text{Content Residual} = \text{Feature}_{\text{raw}} - \text{Positional Template}
$$

Where:
- **Positional Template** = Mean of features across multiple clips.

This operation removed the static "coordinate signal," leaving only the "visual signal."

## 4. Summary of Results

| Feature State | Visualization (`output.png`) | Interpretation |
|:---|:---|:---|
| **Raw Features** | **Top Row**: Checkerboard/Noisy pattern. | Dominated by fixed positional embeddings. The grid pattern likely represents the encoding structure itself rather than image content. |
| **Position-Corrected** | **Bottom Row**: Clear central "blob." | The positional signal is removed. The remaining signal represents the pink cube, which is the primary source of variability (motion/presence) in the clip. |

## 5. Recommendation

For any downstream task involving V-JEPA2 (such as action recognition or motion tracking), do not use raw feature comparisons. Always calculate a baseline positional template from a diverse set of clips and subtract it to work with the content residuals.
