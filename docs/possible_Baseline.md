# Possible Baseline

Since you are already employing a sophisticated training schedule (Curriculum Learning/Scheduled Sampling), you need a baseline that is architecturally simple but explicitly designed to handle state memory (for physics) and spatial data.

The optimal simple baseline for your requirements is an **Action-Conditioned ConvLSTM with Residual Learning**.

Here is the exact blueprint for this baseline:

## 1. Data Interpretation (Crucial Step)

Your "Simple Baseline" will fail if you treat the dimension 256 as a flat vector. In the context of encoded video features, this represents a spatial grid.

*   **Input Tensor:** $(B, 8, 256, 1024)$
*   **Physical Interpretation:** 8 time steps, 1024 feature channels, and a $16 \times 16$ spatial grid ($16 \times 16 = 256$).
*   **Required Reshape:** You must permute and reshape your data to:
    $$(B, \text{Time}, 1024, 16, 16)$$

## 2. The Architecture

This model is much lighter than a Transformer but captures the physics (momentum/velocity) via the LSTM hidden state, which is necessary since you are predicting 7 steps from only 1 input frame.

### Module A: 1x1 Convolution Bottleneck (Encoder)
**Drastically reduce parameters to prevent overfitting.**
*   **Input:** 1024 channels.
*   **Operation:** `Conv2d(in=1024, out=256, kernel=1)`.
*   **Output:** 256 channels.
*   **Reasoning:** Keeps the $16 \times 16$ spatial structure but compresses the channel depth.

### Module B: Action Injection (Spatial Tiling)
**Your actions are 2D vectors $(x, y)$, but your features are $16 \times 16$ grids.**
1.  Take the action vector for time $t$.
2.  Replicate (tile) it across the spatial dimensions ($16 \times 16$).
3.  Concatenate it with the features from Module A.
*   **LSTM Input:** $256 \text{ (Features)} + 2 \text{ (Action)} = \mathbf{258 \text{ Channels}}$.

### Module C: The Core (ConvLSTM Cell)
**Use a Convolutional LSTM, not a standard MLP-LSTM.**
*   **Mechanism:** It uses convolutions for input/state transitions.
*   **Why it fits your training strategy:** LSTMs are naturally suited for the shift from Teacher Forcing to Rolling Prediction. The hidden state $h_t$ and cell state $c_t$ accumulate the "physics" (velocity/inertia) that is missing from the single start frame.

### Module D: Residual Projection (Decoder)
**Predict the change in the physical state, not the new state itself.**
*   **Operation:** `Conv2d(in=256, out=1024, kernel=1)` applied to the LSTM hidden state.
*   **Final Output:** $\hat{z}_{t+1} = z_t + \text{Delta}_t$.
*   **Benefit:** This stabilizes the autoregressive rollout significantly, as the model only needs to learn the gradients of change.

## 3. Implementation Logic (Pseudocode)

This structure is easy to implement in PyTorch and perfect for benchmarking against your Transformer.

```python
class BaselineModel(nn.Module):
    def forward(self, z_start, actions, future_steps=7):
        # z_start shape: (Batch, 1024, 16, 16)
        # actions shape: (Batch, Total_Steps, 2)

        # Initialize Memory (Hidden States) with Zeros
        h, c = self.init_hidden(z_start.shape)

        current_z = z_start
        predictions = []

        for t in range(future_steps):
            # 1. Encode Features (Bottleneck)
            # Reduce 1024 -> 256 channels
            feat = self.encoder(current_z)

            # 2. Prepare Action
            # Get action for step t, reshape to (B, 2, 1, 1) and expand to (B, 2, 16, 16)
            act_map = self.tile_action(actions[:, t])

            # 3. Concatenate
            # Input becomes 258 channels
            lstm_in = torch.cat([feat, act_map], dim=1)

            # 4. ConvLSTM Step
            h, c = self.conv_lstm_cell(lstm_in, (h, c))

            # 5. Decode to Residual (Delta)
            # Expand 256 -> 1024 channels
            delta = self.decoder(h)

            # 6. Update State (Residual Connection)
            next_z = current_z + delta
            predictions.append(next_z)

            # 7. Autoregressive Loop
            # Update current_z for the next iteration
            # (Note: Handle your Teacher Forcing logic here if needed during training)
            current_z = next_z

        return torch.stack(predictions, dim=1)
```

## Why this is the correct Baseline

*   **Handles "1-to-7" Prediction:** Unlike a standard CNN (ResNet), the LSTM state ($c_t$) acts as a memory buffer to infer velocity from the first frame and the action sequence, which is strictly required for physics simulation.
*   **Robustness:** It has significantly fewer parameters than a Vision Transformer, making it a stable "lower bound" to compare against your complex model.
*   **Spatial Awareness:** By reshaping 256 to $16 \times 16$ and using ConvLSTM, you preserve the collision topology of your physics simulation.
