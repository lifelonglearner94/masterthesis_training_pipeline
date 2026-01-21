# The Process of a Single Training Iteration (Step)

In each iteration, the following process occurs with a batch of data:

# Encoder

## 1. **Sampling (Data Selection):**
* A mini-batch of **256 video clips** is randomly sampled from the Droid dataset.
* Each clip is **4 seconds** long.
* The clips are sampled at a frame rate of 4 frames per second (fps), resulting in 16 frames per clip.


## 2. **Data Augmentation:**
* The video clips undergo "Random-Resize-Crop" augmentations (random scaling and cropping) to vary the visual input and make the model more robust.




## 3. **Encoding (Frozen Features):**
* The frames are passed through the **frozen V-JEPA 2 Encoder** (ViT-g).
* Since the encoder is frozen, its weights are *not* updated during this step. It simply converts the raw pixels into a sequence of feature maps (latent representations) .



## 4. **Action and State Preparation:**
* The robot's end-effector states (position, orientation, gripper) are extracted.
* Actions are computed as the change in state between frames.


# Predictor


## 5. **Prediction (The "World Model" Task):**
* The predictor network (a Transformer) receives the current latent representation , the action , and the robot state .
* The model attempts to predict the latent representation of the *next* video frame.




## 6. **Loss Calculation (Error Measurement):**
Two types of losses are calculated and summed up:

* **Teacher-Forcing Loss:** The model predicts the next step () given the ground-truth history up to .
* **Rollout Loss:** The model predicts multiple steps into the future (specifically  steps) autoregressively (using its own previous prediction as input for the next step) to learn long-term dynamics.




## 7. **Optimization (Weight Update):**
* The calculated error (L1 Loss) is backpropagated.
* The weights of the **Predictor** (and only the Predictor) are updated using the AdamW optimizer.





### Summary of the Total Duration

Instead of a fixed number of "Epochs", the training schedule is defined as follows:

* **Warmup:** 4,500 Iterations (Linear increase of Learning Rate).
* **Constant Phase:** 85,500 Iterations.
* **Decay:** 4,500 Iterations (Decay to 0).
* **Total:** Approximately **94,500 Iterations**.
