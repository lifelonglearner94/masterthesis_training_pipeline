### Executive Summary

This experiment evaluates a **Naive Fine-Tuning Baseline** model in a Continual Learning (CL) setting. Unlike specialized memory architectures, this approach relies on standard sequential gradient descent without explicit mechanisms to preserve past representations.

The pipeline consists of a **Base Training Phase** (5,000 clips) followed by sequential fine-tuning on **5 distinct distribution shifts** (1,000 clips each). A critical constraint of this baseline experiment is that no parameters are frozen or protected during the sequential shifts, meaning the network is fully updated on new tasks, setting up a classic scenario to measure unmitigated catastrophic forgetting.

---

### Architectural & Methodological Context

* **Optimization:** The model uses standard gradient descent, sequentially updating all parameters as new task distributions are introduced. There is no inner/outer loop separation or bi-bilevel optimization as seen in advanced memory models.
* **Jump Prediction Target:** The model is tasked with predicting future states at a jump of $k=3$. The target frames are $z_6, z_7,$ and $z_8$. Similar to other architectures, the network consistently struggles most with the furthest temporal target ($z_8$) and finds an optimization "sweet spot" on $z_7$.

---

### Phase-by-Phase Deep Dive

#### 0. Base Training (Clips 0–5000)

* **Base Performance ($R_{jump}[0, 0]$):** The model achieves a very strong initial jump prediction $L_1$ loss of **$0.2783$**.
* **Forward Transfer (Zero-Shot Baseline):** At this stage, the model's zero-shot performance on unseen future tasks is recorded as a baseline to measure future learning efficacy:
* Task 1 (Scaling): $0.3948$
* Task 2 (Dissipation): $0.4216$
* Task 3 (Discretization): $0.4820$ (Hardest zero-shot task)
* Task 4 (Kinematics): $0.4132$
* Task 5 (Compositional OOD): $0.4719$



#### 1. Task 1: Scaling Shift (Clips 5000–6000)

* **Task 1 Performance ($R_{jump}[1, 1]$):** Unrestricted fine-tuning allows the model to sharply improve the jump loss on this split from the zero-shot baseline of $0.3948$ down to **$0.2791$**.
* **Catastrophic Forgetting:** Performance on the Base task immediately and severely degrades from $0.2783 \rightarrow \mathbf{0.3529}$, demonstrating the interference caused by unrestricted parameter updates.

#### 2. Task 2: Dissipation Shift (Clips 6000–7000)

* **Task 2 Performance ($R_{jump}[2, 2]$):** Fine-tuning yields an excellent jump loss of **$0.2953$**, a significant improvement over its zero-shot baseline ($0.4216$).
* **Catastrophic Forgetting:** Base task loss degrades further to **$0.3805$**. Task 1 loss also degrades significantly from $0.2791 \rightarrow 0.3087$.

#### 3. Task 3: Discretization Shift (Clips 7000–8000)

* **Task 3 Performance ($R_{jump}[3, 3]$):** For the hardest in-domain task, the model successfully adapts, driving the loss down from the $0.4820$ Phase 0 zero-shot evaluation to **$0.3419$**.
* **Catastrophic Forgetting:** Base task loss drops further to **$0.4030$**. Task 1 degrades to $0.3538$, and Task 2 degrades to $0.3264$.

#### 4. Task 4: Kinematics Shift (Clips 8000–9000)

* **Task 4 Performance ($R_{jump}[4, 4]$):** The model adapts remarkably well to the kinematics shift, achieving a strong jump loss of **$0.2929$**.
* **Catastrophic Forgetting:** The Base task loss now sits at **$0.4227$**. Task 3 takes an immediate degradation hit, jumping from $0.3419 \rightarrow 0.3853$.

#### 5. Task 5: Compositional OOD (Clips 9000–10000)

* **Task 5 Performance ($R_{jump}[5, 5]$):** The model concludes its learning phase with an end-state jump loss of **$0.3167$** on the compositional out-of-distribution shift.
* **Final Catastrophic Forgetting:** The Base task concludes with a heavily degraded jump loss of **$0.4401$**.

---

### Key Scientific Observations & Trends

1. **The Temporal Target Difficulty Gradient:**
Just like highly specialized architectures, the naive fine-tuning model exhibits a strict correlation between prediction error and the temporal distance of the jump. In the final evaluation on Task 5 data, we see:

* $z_6$ (Closest): Mean Loss = $0.3069$
* $z_7$ (Middle): Mean Loss = $0.2882$ *(The network consistently finds its lowest error margin at $z_7$)*
* $z_8$ (Furthest): Mean Loss = $0.3549$

2. **Continual Learning Metrics (Final State):**

* **$cl\_jump/StreamForgetting$**: The final tracked forgetting metric lands at **$0.0856$**. (Noticeably higher than specialized memory models, confirming the fragility of the naive approach).
* **Memory Decay Matrix:** The continuous, linear degradation is starkly visible in the Base Task ($Exp_0$) column of the $R_{jump}$ matrix: $0.2783 \rightarrow 0.3529 \rightarrow 0.3805 \rightarrow 0.4030 \rightarrow 0.4227 \rightarrow 0.4401$.

3. **Plasticity vs. Stability Trade-off:**
This baseline perfectly illustrates the plasticity/stability dilemma in Neural Networks. Because there are no frozen parameters or pure-retrieval constraints, the model exhibits **exceptional plasticity**—it achieves slightly better in-domain minimums on newly introduced tasks than constrained architectures (e.g., hitting $0.2953$ on Task 2 and $0.2929$ on Task 4). However, it possesses **zero stability**. Learning a new task completely overwrites the synaptic weights necessary for previous domains, leading to rapid, unimpeded catastrophic forgetting across the stream.
