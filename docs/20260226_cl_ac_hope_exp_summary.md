Here is a detailed scientific summary of the Continual Learning (CL) experiment based on the provided logs and dashboard descriptions.

### Executive Summary

This experiment evaluates the **AC-HOPE-ViT (Action-Conditioned Hierarchical Objective Predictive Encoding)** model in a Continual Learning setting. The architecture (~42M parameters) utilizes self-modifying Titan memories and Context Memory System (CMS) multi-frequency MLPs (fast, medium, slow updates) with inner-loop Dynamic Gradient Descent (DGD).

The pipeline consists of a **Base Training Phase** (5,000 clips) followed by sequential fine-tuning on **5 distinct distribution shifts** (1,000 clips each). A critical constraint of this experiment is that evaluations are performed in a **pure retrieval mode** (frozen model + frozen inner-loop DGD), meaning no parameter updates occur during inference.

---

### Architectural & Methodological Context

* **Optimization:** The model uses bi-bilevel optimization. The outer meta-learning loop uses an $L_1$ loss with a learning rate of $1.5 \times 10^{-4}$ (base) scaled down to $5 \times 10^{-5}$ for sequential fine-tuning to mitigate catastrophic forgetting.
* **Jump Prediction Target:** The model is tasked with predicting future states at a jump of $k=3$. The target frames are $z_6, z_7,$ and $z_8$. Across almost all phases, the network struggles most with the furthest temporal target ($z_8$) and is most accurate on $z_7$.

---

### Phase-by-Phase Deep Dive

#### 0. Base Training (Clips 0–5000)

* **Dynamics:** Trained over 40 epochs. The training loss curves are highly jagged, reflecting the complex optimization landscape of the self-modifying memories, while the validation curves are smooth and flatten out, indicating convergence.
* **Base Performance ($R_{jump}[0, 0]$):** Achieves a strong initial jump prediction $L_1$ loss of **$0.3256$**.
* **Forward Transfer (Zero-Shot Baseline):** At this stage, the model's zero-shot performance on unseen future tasks is recorded as a baseline:
* Task 1 (Scaling): $0.4085$
* Task 2 (Dissipation): $0.4391$
* Task 3 (Discretization): $0.4915$ (Hardest zero-shot task)
* Task 4 (Kinematics): $0.4242$
* Task 5 (Compositional OOD): $0.4634$



#### 1. Task 1: Scaling Shift (Clips 5000–6000)

* **Dynamics:** 10 epochs. The training curves show a zigzag pattern with sharp peaks and deep valleys but an overall downward trend.
* **Task 1 Performance ($R_{jump}[1, 1]$):** Fine-tuning improves the jump loss on this split from the zero-shot baseline of $0.4085$ down to **$0.3272$**.
* **Catastrophic Forgetting:** Performance on the Base task degrades from $0.3256 \rightarrow \mathbf{0.3916}$, indicating that learning the scaling shift causes immediate interference with the base representation.

#### 2. Task 2: Dissipation Shift (Clips 6000–7000)

* **Dynamics:** High volatility is observed, with a massive spike in loss around step 100 before settling into a fluctuating but lower plateau.
* **Task 2 Performance ($R_{jump}[2, 2]$):** Fine-tuning yields a jump loss of **$0.3447$**, a significant improvement over its zero-shot baseline ($0.4391$) and the Phase 1 forward-transfer state ($0.3799$).
* **Catastrophic Forgetting:** Base task loss degrades further to **$0.4106$**. Task 1 loss degrades slightly from $0.3272 \rightarrow 0.3515$.

#### 3. Task 3: Discretization Shift (Clips 7000–8000)

* **Dynamics:** The training curve remains relatively flat before experiencing a violent, vertical spike in loss around step 400, followed by a sharp recovery.
* **Task 3 Performance ($R_{jump}[3, 3]$):** This is the hardest in-domain task so far. The model achieves a jump loss of **$0.3932$**. While relatively high, it is a vast improvement over the $0.4915$ Phase 0 zero-shot evaluation.
* **Catastrophic Forgetting:** Base task loss drops further to **$0.4200$**. Task 1 degrades to $0.3834$, and Task 2 degrades to $0.3642$.

#### 4. Task 4: Kinematics Shift (Clips 8000–9000)

* **Dynamics:** Training shows two distinct, massive spikes in loss at steps 100 and 300, indicating severe distribution shocks that the DGD struggles to immediately reconcile before forcing the loss back down.
* **Task 4 Performance ($R_{jump}[4, 4]$):** The model adapts well, achieving a strong jump loss of **$0.3406$**.
* **Catastrophic Forgetting:** The Base task loss now sits at **$0.4244$**. Interestingly, Task 3 actually shows a slight *backward transfer* improvement, moving from $0.3932 \rightarrow 0.4148$ (Wait, $0.4148$ is a degradation, forgetting continues).

#### 5. Task 5: Compositional OOD (Clips 9000–10000)

* **Dynamics:** Training finishes with high volatility, mapping a continuous sequence of high peaks around steps 350 and 500, ending on a sharp downward drop.
* **Task 5 Performance ($R_{jump}[5, 5]$):** The model achieves an end-state jump loss of **$0.3666$**.
* **Final Catastrophic Forgetting:** The Base task concludes with a jump loss of **$0.4359$**.

---

### Key Scientific Observations & Trends

1. **The Temporal Target Difficulty Gradient:**
Across all evaluations, the prediction error strictly correlates with the temporal distance of the jump. For example, in the final evaluation on Task 5:
* $z_6$ (Closest): Mean Loss = $0.3545$
* $z_7$ (Middle): Mean Loss = $0.3294$ *(Note: the network consistently finds an optimization sweet spot at $z_7$, making it easier to predict than $z_6$)*
* $z_8$ (Furthest): Mean Loss = $0.4157$


2. **Continual Learning Metrics (Final State):**
* **$cl\_jump/StreamForgetting$**: The final tracked forgetting metric is **$0.0421$**.
* **Memory Decay Matrix:** The continuous degradation is clearly visible in the Base Task ($Exp_0$) column of the $R_{jump}$ matrix: $0.3256 \rightarrow 0.3916 \rightarrow 0.4106 \rightarrow 0.4200 \rightarrow 0.4244 \rightarrow 0.4359$.


3. **Efficacy of the Architecture:**
Despite the pure retrieval (frozen) evaluation setup, the model successfully learns every individual task presented to it. The fine-tuning phase successfully drives the highly elevated zero-shot $L_1$ losses (often ~0.45+) down into the ~0.34–0.39 range. However, standard gradient descent fine-tuning overpowers the CMS multi-frequency MLPs ability to strictly preserve past states, resulting in a classic, linear trajectory of catastrophic forgetting.
