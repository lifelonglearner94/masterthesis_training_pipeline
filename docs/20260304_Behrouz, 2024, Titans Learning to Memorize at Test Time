### 1. The Long-term Neural Memory Module ($\mathcal{M}$)

The memory is designed as a neural network (typically an MLP with $L_{\mathcal{M}} \ge 1$ layers) trained at test time via online gradient descent.

*
**Key and Value Projections:** The input $x_t$ is projected into keys and values using two linear layers.


*
$k_t = x_t W_K$


*
$v_t = x_t W_V$




*
**Associative Memory Loss:** The memory learns to map keys to values by optimizing the following objective in its inner loop.


*
$$l(\mathcal{M}_{t-1}; x_t) = ||\mathcal{M}_{t-1}(k_t) - v_t||_2^2$$







*
**Memory Update Rule (with Momentum and Forgetting):** The memory parameters are updated using momentary surprise (gradients), past surprise (momentum), and an adaptive forgetting mechanism (decay).


*
$$\mathcal{M}_t = (1 - \alpha_t)\mathcal{M}_{t-1} + S_t$$





*
$$S_t = \eta_t S_{t-1} - \theta_t \nabla l(\mathcal{M}_{t-1}; x_t)$$





*
*(Note: $\alpha_t \in [0, 1]$ is the gating/forgetting mechanism, $\eta_t$ controls surprise decay, and $\theta_t$ is the step size/momentary surprise weight)*.




*
**Memory Retrieval (Inference):** Information is retrieved via a forward pass without weight adjustment, denoted as $\mathcal{M}^*$.


*
$q_t = x_t W_Q$


*
$$y_t = \mathcal{M}^*(q_t)$$








### 2. Parallelizing the Memory Training

To train efficiently, the sequence is split into chunks of size $b \ge 1$. The mini-batch gradient descent update can be tensorized as follows:

*
$$\mathcal{M}_t = \beta_t \mathcal{M}_0 - \sum_{i=1}^t \theta_i \frac{\beta_t}{\beta_i} \nabla l(\mathcal{M}_{t'}; x_i)$$





*
*(Note: $t' = t - \text{mod}(t, b)$, and $\beta_i = \prod_{j=1}^i (1 - \alpha_j)$)*.



### 3. Persistent Memory

Titans use learnable, data-independent parameters to encode task knowledge, prepended to the sequence.

* Given parameters $P = [p_1, p_2, ..., p_{N_p}]$, the input sequence is modified to:


*
$$x_{new} = [p_1, p_2, ..., p_{N_p}] || x$$








### 4. Architectural Variants (Incorporating the Memory)

The authors propose three ways to integrate the neural memory module with attention mechanisms.

#### Variant A: Memory as a Context (MAC)

The sequence is chunked into fixed-size segments $S^{(i)}$, and the memory provides historical context to the attention module.

*
**Retrieve:** $h_t = \mathcal{M}_{t-1}^*(q_t)$ where $q_t = S^{(t)} W_Q$.


* **Attention:** $\tilde{S}^{(t)} = [p_1, p_2, ..., p_{N_p}] || h_t || [cite_start]S^{(t)}$


*
**Process:** $y_t = \text{Attn}(\tilde{S}^{(t)})$


*
**Update & Output:** $\mathcal{M}_t = \mathcal{M}_{t-1}(y_t)$ and $o_t = y_t \otimes \mathcal{M}_t^*(y_t)$



#### Variant B: Gated Memory (MAG)

The model processes the input through two branches: direct input to the neural memory, and a Sliding Window Attention (SWA) branch, combined via gating.

*
$$\tilde{x} = [p_1, p_2, ..., p_{N_p}] || x$$





*
$$y = \text{SW-Attn}(\tilde{x})$$





*
$$o = y \otimes \mathcal{M}(\tilde{x})$$





*
*(Note: $\otimes$ represents a gating mechanism where outputs are normalized using learnable vector-valued weights followed by a non-linearity $\sigma$)*.



#### Variant C: Memory as a Layer (MAL)

The neural memory acts as a sequential layer compressing data before passing it to the attention module.

*
$$\tilde{x} = [p_1, p_2, ..., p_{N_p}] || x$$





*
$$y = \mathcal{M}(\tilde{x})$$





*
$$o = \text{SW-Attn}(y)$$






### 5. Low-Level Implementation Details

*
**Activations:** Use $\text{SiLU}(\cdot)$ as the non-linear activation for computing query, key, and values.


*
**Normalization:** Normalize queries and keys using $l_2$-norm.


*
**Convolutions:** Incorporate a 1D depthwise-separable convolution layer after each of the query, key, and value projections.
