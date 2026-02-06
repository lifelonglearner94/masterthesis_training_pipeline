Here is the complete content of Section 8 from the provided document.

# **8 HOPE: A Self-Referential Learning Module with Continuum Memory**

As we discussed earlier in Section 5.1, architectures in nested learning are uniform, i.e., a set of feedforward neural network blocks, each of which with its own context, update frequency, and internal objective. Sequence models—a common term to refer to blocks often with the highest update frequency that fuse information across tokens in the input sequence—are critical components for memory management and in-context learning ability of models. Following our earlier discussion in Section 5, modern sequence models can be seen as associative memories and so are nested optimization problems. From this perspective, global softmax attention or its more expressive higher-order variants are perfect memories (forcing to cache all past tokens) with frequency update of infinity as they are non-parametric solutions for optimizing (local) $L\_2$-regression objective with Nadaraya-Watson estimators (Fan 2018; Zhang et al. 2022\) (see Equation 62). Therefore, parametric solutions (e.g., modern RNNs) for the similar objectives and when the parameter search space are the same (i.e., matrix-valued memory) are not expected to outperform softmax attention when the model size and data scales. To this end, and to design powerful sequence models, we need to understand where Transformers are limited and how one can overcome such limitations.  
\+4

From the nested learning perspective, Transformers are two-level components, where projections and MLP blocks are optimized in the first level and the second level is responsible for in-context learning with finding the non-parametric solution and so conditioning the output on the context. This design, however, has limited computational depth as also stated in recent studies on state-tracking and similar computational capabilities of models (Merrill et al. 2024; Sanford et al. 2024; Grazzi et al. 2025). Furthermore, Transformers' parameters are static throughout their context, meaning that their found solution to map tokens in the context (since it is non-parametric solution) remains the same and so they lack the ability to modify themselves (at least in-context). More specifically, the initial linear blocks, $W\_k, W\_v$, and $W\_q$, that projects input data to keys, values, and queries, are fixed after the pre-training stage (i.e., are in the first level) and so the Transformer's ability to contextualize and map tokens is bounded by the knowledge stored in these blocks. For example, given a 1-layer Transformer, the projection of each token is a function of the token itself and its position ; therefore, as an example, it can miss the diverse possible encodings of words whose meaning depend on the context, rather than the word itself. Although with increasing the depth of the model this issue might fade in later layers, we should not rely on the depth to compensate the models ability as it still is a bottleneck to unleash the capability of the model in earlier layers.  
\+4

To overcome the above challenge, recently, the use of short convolutions and canon layers (Allen-Zhu 2025\) have became a de facto component in modern models. Despite their success in mixing local tokens, still the models are fundamentally limited to adapt to the context and capture the global information beyond the local mixing. In the next part, we discuss a fundamental solution by presenting self-referential Titans that allows all the components to perform in-context learning, and adapt and modify themselves.  
\+2

## **8.1 Deep Self-Referential Titans**

A general formulation for the associative memory-based blocks is to project the data into keys, values, and queries and learns how to map keys to values and how to retrieve from the mapping based on queries. More formally, for a parametric associative memory, let $x\_t \\in \\mathbb{R}^d$ for $t=1,...,L$ be the input, we have:  
\+1

$$k\_t \= x\_t W\_k, \\quad v\_t \= x\_t W\_v, \\quad \\min\_{\\mathcal{M}} \\sum\_t \\mathcal{L}(\\mathcal{M}; k\_t, v\_t) \\quad \\text{with an optimization algorithm} \\quad (76)$$

$$y\_t \= \\mathcal{M}\_t q\_t \\quad (77)$$

$$q\_t \= x\_t W\_q, \\quad \\eta\_t \= x\_t W\_{\\eta}, \\quad \\alpha\_t \= x\_t W\_{\\alpha} \\quad (78)$$  
For the sake of clarity, we use red (resp. blue) to highlight computations/weight in the upper level (resp. lower level). Similar to example in Figure 3, we can add a new level for each of $W\_k, W\_v, W\_q, W\_{\\eta}$, and $W\_{\\alpha}$ and allow them to be updated in-context. For the sake of efficiency, a simple version is to share the values for all the components in the nested system of associative memories:  
\+2

$$k\_t \= \\mathcal{M}\_{k, t-1}(x\_t), \\quad v\_t \= \\mathcal{M}\_{v, t-1}(x\_t), \\quad q\_t \= \\mathcal{M}\_{q, t-1}(x\_t), \\quad \\eta\_t \= \\mathcal{M}\_{\\eta, t-1}(x\_t), \\quad \\alpha\_t \= \\mathcal{M}\_{\\alpha, t-1}(x\_t) \\quad (79)$$

$$\\min\_{\\mathcal{M}\_{\\Box}} \\sum\_{t} \\mathcal{L}(\\mathcal{M}\_{\\Box}; k\_t, v\_t) \\quad \\text{with an optimization algorithm, } \\Box \\in \\{k, v, q, \\eta, \\alpha, \\text{memory}\\} \\quad (80)$$

$$\\min\_{\\mathcal{M}\_{mem}} \\sum\_{t} \\mathcal{L}(\\mathcal{M}\_{mem}; k\_t, v\_t) \\quad \\text{with an optimization algorithm} \\quad (81)$$

$$y\_t \= \\mathcal{M}\_{mem, t}(q\_t) \\quad (82)$$  
where the initial states of all memories, i.e., $\\mathcal{M}\_{\\Box, 0}$ for any $\\Box \\in \\{k, v, q, \\eta, \\alpha, \\text{memory}\\}$, are meta-learned across all sequences/contexts. As discussed earlier, the meta-learning of the initial states of memories is essential for both fast-adaption, training stability, robustness to noise in the data. This design provides a fully adaptive memory, where all the components can adapt themselves in-context. It, however, (1) still lacks self-modification, where the model in response to new data changes its own parameters or learning process (Schmidhuber 2003); (2) has suboptimal design as it shares of keys and values for all the memories. In continual learning, where the model requires consistent weight/knowledge update in response to new data, it is critical for the model to not solely rely on data, and instead learns how to modify itself when it is needed. Motivated by the above points, and inspired by the self-modifying mechanisms that generate their own values based on the context (Schmidhuber 1993, 2003; Irie et al. 2022b), we present self-modifying deep associative memory where the models generate their own values:  
\+4

$$y\_t \= \\mathcal{M}\_{memory, t-1}(q\_t), \\quad \\hat{v}\_{\\Box, t} \= \\mathcal{M}\_{\\Box, t-1}(v\_t), \\quad \\min\_{\\mathcal{M}\_{\\Box}} \\mathcal{L}(\\mathcal{M}\_{\\Box}; k\_t, \\hat{v}\_{\\Box, t}) \\quad (83)$$

$$k\_t \= \\mathcal{M}\_{k, t-1}(x\_t), \\quad v\_t \= \\mathcal{M}\_{v, t-1}(x\_t), \\quad \\eta\_t \= \\mathcal{M}\_{\\eta, t-1}(x\_t), \\quad \\text{(Generating its own values for each memory)} \\quad (84)$$

$$\\text{with an optimization algorithm, } \\Box \\in \\{k, v, q, \\eta, \\alpha, \\text{memory}\\}, \\quad \\alpha\_t \= \\mathcal{M}\_{\\alpha, t-1}(x\_t) \\quad (85)$$  
where $q\_t \= x\_t W\_q$ is the only non-adaptive projection, $\\eta\_t$ is the learning rate in optimization process, and $\\alpha\_t$ is the retention gate (forget gate or weight decay) in the optimization process. Note that, again, the initial states of all memories, i.e., $\\mathcal{M}\_{\\Box, 0}$ for any $\\Box \\in \\{k, v, q, \\eta, \\alpha, \\text{memory}\\}$ are meta-learned across all sequences/contexts, and so are optimized in the higher levels (or outer-loop).  
\+1

Learning the mappings for associative memory modules (see Equation 85\) requires a choice of optimization algorithm as well as an objective $\\mathcal{L}$ that measures the quality of mappings. A simple and common choice for objective and optimization process are $L\_2$-regression loss, and gradient descent algorithm. As for the objective, we use $L\_2$-regression loss, i.e., $\\mathcal{L}(\\mathcal{M}; k, v) \= ||\\mathcal{M}(k) \- v||^2\_2$. As discussed earlier (see Section 4.5), the choice of optimizer highly depends on the context of optimization. For example, gradient descent from associative memory perspective is based on dot-product similarity and so the update at each step, is solely based on the input and does not incorporate the previous data samples to the update. When performing optimization in the token space, however, we know tokens are highly correlated. Therefore, following our discussion in Section 4.5, we use our DGD with weight decay, resulting in general update rule of:  
\+4

$$y\_t \= \\mathcal{M}\_{memory, t-1}(q\_t) \\quad (86)$$

$$\\hat{v}\_{\\Box, t} \= \\mathcal{M}\_{\\Box, t-1}(v\_t), \\quad k\_t \= \\mathcal{M}\_{k, t-1}(x\_t), \\quad v\_t \= \\mathcal{M}\_{v, t-1}(x\_t), \\quad \\text{(Generating its own values for each memory)} \\quad (87)$$

$$\\mathcal{M}\_{\\Box, t} \= \\mathcal{M}\_{\\Box, t-1}(\\alpha\_t I \- \\eta\_t k\_t k\_t^\\top) \- \\eta\_t \\nabla \\mathcal{L}\_{\\mathcal{M}\_{\\Box, t-1}}(\\mathcal{M}\_{\\Box, t-1}; k\_t, \\hat{v}\_{\\Box, t}), \\quad \\eta\_t \= \\mathcal{M}\_{\\eta, t-1}(x\_t), \\quad \\alpha\_t \= \\mathcal{M}\_{\\alpha, t-1}(x\_t) \\quad (88)$$

$$\\Box \\in \\{k, v, q, \\eta, \\alpha, \\text{memory}\\} \\quad (89)$$  
Here, the architecture of the memories are arbitrary and even we are not forced to use the same architecture for all components. We use a 2-layer MLP block as the architecture of all the memories:  
\+1

$$\\mathcal{M}\_{\\Box}(\\cdot) \= (\\cdot) \+ W\_{\\Box, 1} \\sigma(W\_{\\Box, 2}(\\cdot)) \\quad (89)$$

## **8.2 Fast and Parallelizable Training**

In the above, we discussed how to design a model that can learn to generate its own latent values and so modify itself. The main challenge from the practical point of view is the efficiency of the method and if its training is parallelizable. We follow the chunk-wise training algorithm of non-linear update rules (Sun et al. 2024; Behrouz et al. 2025c) and use update frequency of $f\_{\\Box} \= \\frac{L}{C\_{\\Box}}$ where $L$ is the context length. While there is no limitation to use different chunk-sizes, in our experiments, we use two different value of chunk sizes, one for the update of $\\mathcal{M}\_{memory}(\\cdot)$ and the other for all the other memories in the self-referential Titans. In more details, given an input sequence $\\{x\_t\\}\_{t=1}^L$ and chunk size $1 \\le C \\le L$, we split the sequence into $\\lceil \\frac{L}{C} \\rceil$ chunks of $\\{x\_{((i-1)C+t)}\\}\_{t=1}^C$ for $i=1,...,\\lceil \\frac{L}{C} \\rceil$ and then generate all elements in Equation 86 at the end of each chunk for the next chunk. This allows for generating all the elements for the entire chunk in parallel, before starting the computation for this chunk. Furthermore, to update the memory modules based on Equation 88, we take the gradient with respect to the last state of the previous chunk. Again, this allows for computing all the gradients for the next chunk in parallel. In more details, given this chunk-wise updating procedure, the update rule for the self-referential Titans is computed as:  
\+4

$$y\_t \= \\mathcal{M}\_{memory, C \\times \\lceil \\frac{t}{C} \\rceil}(q\_t), \\quad k\_t \= \\mathcal{M}\_{k, C \\times \\lceil \\frac{t}{C} \\rceil}(x\_t), \\quad v\_t \= \\mathcal{M}\_{v, C \\times \\lceil \\frac{t}{C} \\rceil}(x\_t), \\quad \\eta\_t \= \\mathcal{M}\_{\\eta, C \\times \\lceil \\frac{t}{C} \\rceil}(x\_t), \\quad \\alpha\_t \= \\mathcal{M}\_{\\alpha, C \\times \\lceil \\frac{t}{C} \\rceil}(x\_t) \\quad (90)$$

$$\\hat{v}\_{\\Box, t} \= \\mathcal{M}\_{\\Box, C \\times \\lceil \\frac{t}{C} \\rceil}(v\_t), \\quad \\mathcal{M}\_{\\Box, t} \= \\mathcal{M}\_{\\Box, t-1}(\\alpha\_t I \- \\eta\_t k\_t k\_t^\\top) \- \\eta\_t \\nabla \\mathcal{L}\_{\\mathcal{M}\_{\\Box, C \\times \\lceil \\frac{t}{C} \\rceil}}(\\mathcal{M}\_{\\Box, C \\times \\lceil \\frac{t}{C} \\rceil}; k\_t, \\hat{v}\_{\\Box, t}) \\quad (91)$$  
where $\\Box \\in \\{k, v, q, \\eta, \\alpha, \\text{memory}\\}$. Here, the architecture of the memories are arbitrary and even we are not forced to use the same architecture for all components. We use a 2-layer MLP block as the architecture of all the memories:  
\+1

$$\\mathcal{M}\_{\\Box}(\\cdot) \= (\\cdot) \+ W\_{\\Box, 1} \\sigma(W\_{\\Box, 2}(\\cdot)) \\quad (91)$$  
Since all the gradients as well as new keys, values, learning-rates, and weight decays can be computed in parallel before starting the processing of the current chunk, the above updates accepts the fast parallelizable dual form that is discussed by Sun et al. (2024) and Behrouz et al. (2025c) . To better illustrate the above update rule for self-referential Titans, let us derive the recurrent formula for the simplest case of matrix-valued memory. We derive the recurrent form for two different objectives:  
\+2

**Dot-product similarity** $\\mathcal{L}(\\mathcal{M}; k, v) \= \-\\langle \\mathcal{M}k, v \\rangle$: Given this objective and linear memory, the gradient is calculated as $-vk^\\top$, which results in update rule of:

$$\\mathcal{M}\_{\\Box, t} \= \\mathcal{M}\_{\\Box, t-1}(\\alpha\_t I \- \\eta\_t k\_t k\_t^\\top) \- \\eta\_t \\hat{v}\_{\\Box, t} k\_t^\\top, \\quad \\Box \\in \\{k, v, q, \\eta, \\alpha, \\text{memory}\\} \\quad (92)$$

**$L\_2$-regression loss**: Given this objective and linear memory, the gradient is calculated as $(\\mathcal{M}k \- v)k^\\top$ which results in update rule of:

$$\\mathcal{M}\_{\\Box, t} \= \\mathcal{M}\_{\\Box, t-1}(\\alpha\_t I \- \\eta\_t k\_t k\_t^\\top) \- \\eta\_t (\\mathcal{M}\_{\\Box, C \\times \\lceil \\frac{t}{C} \\rceil} k\_t \- \\hat{v}\_{\\Box, t})k\_t^\\top, \\quad \\Box \\in \\{k, v, q, \\eta, \\alpha, \\text{memory}\\} \\quad (93)$$

## **8.3 Hope Neural Learning Module**

In the previous sections, we first discussed Continuum Memory System (CMS) that allows for more persistent storage of memories and defines memory as a spectrum of blocks with different frequencies of update. Due to the larger capacity and constraints for scaling the parameters, often CMS requires simple learning rule but higher capacity to store more persistent knowledge. On the other hand, in the previous section, we discussed the design of a self-modifying Titans, where it can generate its own keys and so learning update to better adapt to the context. Contrary to CMS, the self-modifying Titans has a small capacity but is using a complex and expressive learning rule. Accordingly, these two systems seem to be complementary and their combination can enhance the model expressiveness from different aspects.  
\+4

To this end, we present HOPE architecture: A neural learning module that incorporates self-modifying Titans followed by Continuum Memory System. The HOPE design is illustrated in Figure 5\. Formally, let $x\_t \\in \\mathbb{R}^d$ for $t=1,...,L$ be the input, the HOPE forward pass is defined as (we remove the normalization and convolution layers for the sake of clarity):  
\+1

$$o\_t \= \\mathcal{M}\_{memory, t-1}(q\_t), \\quad \\hat{v}\_{\\Box, t} \= \\mathcal{M}\_{\\Box, t-1}(v\_t), \\quad k\_t \= \\mathcal{M}\_{k, t-1}(x\_t), \\quad v\_t \= \\mathcal{M}\_{v, t-1}(x\_t), \\quad \\eta\_t \= \\mathcal{M}\_{\\eta, t-1}(x\_t) \\quad (94)$$

$$\\mathcal{M}\_{\\Box, t} \= \\mathcal{M}\_{\\Box, t-1}(\\alpha\_t I \- \\eta\_t k\_t k\_t^\\top) \- \\eta\_t \\nabla \\mathcal{L}\_{\\mathcal{M}\_{\\Box, t-1}}(\\mathcal{M}\_{\\Box, t-1}; k\_t, \\hat{v}\_{\\Box, t}), \\quad \\Box \\in \\{k, v, q, \\eta, \\alpha, \\text{memory}\\} \\quad (95)$$

$$y\_t \= \\text{MLP}^{(f\_K)}(\\text{MLP}^{(f\_{K-1})}(\\cdot\\cdot\\cdot \\text{MLP}^{(f\_1)}(o\_t))) \\quad (96)$$

$$\\alpha\_t \= \\mathcal{M}\_{\\alpha, t-1}(x\_t) \\quad (97)$$  
where the block's output for token $t$ is $y\_t$. In our experiments, we also normalize $q$ and $k$ with $L\_2$ normalization and also use local convolutions with window size of 4\.

**Hope-Attention.** We also use another variant of HOPE, in which we simply replace the self-modifying Titans with softmax global attention (Vaswani et al. 2017).  
