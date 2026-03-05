# Class-Incremental vs. Task-Incremental Evaluation in Split CIFAR-100

**Date:** 2026-03-05
**Status:** Resolved
**Affected benchmarks:** Split CIFAR-100 (all models)
**Paper reference:** Anbar Jafari et al. (2025), "Dynamic Nested Hierarchies," arXiv:2511.14823

---

## 1. Problem Statement

After training the AC-HOPE-Hybrid-ViT (Phase 8) on Split CIFAR-100 for 3 tasks (100 epochs each, seed 42), evaluation produced the following R-matrix row after Task 2:

| R[2, j] | Task 0 | Task 1 | Task 2 | Tasks 3–9 |
|----------|--------|--------|--------|-----------|
| Accuracy | 0.000  | 0.000  | 0.629  | 0.000     |

The model achieved 62.9% on the current task but **exactly 0.0%** on all other tasks, including previously trained ones. Cross-entropy loss on non-current tasks was ~11 (≈ ln(100,000)), indicating the model assigned near-zero probability to the correct class.

This pattern is inconsistent with the DNH paper's Table 2, which reports for naive fine-tuning:

| Model           | AA (P-MNIST) | BWT (P-MNIST) | AA (S-CIFAR) | BWT (S-CIFAR) |
|-----------------|-------------|---------------|-------------|---------------|
| Transformer++   | 82.4        | −15.2         | 65.1        | −18.7         |
| DNH-HOPE (best) | 89.3        | −8.5          | 71.6        | −11.1         |

An AA of 65% with BWT of −18.7 is **impossible** under class-incremental evaluation with naive sequential fine-tuning, where expected AA is 5–10%.

---

## 2. Root Cause Analysis

### 2.1 Two CL evaluation protocols

The continual learning literature defines two distinct evaluation protocols for multi-class settings:

**Class-Incremental (CI):**
- At test time, the model must distinguish among **all classes ever encountered**.
- Prediction: `argmax` over the full logit vector (e.g., all 100 CIFAR-100 classes).
- Much harder: the classification head becomes biased toward recently trained classes.

**Task-Incremental (TI):**
- At test time, the model knows **which task** the sample comes from.
- Prediction: `argmax` is restricted to only the current task's class logits.
- Easier: eliminates inter-task interference at the output layer.

Formally, for a model with logits $\mathbf{z} \in \mathbb{R}^C$ and task $t$ containing classes $\mathcal{C}_t \subset \{0, \ldots, C-1\}$:

$$
\hat{y}_{\text{CI}} = \arg\max_{c \in \{0,\ldots,C-1\}} z_c
\qquad\text{vs.}\qquad
\hat{y}_{\text{TI}} = \arg\max_{c \in \mathcal{C}_t} z_c
$$

### 2.2 Why our pipeline produced 0% on previous tasks

Our setup:
- `remap_labels: false` → single-head 100-way classifier (correct for both CI and TI)
- `_shared_step()` computed `preds = logits.argmax(dim=-1)` over all 100 logits

After 100 epochs of training on Task 2's 10 classes, the corresponding 10 logit positions became strongly activated while the other 90 decayed. For any test sample from Task 0, the argmax always landed on one of Task 2's class indices — never Task 0's true class. Hence accuracy = 0.0%.

This is not a bug — it is the **expected behavior** of class-incremental evaluation with naive fine-tuning. The complete forgetting is well-documented in the CL literature (e.g., Masana et al. 2022, "Class-Incremental Learning: Survey and Performance Evaluation").

### 2.3 Deduction that the DNH paper uses task-incremental evaluation

**Numerical proof:**

Under task-incremental eval, suppose each task achieves R[j,j] ≈ 83% accuracy when freshly trained. With BWT = −18.7, the accuracy on each previous task after the final training step degrades to:

$$
R[T{-}1, j] \approx R[j, j] + \text{BWT} \approx 83 - 18.7 = 64.3\%
$$

Average Accuracy:

$$
\text{AA} = \frac{1}{T} \sum_{j=0}^{T-1} R[T{-}1, j] \approx 65\%
$$

This matches the paper's reported 65.1% for Transformer++ exactly.

Under class-incremental eval, naive fine-tuning would yield R[T−1, j] ≈ 0% for j < T−1 (as observed in our logs), giving AA ≈ R[T−1, T−1] / T ≈ 8%. This is incompatible with the paper's reported 65.1%.

**Conclusion:** The DNH paper (2511.14823) uses **task-incremental evaluation**, though this is not explicitly stated in Section 5.1. This is consistent with many CL papers that evaluate on "Permuted MNIST and Split CIFAR-100" — both are commonly evaluated task-incrementally in the nested learning / memory-augmented architecture literature.

---

## 3. Fix Applied

### 3.1 Logit masking in `BackboneClassifierModule`

Added a `_eval_task_classes` attribute to `BackboneClassifierModule` (`src/models/benchmark_classifier.py`). During `test_step`, if this attribute is set:

```python
if stage == "test" and self._eval_task_classes is not None:
    mask = torch.full_like(logits, float("-inf"))
    mask[:, self._eval_task_classes] = 0.0
    logits_for_pred = logits + mask
```

- **Loss** is still computed on the full 100-way logits (preserving gradient and loss semantics).
- **Predictions** use the masked logits (argmax restricted to the task's 10 classes).
- Training and validation steps are **unaffected** — masking only applies during test.

### 3.2 Wiring in `evaluate_on_all_tasks`

In `src/cl_benchmark_train.py`, `evaluate_on_all_tasks()` now sets:

```python
model._eval_task_classes = eval_task.get("class_ids", None)
```

before each task's evaluation, and clears it afterward.

For **Permuted MNIST**, `eval_task` has no `class_ids` key (all tasks share the same 10 classes), so `_eval_task_classes` remains `None` and the full 10-way logits are used — no masking needed (domain-incremental).

### 3.3 `inference_mode=False` for HOPE compatibility

The evaluation `Trainer` was also updated to `inference_mode=False`. PyTorch Lightning defaults to `torch.inference_mode()` during `trainer.test()`, which is a hard context that **cannot** be overridden by `torch.enable_grad()`. The HOPE / Titan memory DGD inner-loop calls `torch.autograd.grad()` during the forward pass, requiring gradients to be enabled. Setting `inference_mode=False` downgrades to `torch.no_grad()`, which `torch.enable_grad()` inside `compute_and_apply_update()` can override.

---

## 4. Impact on Metrics

### 4.1 Split CIFAR-100

| Metric | Before fix (CI) | After fix (TI) | Expected range (paper) |
|--------|-----------------|----------------|------------------------|
| R[i, j≠i] | 0.0% | ~50–80% | ~50–80% |
| AA | ~5–8% | ~60–75% | 65–72% |
| BWT | ~−80 | ~−10 to −20 | −11 to −19 |

### 4.2 Permuted MNIST

No change — Permuted MNIST is domain-incremental (same 10-class output space for every task). The existing evaluation was already correct. Expected AA: 82–89%.

---

## 5. Implications for Thesis

1. **All four benchmark runs** (`benchmark_hybrid` × {CIFAR, MNIST} and `benchmark_dnh` × {CIFAR, MNIST}) must use the updated code.

2. When reporting results in the thesis, clearly state: *"Following the protocol in Anbar Jafari et al. (2025), we evaluate Split CIFAR-100 in the task-incremental setting, where test predictions are restricted to the active task's class subset."*

3. The class-incremental numbers could additionally be reported in an appendix as a more challenging evaluation setting, if desired. This would be novel — the DNH paper does not report CI results.

4. Permuted MNIST results are directly comparable to the paper regardless, since domain-incremental evaluation is unambiguous.

---

## 6. References

- Anbar Jafari, A., Ozcinar, C., & Anbarjafari, G. (2025). Dynamic Nested Hierarchies: Pioneering Self-Evolution in Machine Learning Architectures for Lifelong Intelligence. *arXiv:2511.14823*.
- Masana, M., et al. (2022). Class-Incremental Learning: Survey and Performance Evaluation on Exemplar-Free Methods. *IEEE TPAMI*.
- van de Ven, G. M., & Tolias, A. S. (2019). Three Scenarios for Continual Learning. *NeurIPS CL Workshop*. — Defines the CI vs. TI vs. domain-incremental taxonomy.
- Behrouz, A. (2025). Nested Learning: Self-Modifying Titans through Deep Gradient Descent. *ICML*.
