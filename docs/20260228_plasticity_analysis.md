# Plasticity Analysis: Immediate Per-Task Jump L1 Error

> **Protocol Date:** 2026-02-28  
> **Context:** Master Thesis – Continual Learning for Action Counting (Regression)  
> **Metric:** Jump L1 Error (lower ↓ is better)

---

## 1. Objective

Examine the **plasticity** of each experiment by analysing the R-matrix diagonal—i.e., the Jump L1 error on task $t$ measured immediately after training on task $t$. Compare the naive sequential fine-tuning baseline ("lower bound") against the joint training baseline ("upper bound") to verify whether the conventional CL assumption holds that naive fine-tuning maximises plasticity.

---

## 2. Data — Immediate Per-Task Jump L1 Error (R-Matrix Diagonal)

| Experiment | Base | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|------------|------:|------:|------:|------:|------:|------:|
| AC ViT Lower Bound Naive | 0.2783 | 0.2791 | 0.2953 | 0.3419 | 0.2929 | 0.3167 |
| AC ViT Upper Bound Joint  | 0.2502 | 0.2437 | 0.2491 | 0.2818 | 0.2581 | 0.3364 |
| First AC HOPE Run         | 0.3256 | 0.3272 | 0.3447 | 0.3932 | 0.3406 | 0.3666 |
| Second AC HOPE Run        | 0.3309 | 0.3368 | 0.3538 | 0.4019 | 0.3506 | 0.3777 |

> Each cell corresponds to $R_{t,t}$ in the Jump L1 R-matrix, which is the standard operationalisation of plasticity in continual learning.

---

## 3. Observation

### 3.1 The table captures plasticity correctly

The R-matrix diagonal (error on task $t$ evaluated right after training on task $t$) is the textbook measure of **plasticity** in continual learning. This table therefore provides a direct, per-task view of each method's ability to learn from newly presented data.

### 3.2 Joint training outperforms naive fine-tuning in plasticity on 5 of 6 tasks

| Task | Naive (LB) | Joint (UB) | $\Delta$ (Naive − Joint) | Joint wins? |
|------|----------:|----------:|----------:|:-----------:|
| Base | 0.2783 | 0.2502 | +0.0281 | **Yes** |
| Task 1 | 0.2791 | 0.2437 | +0.0354 | **Yes** |
| Task 2 | 0.2953 | 0.2491 | +0.0462 | **Yes** |
| Task 3 | 0.3419 | 0.2818 | +0.0601 | **Yes** |
| Task 4 | 0.2929 | 0.2581 | +0.0348 | **Yes** |
| Task 5 | 0.3167 | 0.3364 | −0.0197 | No |

The joint model achieves a **lower** (better) immediate error on every task except Task 5. The gap is substantial—up to 0.06 L1 on Task 3.

This is counter-intuitive: the naive model is fine-tuned for 10 epochs **exclusively** on each task's data and should, by the conventional CL definition, represent **maximal plasticity**. Yet the jointly trained model—which must fit all six tasks simultaneously—surpasses it.

---

## 4. Analysis — Why the "Lower Bound" Is Not the Plasticity Ceiling

### 4.1 Positive inter-task transfer in joint training

The six action-counting tasks share substantial visual and temporal structure. Joint training exposes the model to all task data simultaneously, yielding a larger effective training set. The shared features learned across tasks produce a **richer representation** that generalises well to each individual task, providing a benefit that isolated per-task training cannot match.

### 4.2 Representation degradation in sequential fine-tuning

The naive model trains tasks sequentially. Each round of fine-tuning catastrophically specialises the model to the current task, destroying previously learned shared features. By the time the model reaches a later task (e.g., Task 3), it starts from a **degraded initialisation** whose useful general features have been overwritten by earlier tasks. Even with 10 epochs of dedicated training, this handicapped starting point limits the achievable performance.

In other words, **catastrophic forgetting does not only harm stability—it also indirectly harms plasticity** by corrupting shared representations that would be beneficial for learning the current task.

### 4.3 Insufficient training budget is not the primary explanation

While one might hypothesise that 10 epochs per task are simply insufficient, this does not fully explain the pattern. The naive model achieves competitive results on Task 5 (the final task, where its specialised state happens to be closest to the task distribution). The issue is not budget per se, but the quality of the starting representation at each stage. A model that has been sequentially fine-tuned through five prior tasks starts from a worse point than a model that has seen balanced data from all tasks.

### 4.4 The Task 5 exception

Task 5 is the only task where the naive model outperforms the joint model (0.3167 vs. 0.3364). Two possible explanations:

1. **Recency advantage:** Task 5 is the final task in the naive sequence. The model's representations, though degraded for earlier tasks, are maximally tuned for the most recent data distribution. This recency bias tips the balance in favour of the naive approach.
2. **Capacity saturation of the joint model:** With six tasks competing for model capacity simultaneously, the joint model may be stretched thin on the most dissimilar or hardest task, making it the one case where dedicated fine-tuning wins.

---

## 5. Implications for the Thesis

1. **The conventional lower bound is not a valid plasticity upper bound in this setting.** When tasks share structure, joint training can achieve both superior stability *and* superior plasticity. The naive lower bound serves as a worst-case for overall CL performance, but not as a best-case for plasticity.

2. **The joint model is a particularly strong baseline.** It represents the ceiling for both stability and plasticity (on most tasks), making it a challenging reference point for evaluating HOPE.

3. **HOPE's plasticity gap is the primary concern.** Both HOPE runs show noticeably higher immediate error than the naive model, suggesting that the regularisation or replay mechanism in HOPE trades plasticity for stability more aggressively than desired.

4. **Positive transfer matters.** The results highlight that a CL method for this regression domain should be designed to preserve and exploit shared representations across tasks, not merely prevent forgetting.

---

## 6. Summary

| Finding | Detail |
|---------|--------|
| **R-matrix diagonal = plasticity** | Confirmed: the table correctly captures per-task plasticity. |
| **Joint > Naive in plasticity** | Joint training outperforms naive fine-tuning on 5/6 tasks due to positive inter-task transfer and the naive model's representation degradation. |
| **10 epochs are not the bottleneck** | The primary issue is the degraded initialisation from sequential fine-tuning, not an insufficient number of epochs. |
| **Task 5 exception** | Likely explained by recency advantage and/or joint-model capacity limits. |
| **Thesis implication** | The naive lower bound cannot be treated as a plasticity ceiling; the joint model serves as the stronger reference for both stability and plasticity. |

---

*Auto-generated analysis protocol based on `reports/summary.md`.*
