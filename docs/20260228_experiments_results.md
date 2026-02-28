# Continual Learning – Regression Results

> Auto-generated — 4 experiment(s) from `raw_results/`

> **Regression task.** All metrics use **Jump L1 Error** (lower ↓ is better).

---

## 1 · At a Glance

The three numbers that define a CL method: how good it is overall, how much it forgets, and how well it learns new tasks.

| Experiment | Type | Avg Error ↓ | Rel. Error | Forgetting ↓ | Plasticity ↓ | Gap Closed |
|------------|:----:|----------:|----------:|----------:|----------:|----------:|
| AC ViT Lower Bound Naive | CL | 0.3720 | 11.7% | 0.0856 | 0.3007 | 0.0% |
| AC ViT Upper Bound Joint | Joint | 0.2502 | 7.9% | 0.0000 | 0.2502 | 100.0% |
| First AC HOPE Run | CL | 0.3958 | 12.4% | 0.0554 | 0.3496 | -19.5% |
| Second AC HOPE Run | CL | 0.3987 | 12.5% | 0.0481 | 0.3586 | -21.9% |

> **Putting errors in perspective (σ = 3.18):** The V-JEPA feature representations have a natural standard deviation of σ = 3.18. An average error of 0.3720 corresponds to **11.7% of σ** — the model's predictions deviate by ~12% of the typical data variation.

<details><summary><strong>Metric definitions</strong></summary>

| Metric | What it measures | Source |
|--------|-----------------|--------|
| **Avg Error** | Mean L1 across all tasks after final training phase | `cl_jump/Top1_L1_Stream` |
| **Rel. Error** | Avg Error normalised by feature σ (3.18) — error as % of natural data variation | Computed |
| **Forgetting** | How much old-task error increased (0 = perfect stability) | `cl_jump/StreamForgetting` |
| **Plasticity** | Mean diagonal of Jump R-matrix — immediate error when learning each task (lower = learns faster) | R-matrix diagonal avg |
| **Gap Closed** | % of (Lower Bound − Upper Bound) gap closed by this method. 100% = matches UB | Computed |

</details>

---

## 2 · Stability–Plasticity Profile

The fundamental CL tradeoff. Ideal = bottom-left corner (low forgetting AND low plasticity error).

| Experiment | Forgetting ↓ (stability) | Plasticity Error ↓ | Final Avg Error ↓ |
|------------|----------:|----------:|----------:|
| AC ViT Lower Bound Naive | 0.0856 | 0.3007 | 0.3720 |
| AC ViT Upper Bound Joint | 0.0000 | 0.2502 | 0.2502 |
| First AC HOPE Run | 0.0554 | 0.3496 | 0.3958 |
| Second AC HOPE Run | 0.0481 | 0.3586 | 0.3987 |

**Per-task view:** immediate error (plasticity, R-matrix diagonal) vs. final error (after all training).

*AC ViT Lower Bound Naive:*

| Task | Immediate Error | Final Error | Δ (forgetting) |
|------|----------:|----------:|----------:|
| Base | 0.2783 | 0.4401 | 0.1618 |
| Task 1 | 0.2791 | 0.3990 | 0.1200 |
| Task 2 | 0.2953 | 0.3771 | 0.0818 |
| Task 3 | 0.3419 | 0.3917 | 0.0498 |
| Task 4 | 0.2929 | 0.3076 | 0.0147 |
| Task 5 | 0.3167 | 0.3167 | 0.0000 |

*First AC HOPE Run:*

| Task | Immediate Error | Final Error | Δ (forgetting) |
|------|----------:|----------:|----------:|
| Base | 0.3256 | 0.4359 | 0.1103 |
| Task 1 | 0.3272 | 0.4077 | 0.0806 |
| Task 2 | 0.3447 | 0.3966 | 0.0519 |
| Task 3 | 0.3932 | 0.4172 | 0.0241 |
| Task 4 | 0.3406 | 0.3507 | 0.0101 |
| Task 5 | 0.3666 | 0.3666 | 0.0000 |

*Second AC HOPE Run:*

| Task | Immediate Error | Final Error | Δ (forgetting) |
|------|----------:|----------:|----------:|
| Base | 0.3309 | 0.4325 | 0.1015 |
| Task 1 | 0.3368 | 0.4065 | 0.0697 |
| Task 2 | 0.3538 | 0.3968 | 0.0430 |
| Task 3 | 0.4019 | 0.4210 | 0.0192 |
| Task 4 | 0.3506 | 0.3576 | 0.0070 |
| Task 5 | 0.3777 | 0.3777 | 0.0000 |

---

## 3 · Feature Similarity vs. Forgetting

Catastrophic forgetting is driven by representational overlap: when two tasks occupy similar regions in feature space, training on one can overwrite the other. The inter-task cosine similarity of the V-JEPA feature representations (σ = 3.18) reveals which task transitions cause the most interference.

### AC ViT Lower Bound Naive

**Step-wise interference:** each row shows the error increase on a *victim* task caused by training a subsequent *cause* task, alongside their feature-space cosine similarity.

| Victim Task | Cause Task | Feature Similarity | Step Forgetting (Δ Error) |
|-------------|------------|-------------------:|-------------------------:|
| Base | T1-Scale | 0.9882 | +0.0746 |
| Base | T2-Ice | 0.9835 | +0.0276 |
| Base | T3-Bounce | 0.9814 | +0.0225 |
| Base | T4-Rot | 0.9510 | +0.0197 |
| Base | T5-OOD | 0.9536 | +0.0175 |
| T1-Scale | T2-Ice | 0.9936 | +0.0297 |
| T1-Scale | T3-Bounce | 0.9905 | +0.0450 |
| T1-Scale | T4-Rot | 0.9667 | +0.0251 |
| T1-Scale | T5-OOD | 0.9700 | +0.0202 |
| T2-Ice | T3-Bounce | 0.9964 | +0.0311 |
| T2-Ice | T4-Rot | 0.9758 | +0.0311 |
| T2-Ice | T5-OOD | 0.9791 | +0.0197 |
| T3-Bounce | T4-Rot | 0.9698 | +0.0434 |
| T3-Bounce | T5-OOD | 0.9764 | +0.0064 |
| T4-Rot | T5-OOD | 0.9976 | +0.0147 |

**Pearson *r* = 0.294** — weak positive correlation between feature similarity and step-wise forgetting.

> The largest interference (Δ = +0.0746) occurs between tasks with similarity 0.9882, while the smallest (Δ = +0.0064) corresponds to similarity 0.9764. 

**Per-task summary:**

| Task | Total Forgetting | Avg Sim. to Later Tasks | Feature σ |
|------|----------------:|------------------------:|---------:|
| Base | 0.1618 | 0.9715 | 3.2130 |
| T1-Scale | 0.1200 | 0.9802 | 3.1991 |
| T2-Ice | 0.0818 | 0.9838 | 3.1679 |
| T3-Bounce | 0.0498 | 0.9731 | 3.1541 |
| T4-Rot | 0.0147 | 0.9976 | 3.1304 |
| T5-OOD | — | — | 3.1268 |

### First AC HOPE Run

**Step-wise interference:** each row shows the error increase on a *victim* task caused by training a subsequent *cause* task, alongside their feature-space cosine similarity.

| Victim Task | Cause Task | Feature Similarity | Step Forgetting (Δ Error) |
|-------------|------------|-------------------:|-------------------------:|
| Base | T1-Scale | 0.9882 | +0.0661 |
| Base | T2-Ice | 0.9835 | +0.0190 |
| Base | T3-Bounce | 0.9814 | +0.0095 |
| Base | T4-Rot | 0.9510 | +0.0044 |
| Base | T5-OOD | 0.9536 | +0.0115 |
| T1-Scale | T2-Ice | 0.9936 | +0.0244 |
| T1-Scale | T3-Bounce | 0.9905 | +0.0319 |
| T1-Scale | T4-Rot | 0.9667 | +0.0073 |
| T1-Scale | T5-OOD | 0.9700 | +0.0170 |
| T2-Ice | T3-Bounce | 0.9964 | +0.0196 |
| T2-Ice | T4-Rot | 0.9758 | +0.0165 |
| T2-Ice | T5-OOD | 0.9791 | +0.0159 |
| T3-Bounce | T4-Rot | 0.9698 | +0.0216 |
| T3-Bounce | T5-OOD | 0.9764 | +0.0024 |
| T4-Rot | T5-OOD | 0.9976 | +0.0101 |

**Pearson *r* = 0.412** — moderate positive correlation between feature similarity and step-wise forgetting.

> The largest interference (Δ = +0.0661) occurs between tasks with similarity 0.9882, while the smallest (Δ = +0.0024) corresponds to similarity 0.9764. High feature similarity predicts high forgetting.

**Per-task summary:**

| Task | Total Forgetting | Avg Sim. to Later Tasks | Feature σ |
|------|----------------:|------------------------:|---------:|
| Base | 0.1103 | 0.9715 | 3.2130 |
| T1-Scale | 0.0806 | 0.9802 | 3.1991 |
| T2-Ice | 0.0519 | 0.9838 | 3.1679 |
| T3-Bounce | 0.0241 | 0.9731 | 3.1541 |
| T4-Rot | 0.0101 | 0.9976 | 3.1304 |
| T5-OOD | — | — | 3.1268 |

### Second AC HOPE Run

**Step-wise interference:** each row shows the error increase on a *victim* task caused by training a subsequent *cause* task, alongside their feature-space cosine similarity.

| Victim Task | Cause Task | Feature Similarity | Step Forgetting (Δ Error) |
|-------------|------------|-------------------:|-------------------------:|
| Base | T1-Scale | 0.9882 | +0.0597 |
| Base | T2-Ice | 0.9835 | +0.0202 |
| Base | T3-Bounce | 0.9814 | +0.0099 |
| Base | T4-Rot | 0.9510 | +0.0032 |
| Base | T5-OOD | 0.9536 | +0.0087 |
| T1-Scale | T2-Ice | 0.9936 | +0.0214 |
| T1-Scale | T3-Bounce | 0.9905 | +0.0299 |
| T1-Scale | T4-Rot | 0.9667 | +0.0039 |
| T1-Scale | T5-OOD | 0.9700 | +0.0145 |
| T2-Ice | T3-Bounce | 0.9964 | +0.0171 |
| T2-Ice | T4-Rot | 0.9758 | +0.0122 |
| T2-Ice | T5-OOD | 0.9791 | +0.0138 |
| T3-Bounce | T4-Rot | 0.9698 | +0.0180 |
| T3-Bounce | T5-OOD | 0.9764 | +0.0011 |
| T4-Rot | T5-OOD | 0.9976 | +0.0070 |

**Pearson *r* = 0.431** — moderate positive correlation between feature similarity and step-wise forgetting.

> The largest interference (Δ = +0.0597) occurs between tasks with similarity 0.9882, while the smallest (Δ = +0.0011) corresponds to similarity 0.9764. High feature similarity predicts high forgetting.

**Per-task summary:**

| Task | Total Forgetting | Avg Sim. to Later Tasks | Feature σ |
|------|----------------:|------------------------:|---------:|
| Base | 0.1015 | 0.9715 | 3.2130 |
| T1-Scale | 0.0697 | 0.9802 | 3.1991 |
| T2-Ice | 0.0430 | 0.9838 | 3.1679 |
| T3-Bounce | 0.0192 | 0.9731 | 3.1541 |
| T4-Rot | 0.0070 | 0.9976 | 3.1304 |
| T5-OOD | — | — | 3.1268 |

---

## 4 · Final Per-Task Error

Jump L1 on each task after all training is complete.

| Experiment | Base | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Mean |
|------------|------:|------:|------:|------:|------:|------:|------:|
| AC ViT Lower Bound Naive | 0.4401 | 0.3990 | 0.3771 | 0.3917 | 0.3076 | 0.3167 | 0.3720 |
| AC ViT Upper Bound Joint | 0.2502 | 0.2437 | 0.2491 | 0.2818 | 0.2581 | 0.3364 | 0.2699 |
| First AC HOPE Run | 0.4359 | 0.4077 | 0.3966 | 0.4172 | 0.3507 | 0.3666 | 0.3958 |
| Second AC HOPE Run | 0.4325 | 0.4065 | 0.3968 | 0.4210 | 0.3576 | 0.3777 | 0.3987 |

---

## 5 · Upper Bound Reference

**AC ViT Upper Bound Joint** — joint training on all data (no forgetting by design).

- **Avg Error:** 0.2502
- **Test Loss (jump):** 0.3364
- **Teacher Loss:** 0.3049

| Base | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
| ------: | ------: | ------: | ------: | ------: | ------: |
| 0.2502 | 0.2437 | 0.2491 | 0.2818 | 0.2581 | 0.3364 |

---

## 6 · Experiment Details

### AC ViT Lower Bound Naive

#### Training Progression

| Phase | Avg Error ↓ | Forgetting ↓ | BWT | Test Loss |
|-------|----------:|----------:|----:|----------:|
| base | 0.2783 | 0.0000 | — | 0.4719 |
| task_1 | 0.3160 | 0.0746 | 0.0746 | 0.4561 |
| task_2 | 0.3282 | 0.0659 | 0.0659 | 0.4554 |
| task_3 | 0.3563 | 0.0768 | 0.0768 | 0.3700 |
| task_4 | 0.3674 | 0.0874 | 0.0874 | 0.3440 |
| task_5 | 0.3720 | 0.0856 | 0.0856 | 0.3167 |

#### Jump R-Matrix

Rows = evaluated after phase, Columns = task. **Diagonal = plasticity, below-diagonal = forgetting signal.**

| Phase | Base | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|-------|------:|------:|------:|------:|------:|------:|
| Base | 0.2783 | 0.3948 | 0.4216 | 0.4820 | 0.4132 | 0.4719 |
| Task 1 | 0.3529 | 0.2791 | 0.3494 | 0.4521 | 0.3839 | 0.4561 |
| Task 2 | 0.3805 | 0.3087 | 0.2953 | 0.4202 | 0.3787 | 0.4554 |
| Task 3 | 0.4030 | 0.3538 | 0.3264 | 0.3419 | 0.3368 | 0.3700 |
| Task 4 | 0.4227 | 0.3789 | 0.3575 | 0.3853 | 0.2929 | 0.3440 |
| Task 5 | 0.4401 | 0.3990 | 0.3771 | 0.3917 | 0.3076 | 0.3167 |

---

### First AC HOPE Run

#### Training Progression

| Phase | Avg Error ↓ | Forgetting ↓ | BWT | Test Loss |
|-------|----------:|----------:|----:|----------:|
| base | 0.3256 | 0.0000 | — | 0.4634 |
| task_1 | 0.3594 | 0.0661 | -0.0661 | 0.4497 |
| task_2 | 0.3689 | 0.0547 | -0.0547 | 0.4428 |
| task_3 | 0.3902 | 0.0568 | -0.0568 | 0.4063 |
| task_4 | 0.3903 | 0.0550 | -0.0550 | 0.3861 |
| task_5 | 0.3958 | 0.0554 | -0.0554 | 0.3666 |

#### Jump R-Matrix

Rows = evaluated after phase, Columns = task. **Diagonal = plasticity, below-diagonal = forgetting signal.**

| Phase | Base | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|-------|------:|------:|------:|------:|------:|------:|
| Base | 0.3256 | 0.4085 | 0.4391 | 0.4915 | 0.4242 | 0.4634 |
| Task 1 | 0.3916 | 0.3272 | 0.3799 | 0.4667 | 0.3957 | 0.4497 |
| Task 2 | 0.4106 | 0.3515 | 0.3447 | 0.4287 | 0.3882 | 0.4428 |
| Task 3 | 0.4200 | 0.3834 | 0.3642 | 0.3932 | 0.3699 | 0.4063 |
| Task 4 | 0.4244 | 0.3907 | 0.3807 | 0.4148 | 0.3406 | 0.3861 |
| Task 5 | 0.4359 | 0.4077 | 0.3966 | 0.4172 | 0.3507 | 0.3666 |

---

### Second AC HOPE Run

#### Training Progression

| Phase | Avg Error ↓ | Forgetting ↓ | BWT | Test Loss |
|-------|----------:|----------:|----:|----------:|
| base | 0.3309 | 0.0000 | — | 0.4668 |
| task_1 | 0.3637 | 0.0597 | 0.0597 | 0.4533 |
| task_2 | 0.3742 | 0.0506 | 0.0506 | 0.4466 |
| task_3 | 0.3954 | 0.0527 | 0.0527 | 0.4131 |
| task_4 | 0.3939 | 0.0488 | 0.0488 | 0.3951 |
| task_5 | 0.3987 | 0.0481 | 0.0481 | 0.3777 |

#### Jump R-Matrix

Rows = evaluated after phase, Columns = task. **Diagonal = plasticity, below-diagonal = forgetting signal.**

| Phase | Base | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|-------|------:|------:|------:|------:|------:|------:|
| Base | 0.3309 | 0.4124 | 0.4444 | 0.4952 | 0.4278 | 0.4668 |
| Task 1 | 0.3906 | 0.3368 | 0.3866 | 0.4705 | 0.4013 | 0.4533 |
| Task 2 | 0.4108 | 0.3582 | 0.3538 | 0.4352 | 0.3939 | 0.4466 |
| Task 3 | 0.4206 | 0.3881 | 0.3709 | 0.4019 | 0.3770 | 0.4131 |
| Task 4 | 0.4238 | 0.3920 | 0.3830 | 0.4199 | 0.3506 | 0.3951 |
| Task 5 | 0.4325 | 0.4065 | 0.3968 | 0.4210 | 0.3576 | 0.3777 |

---

## Appendix · Combined (cl/) Metrics Reference

<details><summary>Click to expand combined (non-jump) metrics — included for completeness.</summary>

### AC ViT Lower Bound Naive — Combined Metrics

#### Training Progression (Combined)

| Phase | Stream L1 ↓ | Forgetting | BWT | Test Loss |
|-------|----------:|----------:|----:|----------:|
| base | 0.5528 | 0.0000 | — | 0.8935 |
| task_1 | 0.6312 | 0.1434 | 0.1434 | 0.8498 |
| task_2 | 0.6490 | 0.1201 | 0.1201 | 0.8445 |
| task_3 | 0.6931 | 0.1366 | 0.1366 | 0.7060 |
| task_4 | 0.7176 | 0.1652 | 0.1652 | 0.6541 |
| task_5 | 0.7238 | 0.1623 | 0.1623 | 0.6039 |

#### Combined R-Matrix

| Phase | Base | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|-------|------:|------:|------:|------:|------:|------:|
| Base | 0.5528 | 0.7746 | 0.8180 | 0.9197 | 0.7893 | 0.8935 |
| Task 1 | 0.6962 | 0.5662 | 0.6894 | 0.8589 | 0.7292 | 0.8498 |
| Task 2 | 0.7400 | 0.6192 | 0.5878 | 0.7869 | 0.7182 | 0.8445 |
| Task 3 | 0.7777 | 0.6978 | 0.6411 | 0.6557 | 0.6495 | 0.7060 |
| Task 4 | 0.8212 | 0.7523 | 0.7073 | 0.7427 | 0.5646 | 0.6541 |
| Task 5 | 0.8567 | 0.7861 | 0.7442 | 0.7590 | 0.5928 | 0.6039 |

### First AC HOPE Run — Combined Metrics

#### Training Progression (Combined)

| Phase | Stream L1 ↓ | Forgetting | BWT | Test Loss |
|-------|----------:|----------:|----:|----------:|
| base | 0.7146 | 0.0000 | — | 0.9445 |
| task_1 | 0.7770 | 0.1074 | -0.1074 | 0.9151 |
| task_2 | 0.7911 | 0.0866 | -0.0866 | 0.9031 |
| task_3 | 0.8228 | 0.0890 | -0.0890 | 0.8454 |
| task_4 | 0.8248 | 0.0911 | -0.0911 | 0.8104 |
| task_5 | 0.8334 | 0.0923 | -0.0923 | 0.7795 |

#### Combined R-Matrix

| Phase | Base | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|-------|------:|------:|------:|------:|------:|------:|
| Base | 0.7146 | 0.8575 | 0.9050 | 0.9893 | 0.8748 | 0.9445 |
| Task 1 | 0.8220 | 0.7321 | 0.8124 | 0.9438 | 0.8270 | 0.9151 |
| Task 2 | 0.8534 | 0.7665 | 0.7534 | 0.8812 | 0.8147 | 0.9031 |
| Task 3 | 0.8704 | 0.8148 | 0.7818 | 0.8241 | 0.7860 | 0.8454 |
| Task 4 | 0.8822 | 0.8311 | 0.8127 | 0.8626 | 0.7353 | 0.8104 |
| Task 5 | 0.9038 | 0.8575 | 0.8391 | 0.8693 | 0.7512 | 0.7795 |

### Second AC HOPE Run — Combined Metrics

#### Training Progression (Combined)

| Phase | Stream L1 ↓ | Forgetting | BWT | Test Loss |
|-------|----------:|----------:|----:|----------:|
| base | 0.8269 | 0.0000 | — | 1.0170 |
| task_1 | 0.8805 | 0.0884 | 0.0884 | 1.0017 |
| task_2 | 0.8899 | 0.0682 | 0.0682 | 0.9882 |
| task_3 | 0.9141 | 0.0681 | 0.0681 | 0.9376 |
| task_4 | 0.9201 | 0.0734 | 0.0734 | 0.9176 |
| task_5 | 0.9302 | 0.0752 | 0.0752 | 0.8984 |

#### Combined R-Matrix

| Phase | Base | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|-------|------:|------:|------:|------:|------:|------:|
| Base | 0.8269 | 0.9394 | 0.9866 | 1.0635 | 0.9566 | 1.0170 |
| Task 1 | 0.9153 | 0.8457 | 0.9151 | 1.0312 | 0.9264 | 1.0017 |
| Task 2 | 0.9412 | 0.8678 | 0.8606 | 0.9720 | 0.9122 | 0.9882 |
| Task 3 | 0.9547 | 0.9045 | 0.8783 | 0.9190 | 0.8855 | 0.9376 |
| Task 4 | 0.9679 | 0.9195 | 0.9054 | 0.9532 | 0.8543 | 0.9176 |
| Task 5 | 0.9838 | 0.9424 | 0.9281 | 0.9595 | 0.8687 | 0.8984 |


</details>


---
*Report auto-generated by `generate_reports.py`.*
