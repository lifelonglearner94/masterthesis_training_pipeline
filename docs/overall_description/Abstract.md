# Abstract

## Adaptive World Models with HOPE: Test-Time Adaptation in Latent Video Prediction

World models that can anticipate future visual states are essential for autonomous decision-making, yet they typically fail when encountering domain shifts at deployment time. This thesis investigates **Test-Time Adaptation (TTA) for latent video prediction** using V-JEPA 2 world models, and proposes the **HOPE architecture** (Behrouz, 2025) as a principled alternative to conventional adaptation strategies.

We construct a physics simulation dataset in which objects receive force impulses and slide under varying physical conditions, organized into an **A→B→A protocol**: normal friction (A₁), reduced friction and mass (B), and return to normal (A₂). This design enables systematic evaluation of both adaptation capability and catastrophic forgetting. All RGB frames are pre-encoded through a frozen ViT-L/16 encoder into compact latent representations (256 patches × 1024 dimensions per frame).

As a baseline, we train a **24-layer Vision Transformer with action conditioning (ViT-AC)** that predicts future latent states given context frames and action tokens. The model is optimized using a combined teacher-forcing and autoregressive rollout loss in L1 space. For TTA, only LayerNorm parameters are adapted online via a self-supervised look-back objective, while all attention and MLP weights remain frozen.

Our main contribution is the integration of the **HOPE (Hierarchical Online Prediction and Evolution) architecture** — originally designed for text — into the latent video prediction setting. The AC-HOPE-ViT model replaces standard attention and MLP layers with two novel components: (1) **Titan Memory layers** that update their associative memory weights *during the forward pass* via Delta Gradient Descent and surprise-based gating, and (2) a **Continuum Memory System (CMS)** operating at fast, medium, and slow update frequencies to capture temporal patterns at multiple scales.

Preliminary results demonstrate that AC-HOPE-ViT learns stably on V-JEPA 2 features, converging from a validation loss above 0.6 to 0.37 within few epochs. The architecture's inherent capacity for continuous self-modification — without requiring a separate TTA phase — combined with multi-frequency memory consolidation, positions it as a promising approach for mitigating catastrophic forgetting in the A→B→A domain shift scenario. Full comparative experiments against the ViT-AC + TTA baseline are underway.

**Keywords:** World Models, V-JEPA 2, Test-Time Adaptation, HOPE, Titan Memory, Vision Transformer, Latent Video Prediction, Catastrophic Forgetting
