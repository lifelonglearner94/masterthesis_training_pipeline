# Masterthesis: Adaptive Weltmodelle mit HOPE

## Was ich cooles mache?

Ich arbeite mit **V-JEPA 2 Weltmodellen**.
Diese sind mega gut für eine Maschinen um Videos bzw. die Welt zu verstehen / sehen / antizipieren was passieren wird.
Denn sie transformieren Frames mit Pixeln in einen latenten Raum, der viel Speichereffizienter ist und trotzdem die wichtigsten Infos enthält! Gibt auch coole 3D-Rotary Positional Embeddings um Raum-Zeit Informationen in diese latenten Feature Maps zu integrieren. (V-JEPA 2 Paper)

---

## Aufbau des Experiments: Continual Learning Pipeline

Ich habe einen **Physik Simulations Datensatz** erstellt, in dem Objekte einen Kraftimpuls bekommen und dann ein Stück rutschen!
Anstatt eines simplen A→B→A Szenarios nutze ich nun eine **Continual Learning Pipeline mit 5 progressiven dynamischen Shifts**, um die Modelle auf Forward Transfer und Catastrophic Forgetting zu testen.

**1. Base Training Phase:** 5000 Clips, ca. 40 Epochen.

**2. Sequential Task Curriculum:**
| Task | Shift-Typ | Beschreibung |
|------|-----------|--------------|
| **1** | Scaling Shift | Lineare Anpassung |
| **2** | Dissipation Shift | Eis-Szenario (Disentanglement) |
| **3** | Discretization Shift | Hybride Dynamik / Wände |
| **4** | Kinematics Shift | Rotation & Asymmetrie |
| **5** | Compositional OOD | Das große Finale (Out-of-Distribution) |

Nach jedem Task erfolgt eine **Volle Evaluation** auf einem fixen Validation-Set aller Tasks.

### Methodisches Framework: Lower & Upper Bounds
Um die Continual Learning Fähigkeiten objektiv zu messen, wird die Leistung zwischen zwei Extremen eingeordnet:
- **Lower Bound (Naives Finetuning):** Strikt sequenzielles Training ohne Schutz vor Catastrophic Forgetting (Worst-Case).
- **Upper Bound (Joint Training):** Offline-Training über alle aggregierten Daten gleichzeitig (Best-Case / theoretisches Maximum).

Diese Clips habe ich durch einen **frozen ViT-L/16 Encoder** gejagt. Dadurch erhalte ich meine Feature Maps:
- **Input:** 16 RGB Frames → **Output:** 8 Zeitschritte (tubelet_size=2)
- **Pro Frame:** 256 Patches × 1024 dim

Außerdem habe ich bei der Datengenerierung ein Array mit Stärke des Kraftimpulses (x,y in Newton) zum Zeitschritt n erstellt.

---

## AC-Predictor Architektur (ViT-AC)

**24-Layer Vision Transformer** mit Action/State Conditioning:

| Parameter | Wert |
|-----------|------|
| Embed Dim | 384 (intern) / 1024 (Features) |
| Heads | 16 |
| MLP Ratio | 4.0 |
| RoPE | ✅ (3D Rotary Positional Embedding) |

**Token Interleaving:** Pro Zeitschritt werden Action zwischen Bild-Patches eingefügt → Lokale Konditionierung.

### Loss Funktionen: Stochastic Jump Prediction

Der fehleranfällige autoregressive Rollout-Loss wurde durch eine **Stochastic Jump Prediction** ersetzt. Dies eliminiert Error Compounding (Exposure Bias) und reduziert die Komplexität von $\mathcal{O}(T)$ auf $\mathcal{O}(1)$ Forward-Passes pro Sequenz.

**1. Teacher-Forcing Loss** (parallele Vorhersage aller Schritte zur Stabilisierung):
$$\mathcal{L}_{\text{teacher-forcing}}(\phi) := \frac{1}{T} \sum_{k=1}^{T} || P_{\phi}(\cdot) - E(x_{k+1}) ||_{1}$$

**2. Stochastic Jump Prediction Loss** (direkte Vorhersage von $z_\tau$ aus $z_1$ und $a_1$):
$$\mathcal{L}_{\text{jump}}(\phi) := || P_{\phi}(a_{1:\tau}, s_{1}, z_{1}) - z_{\tau} ||_{1}$$
*(wobei $\tau$ uniform aus den letzten $k$ Frames gesampelt wird)*

**Gesamt (Curriculum Learning Schedule):**
$$L(\phi, e) := (1 - \lambda(e)) \cdot \mathcal{L}_{\text{teacher-forcing}} + \lambda(e) \cdot \mathcal{L}_{\text{jump}}$$
*(Der Teacher-Forcing Anteil nimmt über die Epochen ab, während Jump Prediction dominiert)*

---

## Test-Time Adaptation (TTA) - Baseline

**Protokoll:** Self-Supervised Adaption an Domain Shift ohne Labels.

**Methode:**
- Nur **LayerNorm Parameter** ($\gamma$, $\beta$) werden online adaptiert
- Alle Attention/MLP-Gewichte bleiben **frozen**
- **Look-Back Update:** Vorhersage $\hat{z}_t$ wird mit beobachtetem $z_t$ verglichen

$$\mathcal{L}_{\text{TTA}} = || \hat{z}_{t+1} - \text{sg}(z_{t+1}) ||_1$$

| Risiko | Maßnahme |
|--------|----------|
| Model Collapse | Gradient Clipping (≤1.0) |
| Catastrophic Forgetting | Nur LayerNorm adaptieren |
| Oszillation | Konservative LR (~1e-4) |

---

## Forschungsbeitrag: HOPE Architektur

Mein Ziel ist die **HOPE Architektur** aus dem Nested Learning Paper von Behrouz (2025) in meinem Szenario einzusetzen.

**Neuheit:** HOPE (bisher nur Text) → **Videodaten im latenten Raum** = Adaptive Weltmodelle.

**Erfolg? =** HOPE erzielt bessere Ergebnisse in der Continual Learning Pipeline (höherer Forward Transfer, geringeres Forgetting) als ViT-AC mit Standard-TTA und schließt die Lücke zum Upper Bound.

---

## HOPE Architektur — Implementiert & Funktioniert! ✅

Die HOPE Architektur (Behrouz 2025) ist jetzt vollständig integriert und arbeitet auf **pre-encoded V-JEPA 2 Features**.

**Kernidee:** Das Modell hat ein **selbst-modifizierendes Gedächtnis**, das sich *während des Forward-Passes* anpasst — nicht erst beim Training!

### AC-HOPE-ViT Architektur (`src/models/hope/`)

Drop-in Replacement für den ViT-AC Predictor. Gleiche I/O Schnittstelle, komplett neues Innenleben:

| Stufe | Komponente | Beschreibung |
|-------|------------|--------------|
| **1 — Embedding** | `predictor_embed` | 1024 → 384, Token Interleaving (wie ViT-AC) |
| **2 — Backbone** | × HOPE Blocks | Titan Memory + CMS (ersetzt Attention + MLP) |
| **3 — Output** | `predictor_proj` | 384 → 1024, zurück in V-JEPA 2 Feature Space |

### HOPE Block = Titan Memory + CMS

Jeder der x Blöcke besteht aus zwei Phasen:

**Phase A — Self-Modifying Titan Layer** (ersetzt Standard-Attention):
- MLP-basiertes assoziatives Gedächtnis, das seine eigenen Gewichte **im Forward-Pass** updatet
- **Delta Gradient Descent (DGD):** $M_t = M_{t-1}(\alpha_t I - \eta_t k_t k_t^T) - \eta_t (M_{t-1} k_t - \hat{v}_t) k_t^T$
- **Surprise Gating:** Memory wird nur geschrieben, wenn der Retrieval-Error hoch ist. *Die neue Jump Prediction liefert hierfür ein starkes, makroskopisches Fehlersignal!*
- **Self-Generated Targets:** $\hat{v}_t = M_{t-1}(v_t)$ → Das Modell setzt sich selbst Lernziele

**Phase B — Continuum Memory System (CMS)** (ersetzt Standard-MLP):

| Level | Update-Frequenz | Funktion |
|-------|----------------|----------|
| **Fast** | Jeder Zeitschritt | Schnelle lokale Anpassung |
| **Medium** | Alle 4 Schritte | Mittelfristige Muster |
| **Slow** | Alle 16 Schritte | Langzeit-Gedächtnis |

### Erste Ergebnisse

| Metrik | Start | Nach wenigen Epochen |
|--------|-------|---------------------|
| **Validation Loss** | >0.6 | **0.37** 📉 |

Das Modell lernt stabil und konvergiert schnell auf den V-JEPA 2 Features.

---

## Warum HOPE in der Continual Learning Pipeline überlegen sein sollte

Der entscheidende Unterschied zum Standard ViT-AC + TTA:

| | ViT-AC + TTA | AC-HOPE-ViT |
|--|-------------|-------------|
| **Adaptation** | Nur LayerNorm (γ, β) | Gesamtes Titan Memory |
| **Wann?** | Nur im TTA-Schritt | **Ständig im Forward-Pass** |
| **Mechanismus** | Gradient Descent auf Loss | DGD mit Surprise Gating |
| **Forgetting-Risiko** | Hoch (trotz Einschränkung) | Gering (Multi-Frequenz CMS) |

**Kernargument:** HOPE adaptiert sich *von Natur aus* an neue Daten — ohne separaten TTA-Schritt. Das CMS mit seinen drei Frequenz-Ebenen kann kurzfristige Änderungen (neue Tasks) lernen, ohne langfristiges Wissen (alte Tasks) zu vergessen.

→ **Genau das, was wir für die progressiven dynamischen Shifts brauchen!**

---

## Nächste Schritte

- Vollständige Evaluierung der 5-Task Continual Learning Pipeline mit HOPE vs. ViT-AC + TTA Baseline.
- Training und Evaluierung der **Lower Bound** (Naives Finetuning) und **Upper Bound** (Joint Training) Modelle.
- Quantitativer Vergleich anhand der Metriken:
  - **Relative Forgetting Rate** (Experience & Stream Forgetting)
  - **Performance Gap** zum Upper Bound
  - **Forward Transfer (FWT)** und **Backward Transfer (BWT)**
- Analyse der Titan Memory Diagnostics (Surprise, Gradient Norms, Memory Drift) unter den neuen Jump Prediction Bedingungen.

*Das Modell lernt ständig → starke Hypothese, dass es in der Continual Learning Pipeline deutlich besser performt und sich dem Upper Bound annähert!*
