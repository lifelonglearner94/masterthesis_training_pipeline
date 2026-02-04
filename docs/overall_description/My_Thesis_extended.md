# Masterthesis: Adaptive Weltmodelle mit HOPE

## Was ich cooles mache?

Ich arbeite mit **V-JEPA 2 Weltmodellen**.
Diese sind mega gut für eine Maschinen um Videos bzw. die Welt zu verstehen / sehen / antizipieren was passieren wird.
Denn sie transformieren Frames mit Pixeln in einen latenten Raum, der viel Speichereffizienter ist und trotzdem die wichtigsten Infos enthält! Gibt auch coole 3D-Rotary Positional Embeddings um Raum-Zeit Informationen in diese latenten Feature Maps zu integrieren. (V-JEPA 2 Paper)

---

## Aufbau des Experiments

Ich habe einen **Physik Simulations Datensatz** erstellt, in dem Objekte einen Kraftimpuls bekommen und dann ein Stück rutschen!

| Phase | Bedingungen | Zweck |
|-------|-------------|-------|
| **A₁** | Normale Reibung | Training |
| **B** | Reduzierte Reibung & Masse | Test-Time Adaptation (TTA) |
| **A₂** | Wie A₁ | Catastrophic Forgetting Test |

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

### Loss Funktionen

**1. Teacher-Forcing Loss** (parallele Vorhersage aller Schritte):
$$\mathcal{L}_{\text{teacher-forcing}}(\phi) := \frac{1}{T} \sum_{k=1}^{T} || P_{\phi}(\cdot) - E(x_{k+1}) ||_{1}$$

**2. Rollout Loss** (autoregressive Schleife):
$$\mathcal{L}_{\text{rollout}}(\phi) := || P_{\phi}(a_{1:T}, s_{1}, z_{1}) - z_{T+1} ||_{1}$$

Gesamt: $L(\phi) := \mathcal{L}_{\text{teacher-forcing}} + \mathcal{L}_{\text{rollout}}$

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

**Erfolg? =** HOPE erzielt bessere Ergebnisse im A→B→A Szenario als ViT-AC mit Standard-TTA.

---

## Nächste Schritte

Integration der HOPE Architektur in dieses Framework.

*Wenn es klappt: Großer Beitrag zum Schritt zur autonomen Maschinen Intelligenz!*
