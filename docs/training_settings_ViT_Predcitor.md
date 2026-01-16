# Trainings-Details des V-JEPA 2-AC Predictors

Der V-JEPA 2-AC ist ein Latent World Model. Das bedeutet, er lernt die Dynamik der Welt nicht auf Pixelebene, sondern in einem komprimierten Repräsentationsraum (Latent Space).

## 1) Architektur & Input

**Basis:** Ein vortrainierter V-JEPA 2 Video-Encoder (ViT-g) wird verwendet und dessen Gewichte werden eingefroren ($frozen encoder$). Dieser Encoder wandelt Video-Frames in Feature-Maps $z_k$ um.

**Neues Modul:** Ein neues Transformer-Netzwerk (Predictor, ca. 300M Parameter) wird „on top“ trainiert.

**Eingabedaten:** Der Predictor erhält eine Sequenz aus drei Komponenten:

- Encodierte Video-Features $z_k$
- Robot-Endeffektor-Zustand $s_k$ (Position, Orientierung, Greifer)
- Aktion $a_k$ (Änderung des Endeffektor-Zustands zwischen Frames)

**Mechanismus:** Es wird eine block-causal attention verwendet. Das Modell darf beim Vorhersagen eines Zeitschritts auf vergangene Schritte und den aktuellen Schritt (Aktion/State), aber nicht auf zukünftige Video-Informationen zugreifen.

## 2) Ziele (Objectives / Loss Function)

Ziel ist die Vorhersage, wie sich die Video-Repräsentation durch eine Aktion verändert. Das Training minimiert eine kombinierte Verlustfunktion aus zwei Teilen:

- **Teacher-Forcing Loss (L1):** Das Modell sagt die Repräsentation des nächsten Frames ($\hat{z}_{k+1}$) vorher, basierend auf echten vergangenen Frames, Zuständen und Aktionen. Der Fehler ist die L1-Distanz zur echten Repräsentation $z_{k+1}$.
- **Rollout Loss (L1):** Für robustere Langzeitvorhersagen wird ein Rollout trainiert. Dabei wird der Output des Modells als Input für den nächsten Schritt verwendet. Im Training typischerweise mit $T=2$ Schritten.

Ziel ist es, Fehlerakkumulation zu vermeiden, wenn das Modell später im „Blindflug“ (ohne neue echte Bilder) planen muss.

**Gesamt-Loss:**

$$
L(\phi) := \mathcal{L}_{teacher\text{-}forcing}(\phi) + \mathcal{L}_{rollout}(\phi)
$$

## 3) Metriken

Während des Trainings wird primär der L1-Loss im latenten Raum minimiert. Eine klassische „Accuracy“ gibt es für das World-Model-Training nicht, da es sich um eine Regressionsaufgabe im Vektorraum handelt.

Die eigentliche Erfolgsmessung (Evaluation) erfolgt Zero-Shot in der realen Welt anhand von Roboter-Tasks:

- **Success Rate:** Erfolgsrate bei Aufgaben wie Grasp, Reach und Pick-and-Place (in Prozent)
- **Planungs-Metrik:** Bei der Planung minimiert das Modell die Distanz zur Repräsentation eines Zielbildes (Goal Image)

## 4) Optimizer & Hyperparameter

Details laut Appendix B.1:

- **Optimizer:** AdamW
- **Learning Rate Schedule:** Warmup–Constant–Decay
- **Warmup:** Linear von $7.5 \times 10^{-5}$ auf $4.25 \times 10^{-4}$ über 4.500 Iterationen
- **Constant:** Gehalten für 85.500 Iterationen
- **Decay:** Abfall auf 0 über 4.500 Iterationen
- **Weight Decay:** 0.04 (konstant)
- **Batch Size:** 256 Video-Clips (jeweils 4 Sekunden lang)
- **Gradient Clipping:** 1.01

## 5) Train / Val / Test Splits

Hier unterscheidet sich der Ansatz von klassischen Supervised-Learning-Papers:

- **Trainingsdaten:** ~62 Stunden ungelabelte Robotervideos aus dem Droid Dataset. Keine expliziten Belohnungen (Rewards) oder Task-Labels.
- **Validation/Test:** Kein klassischer Test-Split. Stattdessen Zero-Shot-Anwendung in der realen Welt.
- **Umgebungen:** Zwei verschiedene Labore, die nicht im Trainingsdatensatz (Droid) enthalten waren.

Damit muss das Modell auf völlig neue Umgebungen und Objekte generalisieren.
