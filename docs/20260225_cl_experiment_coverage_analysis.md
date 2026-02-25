# Analyse: Decken die 4 CL-Experiment-Configs den methodischen Text ab?

**Datum:** 2026-02-25
**Gegenstand:** Abgleich der 4 `configs/experiment/cl_*.yaml` mit dem beschriebenen 4-Phasen-Experimentaldesign

---

## Überblick: Die 4 Experiment-Configs

| Config | `pipeline_mode` | `task_training_mode` | Zweck |
|---|---|---|---|
| `cl_upper_bound_cross_validation.yaml` | `cross_validation` | (finetune) | 10-Fold CV auf allen 10 000 Clips → Datenqualität + Upper Bound |
| `cl_lower_bound.yaml` | `sequential` | `finetune` | Naives sequenzielles Finetuning ohne CL-Schutz → Performance-Floor |
| `cl_ac_vit.yaml` | `sequential` | `tta` | AC-ViT mit Test-Time Adaptation → CL-Ansatz 1 |
| `cl_ac_hope.yaml` | `sequential` | `finetune` | AC-HOPE-ViT mit Finetuning + innerer DGD-Schleife → CL-Ansatz 2 |

---

## Phase-für-Phase-Abgleich

### Phase 1: Datengenerierung und Aufgaben-Definition (Tasks)

**Anforderung:** Problem in sequenzielle Tasks zerlegen (z. B. veränderte Physik, Reibung, Geometrie). Hohe Variation innerhalb der Tasks.

**Status: ✅ Abgedeckt**

Alle 3 sequenziellen Configs (`cl_ac_vit`, `cl_ac_hope`, `cl_lower_bound`) definieren identische Task-Definitionen:

| Task | Clips | Physik-Shift |
|---|---|---|
| Base Training | 0–5000 | Grundphysik |
| `scaling_shift` | 5000–6000 | Skalierung |
| `dissipation_shift` | 6000–7000 | Dissipation/Reibung |
| `discretization_shift` | 7000–8000 | Diskretisierung |
| `kinematics_shift` | 8000–9000 | Kinematik |
| `compositional_ood` | 9000–10 000 | Kompositionelle OOD-Kombination |

Die 5 Tasks bilden physikalische Variationen ab, die den im Text genannten Beispielen (Reibung, Geometrie, Physik) gut entsprechen. Die Aufteilung Base (5000) + 5×1000 Task-Clips ist konsistent über alle Configs.

---

### Phase 2: Validierung der Repräsentativität (Joint-Training Check)

**Anforderung:** Vor dem CL-Start: Modell auf **alle Daten gleichzeitig** trainieren. Via n-facher Kreuzvalidierung (10–100-fach) prüfen, ob zufällig weggelassene Daten prädiziert/rekonstruiert werden können.

**Status: ✅ Abgedeckt (mit Einschränkung)**

`cl_upper_bound_cross_validation.yaml` implementiert genau dies:
- `pipeline_mode: "cross_validation"`
- `n_folds: 10` (10-fache CV)
- `clip_start: 0`, `clip_end: 10000` (alle Clips)
- Zufälliges Shuffling der Clips vor der Fold-Aufteilung, geseeded (`seed: 42`)
- Pro Fold: fresh model, Training auf 9000 Clips, Test auf 1000 Clips
- Mean/Std über Folds → W&B + JSON

Der Code in `cl_train.py` (`_run_cross_validation_pipeline`) setzt dies korrekt um:
- Random shuffle mit `np.random.RandomState(seed)`
- Per-Fold Training + Test
- Aggregation der Metriken (Mean/Std) über alle Folds

**Einschränkung:** Der Text trennt konzeptionell zwischen:
1. **Datenqualitäts-Check** (CV prüft, ob Tasks genug gute Samples haben)
2. **Upper Bound** (Joint Training als Vergleichsmaßstab)

Die Config vereint beides unter einem Namen. Das ist methodisch vertretbar, da die CV-Ergebnisse sowohl die Datenqualität validieren als auch als Upper-Bound-Referenz dienen.

---

### Phase 3: Kontinuierliches Lernen & Evaluierungslogik (Metriken)

**Anforderung:** Sequenzielles Training der Tasks mit Metriken für:
- Catastrophic Forgetting
- Wissenstransfer (Behalten)
- Backward Transfer
- Forward Transfer
- Latent-Space-Analyse (JEPA-Features)

**Status: ✅ Vollständig abgedeckt**

#### 3a. Sequenzielles Training

Alle 3 sequenziellen Configs teilen die Pipeline: Base Training → Task 1 → Eval → Task 2 → Eval → … → Task 5 → Eval.

`cl_train.py` (`_run_sequential_pipeline`) implementiert:
- Base Training (Clips 0–4900, 100 Eval-Clips reserviert)
- Pro Task: Training (900 Clips) → Full Evaluation auf **allen** Partitionen (Base + alle Tasks)
- R-Matrix wird nach jeder Phase aktualisiert

#### 3b. CL-Metriken (R-Matrix-Ansatz)

`src/utils/cl_metrics.py` (`ContinualLearningMetricsTracker`) implementiert alle geforderten Metriken:

| Text-Anforderung | Implementierte Metrik | Formel / Logik |
|---|---|---|
| **Catastrophic Forgetting** | `ExperienceForgetting` + `StreamForgetting` | `current_loss − best_past_loss` (positiv = Vergessen) |
| **Wissenstransfer (Behalten)** | `Top1_L1_Exp` + `Top1_L1_Stream` | Absolute Performance pro Task / gemittelt über alle gesehenen Tasks |
| **Backward Transfer** | `BackwardTransfer` (BWT) | `(1/(T-1)) * Σ (R[T,i] − R[i,i])` nach López-Paz & Ranzato (2017) |
| **Forward Transfer** | `ForwardTransfer` (FWT) | Mittlere Performance auf **ungesehenen** zukünftigen Tasks |

Die Metriken werden sowohl für Teacher-Loss als auch Jump-Loss berechnet (zwei separate R-Matrizen: `R_matrix` und `R_matrix_jump`).

#### 3c. Wenn-Dann-Logik aus dem Text

| Bedingung im Text | Wird erkannt durch |
|---|---|
| Leistung auf alter Aufgabe **schlechter** → Catastrophic Forgetting | `ExperienceForgetting > 0` |
| Leistung auf alter Aufgabe **gleich** → Wissenstransfer | `ExperienceForgetting ≈ 0` |
| Leistung auf alter Aufgabe **verbessert** → Backward Transfer | `BWT < 0` (für Loss: negativ = Verbesserung) |
| Unbekannte Aufgabe **über Zufall** → Forward Transfer | `ForwardTransfer` ist ein numerischer Wert; Vergleich mit Zufall muss manuell interpretiert werden |

#### 3d. Latent-Space / JEPA-Features

Der Text vermutet, dass JEPA-Modelle im Latent Space Features so abbilden, dass Forward Transfer verbessert wird. Dies ist **implizit** durch die Architektur abgedeckt (V-JEPA2 Features werden als Input genutzt), aber es gibt **keine explizite Latent-Space-Analyse-Metrik** in den Configs. Das ist akzeptabel, da der Forward-Transfer-Wert diesen Effekt indirekt misst.

---

### Phase 4: Benchmarking und Darstellung

**Anforderung:**
1. Vergleich mit naivem sequenziellem Lernen (Lower Bound)
2. Vergleich mit Joint-Training (Upper Bound)
3. R-Matrix-Darstellung für Genauigkeit, FWT, BWT

**Status: ⚠️ Teilweise abgedeckt — eine Lücke identifiziert**

#### 4a. Lower Bound (Naive Sequential) ✅

`cl_lower_bound.yaml` implementiert dies exakt:
- `pipeline_mode: "sequential"`, `task_training_mode: "finetune"`
- Keine CL-Mechanismen, kein Replay, keine Regularisierung, kein TTA
- Erzwingt maximales Catastrophic Forgetting → Performance Floor
- Gleiche Architektur (AC-ViT, ~43M Params) für fairen Vergleich

#### 4b. Upper Bound (Joint Training) ⚠️ **LÜCKE**

Der Text fordert explizit **Joint-Training** als Vergleich: "Modell auf alle Daten gleichzeitig trainieren".

Die Codebase hat `_run_joint_pipeline()` in `cl_train.py` implementiert, die:
- Alle Clips gemischt trainiert
- Anschließend per-Partition evaluiert (R-Matrix mit einer Zeile)
- Direkten Vergleich mit sequenziellen Experimenten ermöglicht

**Aber:** Es fehlt ein dediziertes Experiment-Config `cl_upper_bound_joint.yaml` mit `pipeline_mode: "joint"` und einer `joint_training`-Sektion. Die vorhandene `cl_upper_bound_cross_validation.yaml` nutzt `pipeline_mode: "cross_validation"`, **nicht** `"joint"`.

**Unterschied:**
- **CV-Pipeline:** Validiert Datenqualität und gibt aggregierte Mean/Std-Metriken → misst **nicht** per-Task-Partition-Performance
- **Joint-Pipeline:** Trainiert auf allen Daten, evaluiert dann auf jeder Partition separat → erzeugt eine R-Matrix-Zeile, direkt vergleichbar mit den sequenziellen Experimenten

→ **Die CV-Pipeline ersetzt die Joint-Pipeline nicht vollständig als Benchmark**, da sie keine per-Partition-R-Matrix-Zeile erzeugt.

#### 4c. R-Matrix-Darstellung ✅

Vollständig implementiert:
- `ContinualLearningMetricsTracker` baut R-Matrix auf
- Wird nach jeder Phase an W&B geloggt
- Finaler Summary-Run loggt `cl/R_matrix_final`
- JSON-Export mit per-Stage-Metriken (`cl_metrics.json`)

---

## Zusammenfassung

| Phase | Anforderung | Status | Details |
|---|---|---|---|
| **1** | Task-Definition (physikalische Variationen) | ✅ | 5 Tasks mit klar definierten Physik-Shifts |
| **2** | Datenqualitäts-Validierung (n-Fold CV) | ✅ | 10-Fold CV auf allen 10 000 Clips |
| **3a** | Sequenzielles CL-Training | ✅ | 3 Configs: AC-ViT, AC-HOPE, Lower Bound |
| **3b** | Catastrophic Forgetting Metrik | ✅ | ExperienceForgetting + StreamForgetting |
| **3c** | Backward Transfer Metrik | ✅ | BWT nach López-Paz & Ranzato (2017) |
| **3d** | Forward Transfer Metrik | ✅ | FWT = mittlere Loss auf ungesehenen Tasks |
| **3e** | Wissenstransfer (Behalten) | ✅ | Top1_L1_Exp / Top1_L1_Stream |
| **3f** | Latent-Space-Rolle (JEPA) | ✅* | Implizit durch V-JEPA2 Features; kein expliziter Metrik-Layer |
| **4a** | Lower Bound Baseline | ✅ | `cl_lower_bound.yaml` — naives Finetuning |
| **4b** | Upper Bound (Joint Training) | ⚠️ | Code existiert (`_run_joint_pipeline`), **aber kein Config dafür** |
| **4c** | R-Matrix Darstellung | ✅ | R-Matrix + W&B Logging + JSON Export |

---

## Identifizierte Lücke: Fehlende Joint-Training Config

### Problem

Der Code `cl_train.py` enthält `_run_joint_pipeline()` mit `pipeline_mode: "joint"`, die:
1. Alle Clips gemischt trainiert (`run_joint_training`)
2. Per-Partition evaluiert (`evaluate_all_tasks` mit R-Matrix)
3. Eine R-Matrix-Zeile erzeugt, direkt vergleichbar mit sequenziellen Experimenten

Es fehlt jedoch ein Config, das `pipeline_mode: "joint"` nutzt. Die bestehende `cl_upper_bound_cross_validation.yaml` nutzt `pipeline_mode: "cross_validation"`, welches eine **andere** Evaluation durchführt (Fold-basiert statt Partition-basiert).

### Empfohlene Lösung

Eine neue Config `cl_upper_bound_joint.yaml` erstellen mit:
- `pipeline_mode: "joint"`
- `joint_training.max_epochs: 40`
- `joint_training.val_split: 0.1`
- Gleiche Task-Definitionen für die per-Partition-Evaluation
- Gleiche Modellarchitektur (AC-ViT) für fairen Vergleich

Dies würde die 4-Phasen-Methodik vollständig implementieren und den direkten Upper-Bound-Vergleich via R-Matrix ermöglichen.

### Alternativ

Wenn die CV-Pipeline bewusst als alleiniger Upper Bound gewählt wurde (da sie robuster ist als ein einzelner Joint-Training-Durchlauf), sollte dies in der Dokumentation explizit begründet werden.
