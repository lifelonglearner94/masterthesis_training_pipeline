# Antworten auf Fragen Ende Februar 2026

> **Datum:** 25. Februar 2026
> **Kontext:** Masterarbeit – Continual Learning mit V-JEPA 2 Action-Conditioned Predictors auf 40 GB Videoclip-Daten

---

## Q1: Ist der Ablauf in `cl_ac_hope.yaml` und `cl_ac_vit.yaml` exakt so wie vorgestellt?

> *Referenz: `docs/20260222_Full_CL_Pipeline.md`*

### Antwort

**Ja, die Pipeline-Grundstruktur stimmt:** Base Training → 5 sequenzielle Tasks → Volle Evaluation nach jedem Schritt. Task-Reihenfolge, -Namen und Clip-Bereiche sind konsistent zwischen Dokumentation und Code.

### Übersicht

| Aspekt | Status |
|---|---|
| Pipeline-Struktur (Base → 5 Tasks → Eval) | ✅ Exakt wie im Dokument |
| Task-Reihenfolge und -Namen | ✅ Identisch |
| AC-ViT: TTA-Modus | ✅ Korrekt implementiert |
| AC-HOPE: Finetune-Modus | ✅ Korrekt implementiert |
| HOPE Inner-Loop Freeze bei Eval | ✅ Korrekt |
| TTA deaktiviert bei Eval | ✅ Korrekt |
| W&B-Gruppierung | ✅ Wie gewünscht |
| CL-Metriken (FWT, Forgetting, Top1) | ✅ Vorhanden |
| **Datenleck Base-Training-Eval** | **❌ KRITISCH** |
| Zweite Metrik (Jump Prediction parallel in R-Matrix) | ⚠️ Fehlt |
| BWT vs. Forgetting | ⚠️ Abweichende Definition |
| Curriculum bei Task-Finetuning | ⚠️ Unsauber |
| Val-Split bei Task-Finetuning | ⚠️ Fehlt |
| LR-Reduktion beim Finetuning | ⚠️ Empfohlen |

### Kritische Diskrepanzen im Detail

#### ❌ Datenleck beim Base Training (KRITISCH)

Die **gravierendste** Diskrepanz: Das Base Training verwendet **alle** 5000 Clips (0–5000), aber die CL-Evaluation testet auf den **letzten 100 Clips** (4900–4999) derselben Partition. Diese 100 Eval-Clips sind also Teil der Trainingsmenge!

Im Kontrast dazu reservieren die **Task-Phasen korrekt** die letzten 100 Clips: `train_clip_end = task_cfg.clip_end - eval_clips`. Beim Base Training fehlt diese Reservierung — `cl_train.py` Zeile 371–374 übergibt den gesamten Clip-Bereich 0–5000 ohne Abzug.

**Fix:** `clip_end` beim Base Training auf `base_cfg.clip_end - cfg.cl.eval.clips_per_task` setzen.

#### ⚠️ Nur eine Metrik pro Partition in der R-Matrix

Das Dokument fordert sowohl **L1** als auch **Jump Prediction** als Kernmetriken. Der Code speichert jedoch nur **einen** Wert pro Partition (Fallback `test/loss` → `test/loss_jump`). Die R-Matrix enthält also nur eine Dimension.

**Empfehlung:** Zwei separate R-Matrizen führen (eine für L1/Teacher-Loss, eine für Jump-Loss).

#### ⚠️ Curriculum Schedule läuft während Task-Finetuning weiter

Beide Modelle implementieren `on_train_epoch_start()` mit Curriculum-Logik. Der Trainer wird pro Phase neu instanziiert, sodass `self.current_epoch` bei 0 startet. In der Praxis kein Problem (HOPE-Finetuning bei 10 Epochen bleibt bei `loss_weight_teacher = 1.0`), aber methodisch unsauber.

**Empfehlung:** Curriculum-Schedule beim Task-Training explizit deaktivieren.

#### ⚠️ Kein Validation-Monitoring beim Task-Finetuning (HOPE)

`val_split: 0.0` bedeutet: kein Validation-Set, kein Early Stopping, keine Overfitting-Erkennung auf den 900 Task-Clips.

#### ⚠️ LR beim Finetuning zu hoch

Die HOPE-Config verwendet die Base-Training-LR (`1.5e-4`) auch für das Task-Finetuning. Eine zu hohe LR begünstigt katastrophisches Vergessen. Empfehlung: 1/3 bis 1/10 der Base-LR.

#### ⚠️ Doc vs. Code: „randomly selected" vs. „last N"

Das Dokument beschreibt Eval-Clips als „randomly selected". Der Code nimmt deterministisch die **letzten N** Clips. Konsistent im Geist (fest und reproduzierbar), aber Formulierung im Dokument sollte korrigiert werden.

### W&B-Frage aus dem Dokument

> *„Ist es möglich, automatisch neue Runs innerhalb desselben W&B-Runs zu starten?"*

**Bereits implementiert:** Jede Phase erstellt einen neuen `WandbLogger`, alle gruppiert unter `cl.wandb_group`.

---

## Q2: Läuft die Evaluierung korrekt? Werden alle Metriken getracked? Was ist FWT / BWT?

> *Referenz: `docs/20260222_Full_CL_Pipeline.md`*

### FWT und BWT — Konzeptuelle Erklärung

#### Forward Transfer (FWT)

FWT misst die **Zero-Shot-Generalisierung** auf zukünftige, noch nicht trainierte Tasks:

$$\text{FWT}(i) = \frac{1}{N - i} \sum_{j=i+1}^{N} R[i, j]$$

wobei $R[i, j]$ die Performance auf Task $j$ nach Training auf Experience $i$ ist. Ein **niedriger FWT-Wert** (bei Loss-Metriken) bedeutet: Das Modell kann bereits gut auf ungesehene Tasks vorhersagen — ein Indikator für positiven Wissenstransfer.

#### Backward Transfer (BWT)

BWT misst, **wie stark das Lernen neuer Tasks die Performance auf bereits gelernte Tasks verschlechtert** — also katastrophisches Vergessen. Die klassische Definition nach López-Paz & Ranzato (2017):

$$\text{BWT} = \frac{1}{T-1} \sum_{i=1}^{T-1} \left( R[T, i] - R[i, i] \right)$$

wobei $R[T, i]$ die Performance auf Task $i$ ganz am Ende ist und $R[i, i]$ die Performance direkt nach dem Training auf Task $i$. Ein **positives BWT** (bei Loss) zeigt katastrophisches Vergessen an.

Die Dokumentation unterscheidet zwei verwandte Varianten:
- **ExperienceForgetting:** Pro einzelnem vergangenen Task: $\text{Fgt}(i, j) = R[i, j] - \min_{k < i} R[k, j]$ (positiv = Verschlechterung)
- **StreamForgetting:** Durchschnittliches Vergetting über alle gesehenen Tasks

### Was der Code tatsächlich tut

Die Funktion `evaluate_all_tasks()` in `src/cl_train.py` implementiert:

1. **Baut Eval-Partitionen:** Base (letzte 100 Clips) + jeder der 5 Tasks (jeweils letzte 100 Clips). ✅
2. **Eigener W&B-Run** pro Evaluationsphase (mit `job_type="evaluation"`). ✅
3. **Frozen Evaluation:** HOPE Inner-Loops eingefroren, TTA deaktiviert. ✅
4. **Loss-Extraktion:** `test/loss` oder `test/loss_jump` aus `trainer.callback_metrics`. ✅
5. **CL-Metriken berechnen & loggen:** via `ContinualLearningMetricsTracker`. ✅

### Metrik-Status

| Metrik | Implementiert | Geloggt an W&B | Korrekt? |
|---|---|---|---|
| Top1_L1_Stream | ✅ | `cl/Top1_L1_Stream` | ✅ |
| Top1_L1_Exp_{j} | ✅ | `cl/Top1_L1_Exp_{j}` | ✅ |
| StreamForgetting | ✅ | `cl/StreamForgetting` | ✅ |
| ExperienceForgetting_{j} | ✅ | `cl/ExperienceForgetting_{j}` | ✅ |
| ForwardTransfer | ✅ | `cl/ForwardTransfer` | ✅ |
| **BackwardTransfer (BWT)** | **❌ Fehlt** | — | — |
| R-Matrix | ✅ | `cl/R_matrix` (als Liste) | ⚠️ als `wandb.Table` besser |

### W&B Logging-Pfade

1. **Training:** `train/loss`, `train/loss_teacher`, `train/loss_jump`, `curriculum/*` ✅
2. **Test/Eval:** `test/loss`, `test/loss_jump`, `test/loss_teacher`, `test/loss_jump_tau_{τ}` ✅
3. **CL-Metriken:** `cl/Top1_L1_Stream`, `cl/StreamForgetting`, `cl/ForwardTransfer`, etc. ✅
4. **TTA-Metriken:** `test/tta_loss_jump`, `test/tta_improvement`, `test/tta_loss_rolling_50`, etc. ✅
5. **Prediction-Callbacks:** `val/prediction_error_histogram`, `val/mean_batch_error` ✅
6. **Final Summary:** Eigener `cl_summary` W&B-Run ✅
7. **JSON-Persistenz:** `cl_metrics.json` mit R-Matrix und allen Metriken pro Stage ✅

### Handlungsbedarf

**BWT als explizite Metrik ergänzen** — wenige Zeilen Code in `src/utils/cl_metrics.py`:
- ExperienceForgetting vergleicht mit dem *besten* vergangenen Wert
- BWT vergleicht mit dem Wert *direkt nach* dem Task-Training ($R[i,i]$)
- Beides sind gängige Metriken, aber BWT ist der in der CL-Literatur am weitesten verbreitete Indikator

---

## Q3: Wieviele Epochen für den Upper Bound / Cross Validation Test?

> *Epochenanzahl? Sollte ich einen Early-Stopping-Pilot-Run machen? Oder einfach alle auf 40 setzen?*

### Aktuelle Konfigurationslage

| Experiment | `base_training.max_epochs` | `cross_validation.max_epochs` | `task.max_epochs` |
|---|---|---|---|
| cl_upper_bound_cross_validation | 40 | **40** | — |
| cl_ac_vit | **40** | — | 10 (TTA) |
| cl_ac_hope | **40** | — | 10 |
| cl_lower_bound | **40** | — | — |

### Empfehlung: 40 Epochen beibehalten — kein Pilot-Run nötig

#### Begründung 1: Der LR-Schedule *ist* das Early Stopping

Der iterationsbasierte Warmup–Constant–Decay-Scheduler (8.5% Warmup, 83% Constant, 8.5% Decay) ist so konstruiert, dass die Learning Rate bei `max_epochs` **exakt Null** erreicht. Das ist kein arbiträres Abschneiden, sondern geplante Konvergenz.

#### Begründung 2: Die Beobachtung bestätigt dies

Die Loss-Kurve wird bei 30 Epochen „sehr deutlich flacher" — genau das erwartete Verhalten. Ab Epoche ~34 (bei 40 Epochen Gesamtlänge) setzt der LR-Decay ein. Dass bei 30 Epochen noch nicht das Minimum erreicht war, liegt daran, dass die Decay-Phase noch aussteht.

#### Begründung 3: V-JEPA 2 Referenz

Das Original-Paper verwendet einen vollständigen Warmup–Constant–Decay-Cycle ohne Early Stopping: *„Early stopping is disabled — training runs for full max_epochs."*

#### Begründung 4: Vergleichbarkeit automatisch gegeben

Base Training (alle Modelle) = 40, CV-Folds = 40, Task-Training = 10 → konsistent.

### Warum KEIN Pilot-Run mit Early Stopping?

1. **Es gibt kein Validation-Set im Upper-Bound-Design:** Die Cross-Validation teilt in 8000 Train / 2000 Val pro Fold. Early Stopping auf Basis des Val-Loss würde pro Fold bei unterschiedlichen Epochen stoppen → Fold-übergreifende Vergleichbarkeit unterminiert.
2. **Der LR-Schedule ist das funktionale Äquivalent zu Early Stopping.**
3. **Kosten-Nutzen:** Ein Pilot-Run bei 40 GB Daten, der nur bestätigt, was der LR-Schedule bereits impliziert, ist verlorene GPU-Zeit.

### Vorsichtsmaßnahme

Val-Loss des ersten CV-Folds in W&B beobachten. Falls Overfitting auftritt, können dank 2-Epochen-Checkpoints retroaktiv optimale Checkpoints gewählt werden — **kostenneutrale Absicherung**.

---

## Q4: L1 vs. L2 Loss — Soll ich einfach zu L2 switchen?

> *Hatte bisher immer L1, fällt leichter damit zu arbeiten und intuitiv zu vergleichen. Habe aber gelernt dass L2 Sinn machen würde...*

### Klare Antwort: Nicht „einfach" switchen.

### Theoretische Unterschiede für diesen Use Case

| Eigenschaft | L1 (MAE) | L2 (MSE) |
|---|---|---|
| Gradienten nahe Optimum | Konstant ($|\nabla| = 1$) → „Zittern" | Abflachend → glattere Konvergenz |
| Große Fehler | Linear bestraft | Quadratisch bestraft |
| Ausreißer-Robustheit | Hoch | Niedrig |
| Relevanz für deterministische Simulation (Kubric) | Ausreißer-Robustheit irrelevant | Theoretisch eleganter |
| V-JEPA 2 Originalpaper | **Verwendet L1** | — |

### Wichtige Code-Details

- **Feature-Dimensionalität:** $16 \times 16 \times 1408 = 360\,448$ Werte pro Frame. Bei L2 wird jeder elementweise Fehler quadriert → **Magnitude der Loss-Werte ist bei L2 nicht vergleichbar mit L1.**
- **`normalize_reps=True`** mildert den Skaleneffekt, hebt ihn aber nicht auf.
- **TTA-Wrapper nutzt hardkodiert L1:** `F.l1_loss(prediction, ground_truth.detach())` — bei L2-Wechsel entsteht Inkonsistenz!
- **Huber Loss bereits unterstützt:** `loss_type: "huber"` mit konfigurierbarem `delta` — eleganter Kompromiss (L2 nahe Null, L1 für große Fehler).

### Entscheidungsmatrix

| Szenario | Empfehlung |
|---|---|
| Bereits finale CL-Ergebnisse mit L1 vorhanden | **Bei L1 bleiben** |
| Ohnehin alles neu trainiert wird | L2-Wechsel vertretbar, aber **alle 5 Modelle + TTA-Wrapper** konsistent umstellen |
| Egal welches Szenario | **Niemals L1 und L2 innerhalb einer Experimentreihe mischen** |

### Beeinflusst die Wahl die Forschungsschlussfolgerungen?

**Wahrscheinlich nicht wesentlich:**
- **Relative Rankings** zwischen CL-Methoden bleiben mit hoher Wahrscheinlichkeit stabil
- **Absolute Loss-Werte** sind nicht vergleichbar
- **Konvergenzverhalten** kann sich unterscheiden

### Empfehlung

**Für die CL-Forschungsfrage ist die Wahl L1 vs. L2 fast sicher nicht ergebnisrelevant.** Beide sind valide. L1 ist das, was V-JEPA 2 nutzt — Konformität mit dem Originalpaper ist in einer Thesis ein starkes Argument.

**Optional als Ablation:** Ein Modell (z.B. AC Predictor Baseline auf Task 1) sowohl mit L1 als auch L2 trainieren und zeigen, dass das relative Ergebnis stabil bleibt — halbe Seite in der Thesis, stärkt die Argumentation.

**Investiere die verbleibende Zeit lieber in die eigentlichen CL-Experimente.**

---

## Q5: Gleiche Hardware für ViT und HOPE?

> *Macht es Sinn beide auf der gleichen Hardware (RTX 4090) laufen zu lassen?*

### VRAM-Abschätzung für RTX 4090 (24 GB)

| Modell | Params | Precision | Checkpointing | Geschätzter Peak-VRAM | RTX 4090 tauglich? |
|---|---|---|---|---|---|
| AC-ViT (24L) | ~43M | 16-mixed | Ja | **8–12 GB** | ✅ Sehr gut |
| HOPE param-matched (6L) | ~42M | FP32 | Nein (DGD-inkompatibel) | **10–16 GB** | ✅ Machbar |
| HOPE depth-matched (24L) | ~83M | FP32 | Nein | **30–50 GB** | ❌ OOM |

### Warum divergieren die Anforderungen?

1. **Precision:** ViT läuft in `16-mixed`, HOPE benötigt `precision: 32` (DGD-Stabilität). Das **verdoppelt** den Speicherbedarf für Gewichte, Aktivierungen *und* Gradienten.
2. **Activation Checkpointing:** Beim ViT aktiviert, bei HOPE deaktiviert (DGD Inner-Loop braucht den Computation Graph).
3. **DGD Computation Graph:** Jeder HOPE-Block hält 5 Titan-Memories mit je 2 Gewichtsmatrizen im Graph.

### Ist identische Hardware methodisch problematisch?

**Nein — im Gegenteil:**
- Gleiche Hardware **eliminiert Hardware-Konfundierung** (unterschiedliche GPUs haben unterschiedliche numerische Eigenschaften)
- **Effektive Batch Size muss identisch sein:** Falls ViT BS=64 schafft und HOPE nur BS=32, verwende `accumulate_grad_batches=2` → effektive BS=64
- **FP32/FP16-Unterschied** ist ein architektonischer Unterschied, kein Hardware-Artefakt → explizit in der Thesis dokumentieren

### Stellschrauben bei VRAM-Engpass (HOPE)

1. Batch Size reduzieren (z.B. 32) + `accumulate_grad_batches` erhöhen
2. `titan_detach_interval: 1` beibehalten
3. `chunk_size: 0` beibehalten
4. Notfalls `precision: "bf16-mixed"` testen (DGD-Stabilität empirisch validieren)

### Empfehlung

**Param-matched Vergleich** (ViT 24L vs. HOPE 6L, beide ~42–43M Parameter) auf der RTX 4090. Das ist das wissenschaftlich stärkere Experiment — bei gleicher Kapazität wird die architektonische Überlegenheit getestet. Depth-matched (24L HOPE) auf Cloud-GPU auslagern falls nötig, aber für die Kernaussage nicht zwingend erforderlich.

---

## Q6: Was muss ich bei der `curriculum_schedule` beachten?

> *Ist es sinnvoll einem Loss weniger Gewicht zu geben? Was ist mit catastrophic forgetting der short-term dynamics wenn $\lambda(e)$ sich zum jump loss verschiebt?*

### Aktuelle Konfiguration

Der kombinierte Loss wird berechnet als:

$$L = \lambda_{\text{teacher}} \cdot L_{\text{teacher}} + \lambda_{\text{jump}} \cdot L_{\text{jump}}$$

**AC-ViT** (`cl_ac_vit.yaml`):

| Epoche | $\lambda_{\text{teacher}}$ | $\lambda_{\text{jump}}$ |
|---|---|---|
| 0 | 1.0 | 1.0 |
| 7 | 0.7 | 1.0 |
| 11 | 0.3 | 1.0 |

**AC-HOPE** (`cl_ac_hope.yaml`):

| Epoche | $\lambda_{\text{teacher}}$ | $\lambda_{\text{jump}}$ |
|---|---|---|
| 0 | 1.0 | 1.0 |
| 15 | 0.7 | 1.0 |
| 25 | 0.5 | 1.0 |

Wichtig: `loss_weight_jump` wird **nie** verändert — es bleibt konstant bei 1.0.

### Ist Loss-Gewichtung theoretisch sinnvoll?

**Ja, absolut — aber mit Vorsicht.** Multi-Task-Learning-Theorie (Kendall et al., 2018) zeigt:

- Unterschiedliche Losses operieren auf unterschiedlichen Skalen
- $L_{\text{teacher}}$ wird über $T \cdot N$ Patches gemittelt, $L_{\text{jump}}$ über $1 \cdot N$ Patches
- Entscheidend ist nicht $\lambda_{\text{teacher}} / \lambda_{\text{jump}}$, sondern die **Gradienten-Ratio:**

$$\frac{||\lambda_{\text{teacher}} \cdot \nabla_\theta L_{\text{teacher}}||}{||\lambda_{\text{jump}} \cdot \nabla_\theta L_{\text{jump}}||}$$

### Die Sorge um Catastrophic Forgetting der Short-Term Dynamics

**Das ist die kritischste Schwachstelle.** Bei Epoche 11 (AC-ViT) gilt:

$$L = 0.3 \cdot L_{\text{teacher}} + 1.0 \cdot L_{\text{jump}}$$

Der Jump-Loss dominiert mit Faktor ~3.3×. Das bedeutet:

- **Kaum noch Lernsignal für zeitlich nahe Vorhersagen.** Der Teacher-Forcing-Loss trainiert die Grundkompetenz für sequenzielle Vorhersagen — essentiell für Planung.
- **Keine Garantie, dass der Jump-Loss diese Kompetenz erhält:** Jump-Prediction nutzt einen völlig anderen Berechnungspfad (ein Forward-Pass mit nur einem Context-Frame und RoPE-Override).
- **Im CL-Kontext Verschärfung:** Erst Abschwächung durch Schedule, dann zusätzliches Vergessen durch Task-Shift → **doppeltes Vergessen**.

### Gradienten-Interaktion

- $L_{\text{teacher}}$ → Gradienten über vollen Kontext ($T$ Frames), tendenziell **klein und gleichmäßig** verteilt
- $L_{\text{jump}}$ → Gradienten aus nur einem Frame, tendenziell **stärker und lokaler**
- Bei $\lambda_{\text{teacher}} = 0.3$ dominieren die Jump-Gradienten die Manifolds, auf denen short-term Dynamics kodiert sind

### Bewertung des aktuellen Schedules

| Aspekt | Bewertung |
|---|---|
| Grundidee (Curriculum) | ✅ Sinnvoll (Bengio et al., 2009) |
| Nur $\lambda_{\text{teacher}}$ runterfahren | ⚠️ Riskant für short-term Dynamics |
| $\lambda_{\text{teacher}} = 0.3$ (AC-ViT, ab Ep. 11) | ❌ **Zu aggressiv** |
| $\lambda_{\text{teacher}} = 0.5$ (AC-HOPE, ab Ep. 25) | ⚠️ Grenzwertig |
| Kein Gradienten-Monitoring | ❌ **Blindflug** |
| Stufenförmiger Schedule | ⚠️ Suboptimal (abrupte Verlustlandschafts-Änderungen) |
| HOPE langsamer als ViT | ✅ Richtig (Inner-Loop konvergiert langsamer) |

### Empfehlungen

#### A. Sofort umsetzbar (ohne neue Runs):

1. **Gradienten-Normen loggen:** In `_shared_step` ein Logging von $||\nabla L_{\text{teacher}}||$ und $||\nabla L_{\text{jump}}||$ hinzufügen. Kostet fast nichts, eliminiert Blindflug.
2. **Teacher-Gewicht nicht unter 0.5 senken:** Faustregel: $\lambda_{\text{teacher}} \geq 0.5 \cdot \lambda_{\text{jump}}$.

#### B. Zentrale Empfehlung — Logik umdrehen:

Statt $\lambda_{\text{teacher}}$ runterzufahren, **$\lambda_{\text{jump}}$ hochfahren:**

| Epoche | $\lambda_{\text{teacher}}$ | $\lambda_{\text{jump}}$ |
|---|---|---|
| 0 | 1.0 | 0.3 |
| 10 | 1.0 | 0.7 |
| 20 | 1.0 | 1.0 |

**Vorteil:** Die short-term Kompetenz bleibt während des gesamten Trainings vollständig erhalten, während die Jump-Kompetenz graduell aufgebaut wird. Mathematisch äquivalent in der Endkonfiguration, aber sicherer.

#### C. Linearer statt stufenförmiger Schedule:

$$\lambda_{\text{jump}}(e) = \min\left(1.0,\; \lambda_{\min} + \frac{e}{E_{\max}} \cdot (1.0 - \lambda_{\min})\right)$$

mit z.B. $\lambda_{\min} = 0.3$. Eliminiert abrupte Verlustlandschafts-Änderungen.

#### D. Fortgeschritten (optional): Uncertainty Weighting (Kendall et al., 2018)

$$L = \frac{1}{2\sigma_1^2} L_{\text{teacher}} + \frac{1}{2\sigma_2^2} L_{\text{jump}} + \log(\sigma_1 \sigma_2)$$

Automatisiert die Balance, aber zusätzlicher Hyperparameter-Freiheitsgrad.

---

## Priorisierung (angesichts 40 GB Daten, Kosten & Zeitdruck)

| Priorität | Maßnahme | Aufwand |
|---|---|---|
| 1 | Datenleck im Base Training fixen | 1 Zeile Code |
| 2 | BWT-Metrik ergänzen | Wenige Zeilen, kein neuer Run |
| 3 | 40 Epochen beibehalten, keinen Pilot-Run | Null Aufwand |
| 4 | Bei L1 bleiben (wenn Ergebnisse existieren) | Null Aufwand |
| 5 | Curriculum-Logik umdrehen ($\lambda_{\text{jump}}$ hochfahren) | Config-Änderung |
| 6 | Gradienten-Normen loggen | Wenige Zeilen Code |
