# Analyse: Experiment-Priorisierung & Cross-Validation-Notwendigkeit

**Datum:** 2026-02-26
**Kontext:** Budget- und zeitkritisches Training auf vast.ai GPUs. Betreuer-Vorgabe war 4-Phasen-Ansatz, CL-Experimente wurden vorgezogen (Phase 3 vor Phase 2).

---

## Status der Experimente

| Config | Status | GPU-Zeit (geschätzt) | Zweck |
|---|---|---|---|
| `cl_ac_hope.yaml` | ✅ Fertig | ~Stunden | CL-Ansatz 2 (Titan Memory + DGD) |
| `cl_lower_bound.yaml` | 🔄 Läuft | ~Stunden | Performance-Floor (naives Finetuning) |
| `cl_upper_bound.yaml` | ⏳ Ausstehend | ~Stunden | Performance-Ceiling (Joint Training) |
| `cl_upper_bound_cross_validation.yaml` | ⏳ Ausstehend | **~10× so lang** (10 Folds!) | Datenqualitäts-Check |
| `cl_ac_vit.yaml` | ⏳ Ausstehend | ~Stunden | CL-Ansatz 1 (TTA) |

---

## Frage 1: Kann `cl_ac_vit.yaml` (TTA) erstmal weggelassen werden?

### Empfehlung: **Ja, kann erstmal weggelassen werden.**

**Begründung:**

Die **minimale vollständige CL-Story** für die Thesis braucht genau drei Dinge:

1. **Lower Bound** (`cl_lower_bound`) — Performance-Floor: "So schlecht ist naives Finetuning"
2. **Upper Bound** (`cl_upper_bound`) — Performance-Ceiling: "So gut geht es maximal mit allen Daten"
3. **Ein CL-Ansatz** (`cl_ac_hope`) — "Meine Methode liegt zwischen Floor und Ceiling"

Das AC-ViT + TTA Experiment ist ein **zweiter CL-Ansatz** — nice to have für einen Methodenvergleich, aber nicht essentiell für die Kernaussage. Die AC-HOPE-Ergebnisse sind bereits stark (StreamForgetting = 0.042), das reicht als CL-Demonstration.

**Priorität:** Niedrig. Erst nach Lower Bound + Upper Bound, falls Budget übrig ist.

---

## Frage 2: Kann die Cross-Validation weggelassen werden?

### Empfehlung: **Ja, mit Einschränkung — die AC-HOPE-Ergebnisse liefern bereits indirekte Evidenz.**

### Was der Betreuer eigentlich prüfen wollte

> "Man trainiert ein Modell auf alle Daten gleichzeitig (Joint-Training). Man lässt zufällig Daten weg und testet via n-facher Kreuzvalidierung, ob das Modell diese prädizieren/rekonstruieren kann."

Der Zweck ist: **Sicherstellen, dass die Tasks genügend gute, repräsentative Samples enthalten**, bevor man teure CL-Experimente startet. Es soll verhindern, dass man CL-Metriken auf Daten berechnet, die das Modell grundsätzlich gar nicht lernen kann.

### Warum die AC-HOPE-Ergebnisse diese Frage bereits beantworten

Die `cl_ac_hope`-Ergebnisse zeigen **indirekt aber eindeutig**, dass die Daten ausreichend lernbar sind:

| Evidenz | Wert | Was es beweist |
|---|---|---|
| Base Training Val-Loss | ~0.326 | Modell kann Base-Daten lernen |
| Task 1–5 Pure Retrieval Loss | 0.327 – 0.393 | Alle 5 Task-Partitionen sind lernbar |
| Backward Transfer nach Task 1 | -0.212 (negativ = Verbesserung!) | Task-Daten enthalten transferierbares Wissen |
| Losses über Tasks hinweg konsistent | 0.32 – 0.39 Range | Kein Task ist ein Ausreißer / unlernbar |
| StreamForgetting final | 0.042 | Stabile Repräsentationen über alle Tasks |

**Kernargument:** Wenn ein Modell im **härtesten Setting** (sequenzielles CL mit frozen inner-loop bei Eval) alle Tasks auf 0.32–0.39 Loss bringt und dabei kaum vergisst, dann sind die Tasks offensichtlich lernbar und repräsentativ. Ein Joint-Training CV-Check würde **zwangsläufig gleich gute oder bessere Ergebnisse zeigen**, da Joint Training strikt einfacher ist als sequenzielles CL.

### Das Upper Bound Experiment deckt den CV-Check teilweise mit ab

`cl_upper_bound.yaml` trainiert auf **allen 10.000 Clips gleichzeitig** und evaluiert dann per Task-Partition. Wenn das Upper-Bound-Modell auf allen Task-Partitionen gute Losses zeigt, ist das **funktional äquivalent** zum CV-Check — es beweist, dass das Modell die Daten aller Tasks gleichzeitig lernen kann.

Der einzige Unterschied: CV mit 10 Folds gibt dir zusätzlich Varianz-Schätzungen (Std über Folds). Aber für den Zweck "sind die Daten lernbar?" reicht ein einzelnes Joint Training.

### Risiko-Bewertung

| Szenario | Risiko | Konsequenz |
|---|---|---|
| Betreuer fragt nach CV-Ergebnissen | **Mittel** | Du kannst argumentieren: "Upper Bound + AC-HOPE-Ergebnisse zeigen Lernbarkeit. CV hätte redundante Information geliefert." |
| Reviewer fragt nach Datenvalidierung | **Niedrig** | Upper Bound IS die Validierung. Paper-Reviewer erwarten keine separate CV vor CL. |
| Ein Task wäre nicht lernbar gewesen | **Bereits widerlegt** | AC-HOPE hat alle Tasks erfolgreich gelernt. |

---

## Empfohlene Reihenfolge (Budget-optimiert)

```
1. ✅ cl_ac_hope          — FERTIG
2. 🔄 cl_lower_bound      — LÄUFT → abwarten
3. ⭐ cl_upper_bound       — ALS NÄCHSTES (essentiell + ersetzt teilweise CV)
4. ❓ cl_ac_vit            — NUR falls Budget übrig
5. ❓ cl_cross_validation  — NUR falls Betreuer explizit darauf besteht
```

### Warum Upper Bound jetzt Priorität hat

- Ohne Upper Bound fehlt die **obere Referenzlinie** — du kannst nicht zeigen, wo AC-HOPE relativ zum Optimum liegt
- Upper Bound trainiert **einmal** auf allen Daten (40 Epochs, 10k Clips) — deutlich billiger als 10-Fold CV (10× Training!)
- Upper Bound + Lower Bound + AC-HOPE = **vollständige CL-Evaluation** mit allen nötigen Referenzpunkten

---

## Zusammenfassung

| Frage | Antwort |
|---|---|
| `cl_ac_vit` weglassen? | **Ja.** Nicht essentiell für die Kernaussage. |
| Cross-Validation weglassen? | **Ja, vertretbar.** AC-HOPE + Upper Bound liefern die gleiche Evidenz. |
| Was ist jetzt essentiell? | **Lower Bound (läuft) + Upper Bound (als nächstes starten).** |
| Dem Betreuer kommunizieren? | Ja — proaktiv erklären, dass Upper Bound + AC-HOPE-Ergebnisse die Lernbarkeit der Tasks bereits belegen, und fragen ob er trotzdem auf CV besteht. |
