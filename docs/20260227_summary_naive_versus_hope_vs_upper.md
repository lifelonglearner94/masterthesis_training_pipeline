Hier ist ein detaillierter Bericht, der die neuen Ergebnisse des Joint Trainings als "Upper Bound" (obere Leistungsgrenze) in Relation zu den beiden vorherigen Continual Learning (CL) Experimenten setzt.

---

# Vergleichende Analyse: Continual Learning vs. Joint Training (Upper Bound)

Das **Joint Training** (gemeinsames Training auf allen 10.000 Clips) dient als perfektes Referenzmodell. Da es nicht sequenziell trainiert wird, unterliegt es keinem Catastrophic Forgetting (StreamForgetting = 0.0000) und zeigt die maximale Kapazität der Architektur, die zugrunde liegende Physik und Dynamik des gesamten Datensatzes simultan zu erfassen.

## 1. Direkter Leistungsvergleich (Die "Upper Bound")

Die folgende Tabelle stellt die L1-Jump-Loss-Werte der drei Modelle gegenüber. Wir vergleichen die initiale Leistung auf der Base-Task (Phase 0), die finale Leistung auf der Base-Task nach dem Durchlaufen aller Phasen, sowie die Forgetting-Metrik.

| Metrik / Task | Exp 1: Naive Baseline (CL) | Exp 2: AC-HOPE-ViT (CL) | Exp 3: Joint Training (Upper Bound) |
| --- | --- | --- | --- |
| **Initialer Base Loss (Task 0)** | 0.2783 | 0.3256 | **0.2502** |
| **Finaler Base Loss (nach Task 5)** | 0.4401 | 0.4359 | **0.2502** (Konstant) |
| **StreamForgetting** | 0.0856 | 0.0421 | **0.0000** |
| **Beste Task (Loss)** | 0.2783 (Task 0) | 0.3256 (Task 0) | **0.2437** (Task 1: Scaling) |
| **Schwerste Task (Loss)** | *N/A (Degradation dominiert)* | *N/A (Degradation dominiert)* | **0.3364** (Task 5: Compositional) |

**Analyse:**
Das Joint Training definiert die absolute Obergrenze der Vorhersagegenauigkeit. Mit einem Base Loss von **0.2502** übertrifft es die initiale Plastizität des naiven Modells (**0.2783**) leicht und deklassiert die initiale Leistung des stark regulierten AC-HOPE-Modells (**0.3256**) deutlich. Dies beweist quantitativ die "Complexity Tax" (Komplexitätssteuer), die das AC-HOPE-Modell für seine stabilisierende Architektur zahlt.

---

## 2. Aufgabenschwierigkeit vs. Modellschwäche

Ein großer Vorteil des Joint Trainings ist, dass es die *tatsächliche* Schwierigkeit der einzelnen Distribution Shifts offenlegt, ungetrübt von sequenziellen Lerneffekten oder Forgetting.

* **Task 1 (Scaling) & Task 2 (Dissipation):** Diese Shifts fallen dem Joint-Modell extrem leicht (Loss: 0.2437 und 0.2491). Sie sind sogar leichter oder gleichauf mit der Base-Task.
* **Task 3 (Discretization) & Task 4 (Kinematics):** Hier steigt der Loss spürbar an (0.2818 und 0.2581).
* **Task 5 (Compositional OOD):** Dies ist auch für das omnisziente Joint-Modell bei Weitem die schwerste Aufgabe (Loss: **0.3364**).

**Was das für die CL-Modelle bedeutet:**
Wenn die CL-Modelle (Exp 1 und Exp 2) in Phase 5 massive Probleme zeigten, lag das nicht *nur* am Catastrophic Forgetting, sondern auch an der inhärenten "Out-of-Distribution"-Natur (OOD) dieser spezifischen Aufgabe. Das Joint Training beweist, dass "Compositional" grundlegend schwerer zu approximieren ist als die anderen physikalischen Parameter.

---

## 3. Die Grenzen der Plastizität im Continual Learning

Das Joint Training zeigt uns, wie gut das Netzwerk die Repräsentationen *theoretisch* anpassen könnte.

* **Exp 1 (Naiv):** War überraschend nah an der Upper Bound, was die reine Plastizität angeht (0.2783 vs. 0.2502). Es hat jedoch bewiesen, dass unregulierte Plastizität in einem sequenziellen Setup unweigerlich zur kompletten Zerstörung vorheriger Repräsentationen führt (Finaler Base Loss: 0.4401).
* **Exp 2 (AC-HOPE):** Ist von der Upper Bound am weitesten entfernt. Der "Puffer" zwischen der Upper Bound (0.2502) und der initialen AC-HOPE-Leistung (0.3256) ist der Preis für das halbierte Forgetting (0.0421).

---

## 4. Temporale Dynamik (z_6, z_7, z_8)

Ein Muster bleibt über **alle drei** Experimente hinweg absolut konsistent:
Die Vorhersage des weitesten temporalen Ziels (z_8) ist immer am schlechtesten, während z_7 oft einen "Sweet Spot" darstellt oder zumindest signifikant besser performt als z_8.

* Im Joint Training (Task 0): z_6 = 0.245, z_7 = 0.228, z_8 = 0.276.
Dies bestätigt endgültig, dass diese Varianzen keine Artefakte der Optimierungsstrategie (SGD vs. DGD) sind, sondern eine grundlegende Eigenschaft der zeitlichen Vorhersagbarkeit innerhalb dieses physikalischen Datensatzes.

---

### Konklusion

Das Einbeziehen des Joint Trainings schließt den Kreis dieser Experimentreihe und liefert den nötigen physikalischen Kontext:

1. **Das Naive Baseline-Modell (Exp 1)** ist ein "Glaskasten": Es nimmt neue Informationen exzellent auf (nah an der Upper Bound), zerbricht aber sofort unter dem Druck neuer, konkurrierender Aufgabenstellungen.
2. **Das AC-HOPE-ViT (Exp 2)** funktioniert genau wie beabsichtigt. Es fungiert als Isolator gegen Catastrophic Forgetting und halbiert den Wissensverlust. Der Vergleich mit dem Joint Training zeigt jedoch, dass die aktuelle Architektur zu restriktiv ist. Sie opfert zu viel von der theoretisch möglichen Leistung (Upper Bound), um die Stabilität zu wahren.
3. **Die Datensatz-Dynamik:** Das Joint Training beweist, dass späte Shifts (wie Task 5: Compositional) selbst für ein simultan trainiertes Modell schwer zu greifen sind.

**Fazit für die Masterarbeit:**
Das AC-HOPE-Modell löst das Kernproblem (Forgetting), kreiert aber ein neues (geringere Base-Plastizität und Instabilität beim Training). Der "Holy Grail" für zukünftige Arbeiten liegt genau im Delta zwischen dem AC-HOPE-Modell und dem Joint Training: Wie kann man die Speichermechanismen von AC-HOPE so modifizieren, dass sie die Plastizität der Upper Bound erreichen (näher an 0.2502) ohne die Stabilität wieder einzureißen?
