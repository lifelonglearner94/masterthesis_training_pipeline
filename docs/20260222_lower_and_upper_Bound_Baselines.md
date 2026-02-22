Hier ist das überarbeitete wissenschaftliche Protokoll. Der Fokus liegt nun ganz auf der methodischen Einordnung durch die Lower und Upper Bounds, während das spezifische Curriculum (die einzelnen Tasks) weggelassen wurde.

---

# Scientific Protocol: Bounding Methodology for Continual Learning in Physical Dynamics

## 1. Zielsetzung (Objective)

Das primäre Ziel dieser Evaluierungsstrategie ist die objektive Quantifizierung der *Continual Learning* (CL) Fähigkeiten repräsentationsbasierter Modelle (wie Action-Conditioned Vision Transformers). Da kontinuierlich lernende Systeme naturgemäß einem Zielkonflikt zwischen Plastizität (Erlernen neuer physikalischer Gesetzmäßigkeiten) und Stabilität (Erhalt alten Wissens) unterliegen, erfordert die Leistungsbewertung einen klar definierten methodischen Rahmen. Dieser Rahmen wird durch empirische untere und obere Leistungsgrenzen (Bounds) aufgespannt.

## 2. Methodisches Framework: Definition der Baselines (Bounds)

Um die Effektivität der vorgeschlagenen CL-Architekturen validieren zu können, wird dieselbe zugrundeliegende Modellarchitektur unter extremen, gegensätzlichen Trainingsbedingungen evaluiert. Diese Bedingungen definieren den Referenzkorridor für alle weiteren Experimente.

### 2.1. Lower Bound (Naives sequenzielles Lernen / Finetuning)

Der Lower Bound repräsentiert das methodische Minimum ("Worst-Case-Szenario").

* **Setup:** Das Modell wird strikt sequenziell über den Strom der physikalischen Tasks trainiert (Task 1  Task 2  ...  Task ). Dabei werden die Gewichte fortlaufend aktualisiert, ohne dass Mechanismen zum Wissenserhalt (wie Replay-Buffer, Regularisierung oder isolierte Parameter-Updates) angewendet werden.
* **Funktion im Experiment:** Dieses Setup erzwingt das sogenannte *Catastrophic Forgetting*. Es dient als Basislinie, um zu beweisen, dass eine signifikante Degradation der Vorhersagegenauigkeit auf vergangenen Tasks existiert, wenn nicht aktiv gegengesteuert wird. Jeder validierte CL-Ansatz muss diese Basislinie statistisch signifikant übertreffen.

### 2.2. Upper Bound (Joint Training / Offline Learning)

Der Upper Bound definiert das theoretische und empirische Leistungsmaximum ("Best-Case-Szenario") der gewählten Modellarchitektur auf dem vorliegenden Datensatz.

* **Setup:** Die sequenzielle Natur des Problems wird für diese Baseline vollständig aufgehoben. Die Datensätze aller Tasks werden aggregiert, gemischt und das Modell wird "offline" über den gesamten Datenbestand (i.i.d. – independent and identically distributed) trainiert.
* **Funktion im Experiment:** 1. **Empirisches Limit:** Es zeigt die maximale Genauigkeit, die die ViT-Architektur erreichen kann, wenn sie uneingeschränkten Zugriff auf alle physikalischen Phänomene gleichzeitig hat. Kein CL-Modell kann systematisch besser abschneiden als dieser Goldstandard.
2. **Sanity Check der Datenbasis:** Bevor die CL-Experimente starten, wird das Joint-Training-Modell mittels Kreuzvalidierung (Cross-Validation) getestet. Nur wenn dieses Modell die physikalische Dynamik fehlerfrei (ohne Generalisierungsfehler) prädizieren kann, ist bewiesen, dass die Modellkapazität und das Datensampling ausreichen.

## 3. Einordnung der zu evaluierenden Methoden

Innerhalb des durch Lower und Upper Bound aufgespannten Leistungsraums werden die eigentlichen Untersuchungsgegenstände platziert und evaluiert:

* **Das Zwischenmodell (Test-Time Adaptation - TTA):** Ein Ansatz, bei dem spezifische Layer (Layer Norm, Attention Output Projections) während der Inferenz dynamisch aktualisiert werden.
* **Der primäre Ansatz (HOPE / Nested Learning):** Das hochadaptive Zielmodell.

Der Erfolg dieser Methoden bemisst sich nicht an absoluten Genauigkeitswerten, sondern an ihrer **relativen Positionierung** zwischen dem Lower und dem Upper Bound. Ein idealer CL-Ansatz schließt die Lücke zum Upper Bound weitestgehend, während er sich maximal vom Lower Bound absetzt.

## 4. Evaluierungsmetriken im Kontext der Bounds

Um die Leistung präzise zu messen, wird die Vorhersagegenauigkeit (z.B. *Last-Frame-Prediction*) in einer Evaluationsmatrix über alle betrachteten Zeitschritte erfasst. Die Bounds dienen hierbei als Normalisierungsfaktoren für folgende Metriken:

* **Relative Forgetting Rate:** Der prozentuale Genauigkeitsverlust im Vergleich zur Performance des Upper Bounds auf demselben Task.
* **Performance Gap:** Die Differenz zwischen der Gesamtgenauigkeit des CL-Ansatzes und dem Joint Training (Upper Bound) am Ende des Curriculums.
* **Forward / Backward Transfer:** Die Fähigkeit des Modells, durch das Lernen von Task  die Leistung auf alten () oder neuen () Tasks zu verbessern, gemessen gegen die isolierte Performance des Lower Bounds auf dem jeweiligen Task.
