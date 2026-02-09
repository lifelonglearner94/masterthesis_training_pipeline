📝 Scientific Notes: V-JEPA 2-AC Planning & Control
1. Definition des Ziels (Goal Specification)
Im Gegensatz zu sprachbasierten Modellen ("Prompting") definiert V-JEPA 2-AC das Ziel rein visuell im latenten Raum.
Input: Ein Zielbild ($x_g$), das den gewünschten Endzustand zeigt (z. B. "Ball im Loch").


Encoding: Dieses Bild wird durch den fixierten Encoder geschickt, um die Ziel-Feature-Map ($z_g$) zu erhalten.


Vergleichsgröße: Das System versucht, den Abstand (L1-Distanz) zwischen der vorhergesagten Zukunft ($z_T$) und diesem statischen Ziel ($z_g$) zu minimieren.


2. Der Optimierungsprozess (Inference Strategy)
Der Roboter führt kein simples "Trial-and-Error" nacheinander aus, sondern nutzt eine populationsbasierte Optimierungsmethode (Cross-Entropy Method - CEM).

Parallelisierung: Es werden initial mehrere (z. B. 100) zufällige Handlungssequenzen (Trajektorien) aus einer Gauß-Verteilung gesampelt.


Simulation (Forward Pass): Alle Sequenzen werden parallel durch den Predictor geschickt. Dieser sagt für jede Sequenz den resultierenden Zustand im latenten Raum voraus.


Selektion (Elites): Die besten Trajektorien (die mit dem geringsten Abstand zum Ziel $z_g$) werden ausgewählt ("Top-k").


Refinement: Aus diesen Top-Trajektorien werden Mittelwert und Varianz der Aktions-Verteilung aktualisiert. Dieser Prozess wird über mehrere Iterationen wiederholt, bis die Lösung konvergiert.


3. Forward Model vs. Backward Planning
Es ist entscheidend, zwischen der Funktion des neuronalen Netzes und der Funktion des Planers zu unterscheiden:
Der Predictor (Das neuronale Netz): Arbeitet strikt kausal vorwärts.
Input: Zustand $t$ + Aktion.
Output: Zustand $t+1$.
Er weiß nichts vom Ziel, er kennt nur die Physik.


Der Planer (CEM-Algorithmus): Nutzt den Predictor, um teleologisch (zielgerichtet) zu planen. Er sucht die Eingabe (Aktion), die den gewünschten Output (Zielzustand) erzeugt.


4. Ausführung: Receding Horizon Control
Auch wenn das Modell eine lange Sequenz plant (z. B. "Arm heben -> rüber schwenken -> runter drücken"), wird nicht der gesamte Plan blind ausgeführt.
Erster Schritt: Nur die allererste geplante Aktion ($a_1$) wird an den Roboter gesendet.


Re-Planning: Nach der Ausführung beobachtet der Roboter den neuen echten Zustand und beginnt den gesamten Planungsprozess von vorn.


Grund: Dies korrigiert Fehler, die entstehen, wenn die Modellvorhersage leicht von der Realität abweicht (Closed-Loop Control).
Zusammenfassung der Analogie (Golf)
Der Roboter "träumt" vor jedem Schlag dutzende Varianten parallel, vergleicht deren geträumtes Endergebnis mit dem Foto eines erfolgreichen Schlags, wählt die beste Bewegung aus, führt sie ein kleines Stück weit aus und bewertet die Situation neu.
