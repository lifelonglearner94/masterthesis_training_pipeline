# 📝 V-JEPA 2-AC — Planning & Control

*Scientific notes and concise explanation of the planning and control pipeline used in V-JEPA 2-AC.*

---

## Inhaltsverzeichnis
1. [Zieldefinition (Goal Specification)](#zieldefinition-goal-specification) ✅
2. [Optimierungsprozess (Inference Strategy)](#optimierungsprozess-inference-strategy) ⚙️
3. [Forward Model vs. Backward Planning](#forward-model-vs-backward-planning) 🔁
4. [Ausführung: Receding Horizon Control](#ausführung-receding-horizon-control) 🔄
5. [Zusammenfassung — Analogie (Golf)](#zusammenfassung--analogie-golf) ⛳️

---

## 1. Zieldefinition (Goal Specification)

- **Grundidee:** Statt sprachlicher Prompts wird das Ziel visuell im latenten Raum definiert.
- **Input:** Ein Zielbild `x_g` (z. B. "Ball im Loch").
- **Encoding:** Das Zielbild wird durch den fixierten Encoder geleitet, wodurch die Ziel-Feature-Map `z_g` entsteht.
- **Zielsetzung (Loss / Vergleichsgröße):** Minimierung der Distanz (z. B. L1) zwischen der vorhergesagten Zukunft `z_T` und `z_g`.

**Kurzformel:**
- Gegeben: `x_g` → Encoder → `z_g`
- Ziel: minimize || `z_T` - `z_g` ||_1

---

## 2. Der Optimierungsprozess (Inference Strategy)

Das System plant nicht sequenziell per Trial-and-Error, sondern verwendet eine populationsbasierte Optimierung: die Cross-Entropy Method (CEM).

**Schritte:**

1. **Initialisierung (Sampling):** Es werden mehrere (z. B. 100) Aktionssequenzen (Trajektorien) aus einer Gauß-Verteilung gezogen.
2. **Forward Pass (Simulation):** Alle Trajektorien werden parallel durch den Predictor simuliert — jeder ergibt ein latentes Ergebnis `z_T`.
3. **Selektion (Elites / Top-k):** Auswahl der besten Trajektorien mit den kleinsten Abständen zu `z_g`.
4. **Refinement:** Aus den Top-Trajektorien werden Mittelwert und Varianz der Aktionsverteilung neu geschätzt.
5. **Iterieren:** Schritte 1–4 werden mehrere Iterationen wiederholt, bis Konvergenz erreicht oder Budget (Iteration/Time) erschöpft ist.

**Wichtig:** CEM ist populationsbasiert und parallel — das erlaubt robustere Suche nach guten Aktionssequenzen als einfache lokale Heuristiken.

---

## 3. Forward Model vs. Backward Planning

Es ist hilfreich, zwischen zwei Komponenten zu unterscheiden:

### Predictor (Neuronales Netz) 🧠
- **Funktion:** Kausaler Vorwärtsmodellierer.
- **Input → Output:** (`z_t`, `a_t`) → `z_{t+1}`.
- **Eigenschaft:** Kennt das Ziel nicht; modelliert physikalische Dynamik bzw. Übergänge.

### Planer (CEM-Algorithmus) 🧭
- **Funktion:** Zielgerichtete Suche nach Aktionen, die das gewünschte Ergebnis erzeugen.
- **Wie:** Nutzt den Predictor mehrfach (simuliert Vorhersagen) und bewertet Trajektorien anhand des Abstands zu `z_g`.
- **Teleologisch:** Obwohl der Predictor nur kausal vorwärts arbeitet, ist der Planer teleologisch, weil er Aktionen auswählt, die ein Endziel maximieren/minimieren.

---

## 4. Ausführung: Receding Horizon Control

- **Langfristiger Plan, kurzfristige Ausführung:** Obwohl lange Aktionssequenzen geplant werden (z. B. "Arm heben → schwenken → drücken"), wird nur die erste Aktion ausgeführt.
- **Erster Schritt:** Nur `a_1` (die geplante erste Aktion) wird an den Roboter gesendet.
- **Re-Planning:** Nach Ausführung wird der reale neue Zustand beobachtet und der gesamte Planungsprozess erneut gestartet.

**Grund:** Dadurch werden Modellfehler korrigiert (Closed-Loop Control) — Anpassung an reale Abweichungen von den Vorhersagen.

---

## 5. Zusammenfassung — Analogie (Golf) ⛳️
> Der Roboter "träumt" vor jedem Schlag dutzende Varianten parallel, vergleicht deren geträumtes Endergebnis mit dem Foto eines erfolgreichen Schlags, wählt die beste Bewegung, führt sie für einen kleinen Schritt aus und bewertet die Situation neu.

---

## Key takeaways ✅
- Ziel wird visuell als latenter Vektor `z_g` definiert.
- CEM erlaubt parallele, populationsbasierte Suche nach guten Aktionssequenzen.
- Predictor ist ein kausales Vorwärtsmodell; der Planer ist teleologisch.
- Receding Horizon Control sorgt für Robustheit gegenüber Modellfehlern.

---

*Version: formatiert for clarity. Feel free to request translation to English or further expansions (e.g., pseudo-code, diagrams).*
