# Versuchsprotokoll: Validierung latent-basierter Videoprädiktion in physikalischen Simulationen

**Datum:** 15. Februar 2026
**Thema:** Evaluierung der Modellgüte und Ausschluss trivialer Lösungsstrategien (Identity Baseline)
**Datensatz:** PyBullet Kubric (Synthetische physikalische Interaktionen)

## 1. Problemstellung und Hypothese

Es wurde ein Prädiktionsmodell trainiert, um die zeitliche Entwicklung physikalischer Objekte (Position, Bewegung unter Krafteinwirkung) basierend auf *Frozen V-JEPA 2* Feature-Maps vorherzusagen.
Da der Datensatz einen statischen Hintergrund und eine feste Kameraposition aufweist, bestand die **Hypothese der Trivialität**: Es bestand die Sorge, dass das Modell keine physikalische Kausalität lernt, sondern lediglich den statischen Hintergrund rekonstruiert und somit einen künstlich niedrigen Loss erzielt, ohne die Objektdynamik tatsächlich zu erfassen.

## 2. Methodik: Die Identity Baseline

Um die Lernleistung des Modells objektiv zu bewerten und die Hypothese der Trivialität zu widerlegen, wurde eine **Identity Baseline (No-Motion Baseline)** als Referenzwert eingeführt.

* **Definition:** Die Identity Baseline ist eine nicht-trainierte Heuristik, die annimmt, dass . Sie sagt vorher, dass keinerlei Änderung im Bildinhalt stattfindet (Persistenz-Annahme).
* **Relevanz für diesen Datensatz:** Da der Hintergrund einen Großteil des Bildbereichs einnimmt und sich nicht verändert, erreicht diese naive Heuristik bereits einen extrem niedrigen Fehlerwert (Loss). Sie fungiert als "Sanity Check": Ein Modell, das diesen Wert nicht signifikant unterschreitet, hat lediglich gelernt, den statischen Zustand zu kopieren.

## 3. Experimentelle Ergebnisse

### A. Vergleich mit der Baseline (In-Distribution)

Das trainierte Prädiktionsmodell wurde auf dem Validierungsset evaluiert und gegen den Loss der Identity Baseline gestellt.

* **Beobachtung:** Der Validierungs-Loss des Modells liegt **signifikant** unter dem Loss der Identity Baseline. Der Abstand ist substanziell (nahezu eine Halbierung des Fehlers).
* **Interpretation:** Das Modell übertrifft die Strategie des bloßen "Kopierens" deutlich. Dies beweist mathematisch, dass das Netzwerk aktiv die Veränderung der Feature-Maps (Bewegung des Objekts) vorhersagt und nicht nur Rauschen oder statische Hintergründe modelliert.

### B. Out-of-Distribution (OOD) Robustheitstest

Um zu prüfen, ob das Modell physikalische Prinzipien (Kausalität von Kraft und Reibung) verstanden hat oder lediglich die Trainingsdatenverteilung auswendig gelernt hat ("Overfitting"), wurde ein Stresstest mit unbekannten physikalischen Parametern durchgeführt (z.B. extrem abweichende Reibungswerte).

* **Beobachtung:** Bei Konfrontation mit OOD-Daten stieg der Loss des Modells zwar moderat an, blieb jedoch weiterhin **deutlich unterhalb** der Identity Baseline. Es kam zu keiner "Explosion" des Fehlers.
* **Interpretation:** Das Modell zeigt eine robuste Generalisierung. Der moderate Anstieg des Fehlers resultiert aus der Unkenntnis der exakten Parameter (Quantität), während die grundlegende Dynamik (Qualität der Bewegung) weiterhin korrekt vorhergesagt wird. Wäre das Modell ein reiner "Pattern Matcher", wäre der Loss auf das Niveau der Baseline oder darüber angestiegen.

## 4. Konklusion

Die Sorge, dass das Szenario zu einfach sei oder das Modell nur Triviallösungen lernt, konnte **falsifiziert** werden.

1. Der signifikante Abstand zur **Identity Baseline** bestätigt, dass das Modell erfolgreich Dynamik und Bewegung lernt.
2. Die Stabilität im **OOD-Test** bestätigt, dass das Modell rudimentäre physikalische Gesetzmäßigkeiten abstrahiert hat, anstatt nur Video-Samples zu memorieren.

Das Modell validiert somit erfolgreich auf semantischer Ebene (V-JEPA Features) und physikalischer Ebene.
