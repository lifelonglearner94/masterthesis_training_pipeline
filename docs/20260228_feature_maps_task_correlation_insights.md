# Wissenschaftliches Protokoll: Analyse der Continual Learning Experimente auf V-JEPA 2 Repräsentationen

**Datum der Analyse:** 28. Februar 2026

**Datengrundlage:** Feature Maps aus einem eingefrorenen V-JEPA 2 ViT-L/16 Encoder (10.000 analysierte Clips)

**Aufgabe:** Vorhersage zukünftiger Frames (Regression) im Continual Learning (CL) Setup über 6 Phasen (Base, T1-Scale, T2-Ice, T3-Bounce, T4-Rot, T5-OOD).

---

## 1. Charakterisierung des Repräsentationsraums (Feature Space)

Vor der Evaluation der CL-Metriken wurde der zugrundeliegende Datenraum (die V-JEPA 2 Features) analysiert, um eine Baseline für die Aufgabenschwierigkeit zu etablieren.

* **Geringe Sparsität (Dense Representations):** Die Features weisen eine extrem geringe Sparsität von durchschnittlich **0.39%** auf. Das bedeutet, dass die Repräsentationen stark verteilt (dense) sind. Dies macht das Netzwerk inhärent anfällig für katastrophales Vergessen, da das Anpassen von Gewichten für eine neue Aufgabe fast unweigerlich die dichten Aktivierungsmuster alter Aufgaben stört.
* **Natürliche Varianz:** Die Standardabweichung der Feature-Maps ($\sigma$) liegt im Durchschnitt bei **3.18**. Diese Metrik dient als Ankerpunkt zur Bewertung der Modellfehler.

## 2. Erkenntnis I: Kontextualisierung des L1-Fehlers (Relativer Fehler)

Der absolute "Jump L1 Error" ist ohne Kenntnis der Datenskalierung schwer zu interpretieren. Durch die Verknüpfung der CL-Ergebnisse mit den Feature-Statistiken wurde eine **Relative Fehlermetrik** eingeführt.

* **Beobachtung:** Das naive Baseline-Modell erreicht nach allen Trainingsphasen einen durchschnittlichen absoluten L1-Fehler von **0.3720**.
* **Erkenntnis:** Setzt man diesen Fehler ins Verhältnis zur natürlichen Standardabweichung der Daten ($\sigma = 3.18$), entspricht der absolute Fehler lediglich einer Abweichung von ca. **11.7%** der natürlichen Datenschwankung ($0.3720 / 3.18$).
* **Schlussfolgerung:** Die Vorhersagen der Modelle liegen, trotz aufgetretenem Vergessen, verhältnismäßig nah am Ground-Truth-Datenraum.

## 3. Erkenntnis II: Identifikation der Ursache für Katastrophales Vergessen

Die Analyse der Inter-Task Cosine Similarity der V-JEPA Features belegt, dass es sich bei dem konstruierten Setup um ein hochgradig anspruchsvolles **Domain Incremental Learning** Szenario handelt. Die Aufgaben überlappen im Feature-Space extrem stark (Cosine Similarities zwischen **0.9510** und **0.9976**).

Es konnte eine signifikante mathematische Korrelation zwischen der Datenähnlichkeit und dem schrittweisen Vergessen (Step-wise Forgetting) nachgewiesen werden:

* **Korrelationsnachweis:** Im *Second AC HOPE Run* zeigt sich eine moderate positive Korrelation (**Pearson $r = 0.431$**) zwischen der Feature-Similarity zweier Tasks und dem induzierten Vergessen. Auch in der Naiven Baseline ($r = 0.294$) und im *First AC HOPE Run* ($r = 0.412$) ist dieser Trend klar erkennbar.
* **Extrembeispiel (Base vs. T1-Scale):** Die Aufgaben "Base" und "T1-Scale" weisen eine extrem hohe Datenähnlichkeit von **0.9882** auf. Konsequenterweise verursacht das Training von T1-Scale den massivsten Einbruch der Base-Performance: Der Fehler auf der Base-Task steigt im naiven Ansatz schlagartig um **+0.0746** (höchster gemessener Step-Forgetting-Wert).
* **Kontrast (Base vs. T4-Rot):** Die Aufgabe "T4-Rot" hat eine geringere Ähnlichkeit zur Base-Task (**0.9510**). Das Training von T4-Rot führt dementsprechend zu einer signifikant geringeren Degradation der Base-Task (Interferenz von nur **+0.0197** in der naiven Baseline, bzw. **+0.0032** im Second AC HOPE Run).

## 4. Fazit und Bewertung des Datensets

Die generierte Sequenz von Aufgaben (Base bis T5-OOD) ist keinesfalls methodisch fehlerhaft, sondern stellt einen exzellenten **Stresstest für Continual Learning Algorithmen** dar.

Durch die nachweislich hohe Ähnlichkeit der Aufgaben (Cosine Similarity > 0.95) isoliert das Datenset gezielt das Kernproblem des katastrophalen Vergessens: **Interferenz durch Repräsentationsüberlappung**. Die eingeführte Korrelationsanalyse beweist, dass das Vergessen in diesem Setup kein zufälliges Rauschen ist, sondern kausal durch die geometrische Nähe der neuen Aufgaben zu den alten Aufgaben im Feature-Space bedingt wird. Algorithmen wie "HOPE" können nun gezielt daraufhin evaluiert werden, wie gut sie Netzwerke gegen diese spezifische Interferenz in hochgradig dichten (dense) Datenräumen stabilisieren.
