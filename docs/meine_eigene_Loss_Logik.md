Das ist eine extrem spannende Frage, weil du dich damit vom reinen "Nachmachen" (Pattern Matching) hin zum Lernen von **echter physikalischer Kausalität** (Dynamics Modeling) bewegst.

Dein Szenario (Würfel rutscht durch einmaligen Impuls) ist physikalisch deterministisch:
 (nur am Anfang) und danach  (als Verzögerung).

Hier ist meine Einschätzung zu deiner Idee, den **Teacher Forcing Loss komplett zu entfernen**:

### 1. Die kurze Antwort

**Tue es nicht sofort komplett.** Es ist sehr riskant und führt oft dazu, dass das Modell gar nichts lernt ("Model Collapse").

Aber: Deine Intuition ist richtig. Für dein Ziel (Robustheit gegen Friction-Shift) muss der **Rollout-Anteil dominieren**.

### 2. Warum reiner Rollout (ohne Teacher) gefährlich ist

Stell dir vor, du bringst einem Kind Fahrradfahren bei.

* **Teacher Forcing:** Du hältst den Sattel fest. Das Kind lernt die Bewegung der Beine.
* **Pure Rollout:** Du setzt das Kind auf das Rad und schubst es an. Es fällt nach 1 Meter um. Beim nächsten Versuch fällt es wieder um.

**Das technische Problem:**
Zu Beginn des Trainings sind die Gewichte des Modells zufällig.

1. Das Modell bekommt Input (Zeitschritt 0-2).
2. Es rät Zeitschritt 3. Da es noch nichts kann, kommt "Müll" raus.
3. Jetzt muss es basierend auf diesem "Müll" (Input 3) den Zeitschritt 4 vorhersagen.
4. Der Fehler explodiert exponentiell ("Error Accumulation").
5. Der Gradient (das Lernsignal) wird so verrauscht, dass das Modell nicht konvergiert.

Da dein "Action"-Input nach Frame 1 wegfällt (alles 0), muss das Modell die **Geschwindigkeit** und die **Reibung** intern im Hidden State speichern. Wenn es einmal die Geschwindigkeit falsch einschätzt, landet der Würfel im Rollout an einer völlig falschen Stelle.

### 3. Warum Teacher Forcing für dein Physik-Szenario wichtig ist

In deinem Fall muss das Modell zwei Dinge lernen:

1. **Repräsentation:** "Wie sieht ein rutschender Würfel im V-JEPA Latent Space aus?"
2. **Dynamik:** "Wie schnell wird er langsamer?"

Der Teacher Loss hilft enorm bei Punkt 1. Er "ankert" das Modell immer wieder an der Realität.

* *Modell:* "Ich glaube, bei Schritt 5 ist der Würfel hier."
* *Teacher:* "Nein, er ist dort. Korrigiere dich, bevor du Schritt 6 vorhersagst."

Ohne diesen Anker kann das Modell physikalisch unmögliche Zustände halluzinieren, die für den Decoder (falls du einen nutzt) oder den Loss keinen Sinn ergeben.

### 4. Meine Empfehlung: Curriculum Learning (Der Königsweg)

Statt "Entweder/Oder" solltest du den Trainingsprozess **dynamisch anpassen**. Das nennt man "Scheduled Sampling" oder "Curriculum Learning".

**Strategie für dein Experiment:**

* **Phase 1 (Die ersten 10-20 Epochen):**
* Lass Teacher Forcing an.
* Nutze `T_rollout = 2` (wie im Paper).
* *Ziel:* Das Modell lernt erst mal die Grundlagen: Wie sieht der Würfel aus? Was bedeutet der Kraftvektor?


* **Phase 2 (Der "Loslass"-Moment):**
* Erhöhe `T_rollout` schrittweise (z.B. auf 4, dann 6, dann 8).
* Reduziere das Gewicht von `loss_teacher`.
* *Ziel:* Das Modell wird gezwungen, seine eigenen Vorhersagen stabil zu halten.


* **Phase 3 (Für dein Reibungs-Experiment):**
* Hier kannst du Teacher Forcing fast auf 0 setzen oder `T_teacher` auf das Minimum (1-2 Frames) reduzieren, um nur den Startzustand zu geben.



### 5. Speziell zu deinem A-B-A Szenario (Reibungs-Änderung)

Du willst, dass das Modell generalisiert (weniger Reibung = längeres Rutschen).

Das ist ein klassisches "Out-of-Distribution" (OOD) Problem.

* Wenn du **zu viel Teacher Forcing** nutzt, lernt das Modell nur: "Nach 5 Frames ist der Würfel meistens an Position X". Es lernt die **Statistik** des Datensatzes auswendig.
* Wenn du **viel Rollout** nutzt, muss das Modell die **Differentialgleichung** () approximieren, um den nächsten Schritt zu berechnen.

**Fazit:**
Um dein Ziel zu erreichen, ist ein **hoher Rollout-Anteil (langer Horizont)** absolut notwendig, damit das Modell die Physik (den *Prozess* des Rutschens) lernt und nicht nur die Positionen auswendig lernt.

**Mein Vorschlag für deine Config jetzt:**
Lass den Teacher Loss drin, aber erhöhe `T_rollout` aggressiv, sobald das Training stabil läuft.

* Start: `T_teacher: 2`, `T_rollout: 2` (Model sieht Startimpuls + 1 Frame, muss 2 raten).
* Später: `T_teacher: 2`, `T_rollout: 6` (Muss fast die ganze Rutschpartie allein simulieren).

Das zwingt das Modell, die interne Variable "Geschwindigkeit" korrekt fortzuschreiben.
