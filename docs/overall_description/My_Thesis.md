Was ich cooles mache?

Ich arbeite mit V-JEPA 2 Weltmodellen.
Diese sind mega gut für eine Maschinen um Videos bzw. die Welt zu verstehen / sehen / antizipieren was passieren wird.
Denn sie transformieren Frames mit Pixeln in einen latenten Raum, der viel Speichereffizienter ist und trotzdem die wichtigsten Infos enthält! Gibt auch coole 3D-Rotary Positional Embeddings um Raum-Zeit Informationen in diese latenten Feature Maps zu integrieren. (V-JEPA 2 Paper)




Aufbau des Experiments:

Ich habe einen Physik Simulations Datensatz erstellt, in dem Objekte einen Kraftimpuls bekommen und dann ein Stück rutschen!
Es wird eine Phase A_1 geben mit normaler Reibung.
Eine Phase B mit deutlich reduzierter Reibung (und Masse): Zum Testen auf Test Time Adaption (TTA) / Test Time Training (TTT) Fähigkeiten.
Und eine Phase A_2 mit gleichen Bedingungen wie Phase A_1: zum Testen auf Catastrophic Forgetting.

Diese Clips habe ich alle durch einen kleinen frozen Encoder gejagt, dem "ViT-L/16". Dadurch erhalten ich meine Feature Maps, welche alle Informationen aus den Clips enthalten. (Aus 16 Frames werden 8 Zeitschritte)
Außerdem habe ich bei der Datengenerierung ein Array mit Stärke des Kraftimpulses (x,y in Newton) zum Zeitschritt n erstellt.



Bisher habe ich den Action Conditioned Vision Transformer Predictor aus dem V-JEPA 2 Paper, leicht adaptiert und für mein Szenario genutzt. Die Loss Funktionen habe ich auch aus dem Paper übernommen, wobei der Rolling Loss im Laufe des Trainings immer mehr Gewicht bekommt.
1. Teacher-Forcing Loss
This loss function measures the difference between the predicted and actual state representations at each time step:


$$\mathcal{L}_{\text{teacher-forcing}}(\phi) := \frac{1}{T} \sum_{k=1}^{T} || \hat{z}_{k+1} - z_{k+1} ||_{1} = \frac{1}{T} \sum_{k=1}^{T} || P_{\phi}((a_{t}, s_{t}, E(x_{t}))_{t \le k}) - E(x_{k+1}) ||_{1}$$
2. Rollout Loss
This loss improves the model's ability to perform autoregressive rollouts by comparing the final predicted state after a sequence of actions to the actual future state:
+1


$$\mathcal{L}_{\text{rollout}}(\phi) := || P_{\phi}(a_{1:T}, s_{1}, z_{1}) - z_{T+1} ||_{1}$$

Note: The overall training objective minimizes the sum of these two losses: $L(\phi) := \mathcal{L}_{\text{teacher-forcing}}(\phi) + \mathcal{L}_{\text{rollout}}(\phi)$.


---

Mein Ziel und der Beitrag zur Forschung, ist aber eigentlich die HOPE Architektur aus dem Nested Learning Paper von Behrouz (2025) in meinem Szenario einzusetzen.
Die Neuheit dabei ist dass die HOPE Architektur (die nur auf Text eingesetzt wurde) nun auf Videodaten im latenten Raum eingesetzt wird.
Um sozusagen Adaptive Weltmodelle möglich zu machen.
Das heißt im besten Fall schaffe ich es die Architektur auf meine Daten und meinen Fall zu übertragen und dass diese bessere Ergebnisse im A - B - A Szenario erreicht als andere Modelle.
Als einfachstes Vergleich Modell habe ich einfach ein auf Szenario A_1 vor-trainiertes "Action Conditioned Vision Transformer Predictor aus dem V-JEPA 2 Paper" genommen und Test Time Adaption (TTA) -Optionen hinzugefügt (Adaption der Layer Norm nach jedem Teststep)

Dadurch habe ich eine gute Baseline.

---

Als nächstes werde ich nun versuchen die HOPE Architektur hier sinnvollzu integrieren.

Wenn es klappen würde, hätte ich damit einen großen Beitrag zur wissenschaft und zum Schritt zur autonomen Maschinen Intelligenz geleistet!
