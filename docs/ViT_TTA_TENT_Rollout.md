Scientific Protocol: Self-Supervised Test-Time Adaptation für ViT-Prädiktoren im Latent Space
Titel: Online-Adaption von Vision Transformer Prädiktoren mittels Rollout-Loss-Minimierung (Latent Space TTA)
Datum: 03. Februar 2026
Kontext: Anpassung an Domänenverschiebung (Domain Shift) ohne Labels
1. Zielsetzung und Hypothese
Ziel: Die Vorhersagegenauigkeit eines pre-trainierten Vision Transformer (ViT) Prädiktors soll in einer neuen Zielumgebung (Target Domain) zur Laufzeit verbessert werden.
Hypothese: Durch die Minimierung des Vorhersagefehlers (Rollout Loss) zwischen dem vorhergesagten latenten Zustand $\hat{z}_{t+1}$ und dem tatsächlich beobachteten Zustand $z_{t+1}$ können die affinen Parameter der Normalisierungsschichten (LayerNorm) so angepasst werden, dass der Domain Shift kompensiert wird.
2. Theoretischer Rahmen
Im Gegensatz zu klassischem TENT (Entropieminimierung bei Klassifikation) operiert dieses Protokoll auf einem Regressionsproblem im Feature-Space.
2.1 Das Optimierungsziel
Sei $P_\phi$ der Prädiktor mit Parametern $\phi$. Zum Zeitpunkt $t$ sagt das Modell den Zustand $\hat{z}_{t+1}$ voraus. Zum Zeitpunkt $t+1$ wird der echte Zustand $z_{t+1}$ durch den Encoder beobachtet.
Die Verlustfunktion für die Adaption ist der $L_1$-Rollout-Loss über einen Schritt ($T=1$):

$$\mathcal{L}_{\text{TTA}}(\phi) = \| \hat{z}_{t+1} - \text{sg}(z_{t+1}) \|_1$$
Hinweis: $\text{sg}(\cdot)$ steht für "stop gradient". Der Zielwert $z_{t+1}$ wird als Konstante betrachtet; es fließen keine Gradienten in den Encoder zurück.
2.2 Beschränkung des Suchraums
Um "Catastrophic Forgetting" zu verhindern, werden nur die Parameter $\theta \subset \phi$ der Layer Normalization optimiert:

$$\theta = \{ \gamma_l, \beta_l \mid l \in \text{LayerNorm Layers} \}$$
Alle Attention-Gewichte und MLP-Parameter bleiben eingefroren (frozen).
3. Experimentelles Setup
3.1 Modell-Architektur
Typ: Vision Transformer (ViT) Predictor.
Input: Konkatenation aus State $s_t$, latent Feature $z_t$ und Action $a_t$.
Output: Latent Feature Map $\hat{z}_{t+1} \in \mathbb{R}^{H \times W \times D}$.
Normalisierung: Standard nn.LayerNorm (Pre-LN Konfiguration empfohlen).
3.2 Datenstrom
Die Daten werden sequenziell verarbeitet (Batch Size = 1). Es gibt keine Epochen im klassischen Sinne, sondern einen kontinuierlichen Datenstrom (Online-Setting).
4. Methodik (Verfahrensanweisung)
Das Verfahren folgt einem "Predict-Observe-Update" Zyklus mit einem Ein-Schritt-Verzögerungsmechanismus (Look-Back Update).
Phase A: Initialisierung
Lade die pre-trainierten Gewichte $\phi_{source}$.
Setze das Modell in den Trainingsmodus (model.train()), um Gradientenberechnung zu ermöglichen (beachte: LayerNorm Statistiken sind bei ViT ohnehin sample-basiert).
Setze requires_grad = False für alle Parameter.
Setze requires_grad = True nur für layer_norm.weight und layer_norm.bias.
Initialisiere Optimizer: AdamW mit Learning Rate $\eta \approx 10^{-4}$.
Phase B: Der Online-Loop (Pro Zeitschritt $t$)
Beobachtung: Empfange aktuellen State $(s_t, z_t)$ und Aktion $a_t$.
Adaptions-Schritt (Look-Back):
Bedingung: Ist $t > 0$? (Existiert eine Vorhersage aus dem vorherigen Schritt $\hat{z}_t$?)
Loss-Berechnung: Berechne $L = \| \hat{z}_t - z_t \|_1$. Hierbei ist $\hat{z}_t$ die gespeicherte Vorhersage aus dem letzten Schritt und $z_t$ die jetzt aktuelle Beobachtung.
Gradient Calculation: loss.backward() nur auf die LayerNorm-Parameter.
Safety Mechanism: Gradient Clipping anwenden (Norm $\le 1.0$).
Update: optimizer.step() und optimizer.zero_grad().
Inferenz-Schritt:
Berechne $\hat{z}_{t+1} = P_{\phi'}(s_t, z_t, a_t)$ mit den soeben aktualisierten Parametern $\phi'$.
Speichere $\hat{z}_{t+1}$ im Buffer für den nächsten Zyklus.
Gebe $\hat{z}_{t+1}$ (detached) für Downstream-Tasks weiter.
5. Sicherheitsmaßnahmen & Stabilitätskontrolle
Da ViTs unter TTA zu Instabilität neigen, sind folgende Protokollpunkte strikt einzuhalten:
Risiko
Maßnahme
Begründung
Model Collapse
Gradient Clipping
ViTs haben keine Batch-Statistik-Puffer wie CNNs. Ein einzelner Ausreißer (Outlier) kann die Features zerstören. Clipping limitiert die Update-Größe drastisch.
Fehler-Akkumulation
Episodischer Reset (Optional)
Falls die Performance nach $N$ Schritten abfällt, setze $\phi$ auf $\phi_{source}$ zurück. Empfohlen für Szenarien mit starken Diskontinuitäten.
Encoder Drift
Detach Targets
Wende .detach() auf den Target-State $z_{t+1}$ an. Der Encoder darf nicht verändert werden.
Oszillation
Konservative LR
Starte mit niedriger LR ($10^{-4}$ oder $10^{-5}$). ViTs reagieren empfindlicher als ResNets.

6. Evaluations-Metriken
Um den Erfolg des Protokolls zu verifizieren, werden folgende Metriken während des Rollouts aufgezeichnet:
Accumulated Rollout Loss: $\sum \| \hat{z} - z \|_1$ (Niedriger ist besser).
Vergleich zur Baseline: Performance des TTA-Modells vs. "Frozen Source Model" auf denselben Out-of-Distribution Daten.
Parameter Drift: Messung der L2-Distanz $\| \phi_{adapt} - \phi_{source} \|_2$ der LayerNorm-Parameter, um sicherzustellen, dass das Modell nicht "wegläuft".
Freigabe:
Dieses Protokoll dient als Grundlage für die Implementierung des RolloutTentAgent. Abweichungen im T-Horizont (z.B. $T=2$ im Loss) sind möglich, erhöhen aber die Rechenkomplexität und Latenz; $T=1$ wird als Startpunkt empfohlen.
