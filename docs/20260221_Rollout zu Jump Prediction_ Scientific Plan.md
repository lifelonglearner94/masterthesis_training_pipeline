

# **Scientific Plan: Transition from Autoregressive Rollout to Jump Prediction in AC-HOPE-ViT**

## **1\. Abstract & Zielsetzung (Was wird gemacht?)**

Im Rahmen der Entwicklung adaptiver Weltmodelle basierend auf V-JEPA 2 Features und der HOPE-Architektur wird das Trainingparadigma des Predictors modifiziert. Der rechenintensive, autoregressive **Rollout Loss** wird vollständig durch eine stochastische **Jump Prediction** ersetzt.  
Anstatt ausgehend von Zeitschritt $1$ sequenziell bis Zeitschritt $T$ zu iterieren, lernt das Modell, direkt aus dem Initialzustand ($z\_1$) und der initialen Aktion ($a\_1$) in einen zufällig gewählten Zeitschritt $\\tau$ am Ende der Sequenz zu springen (z. B. Frame 6, 7 oder 8 bei einer 8-Frame-Sequenz).

## **2\. Rationale (Warum machen wir das?)**

Der Wechsel ist durch drei primäre Faktoren motiviert, die perfekt mit der Physik-Simulation und der HOPE-Architektur harmonieren:

* **Eliminierung von Error Compounding & Exposure Bias:** Autoregressive Vorhersagen tendieren dazu, Fehler bei jedem Schritt aufzuaddieren. Bei Physikdaten mit hohen Framerates führt das oft dazu, dass das Modell einfach den Vorherigen Frame kopiert (Copy-Paste-Heuristik). Die direkte Vorhersage weit in die Zukunft zwingt die latenten Features, tiefere physikalische Gesetzmäßigkeiten (Impulserhaltung, Reibungsverlust) zu abstrahieren.  
* **Synergie mit dem Continuum Memory System (CMS):** Da die HOPE-Architektur in diesem Setup über Clips hinweg (Inter-Clip) adaptiert und die Aktion nur zum Zeitpunkt $t=1$ stattfindet, liefert ein "Jump" ein viel klareres, makroskopisches Fehlersignal. Wenn sich in der Test-Time Adaptation (Phase B) die Reibung ändert, ist der Vorhersagefehler bei Frame 8 signifikant höher als bei Frame 2\. Dieses starke Fehlersignal ist ideal, um das **Surprise Gating** der Delta Gradient Descent (DGD) in der Titan Memory auszulösen.  
* **Massive Reduktion der Komplexität:** Ein einziger Forward-Pass pro Trainingsiteration statt $T-1$ Passes. Dies schont VRAM, beschleunigt die Konvergenz und erlaubt größere Batch-Sizes.

## ---

**3\. Mathematische Formulierung (Formeln)**

### **Bisheriger Zustand: Autoregressiver Rollout**

Bisher wurde der latente Zustand iterativ berechnet, wobei die Vorhersage des letzten Schrittes bestraft wurde:

$$\\mathcal{L}\_{\\text{rollout}}(\\phi) := || P\_{\\phi}(a\_{1:T}, s\_{1}, z\_{1}) \- z\_{T+1} ||\_{1}$$  
*(Nachteil: Erfordert eine Schleife über alle Zeitschritte, Rechenaufwand skaliert linear mit $T$.)*

### **Neuer Zustand: Stochastic Jump Prediction**

Wir definieren einen Ziel-Zeitschritt $\\tau$, der uniform aus den letzten $k$ Frames der Sequenz (Länge $T$) gezogen wird. Für $T=8$ und $k=3$ gilt $\\tau \\in \\{6, 7, 8\\}$.  
Da die Aktion (Kraftimpuls) nur zum Startzeitpunkt wirkt, vereinfacht sich der Input auf $z\_1$ und $a\_1$. Um dem Modell mitzuteilen, *welchen* Frame es generieren soll, nutzen wir das 3D Rotary Positional Embedding (RoPE) der V-JEPA 2 Architektur des Ziel-Frames, markiert als $p\_{\\tau}$.  
Der neue Loss definiert sich als Erwartungswert über die Verteilung der Ziel-Frames:

$$\\mathcal{L}\_{\\text{jump}}(\\phi) := \\mathbb{E}\_{\\tau \\sim \\mathcal{U}(T-k+1, T)} \\left\[ || P\_{\\phi}(z\_1, a\_1, p\_{\\tau}) \- z\_{\\tau} ||\_{1} \\right\]$$

### **Gesamt-Loss Funktion (Curriculum Learning)**

Das Training nutzt weiterhin einen Teacher-Forcing Loss (1-Step Vorhersage) als stabilisierende Basis, um die unmittelbare Pixel/Feature-Dynamik zu lernen. Die Gesamt-Loss-Funktion wird durch einen dynamischen Gewichtungsfaktor $\\lambda(e)$ gesteuert, der über die Epochen $e$ hinweg den Fokus von Teacher-Forcing auf Jump Prediction verschiebt:

$$L(\\phi, e) := (1 \- \\lambda(e)) \\cdot \\mathcal{L}\_{\\text{teacher-forcing}} \+ \\lambda(e) \\cdot \\mathcal{L}\_{\\text{jump}}$$  
Wobei $\\lambda(e)$ monoton von einem Startwert (z.B. 0.2) auf einen Maximalwert (z.B. 0.8) ansteigt.

## ---

**4\. Architektonische Umsetzung im ViT-AC Predictor**

| Komponente | Änderung | Begründung |
| :---- | :---- | :---- |
| **Input-Token** | Nur noch $z\_1$ und $a\_1$ pro Clip verarbeiten. | Aktion wirkt nur bei $t=1$, restliche Steps sind reine Trägheits-Dynamik. |
| **Conditioning** | Addition von RoPE($\\tau$) auf die Query-Tokens im Transformer. | Das Modell muss "wissen", wie weit es in die Zukunft blicken soll. |
| **HOPE Blocks** | Keine Änderung nötig (Drop-In). | Das CMS adaptiert sich weiterhin basierend auf dem $\\mathcal{L}\_{\\text{jump}}$ über aufeinanderfolgende Clips. |

