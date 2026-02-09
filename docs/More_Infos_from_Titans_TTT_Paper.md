Basierend auf dem Paper "Titans: Learning to Memorize at Test Time" sind hier die essentiellen Details, um die **Titans-Komponente** der HOPE-Architektur von Grund auf zu bauen und zu trainieren.  
Das Herzstück ist das **Neural Long-Term Memory Module**, das während der Testphase ("Test Time") weiter lernt (d.h. seine Gewichte aktualisiert), was dem "Inner Loop" in Nested Learning entspricht.

### **1\. Das Kernmodul: Neural Long-Term Memory (LMM)**

Anstatt eines statischen Zustandsvektors (wie bei RNNs) ist das Gedächtnis selbst ein **neuronales Netzwerk (MLP)**, dessen Gewichte $M\_t$ sich über die Zeit ändern.

#### **Architektur des Speichers**

* **Struktur:** Ein einfaches MLP (Multi-Layer Perceptron) mit $L\_M \\ge 1$ Schichten. Tiefe Speicher ($L\_M \\ge 2$) werden empfohlen, um nicht-lineare Abhängigkeiten zu modellieren.  
* **Input-Projektionen:** Ähnlich wie bei Attention wird der Input $x\_t$ in drei Vektoren projiziert:  
  * Query: $q\_t \= x\_t W\_Q$ (zum Abrufen)  
  * Key: $k\_t \= x\_t W\_K$ (zum Lernen/Speichern)  
  * Value: $v\_t \= x\_t W\_V$ (zum Lernen/Speichern)

#### **Der Lernprozess (Update Rule)**

Das Modul lernt "in-context", indem es den Fehler bei der Rekonstruktion von Key-Value-Paaren minimiert. Dies ist der "Surprise"-Mechanismus.

1. **Objective (Verlustfunktion):** Das Modul versucht, die Zuordnung von Keys zu Values zu lernen.  
   $$\\ell(M\_{t-1}; x\_t) \= \\| M\_{t-1}(k\_t) \- v\_t \\|\_2^2$$  
2. **Berechnung der "Surprise" (Gradient):** Die Überraschung ist definiert als der Gradient des Fehlers bezüglich der aktuellen Speicher-Gewichte.  
   $$\\nabla \\ell(M\_{t-1}; x\_t)$$  
3. **Momentum-basiertes Update:** Um kurzfristige Schwankungen auszugleichen und langfristige Abhängigkeiten zu erfassen, wird ein Momentum $S\_t$ genutzt.  
   $$S\_t \= \\eta\_t S\_{t-1} \- \\theta\_t \\nabla \\ell(M\_{t-1}; x\_t)$$  
   * $\\eta\_t$: Data-dependent Surprise Decay (bestimmt, wie lange eine Überraschung im "Momentum-Gedächtnis" bleibt).  
   * $\\theta\_t$: Lernrate für den aktuellen Schritt.  
4. **Vergessen (Forgetting Mechanism):** Ein expliziter Decay-Term $\\alpha\_t$ (Weight Decay) verhindert Speicherüberlauf.  
   $$M\_t \= (1 \- \\alpha\_t) M\_{t-1} \+ S\_t$$  
   * Wenn $\\alpha\_t \\to 1$, wird das alte Wissen gelöscht (Reset).

*Hinweis:* Die Parameter $\\alpha\_t, \\eta\_t, \\theta\_t$ sind oft datenabhängig (Funktionen von $x\_t$) oder chunk-abhängig, um die Dynamik zu steuern.

#### **Abrufen (Inference)**

Um Informationen aus dem Speicher zu lesen, wird einfach ein Forward-Pass mit dem Query-Vektor durchgeführt (ohne Gewichtsupdate):

$$y\_t \= M\_t^\*(q\_t)$$

### ---

**2\. Integration in die Architektur (Die 3 Varianten)**

Das Paper stellt drei Wege vor, dieses Speichermodul mit einem kurzfristigen Speicher (Attention) zu kombinieren. Für HOPE ist besonders "Memory as a Context" interessant.

#### **A. Memory as a Context (MAC) – *Empfohlen für komplexe Zusammenhänge***

Hier dient das Langzeitgedächtnis als Kontext für den Attention-Mechanismus.

1. **Chunking:** Die Sequenz wird in Segmente $S^{(t)}$ unterteilt.  
2. **Retrieval:** Für das aktuelle Segment wird relevanter Kontext aus dem Speicher $M\_{t-1}$ abgerufen: $h\_t \= M^\*\_{t-1}(q\_t)$.  
3. **Konkatenation:** Der abgerufene Kontext wird mit dem aktuellen Segment und einem "Persistent Memory" (siehe unten) verbunden.  
   $$\\tilde{S}^{(t)} \= \[\\text{Persistent Memory} \\ || \\ h\_t \\ || \\ S^{(t)}\]$$  
4. **Attention:** Ein Standard-Attention-Modul verarbeitet diese erweiterte Sequenz.  
5. **Update:** Das Ergebnis wird genutzt, um das Gedächtnis für den nächsten Schritt zu aktualisieren.

#### **B. Memory as a Gate (MAG)**

Gedächtnis und Attention laufen parallel.

* Zweig 1: Sliding Window Attention (kurzfristig).  
* Zweig 2: Neural Memory (langfristig).  
* Kombination: Die Ausgaben werden durch ein Gating (z.B. Sigmoid-gewichtet) fusioniert: $o \= y \\otimes M(\\tilde{x})$.

#### **C. Memory as a Layer (MAL)**

Sequenzielles Stapeln: Das Neural Memory ist eine Schicht, deren Ausgabe in eine Attention-Schicht fließt (oder umgekehrt).

### ---

**3\. Zusätzliche Komponenten**

* **Persistent Memory:** Ein Satz lernbarer, aber *daten-unabhängiger* Parameter $P$, die vor die Sequenz geschaltet werden ($x\_{new} \= P || x$). Sie speichern statisches Wissen über den Task (ähnlich wie "System Prompts" oder festes Weltwissen).  
* **Architektur-Details:**  
  * Verwendung von **SiLU** Aktivierung.  
  * **L2-Norm** für Queries und Keys.  
  * **1D Depthwise-Separable Convolutions** nach den Q, K, V Projektionen (für lokale Glättung).

### **4\. Training (Parallelisierung)**

Das Training des Neural Memory ist im Prinzip sequenziell (rekursiv), kann aber für GPUs parallelisiert werden ("Chunk-wise Parallel Training"):

* **Idee:** Zerlege die Sequenz in Chunks. Innerhalb eines Chunks werden die Updates akkumuliert.  
* **Matrix-Formulierung:** Das Update (Gradient Descent mit Momentum & Decay) lässt sich als Matrix-Operation umschreiben. Für einen Chunk der Größe $b$ gilt:  
  $$M\_t \= \\beta\_t M\_0 \- \\sum\_{i=1}^t \\frac{\\theta\_i \\beta\_t}{\\beta\_i} \\nabla \\ell(M\_{t'}; x\_i)$$  
  (wobei $\\beta$ den kumulativen Decay darstellt).  
* **Parallel Associative Scan:** Für den Momentum-Term $S\_t$ kann ein paralleler Scan-Algorithmus verwendet werden, was das Training extrem beschleunigt (ähnlich wie bei Mamba/S4).

**Zusammenfassend für den Bau von HOPE:** Bauen Sie das "Neural Long-Term Memory" als MLP, implementieren Sie die Update-Regel (Gleichungen 9-14) als differenzierbare Operation innerhalb des Forward-Passes, und nutzen Sie die "Memory as a Context" (MAC) Architektur, um es mit Attention zu verbinden.