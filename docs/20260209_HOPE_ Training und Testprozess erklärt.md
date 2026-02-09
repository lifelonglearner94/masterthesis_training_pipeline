### **1\. Grundphilosophie: Abschaffung der Trennung von Training und Test**

Das Paper argumentiert, dass Modelle als "Neural Learning Modules" betrachtet werden sollten, die kontinuierlich lernen.

* **Keine Unterscheidung:** In einem Nested Learning Modul gibt es keine harte Grenze. Das Modell befindet sich entweder in einem Zustand, in dem es Informationen verarbeitet (und dabei lernt), oder es ist isoliert."For a neural learning module, there is no boarder and clear distinction between training and test time. The model only experiences two different states: when it receives information as input, or when it is an isolated learning system."

* **Pre-Training als In-Context Learning:** Das traditionelle Pre-Training wird lediglich als die Optimierung der langsamsten Ebene (niedrigste Frequenz) über einen extrem langen Kontext betrachtet."From NL's viewpoint pre-training is only one of the possible instances of in-context learning, where the context is the entire pre-training data."

### ---

**2\. Der Trainingsprozess der HOPE-Architektur**

HOPE kombiniert zwei Hauptkomponenten: **Self-Modifying Titans** (schnelle Anpassung) und das **Continuum Memory System (CMS)** (langfristige Speicherung). Der Prozess läuft wie folgt ab:

#### **A. Initialisierung (Meta-Learning)**

Bevor der eigentliche Datenstrom verarbeitet wird, müssen die Startzustände der Speicher ("Initial States") gelernt werden. Dies geschieht oft durch eine "Outer-Loop"-Optimierung (Meta-Learning).  
"The initial states of all memories, i.e., $M\_{\\Box0}$ for any $\\in \\{k, v, q, \\eta, \\alpha, \\text{memory}\\}$ are meta-learned across all sequences/contexts, and so are optimized in the higher levels (or outer-loop)."

#### **B. Chunk-wise Processing (Parallele Verarbeitung)**

Um das Training effizient zu gestalten, werden Sequenzen in Chunks (Blöcke) unterteilt. Dies ermöglicht Parallelisierung trotz rekurrenter Natur.  
"We follow the chunk-wise training algorithm... given an input sequence $\\{x\_{t}\\}\_{t=1}^{L}$ and chunk size $1 \\le C \\le L$; we split the sequence into $\\lceil\\frac{L}{C}\\rceil$ chunks... and then generate all elements... at the end of each chunk for the next chunk."  
\+1

#### **C. Der Update-Schritt (Der eigentliche Lernvorgang)**

HOPE nutzt einen **selbst-referenziellen Mechanismus**. Das Modell generiert seine eigenen Lernraten ($\\eta$), Vergessensraten ($\\alpha$) und sogar die Zielwerte für die Optimierung.

1. **Forward Pass & Parameter-Generierung:** Das Modell sagt für den aktuellen Zeitschritt $t$ nicht nur den Output vorher, sondern generiert auch die Hyperparameter für sein eigenes Update."$\\eta\_{t}=\\mathcal{M}\_{\\eta,t-1}(x\_{t})$" (Lernrate) "$\\alpha\_{t}=M\_{\\alpha,t-1}(x\_{t})$" (Decay/Vergessen)  
   \+1

2. **Optimierung mittels Delta Gradient Descent (DGD):** Anstatt eines einfachen Gradient Descent wird DGD verwendet, das den vorherigen Zustand der Gewichte berücksichtigt (besseres Memory Management)."This new algorithm... updates the weights not only with respect to the current elements, but it also incorporates the previous state of weights, resulting in an adaptive decay term based on the current data sample."

3. **Update der Speicher:** Die Speicher-Matrizen ($M$) werden basierend auf dem generierten Fehler aktualisiert."$M\_{\\Box,t}=M\_{\\Box,t-1}(\\alpha\_{t}I-\\eta\_{t}k\_{t}k\_{t}^{\\top})-\\eta\_{t}\\nabla\\mathcal{L}...$"

#### **D. Multi-Level Updates (CMS)**

Zusätzlich zu den schnellen Updates in den Titans gibt es das **Continuum Memory System (CMS)**. Hier werden MLPs (Feed-Forward Netzwerke) in verschiedenen Intervallen aktualisiert.  
"Continuum Memory System (CMS) is formalized as a chain of MLP blocks... where the parameters of $l$-th MLP block... are updated every $C^{(l)}$ steps"  
\+1

Das bedeutet:

* Einige Parameter werden bei jedem Token aktualisiert (hohe Frequenz).  
* Andere Parameter werden nur alle $X$ Schritte aktualisiert (mittlere Frequenz).  
* Dies verhindert "Catastrophic Forgetting", da altes Wissen in langsameren Schichten erhalten bleibt.

### ---

**3\. Die "Testphase" (Inference / Continual Learning)**

In der traditionellen Deep-Learning-Sichtweise werden die Gewichte nach dem Training eingefroren. In der Nested Learning Architektur (HOPE) läuft der Optimierungsprozess während der Testphase (Inference) einfach weiter, jedoch oft auf einer höheren Frequenzebene (In-Context Learning).

* **Test-Time Training (TTT) ist In-Context Learning:** Wenn das Modell während der "Testphase" neue Daten sieht, führt es weiterhin Gradienten-Updates auf seinen internen Speichern (Memory States) durch."The concepts commonly referred to as test-time training and test-time memorization are in fact instances of parametric in-context learning, where the acquired in-context knowledge does not persist once the current context is removed."

* **Continual Learning Setup:** Das Modell passt sich *in-context* an. Es generiert weiterhin seine eigenen Lernraten und aktualisiert seine Matrix-Speicher basierend auf dem Kontext, den es gerade liest (z.B. ein langes Buch oder eine neue Sprache)."Therefore, from NL perspective, all the levels are performing in-context learning but on their own context flow with their own learning update and optimization process."

### **Zusammenfassung des Ablaufs**

1. **Start:** Initialisierung der Speicher-Matrizen (oft durch Meta-Learning auf großen Datensätzen gelernt).

2. **Laufender Prozess (Training & Test identisch):**  
   * Input $x\_t$ kommt rein.  
   * **Level 1 (Schnell):** Self-Modifying Titans generieren Lernparameter ($\\eta, \\alpha$) und aktualisieren ihren internen Zustand sofort mittels Delta Gradient Descent.

   * **Level 2+ (Langsam):** CMS-Blöcke sammeln Gradienten/Informationen und führen ein Update ihrer Gewichte nur alle $C$ Schritte durch.

   * **Output:** Das Ergebnis ist eine Funktion durch alle diese verschachtelten (nested) und aktualisierten Speicher.  
3. **Wissens-Transfer:** Wissen fließt von schnellen Speichern zu langsamen Speichern (durch Backpropagation oder direkten Transfer), wodurch kurzfristige Anpassung zu langfristigem Wissen wird.

Der entscheidende Unterschied zu klassischen Transformatoren ist, dass HOPE sich **selbst modifiziert** während es Daten verarbeitet (auch im "Test"-Modus), anstatt statische Gewichte zu verwenden.  
"NL suggests a philosophy to design more expressive learning algorithms with more 'levels', resulting in higher-order in-context learning and potentially unlocking effective continual learning capabilities."  
