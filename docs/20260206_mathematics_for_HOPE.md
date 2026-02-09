# Mathematik hinter der HOPE-Architektur

Die HOPE-Architektur kombiniert **Self-Modifying Titans** (für kurzfristige, adaptive Anpassung) mit einem **Continuum Memory System (CMS)** (für langfristiges, hierarchisches Wissen).

## 1. Initialisierung und Query-Projektion

Zuerst wird für den Input-Token $x_t \in \mathbb{R}^d$ die Query $q_t$ berechnet. Dies ist die einzige Projektion, die nicht adaptiv ist (d.h. sie wird durch feste Gewichte $W_q$ bestimmt).

$$
q_t = x_t W_q
$$

## 2. Generierung der adaptiven Parameter (Self-Modifying Titans)

HOPE verwendet vergangene Memory-Zustände $\mathcal{M}_{t-1}$, um die Parameter für den aktuellen Zeitschritt selbst zu generieren. Dies umfasst Keys ($k$), Values ($v$), die Lernrate ($\eta$) und den Forget-Gate/Weight-Decay ($\alpha$).

Die Notation $\mathcal{M}_{\Box}$ steht hierbei für das jeweilige Speichermodul (z. B. Key-Memory, Value-Memory usw.).

$$
k_{t} = \mathcal{M}_{k,t-1}(x_{t})
$$
$$
v_{t} = \mathcal{M}_{v,t-1}(x_{t})
$$
$$
\eta_{t} = \mathcal{M}_{\eta,t-1}(x_{t})
$$
$$
\alpha_{t} = \mathcal{M}_{\alpha,t-1}(x_{t})
$$

Zusätzlich werden Zielwerte ($\hat{v}$) generiert, die für das Training der internen Module benötigt werden:

$$
\hat{v}_{\Box,t} = \mathcal{M}_{\Box,t-1}(v_{t})
$$

**Hinweis:** Die Struktur dieser Memory-Funktionen $\mathcal{M}(\cdot)$ ist oft ein 2-Layer MLP block:

$$
\mathcal{M}_{\Box}(\cdot) = (\cdot) + W_{\Box,1}\sigma(W_{\Box,2}(\cdot))
$$

## 3. Generierung des Outputs (Memory Abruf)

Bevor der Speicher aktualisiert wird, generiert das Modell den Output $o_t$ basierend auf dem aktuellen Query $q_t$ und dem Speicherzustand des vorherigen Schritts:

$$
o_{t} = \mathcal{M}_{memory,t-1}(q_{t})
$$

## 4. Update der Speicher (Self-Modification)

Alle Speichermodule (für Keys, Values, Outputs, Lernraten etc.) werden "in-context" aktualisiert. Die allgemeine Update-Regel, die einem Gradientenabstiegsschritt mit Momentum und Weight-Decay entspricht, lautet:

$$
\mathcal{M}_{\Box,t} = \mathcal{M}_{\Box,t-1}(\alpha_{t}I - \eta_{t}k_{t}k_{t}^{\top}) - \eta_{t}\nabla \mathcal{L}_{\mathcal{M}_{\Box,t-1}}(\mathcal{M}_{\Box,t-1}; k_{t}, \hat{v}_{\Box,t})
$$

Dabei steht $\Box$ für die Menge aller adaptiven Komponenten $\{k, v, q, \eta, \alpha, memory\}$.

Wenn als Verlustfunktion ($\mathcal{L}$) die $L_2$-Regression verwendet wird (wie im Text vorgeschlagen), konkretisiert sich der Gradiententeil der Formel zu:

$$
\mathcal{M}_{\Box,t} = \mathcal{M}_{\Box,t-1}(\alpha_{t}I - \eta_{t}k_{t}k_{t}^{\top}) - \eta_{t}(\mathcal{M}_{\Box,t-1}k_{t} - \hat{v}_{\Box,t})k_{t}^{\top}
$$

## 5. Verarbeitung durch das Continuum Memory System (CMS)

Der Output $o_t$ aus dem Titan-Modul wird anschließend durch eine Hierarchie von MLP-Blöcken geleitet. Diese Blöcke arbeiten auf verschiedenen Frequenzebenen (von schnell zu langsam), was als "Continuum Memory System" bezeichnet wird.

Der finale Output $y_t$ ergibt sich aus der sequenziellen Anwendung dieser MLP-Blöcke:

$$
y_{t} = \text{MLP}^{(f_{k})}(\text{MLP}^{(f_{k-1})}(\dots \text{MLP}^{(f_{1})}(o_{t})\dots))
$$

Hierbei repräsentiert $f_k$ die Frequenz des $k$-ten Blocks, wobei höhere Level oft niedrigere Update-Frequenzen haben, um persistenteres Wissen zu speichern.
