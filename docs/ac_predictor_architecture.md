Hier ist eine detaillierte Analyse des **ViT-AC-Predictor** (Vision Transformer - Action/Conditioned - Predictor), basierend auf den bereitgestellten Logs.

Die Architektur ist ein **autoregressives Transformer-Modell**, das visuelle Features (Patches), Aktionen und States in einem gemeinsamen Latent-Space verarbeitet, um zukünftige visuelle Features vorherzusagen.

---

### 1. Detaillierter Ablaufplan (Der Flow)

Dieser Ablauf beschreibt den "Forward Pass", wie er in den Logs (sowohl beim *Teacher Forcing* als auch beim *Rollout*) zu sehen ist.

#### Schritt 1: Input & Normalisierung

* **Visuelle Features:** Der Input sind extrahierte Feature-Maps (wahrscheinlich von einem DINOv2 oder ähnlichen Encoder).
* Shape:  (Batch=4, Time=7, Patches=256, Dim=1024).
* Die Features werden abgeflacht zu . Im Log entspricht das `[4, 1792, 1024]` (da ).
* **LayerNorm:** Eine Normalisierung wird direkt auf die Features angewendet.



#### Schritt 2: Embedding & Projektion (Encoder)

Hier werden alle Inputs auf die interne Dimension des Transformers () gebracht.

* **Feature Projection (`predictor_embed`):** Die visuellen Features (1024 dim) werden via Linear Layer auf 384 dim projiziert.
* **Action Encoding (`action_encoder`):** Die Aktionen (2 dim) werden auf 384 dim projiziert.
* **State Encoding (`state_encoder`):** Die States (2 dim) werden auf 384 dim projiziert.

#### Schritt 3: Token Interleaving (Fusion)

Dies ist ein entscheidender architektonischer Schritt. Die Konditionierung (Aktionen/States) wird nicht einfach vorne angehängt, sondern **zeitlich verschränkt**.

* Pro Zeitschritt (Frame) gibt es 256 Bild-Tokens.
* Pro Zeitschritt werden 2 Konditionierungs-Tokens (1x Action, 1x State) hinzugefügt.
* **Rechnung:** .
* **Resultat:** Ein Tensor der Form `[4, 1806, 384]`. Das Modell weiß nun lokal an jedem Zeitschritt, welche Aktion zu welchem Bild gehört.

#### Schritt 4: Transformer Backbone (Deep Processing)

Die Sequenz durchläuft nun den tiefen Transformer-Stack.

* **Tiefe:** 24 Blöcke (`Block 0` bis `Block 23`).
* **Architektur pro Block:**
1. **LayerNorm 1**
2. **Attention (`ACRoPEAttn`):**
* Hier wird **Rotary Positional Embedding (RoPE)** verwendet (erkennbar am Modulnamen). Das erlaubt dem Modell, relative Positionen besser zu verstehen, was für Zeitreihen essentiell ist.
* Heads: 16 Attention Heads.
* Dimension pro Head: .


3. **Residual Connection**
4. **LayerNorm 2**
5. **MLP (Feed Forward):** Expansion auf 1536 dim () und Projektion zurück auf 384.
6. **Residual Connection**



#### Schritt 5: Output Handling (Decoder)

Nach den 24 Blöcken muss die Sequenz bereinigt werden, um nur die visuellen Vorhersagen zu erhalten.

* **Remove Action Tokens:** Die beim Interleaving hinzugefügten Action/State-Tokens werden entfernt. Die Sequenzlänge schrumpft von 1806 zurück auf 1792.
* **Final Norm (`predictor_norm`):** LayerNorm Abschluss.
* **Final Projection (`predictor_proj`):** Projektion von der latenten Dimension () zurück auf die originale Feature-Dimension ().

#### Schritt 6: Loss Berechnung

* **Teacher Forcing (Training):** Das Modell sagt alle Zeitschritte parallel vorher. Der Loss ist die Differenz zwischen Vorhersage und Ground Truth ( Loss laut Log: `|diff|^1.0`).
* **Rollout (Inference/Validation):** Das Modell sagt Schritt  vorher, nutzt diesen Output als Input für , usw. (Autoregressive Schleife).

---

### 2. Architekturdiagramm (Markdown/Mermaid)

```mermaid
graph TD
    subgraph Inputs
        IMG[Input Features\n(Batch, Time, 256, 1024)]
        ACT[Actions\n(Batch, Time, 2)]
        STATE[States\n(Batch, Time, 2)]
    end

    subgraph Encoder_Stage["Encoder Stage (Dim Transformation)"]
        LN_IN[LayerNorm]
        PROJ_IMG[Linear Projection\n1024 -> 384]
        PROJ_ACT[Action Encoder\n2 -> 384]
        PROJ_STATE[State Encoder\n2 -> 384]
    end

    subgraph Sequence_Construction["Sequence Construction"]
        INTERLEAVE[Token Interleaving\nInsert Action/State tokens\ninto sequence per timestep\nSeq Len: 1792 -> 1806]
        MASK[Attention Mask\n(Causal/Full)]
    end

    subgraph Vision_Transformer_Backbone
        direction TB
        BLOCK_0[Transformer Block 0\n(ACRoPEAttn + MLP)]
        BLOCK_MID[... Blocks 1-22 ...]
        BLOCK_23[Transformer Block 23\n(ACRoPEAttn + MLP)]

        info1[Details:\n- 24 Layers\n- Dim: 384\n- Heads: 16\n- RoPE Embeddings]
    end

    subgraph Decoder_Stage["Decoder Stage"]
        REMOVE[Remove Tokens\nDiscard Action/State tokens\nSeq Len: 1806 -> 1792]
        LN_OUT[Predictor Norm\nLayerNorm]
        PROJ_OUT[Output Projection\nLinear 384 -> 1024]
    end

    OUT[Predicted Features\n(Batch, Time, 256, 1024)]
    LOSS((L1 Loss))

    %% Connections
    IMG --> LN_IN --> PROJ_IMG
    ACT --> PROJ_ACT
    STATE --> PROJ_STATE

    PROJ_IMG --> INTERLEAVE
    PROJ_ACT --> INTERLEAVE
    PROJ_STATE --> INTERLEAVE

    INTERLEAVE --> BLOCK_0
    MASK -.-> BLOCK_0
    BLOCK_0 --> BLOCK_MID --> BLOCK_23

    BLOCK_23 --> REMOVE
    REMOVE --> LN_OUT --> PROJ_OUT --> OUT
    OUT --> LOSS

    style INTERLEAVE fill:#f9f,stroke:#333,stroke-width:2px
    style BLOCK_0 fill:#e1f5fe,stroke:#333
    style BLOCK_23 fill:#e1f5fe,stroke:#333
    style info1 fill:#fff,stroke:#999
```

### 3. Analyse der Log-Spezifika

Hier sind spezifische Beobachtungen aus den Logs, die für das Verständnis wichtig sind:

1. **Hohe Varianz in den Inputs:**
Die Input Features haben Werte zwischen `-40` und `+38` (siehe `Context stats`). Die LayerNorm (`After LayerNorm`) normalisiert dies effektiv auf einen Bereich um `[-12, 12]`, was für den Transformer bekömmlicher ist.
2. **Dimensionalität:**
Der Wechsel von der Feature-Dimension `1024` auf die Transformer-Dimension `384` ("Bottleneck") spart extrem viel Rechenleistung. Das Modell hat "nur" 43.4 Millionen Parameter, was für einen ViT mit 24 Layern sehr schlank ist (üblich sind oft >80M oder >300M bei Dim 768/1024). Das macht es schnell (`Modules in train mode: 344`).
3. **RoPE Attention:**
Die Logs zeigen `[ACRoPEAttn]`. Rotary Embeddings sind besser als absolute Position Embeddings für Sequenzen unterschiedlicher Länge (wichtig beim Rollout, wo die Sequenz wächst: `Step 1` -> `Step 2` -> `Step 3`).
4. **Verlustfunktion:**
Der Loss (L1) liegt bei ca. `1.08`. Da die Daten normalisiert sind, ist das ein solider Wert, bedeutet aber, dass die Vorhersage im Durchschnitt noch um 1.0 (in normalisierten Einheiten) abweicht.

Das Modell ist also ein spezialisierter, effizienter **Conditional Video Predictor**, der darauf ausgelegt ist, lange Sequenzen (durch die kleine Dimension 384) stabil vorherzusagen.
