# Hardware-Anforderungen für das V-JEPA2 AC Predictor Training

Diese Analyse basiert auf der Konfiguration `configs/experiment/vjepa2_ac.yaml` und dem zugehörigen Modellcode.

## Zusammenfassung
Aufgrund der Deaktivierung von Activation Checkpointing und der Erhöhung von `T_rollout` auf 7 steigen die Anforderungen deutlich. Für **Batch Size 64** ist nun Enterprise-Class Hardware erforderlich.

| Komponente | Empfehlung | Begründung |
| :--- | :--- | :--- |
| **GPU (VRAM)** | **80 GB** (A100) | Durch `use_activation_checkpointing: false` und `T_rollout: 7` vervielfacht sich der Speicherbedarf für Aktivierungen. 24GB reichen für Batch Size 64 nicht mehr aus. |
| **Alternative GPU** | **24 GB** (RTX 3090/4090) | Nur nutzbar mit stark reduzierter Batch Size (z.B. **8-16** statt 64). |
| **GPU (Compute)** | **Tensor Cores** (Ampere/Ada/Hopper) | FlashAttention-Support ist kritisch für die Performance bei Sequenzlänge 2048. |
| **Storage** | **NVMe SSD** (>3000 MB/s lesen) | Der Dataloader streamt bei BS 64 ca. **500 MB/s** rohe Daten. Bei BS 16 sind es ~125 MB/s. |
| **CPU** | **8-16 Kerne** | 8 Worker-Prozesse (`num_workers: 8`) müssen die Daten parallel laden und decodieren. |
| **RAM** | **32 GB** | Ausreichend, da die Daten gestreamt werden und nicht komplett im Speicher liegen. |

---

## Detaillierte Analyse

### 1. GPU VRAM (Extremer Engpass)
Durch die Änderungen (`use_activation_checkpointing: false`, `T_rollout: 7`) verschiebt sich der Engpass massiv auf den VRAM.

*   **Impact von `use_activation_checkpointing: false`:**
    *   **Vorher:** Nur Aktivierungen weniger Layer wurden gespeichert (~50% Speicherersparnis im Tausch gegen Rechenzeit).
    *   **Jetzt:** Alle Intermediate-Activations aller 24 Layer müssen für den Backward-Pass im Speicher gehalten werden. Dies verursacht bei tiefen Transformern den Großteil des Speicherverbrauchs.
*   **Impact von `T_rollout: 7`:**
    *   Der Autoregressive Rollout läuft jetzt über die maximale Länge (8 Timesteps total = 1 Context + 7 Predictions).
    *   Dies maximiert den Computation Graph für die Rollout-Loss-Komponente, was zusätzlichen Speicher für Gradients benötigt.

**Kalkulation (Schätzung):**
*   Batch Size 64 × 2048 Tokens × 24 Layer × Hidden Size 1024 + Attention Matrices + Optimizer States + Gradients.
*   Mit Checkpointing `false` könnte der Bedarf für einen Batch Size 64 Forward/Backward-Pass auf **über 40-50 GB VRAM** ansteigen.

**Empfehlung:**
*   **Für Batch Size 64 (Original Setting):** NVIDIA A100 (80GB). Selbst die 40GB Version könnte knapp werden.
*   **Für RTX 3090/4090 (24GB):**
    *   **Reduzieren Sie die Batch Size auf ca. 16.**
    *   Gradient Accumulation kann genutzt werden, um effektiv BS 64 zu simulieren (z.B. `accumulate_grad_batches: 4` bei BS 16).
*   **Achtung:** Ohne Activation Checkpointing erreichen Sie schnelleres Training (höhere Throughput), aber nur wenn Sie genug VRAM haben. Wenn Sie OOM (Out of Memory) gehen, müssen Sie die Batch Size so weit reduzieren, dass der Speedup durch die fehlende volle Parallelisierung wieder negiert wird.

### 2. Speicher-Geschwindigkeit (Storage I/O)
Das Datamodule `src.datamodules.precomputed_features.PrecomputedFeaturesDataset` lädt bei jedem Trainingsschritt `.npy`-Dateien von der Festplatte. Es gibt kein Caching im RAM.

*   **Datenvolumen pro Batch:**
    *   1 Sample = 8 Timesteps × 256 Patches × 1024 Dim × 4 Bytes (Float32) ≈ **8 MB**.
    *   Batch Size 64 = 64 × 8 MB = **512 MB pro Batch**.
*   **Durchsatz:** Um 1 Batch pro Sekunde zu trainieren (realistisches Ziel), muss die Festplatte **512 MB/s dauerhaft lesen** können.
*   Zusätzlich entsteht Overhead durch das Öffnen vieler kleiner Dateien (IOPS).

**Empfehlung:**
*   Verwenden Sie zwingend eine interne **NVMe M.2 SSD**.
*   Vermeiden Sie das Training von externen USB-Festplatten oder Netzlaufwerken (NFS/SMB), da die Latenz beim Öffnen der vielen Dateien das Training extrem verlangsamt (GPU Utilization sinkt auf <50%).

### 3. CPU Anforderungen
Die Konfiguration nutzt `num_workers: 8`.

*   Jeder Worker lädt Daten, konvertiert Numpy-Arrays zu PyTorch-Tensoren und führt Reshapes durch.
*   Diese Arbeit ist I/O-lastig und speicherintensiv für den Datentransfer.
*   **8 physische Kerne** (oder starke vCPUs) sind empfohlen, um die GPU nicht verhungern zu lassen.

### 4. System RAM
Die Konfiguration erwähnt "tune these for your 32GB RAM".

*   Das Dataset wird **nicht** komplett in den RAM geladen (`__getitem__` lädt on-the-fly).
*   32 GB RAM sind daher völlig ausreichend, solange keine anderen speicherintensiven Prozesse laufen.
*   Wichtig: Linux nutzt freien RAM als Disk-Cache. Je mehr RAM frei ist, desto flüssiger läuft das Training bei wiederholten Epochen, da die `.npy` Files im Cache landen können.
