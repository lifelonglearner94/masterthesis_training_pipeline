# Phase 8: Subspace-Aware DGD for Highly Correlated Tasks — THEORETICAL PLAN

> **⚠️ STATUS: NICHT IMPLEMENTIERT — Nur theoretischer Plan.**
> **Erfordert empirische Validierung bevor Implementierung sinnvoll ist.**

**Date:** 2026-03-02
**Builds on:** Phase 7 (enhanced longterm memory)
**Motivation:** Catastrophic Forgetting bei Tasks mit Cosine-Similarity > 0.95

---

## 1. Problemanalyse

### 1.1 Warum hochkorrelierte Tasks besonders schwer sind

Bei Tasks mit > 95% Feature-Overlap versagen klassische CL-Methoden:

| Methode | Problem bei ähnlichen Tasks |
|---------|----------------------------|
| Task-Maskierung (HAT/Supermask) | Masken sind fast identisch → keine Isolation, Kapazitätsverschwendung |
| Replay | Hilft bei disjunkten Tasks; bei ähnlichen Tasks ist der Replay-Puffer redundant |
| Orthogonal Gradient Descent | Task-1-Subspace ≈ gesamter Raum → kein Platz für Task-2-Updates |
| CMS-Frequenz-Steuerung | CMS operiert auf Frames innerhalb eines Clips, nicht auf Task-Ebene |

### 1.2 Wo genau Forgetting in AC-HOPE-ViT entsteht

Drei Vergessens-Pfade existieren:

1. **Äußerer Optimizer (AdamW) auf `nn.Parameters`:** Finetuning auf neuen Clips überschreibt meta-gelernte Initialzustände
2. **DGD Inner-Loop auf M_memory:** Designed so (reset pro Clip) — kein Forgetting-Problem
3. **DGD Inner-Loop auf M_longterm:** Phase 7 adressiert dies bereits (asymmetric decay, own surprise)

**→ Pfad 1 ist der Hauptverdächtige für restliches Forgetting nach Phase 7.**

### 1.3 Kontext: Bewertung externer Vorschläge

Ein externer Review schlug drei Ansätze vor. Wissenschaftliche Bewertung:

| Ansatz | Note | Begründung |
|--------|:----:|-----------|
| Selective Masking der CMS Slow Weights | C+ | **Falscher Hebel:** CMS-Blocks sind Standard-MLPs mit Frame-level Scheduling, nicht Task-level Retention. `LevelSpec(period=1,4,16)` bezieht sich auf Frames innerhalb eines Clips. Task-Masken auf CMS adressieren nicht die Forgetting-Ursache. Bei 95% Feature-Overlap fragmentiert Maskierung die Kapazität statt sie zu nutzen. |
| Inner-Loop Orthogonalisierung (DGD) | B+ | **Richtiger Hebel, aber riskant:** Greift korrekt am DGD-Update in `compute_and_apply_update()` an. Problem: Bei 95% Overlap bleibt nur ~5% des Raums ($\approx 19$ von $384$ Dimensionen) orthogonal → Plastizität könnte einbrechen. |
| Dynamic CMS Frequencies | C | **Falscher Hebel:** CMS-Update-Frequenzen steuern Intra-Clip-Verarbeitung, nicht Inter-Task-Retention. |

---

## 2. Vorgeschlagene Phase-8-Mechanismen

### 2.1 Soft Orthogonal Projection im DGD Inner-Loop

**Ziel:** DGD-Gradienten teilweise orthogonal zu Task-N-Repräsentationen projizieren.

**Formalisierung:**

Sei $U_\text{prev} \in \mathbb{R}^{D \times r}$ die Matrix der Top-$r$ Principal Components der Schlüssel-Repräsentationen aus abgeschlossenen Tasks ($r \ll D$, z.B. $r = 20$).

Standard-DGD-Gradient:
$$\nabla L = \nabla_{W} \text{MSE}(W \cdot k, \hat{v})$$

Soft-projizierter Gradient:
$$\nabla L_\text{soft} = \nabla L - \beta \cdot U_\text{prev} U_\text{prev}^\top \nabla L$$

wobei $\beta \in [0, 1]$ die Projektionsstärke steuert:
- $\beta = 0$: Standard-DGD (Phase 7 Verhalten)
- $\beta = 1$: Volle Orthogonalisierung (maximaler Forgetting-Schutz, minimale Plastizität)
- $\beta = 0.3$: Empfohlener Startwert — entfernt 30% der Überlappungskomponente

**Ansatzpunkt im Code:**

```python
# In titan_memory.py → compute_and_apply_update(), nach grad-Berechnung:
# Aktuell (Phase 7):
grad = grad.detach()

# Phase 8 (Pseudo-Code):
if self._prev_subspace is not None and self.ortho_beta > 0:
    # U: [D, r] — gespeicherte Principal Components
    U = self._prev_subspace
    # Projiziere grad-Zeilen orthogonal zum Task-Subspace
    proj = grad @ U @ U.T  # Komponente IN Task-Subspace
    grad = grad - self.ortho_beta * proj  # Entferne Anteil β davon
```

**Principal Components sammeln:**

Nach Abschluss eines Tasks: Key-Statistiken aus dem DGD-Durchlauf aggregieren und SVD berechnen:

```python
# Pseudo-Code — am Task-Ende aufrufen:
def compute_task_subspace(key_buffer: Tensor, rank: int = 20) -> Tensor:
    """Berechne Top-r PCs der Schlüssel-Repräsentationen eines Tasks."""
    # key_buffer: [N_total, D] — alle Keys des Tasks (akkumuliert)
    key_centered = key_buffer - key_buffer.mean(dim=0)
    U, S, V = torch.linalg.svd(key_centered, full_matrices=False)
    return V[:rank, :].T  # [D, r]
```

**Risiko-Analyse:**

| Aspekt | Bewertung | Detail |
|--------|:---------:|--------|
| Forgetting ↓ | Wahrscheinlich | Gradient-Komponente im Task-1-Subspace wird reduziert |
| Plastizität ↓ | **Risiko** | Bei 95% Overlap: Soft-Projektion entfernt Gradient-Komponente die für neue Features relevant sein könnte |
| Speicher | +$r \times D$ Floats pro Memory pro Block | Bei $r=20, D=384$: ~7.7K Floats × 6 Memories × 5 Blocks = ~900KB |
| Latenz | Vernachlässigbar | Ein Matrix-Multiply pro DGD-Step |

### 2.2 EWC-Regularisierung auf nn.Parameters (Äußerer Optimizer)

**Ziel:** Fisher-Information-gewichtete Regularisierung verhindert, dass AdamW wichtige meta-gelernte Parameter überschreibt.

**Formalisierung:**

Standard-EWC (Kirkpatrick et al., 2017):
$$\mathcal{L}_\text{total} = \mathcal{L}_\text{task} + \frac{\lambda_\text{EWC}}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$$

wobei:
- $\theta^*$: Parameter-Snapshot nach Task N
- $F_i$: Diagonale Fisher-Information für Parameter $i$
- $\lambda_\text{EWC}$: Regularisierungsstärke

**Ansatzpunkt im Code:**

```python
# In ac_hope_module.py → training_step():
# Aktuell (Phase 7):
loss = reconstruction_loss + aux_weight * aux_loss

# Phase 8 (Pseudo-Code):
if self.ewc_lambda > 0 and self._fisher_diag is not None:
    ewc_loss = 0.0
    for name, param in self.named_parameters():
        if name in self._fisher_diag:
            fisher = self._fisher_diag[name]
            old_param = self._param_snapshot[name]
            ewc_loss += (fisher * (param - old_param) ** 2).sum()
    loss = loss + (self.ewc_lambda / 2) * ewc_loss
```

**Fisher-Berechnung am Task-Ende:**

```python
# Pseudo-Code — am Task-Ende aufrufen:
def compute_fisher_diagonal(model, dataloader, num_samples=500):
    """Diagonale Fisher-Information aus Task-N-Daten."""
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    for batch in itertools.islice(dataloader, num_samples):
        loss = model.training_step(batch)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data ** 2 / num_samples
    return fisher
```

**Risiko-Analyse:**

| Aspekt | Bewertung | Detail |
|--------|:---------:|--------|
| Forgetting ↓ | Wahrscheinlich | Wichtige Parameter werden geschützt |
| Plastizität ↓ | **Hohes Risiko** | Bei korrelierten Tasks sagt Fisher: "fast alles ist wichtig" → globale Regularisierung → kaum Adaptation möglich |
| $\lambda$-Tuning | Schwierig | Zu hoch → keine Plastizität; zu niedrig → kein Schutz |
| Speicher | +2× Modellgröße | Fisher-Diag + Parameter-Snapshot: ~93M Floats bei 46.5M Params |

---

## 3. Kombinationsstrategie

Die beiden Mechanismen adressieren **verschiedene Vergessens-Pfade**:

| Pfad | Mechanismus | Was es schützt |
|------|------------|---------------|
| DGD Inner-Loop (M_longterm) | Soft Orthogonal Projection | Longterm-Memory-Gewichte vor Überschreibung durch ähnliche Tasks |
| Äußerer Optimizer (AdamW) | EWC auf nn.Parameters | Meta-gelernte Initialzustände aller Module |
| DGD Inner-Loop (M_memory) | — (kein Schutz nötig) | Reset pro Clip, Forgetting by Design |

**Abhängigkeiten:**
- Beide Features sind unabhängig voneinander → einzeln oder zusammen aktivierbar
- Beide erfordern einen "Task-Ende"-Hook im CL-Pipeline für Subspace-/Fisher-Berechnung
- Phase 7 Consolidation (`consolidate_all_longterm_memories()`) kann am selben Hook aufgerufen werden

---

## 4. Konfigurationsentwurf

Alle Features per Default deaktiviert (backward-kompatibel):

```yaml
# configs/model/ac_hope_vit.yaml — neue Defaults:
dgd_ortho_projection_beta: 0.0      # 0.0 = disabled; empfohlen: 0.3
dgd_ortho_projection_rank: 20       # Anzahl PCs pro Task-Subspace
ewc_lambda: 0.0                     # 0.0 = disabled; empfohlen: 100-5000
ewc_online: false                    # true = Online-EWC (Running Fisher)
```

---

## 5. Ablation-Plan

| Experiment | Ortho $\beta$ | EWC $\lambda$ | Was es testet |
|-----------|:---:|:---:|------------|
| Phase 7 Baseline | 0.0 | 0.0 | Referenz |
| +Ortho sanft | 0.1 | 0.0 | Minimaler Subspace-Schutz |
| +Ortho mittel | 0.3 | 0.0 | Empfohlener Sweet Spot |
| +Ortho stark | 0.7 | 0.0 | Aggressiver Schutz (Plastizitäts-Risiko) |
| +EWC schwach | 0.0 | 100 | Minimale Parameter-Regularisierung |
| +EWC stark | 0.0 | 5000 | Starke Parameter-Regularisierung |
| +Beide | 0.3 | 1000 | Kombinierter Schutz beider Pfade |

**Metriken je Experiment:**
- Plastizität (Error auf aktuellem Task)
- Forgetting (Degradation auf vorherigen Tasks)
- Forward Transfer (wie schnell lernt Task N+1 vs. von Scratch)
- Gesamtfehler (gewichteter Durchschnitt)

---

## 6. Erwartete Herausforderungen

### 6.1 Plastizitäts-Risiko bei Soft Orthogonalisierung

Bei Task-Ähnlichkeit > 0.95 spannen die Top-20 PCs von Task 1 bereits ~99% der Varianz auf. Selbst mit $\beta = 0.3$ wird ein erheblicher Teil des Gradients entfernt. **Mögliche Abschwächung:**

- **Adaptives $\beta$**: $\beta_t = \beta_0 \cdot \sigma(\text{task\_similarity})$ — bei hoher Ähnlichkeit automatisch reduzieren
- **Nur auf M_longterm anwenden**: M_memory-Updates bleiben ungeprojiziert (Plastizität intakt)
- **Niedrigerer Rank**: $r = 5$ statt 20 — weniger Dimensionen gesperrt

### 6.2 Fisher-Degeneration bei korrelierten Tasks

EWC-Fisher bei korrelierten Tasks: $F_i \approx c$ für fast alle $i$ (uniform hoch) → $\lambda_\text{EWC} \cdot F_i$ wirkt wie globaler $L_2$-Decay, nicht wie selektiver Schutz.

**Mögliche Abschwächung:**
- **Online-EWC** (Schwarz et al., 2018): Running Average der Fisher-Diagonale über Tasks statt Snapshots
- **Nur auf CMS-Parameter anwenden**: Titan-Memory-Parameter sind meta-gelernte Initialzustände → EWC hier sinnvoller als auf DGD-Abkömmlinge
- **Sparsified Fisher**: Nur Top-$k$% der Fisher-Werte behalten, Rest auf 0

### 6.3 Interaktion mit Phase 7

Phase 7's asymmetrische Decay ($\alpha_\text{min} = 0.95$) schützt M_longterm bereits partiell. Soft Orthogonalisierung im DGD wäre ein **zusätzlicher** Schutz auf derselben Ebene → mögliche Redundanz:

- Wenn $\alpha_\text{min}$ ausreicht → Ortho bringt wenig Zusatznutzen
- Wenn $\alpha_\text{min}$ nicht ausreicht → Ortho könnte den Unterschied machen

**→ Ablation klärt, welcher Mechanismus dominiert.**

---

## 7. Vorhandene Code-Bausteine

Relevante existierende Implementierungen, die wiederverwendet werden können:

| Baustein | Datei | Was es tut |
|---------|-------|------------|
| Rank-1 Orthogonal-Projektion | `src/models/hope/m3_optimizer/deep_momentum.py` → `_nl_precondition()` | Projiziert Gradient orthogonal zu Context-Vektor |
| DGD-Update mit Gradient-Manipulation | `src/models/hope/titan_memory.py` → `compute_and_apply_update()` | Exakter Ansatzpunkt für Soft-Orthogonalisierung |
| Task-Boundary-Hook | `ac_hope_vit.py` → `consolidate_all_longterm_memories()` | Pattern für Task-Ende-Aktionen bereits vorhanden |
| Config-System | `HOPEBlockConfig` Dataclass | Pattern für opt-in Features mit Defaults |

---

## 8. Entscheidungskriterien für Implementierung

Phase 8 sollte implementiert werden, **wenn**:

1. Phase 7 Experimente abgeschlossen sind und Forgetting weiterhin > 0.03 auf abgeschlossenen Tasks
2. Analyse bestätigt, dass Vergessens-Quelle primär Pfad 1 (AdamW auf nn.Parameters) oder M_longterm DGD ist
3. Phase 7 Plastizität akzeptabel ist (≤ 0.28) — andernfalls wäre Ortho/EWC kontraproduktiv

**Phase 8 sollte NICHT implementiert werden, wenn:**
- Phase 7 bereits Forgetting ≤ 0.035 erreicht → kein Bedarf
- Plastizität in Phase 7 grenzwertig ist → zusätzliche Regularisierung wäre schädlich
- Task-Ähnlichkeit so hoch ist, dass der orthogonale Subspace praktisch leer wird
