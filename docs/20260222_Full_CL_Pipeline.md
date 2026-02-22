### **Title:** Comparative Continual Learning Pipeline Under Progressive Dynamic Shifts

#### **1. Overview and Objective**

The diagram outlines a comparative experimental study designed to evaluate the performance of two distinct learning-based control architectures—**AC-VIT with TTA** and **AC-HOPE**—in a continual learning setting. The objective is to assess their ability to adapt sequentially to five increasingly complex dynamic environments (tasks) following an initial base training phase, emphasizing their capability for forward transfer and resistance to catastrophic forgetting.

#### **2. Experimental Architectures**

The study compares two methods:

* **AC-VIT, TTA:** Likely an framework utilizing a Vision Transformer (ViT) architecture, augmented with Test-Time Adaptation (TTA).
* **AC-HOPE:**

#### **3. The Main Experimental Pipeline (Upper Section)**

Both architectures undergo identical training protocols, processed in parallel streams.

**3.1. Base Training Phase**
The pipeline begins with a common initialization phase termed **"Base Training."**

* **Data:** 5000 Clips.
* **Duration:** 40 Epochs (noted with a question mark, indicating a tentative parameter).

**3.2. Sequential Task Curriculum**
Following base training, the models are subjected to a curriculum of five distinct tasks, representing progressive "shifts" in environment dynamics. Each task phase consists of a **Training** session specific to that task, followed immediately by a **"Volle Evaluation" (Full Evaluation)**.

The task sequence is defined as follows:

* **Task 1: The Scaling Shift (Linear Adaptation)** – *Der Skalierungs-Shift (Lineare Anpassung)*.
* **Task 2: The Dissipation Shift & Disentanglement (Ice Scenario)** – *Der Dissipations-Shift & Disentanglement (Eis-Szenario)*.
* **Task 3: The Discretization Shift (Hybrid Dynamics / Walls)** – *Der Diskretisierungs-Shift (Hybride Dynamik / Wände)*.
* **Task 4: The Kinematics Shift (Rotation & Asymmetry)** – *Der Kinematik-Shift (Rotation & Asymmetry)*.
* **Task 5: The Compositional OOD-Shift (The Grand Finale)** – *Der kompositionelle OOD-Shift (Das große Finale)*. This represents a final Out-of-Distribution generalization test.

*Implementation Note:* A yellow annotation under AC-HOPE Task 1 queries the possibility of automatically starting new runs within the same Weights & Biases (WandB) logging run.

#### **4. The Full Evaluation Methodology (Lower Section)**

The lower section of the diagram details the standardized **"Volle Evaluation"** procedure performed after each task training phase.

**4.1. Evaluation Data Protocol**
The evaluation process asks: "How exactly does full evaluation work?" The define protocol is:

* **Fixed Validation Set:** A specific number of clips are randomly selected from *each* task. Crucially, this selection is fixed ("always the same ones!!") to ensure consistent cross-task comparison.
* **Inference Mode:** The models are frozen (weights are not updated) and run over this aggregated validation set.

**4.2. Core Performance Metrics**
The raw performance on the validation set is measured using specific metrics, including:

* **L1:** Likely L1 error/loss.
* **Jump Prediction:** A task-specific metric related to predicting sudden changes or discontinuities.

**4.3. Continual Learning Metrics**
The core metrics are processed to yield high-level continual learning indicators:

* **FWT (Forward Transfer):** Measures the model's zero-shot performance on future, unseen tasks based on current knowledge.
* **BWT (Backward Transfer):** Measures how learning the current task has affected performance on previously learned tasks. The diagram further explicitly breaks this down into:
* **ExperienceForgetting**
* **StreamForgetting**


* **Point Metrics:** Specific performance snapshots, denoted as **Top1_L1_Exp** and **Top1_L1_Stream** (likely referring to the best L1 scores in "Experience" and "Stream" evaluation contexts).
