# VI-JEPA 2 AC-Predictor Training Analysis

## V-JEPA 2 AC-Predictor Training: Splits & Validation

### 1. During the training, was there a validation split?
**No.** The training process for the AC-Predictor (part of the `vjepa_droid` application) does not include an online validation step. The training script runs exclusively on the provided training dataset to optimize the JEPA loss.

*   **Evidence**: The main training script [app/vjepa_droid/train.py](app/vjepa_droid/train.py) initializes only an `unsupervised_loader`. There is no initialization of a validation loader, nor is there a validation loop within the `main` execution flow.

### 2. How was train / val / test split?
The data splits were determined **offline** (statically) via separate file lists. The codebase does not randomly split the data at runtime; instead, it relies on the user providing specific CSV/text files containing the list of video trajectories to be used for a specific phase (training vs. evaluation).

*   **Mechanism**: The dataset class `DROIDVideoDataset` in [app/vjepa_droid/droid.py#L86](app/vjepa_droid/droid.py#L86) reads all paths from the provided input file.
*   **Configuration**: The training configuration (e.g., [configs/train/vitg16/droid-256px-8f.yaml](configs/train/vitg16/droid-256px-8f.yaml)) explicitly points to a single file for the dataset, conventionally named with `_train_` (e.g., `droid_train_paths.csv`).

### 3. How did they handle this?
The project handles splits by decoupling **Training** from **Evaluation**.

*   **Training Phase**: The `vjepa_droid` training script consumes the **Training Split** (via the CSV defined in `data.datasets`). It focuses solely on learning representations.
*   **Evaluation Phase**: Validation and Testing are treated as distinct downstream tasks (such as *Action Anticipation*). These are executed using separate scripts (e.g., [evals/action_anticipation_frozen/eval.py](evals/action_anticipation_frozen/eval.py)), which accept strict configuration parameters for `dataset_train` and `dataset_val` to load the respective split files.

This separation ensures that the computationally expensive pre-training is not slowed down by frequent validation, and that evaluation metrics are standardized in dedicated evaluation suites.
