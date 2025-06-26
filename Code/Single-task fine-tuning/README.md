# Fine-tuning SRL Models on VOC Data

> This repository contains scripts for fine-tuning transformer models on Semantic Role Labeling (SRL) and Named Entity Recogntion and Classification tasks using annotated `.conllu` files. The data is also listed in this repository and is preprocessed for token classification.

## Contents

NERC:

| File | Purpose |
|------|---------|
| `fine_tune_NERC.py` | Main training and evaluation script for NERC models. |
| `functions_NERC.py` | Utility functions for preprocessing, tokenizing, evaluation, reporting. |
| `label_mapping_NE.json` | JSON dictionary mapping NE labels to numeric IDs. |
| `script_NER.sh` | Slurm batch script for launching training jobs on a GPU cluster. |

SRL:

| File | Purpose |
|------|---------|
| `fine_tune_SRL.py` | Main training and evaluation script for SRL models. |
| `functions_SRL.py` | Utility functions for preprocessing, tokenizing, evaluation, reporting. |
| `label_mapping.json` | JSON dictionary mapping SRL labels to numeric IDs. |
| `script.sh` | Slurm batch script for launching training jobs on a GPU cluster. |



---

## Quick Start

### 1. Prepare your data
Place your `.conllu` training files in a folder named `../Data/SRL_train_with_entities/`. These should follow a format where:
- Column 5 = token  
- Column 6 = predicate flag (`O` = no predicate)  
- Column 7 = SRL label (e.g., `B-Agent`, `O`, etc.)
- Column 8 = NE label (e.g. 'B-person_name, 'O', etc.)

### 2. Run training locally
NERC: 
```bash
python fine_tune_NERC.py \
  --learning_rate 5e-5 \
  --epoch 30 \
  --batch_size 16 \
  --model_checkpoint globalise/GloBERTise \
  --model_type RoBERTa
```
SRL:
```bash
python fine_tune_SRL.py \
  --learning_rate 5e-5 \
  --epoch 30 \
  --batch_size 16 \
  --model_checkpoint globalise/GloBERTise \
  --model_type RoBERTa
```

### 3. Run on a cluster (Slurm)
NERC:
```bash
sbatch script_NER.sh
```
SRL:
```bash
sbatch script.sh
```

---

## Code Summary

Each description is the same for the NERC files, except the file names.

### `fine_tune_SRL.py` 
- Implements **leave-one-document-out cross-validation**.
- Loads tokenized `.conllu` files using `process_file_to_dict`.
- Inserts `[PRED]` token before predicates.
- Uses HuggingFace `Trainer` for training and evaluation.
- Outputs:
  - `classification_reports.txt`
  - Confusion matrix images and `.csv` files
  - `all_metrics_complete.json` with per-fold metrics
  - Per-token prediction logs in `.txt`

### `functions_SRL.py`
Includes reusable methods for:
- Recursively loading `.conllu` files
- Tokenizing and padding with predicate marker
- Aggregating subword predictions to word level
- Saving confusion matrices and long-format CSVs
- Computing macro scores and per-class metrics

### `label_mapping`
A JSON file mapping label strings (e.g. `"B-Agent"`) to numeric class IDs and vice versa. Used for:
- Label encoding
- Mapping predictions back to string labels during output

### `script.sh`
Batch script for HPC clusters:
- Loads Python/PyTorch modules
- Sets learning rate, epochs, batch size
- Runs `fine_tune_SRL.py` with those params
- Copies results from `$TMPDIR` to a persistent directory (`$DEST`)

---

## Outputs (per fold)

| File | Description |
|------|-------------|
| `complete_pred_versus_gold_foldX.txt` | Per-token predictions vs gold labels |
| `SRL_CM_foldX.png` | Confusion matrix heatmap |
| `all_matrices.csv` | Fold-wise confusion matrices |
| `long_format_matrix.csv` | Tidy format matrix for plotting |
| `classification_reports.txt` | Full sklearn reports |
| `all_metrics_complete.json` | JSON with F1/Precision/Recall/Accuracy per fold |

---

## 📝 Notes
- Uses `seqeval` to compute entity-level metrics.
- Tokenizer automatically adds `[PRED]` as special token.
- `Trainer` is configured to evaluate every epoch.
- Model supports BERT, RoBERTa, and XLM-R variants.
