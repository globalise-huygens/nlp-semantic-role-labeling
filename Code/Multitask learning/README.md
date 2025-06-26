# Multitask SRL & NER Fine-Tuning Scripts

This folder contains the main scripts and utilities used to train multitask models for Semantic Role Labeling (SRL) and Named Entity Recognition (NER) on VOC data.

---

## Contents

| File | Description |
|------|-------------|
| `MT1.py` | Main training script that performs multitask learning using BERT, RoBERTa, or XLM-R. Handles cross-validation, data preprocessing, training, evaluation, and saving predictions/metrics. |
| `functions_MT.py` | Utility functions used in `MT1.py`, such as data loading, splitting, augmentation, metric computation, and result saving. |
| `label_mapping` | JSON dictionary mapping SRL labels to integer IDs. |
| `label_mapping_NER` | JSON dictionary mapping NER labels to integer IDs. |
| `script_MT.sh` | SLURM script to submit multitask training jobs on a GPU cluster. Includes model config and output copying to permanent storage. |

---

## Usage

To run multitask training locally:
```bash
python MT1.py \
    --learning_rate 5e-5 \
    --epoch 30 \
    --batch_size 16 \
    --model_checkpoint "xlm-roberta-base" \
    --model_type "XLM-R"
```

To run it on a SLURM cluster:
```bash
sbatch script_MT.sh
```

---

## Features

- **Cross-validation setup**: Each document acts as test set once.
- **Multitask architecture**: Shared encoder, separate heads for SRL and NER.
- **Token augmentation**: Predicate marker `[PRED]` injected during preprocessing.
- **Metrics**: Per-task F1, accuracy, confusion matrices, and full classification reports.
- **Auto-saving**: Predictions and metrics are stored per fold.

---

## Output

The script automatically stores:
- `.txt`: Per-token prediction vs gold (for SRL and NER)
- `.csv`: Confusion matrices and long-format data
- `.json`: Fold-wise F1, precision, recall, accuracy, and losses
- `.png`: Confusion matrix plots

These outputs are copied to:
```bash
/the/specified/path/in/the/shell/script
```

---

## Notes
- `label_mapping` and `label_mapping_NER` are based on the labels used in the data provided in this repository 
- `MT1.py` uses `Trainer` from HuggingFace with a custom wrapper to handle multitask batches.
- Uses `DataCollatorWithTaskName` to inject task labels into training batches.
