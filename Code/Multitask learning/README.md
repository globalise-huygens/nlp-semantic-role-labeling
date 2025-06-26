# Multitask Learning of SRL & NERC

> This repository contains scripts for **multitask** fine‑tuning of transformer models on both Semantic Role Labeling (SRL) *and* Named‑Entity Recognition and Classification (NERC).  
> The training data is provided in this repository and is in `.conllu` format.

## Contents

| File | Purpose |
|------|---------|
| `MT1.py` | Main multitask training & evaluation script (shared encoder, separate heads). |
| `functions_MT.py` | Helper functions for preprocessing, tokenization, metrics, and I/O. |
| `label_mapping` | JSON mapping SRL labels → numeric IDs. |
| `label_mapping_NER` | JSON mapping NER labels → numeric IDs. |
| `script_MT.sh` | SLURM batch script to launch multitask jobs on an HPC cluster. |

---

### Run training locally
```bash
python MT1.py   --learning_rate 5e-5   --epoch 30   --batch_size 16   --model_checkpoint "xlm-roberta-base"   --model_type XLM-R
```

### Run on a cluster (Slurm)
```bash
sbatch script_MT.sh
```

---

## Code Summary

### `MT1.py`
* Implements **leave‑one‑document‑out cross‑validation** for robust evaluation.  
* Builds a **shared encoder / dual‑head** architecture (SRL & NER).  
* Inserts a `[PRED]` special token before each predicate.  
* Uses a custom `MultitaskTrainer` to alternate SRL & NER batches.  
* Saves all logs, metrics, confusion matrices, and per‑token predictions.

### `functions_MT.py`
Reusable utilities for:
* Recursive file discovery and train/test splitting  
* Sentence augmentation with predicate token  
* Word‑level aggregation of sub‑token predictions  
* Creating and saving confusion matrices (wide & long format)  
* Macro metric computation (ignoring zero‑support labels)

### `label_mapping` & `label_mapping_NER`
Two JSON dictionaries mapping label strings (e.g. `"B-Agent"`, `"B-PER_NAME"`) to integer IDs and back.  
Used for encoding labels and translating predictions to human‑readable form.

### `script_MT.sh`
Batch script for GPU clusters:
* Loads appropriate Python & PyTorch modules  
* Sweeps over learning‑rate / epoch / batch‑size combos  
* Runs `MT1.py` with the specified model checkpoint  
* Copies all outputs from `$TMPDIR/outputs/` to a persistent directory

---

## Outputs (per fold)

| File | Description |
|------|-------------|
| `complete_pred_versus_gold_foldX.txt` | Per‑token predictions vs. gold labels for **both** tasks (extra column `Task`). |
| `SRL_CM_foldX.png` / `NER_CM_foldX.png` | Heatmap confusion matrices for each task. |
| `SRL_all_matrices.csv` / `NER_all_matrices.csv` | Fold‑wise confusion‑matrix values. |
| `SRL_long_format_CM.csv` / `NER_long_format_CM.csv` | Tidy (long) matrix format for plotting. |
| `srl_classification_reports.txt` / `ner_classification_reports.txt` | Full `classification_report` outputs. |
| `MT_all_metrics_complete.json` | JSON with precision / recall / F1 / accuracy & losses for **both** tasks. |

---

## Notes
* Shared encoder is instantiated from BERT, RoBERTa, or XLM‑R; tokenizer is resized to add `[PRED]`.  
* Custom `DataCollatorWithTaskName` injects the task identifier into each batch.  
* `MultitaskTrainer` alternates batches proportionally between tasks.  
* Evaluation runs **every epoch**, and best checkpoints are saved (limit 1 per fold).  
* Metrics computed with `seqeval` and `sklearn`, ensuring comparability with single‑task runs.
