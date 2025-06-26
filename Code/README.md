# Multitask Learning (MTL)

> Files necesary for implementing MTL setup of Semantic Role Labeling (SRL) and Named Entity Recognition and Classification (NERC)

# Single-task fine-tuning

> Files necessary for implementing the seperate single-task fine-tuning of SRL and NERC

# Data_distribution.py

> Python module for analyzing and visualizing sentence, token, semantic role, and named entity distributions in the VOC data files as listed in the Data folder.

## Features
- Recursively gather all `.conllu` files inside a directory.
- Compute sentence and token counts per document.
- Extract and count non‑`O` semantic role (SRL) and named entity (NER) tags.
- Aggregate corpus‑level statistics including yearly breakdowns derived from file names.
- Convert BIO‑encoded tags into mention‑, token‑, or raw‑level representations.
- Generate frequency bar plots and top‑10 horizontal bar charts.
- Build detailed overlap tables showing how NER categories intersect with SRL arguments.
- Export all tables to tab‑separated values (TSV) files for easy downstream analysis.

## Quick Start
```python
from data_distribution import *
# 1. Collect all .conllu paths inside ./corpus
paths = list_of_files("./corpus")

# 2. Compute document‑level statistics
stats = document_statistics(paths)

# 3. Extract tag inventories
roles, ner = count_role_and_ne_tokens(paths)

# 4. Convert tags to mention level (or token-level, bio-tag level, based on type specification)
role_mentions, ne_mentions = distribution(ner, roles, type="Mention")

# 5. Plot frequency distribution
barplot(role_mentions, xlabel="SRL Role", rotation=45)
```

## API Reference
### `list_of_files(directory) -> List[str]`
Recursively returns absolute paths to every `.conllu` file located under `directory`.

### `count_sents(path) -> int`
Counts sentences in a single file; sentences are delimited by blank lines.

### `count_tokens(path) -> int`
Counts tokens in a single file, respecting CoNLL‑U sentence boundaries.

### `count_role_and_ne_tokens(file_paths) -> Tuple[List[str], List[str]]`
Collects all SRL and NER BIO tags that are not `'O'`.

### `document_statistics(file_paths) -> Dict[int, Dict]`
Returns sentence, token, SRL, and NER counts per document plus a year extracted from the filename.

### `distribution(ner, roles, type="Mention") -> Tuple[List[str], List[str]]`
Normalises BIO tags. `type` can be:
- `"Mention"` – keep only `B-` tags (distinct mentions).
- `"Token"` – keep both `B-` & `I-` but strip prefix.
- `"BIO"` – leave tags untouched.

### `barplot(items, xlabel="value", ylabel="count", rotation=0)`
Simple frequency bar plot for any list of categorical items.

### `overlap_role_and_ne_tokens(file_paths) -> Tuple[List[str], List[str], List[str], List[str]]`
Returns SRL tokens, NER tokens, and their overlaps.

### `percentages_overlap(roles, ner, overlap_ner)`
Prints overlap statistics.

### `table_overlap(roles, overlap_role, ner, overlap_ner, file_name) -> pandas.DataFrame`
Constructs a DataFrame with detailed overlap metrics and saves it as TSV.

### `top_10_plot(df_counts, new_file)`
Plots the ten role–NER pairs that occupy the highest share of their role.


## Visualizations
Plots are produced with `matplotlib`. All figure windows are displayed immediately; save them using `plt.savefig()` if needed.

---

# Error_analysis.py

> Python helper for categorizing and summarizing prediction errors in **Semantic Role Labeling (SRL)** experiments. Feed it a token‑level DataFrame of gold and predicted labels, and the script will label every token with a fine‑grained error type, output concise summary statistics, and write detailed CSVs for further inspection.

## Features

- **Fine‑grained error taxonomy**
  - *Correct* — prediction equals gold (and gold ≠ `O`).
  - *False Positive* — predicted label ≠ `O`, gold is `O`.
  - *False Negative* — gold label ≠ `O`, prediction is `O`.
  - *Boundary Error* — same role type but mismatched BIO prefix (`B-` vs `I-`).
  - *Label Confusion* — both non‑`O` but different role types.
- **Token‑level annotation**: assigns an *Error Type* to every row.
- **Summary statistics**: counts & percentages of the four error classes.
- **CSV dumping**: automatically writes three files per run
  - `*_srl_error_summary_fold{n}.csv`
  - `*_srl_token_errors_fold{n}.csv`
  - `*_srl_correct_predictions_fold{n}.csv`
- **Single‑task & multi‑task aware**: filter by `Task` column when needed.
- **Fold support**: embed fold number in output filenames.

## Quick Start

```python
import pandas as pd
from error_analysis import *

# Load predictions (must contain Token, Gold, Prediction — and Task if multi‑task)
df = pd.read_csv("predictions_fold1.csv")

# Run error analysis for a single‑task SRL model
error_analysis(df, train_setup="single-task", fold=1)

# For a multi‑task run focusing on SRL only
error_analysis(df, train_setup="multi-task", task="SRL", fold=3)
```

After execution you will see console summaries and three CSVs in your working directory.

## API Reference

### `categorize_error(gold: str, pred: str) -> str | None`

Returns the error class for a single gold/predicted tag pair.

### `error_analysis(df: pd.DataFrame, train_setup='single-task', task='SRL', fold=1) -> None`

Main pipeline:

1. Cleans DataFrame (drops header lines that start with `===`).
2. Filters by *Task* when `train_setup='multi-task'`.
3. Adds an **Error Type** column via `categorize_error()`.
4. Builds counts/percentages for *False Positive, False Negative, Boundary Error, Label Confusion*.
5. Saves summary and detailed CSVs, prints results.
