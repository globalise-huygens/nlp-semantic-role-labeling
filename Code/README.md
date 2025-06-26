# Data Distribution Utilities

> Python module for analyzing and visualizing sentence, token, semantic role, and named entity distributions in the VOC data files as listed in the Data folder.

## Table of Contents
- [Features](#features)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Data Expectations](#data-expectations)
- [Visualizations](#visualizations)


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

# 4. Convert tags to mention level
role_mentions, ne_mentions = distribution(ner, roles, type="Mention")

# 5. Plot frequency distribution
barplot(role_mentions, xlabel="SRL Role", rotation=45)
```
Detailed examples are provided in `examples/demo.ipynb`.

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


