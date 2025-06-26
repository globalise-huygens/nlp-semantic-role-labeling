# Multitask Learning results
> This folder contains the results of the multitask learning framework for the three models implemented: GloBERTise, XLM-RoBERTa, and multilingual BERT. Each model's results consist of six files.

## File Overview
MT_all_metrics_complete.json – Consolidated run-level metrics (precision, recall, F1, accuracy) for each cross-validation fold and task, plus per-class scores. 
Structure overview:
```text
{
  "learning_rate": 5e-05,
  "epochs": 30,
  "model_checkpoint": "globalise/GloBERTise",
  "batch_size": 16,
  "metrics": {
    "precision": [{"srl": ..., "ner": ...},{....}],
    "recall": [{"srl": ..., "ner": ...},{....}],
    "f1": [{"srl": ..., "ner": ...},{....}],
    "accuracy": [{"srl": ..., "ner": ...},{....}],
    "per_class_scores": [...]
  }
}
```

NER_all_matrices.csv – Wide-format confusion matrices (one matrix per fold) covering all NER labels.

SRL_all_matrices.csv – Wide-format confusion matrices (one matrix per fold) for all SRL labels.

ner_classification_reports.txt – Classification reports for every NER fold. 

srl_classification_reports.txt – Classification reports for every SRL fold. 

## Other output
> The code provided in this repository also creates files per fold that show per token the prediction and the gold label, for each task. These files are not included here due to space constraints: each model-setup combination produces 16 such files (one per fold), quickly adding up in size. These files are used in the error_analysis.py file. 
Each of these files has the following structure:

```text
# Fold 3
Token         Gold        Pred       Task
----------------------------------------------
het           O           O          SRL
schip         B-AGENT     B-AGENT    SRL
vertrok       O           O          SRL
...
het           O           O          NER
schip         SHIP_TYPE   O          NER
vertrok       O           O          NER
...
```
