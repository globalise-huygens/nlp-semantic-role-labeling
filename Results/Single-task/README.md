# Single-task fine-tuning results
> This folder contains the results of the single-task fine-tuning framework for semantic role labeling (SRL) and named entity recognition and classification (NERC) for the three models implemented: GloBERTise, XLM-RoBERTa, and multilingual BERT. Each model's results consist of three files.

## File Overview
all_metrics_complete.json – Consolidated run-level metrics (precision, recall, F1, accuracy) for each cross-validation fold and task, plus per-class scores. 
Structure overview:
```text
{
  "learning_rate": 5e-05,
  "epochs": 30,
  "model_checkpoint": "globalise/GloBERTise",
  "batch_size": 16,
  "metrics": {
    "precision": [],
    "recall": [],
    "f1": [],
    "accuracy": [],
    "per_class_scores": [...]
  }
}
```

all_matrices.csv – Wide-format confusion matrices (one matrix per fold) covering all NERC or SRL labels.

classification_reports.txt – Classification reports for every SRL or NERC fold. 

## Other output
> The code provided in this repository also creates files per fold that show per token the prediction and the gold label, for each task. These files are not included here due to space constraints: each model-setup combination produces 16 such files (one per fold), adding up in size. These files are used in the error_analysis.py file. 
Each of these files has the following structure:

```text
For NERC:

# Fold 3
Token         Gold        Pred      
----------------------------------
het           O            O          
schip         B-SHIP_TYPE  O    
vertrok       O            O

For SRL:
         
# Fold 3
Token         Gold        Pred      
----------------------------------
het           O           O          
schip         B-AGENT     B-AGENT    
vertrok       O           O          
...
or 
...
```
