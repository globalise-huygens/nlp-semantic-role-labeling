# Multitask Learning results
> This folder contains the results of the multitask learning framework for the three models implemented: GloBERTise, XLM-RoBERTa, and multilingual BERT. Each model's results consist of six files.

## File Overview
MT_all_metrics_complete.json – Consolidated run-level metrics (precision, recall, F1, accuracy) for each cross-validation fold and task, plus per-class scores. 

NER_all_matrices.csv – Wide-format confusion matrices (one matrix per fold) covering all NER labels.

SRL_all_matrices.csv – Wide-format confusion matrices for all SRL labels.

ner_classification_reports.txt – Scikit-learn-style classification reports for every NER fold. 

srl_classification_reports.txt – Scikit-learn-style classification reports for every SRL fold. 

## Other output
> The code provided in this repository also creates files per fold that show per token the prediction and the gold label. These files are not included here due to space constraints: each model-setup combination produces 16 such files (one per fold), quickly adding up in size.
# Fold 3
Token         Gold        Pred
-------------------------------
het           O           O
schip         B-AGENT     B-AGENT
vertrok       B-PRED      B-PRED
...

