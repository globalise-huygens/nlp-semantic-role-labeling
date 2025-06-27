# Multitask Learning of Semantic Role Labeling and Named Entity Recognition for domain‑specific documents from the Dutch East‑India Company archives

## Master's Degree in 'Linguistics: Text‑Mining' (currently Language & AI), VU Amsterdam, 2024‑2025

---

**Overview**

This repository belongs to the Master's thesis Project 'Multitask Learning of Semantic Role Labeling and Named Entity Recognition for domain-specific documents from the Dutch East-India Company archives' by Hannah Goossens, supervised by Stella Verkijk and Piek Vossen. The project was part of the GLOBALISE research project, which aims to build a search engine for the Dutch East-India Company (VOC) archives. 

The thesis investigated the effect of Multitask Learning (MTL) on Semantic Role Labeling (SRL) using annotated documents from the Dutch East-India Company (VOC) archives, written in Early-Modern Dutch. Several Transformer-based models were evaluated to determine whether an MTL framework, jointly training for SRL and Named Entity Recognition and Classification (NERC), improves SRL performance compared to single-task finetuning for SRL. The MTL approach was chosen based on the limited availability of labeled data, as MTL can allow the sharing of parameters across tasks, which could reduce the need for vast amounts of labeled data. The models that were implemented are multilingual BERT, XLM-RoBERTa, and GloBERTise: a domain-specific pre-trained RoBERTa model. The models were trained and tested using cross-validation of the 16 annotated documents. An evaluation of the results of SRL was carried out for each model, comparing both the different models in the single-task finetuning for SRL and NERC and multitask learning of SRL and NERC, as well as the same models in the different implementations (single-task vs. multitask) for SRL. An extensive error analysis was reported on for best performing model, based on SRL performance.

The complete background, data description, methodology, results, error analysis and discussion can be found in the thesis report, which is listed below.


---

## Project Structure

```text
Thesis Project Structure
├── code
│   ├── Multitask Learning
│   │   ├── MTL.py
│   │   ├── functions_MT.py
│   │   ├── label_mapping.json
│   │   ├── label_mapping.NER.json
│   │   └── script_MT.sh
│   ├── Single‑task fine‑tuning
│   │   ├── NERC
│   │   │   ├── fine_tune_NERC.py
│   │   │   ├── functions_NER.py
│   │   │   ├── label_mapping_NER.json
│   │   │   └── script_NER.sh
│   │   └── SRL
│   │       ├── fine_tune_SRL.py
│   │       ├── functions_SRL.py
│   │       ├── label_mapping.json
│   │       └── script.sh
│   ├── data_distribution.py
│   └── error_analysis.py
│
├── data
│   ├── SRL_train
│   │   ├── train_2/ …
│   │   ├── train_3/ …
│   │   └── train_4/ …
│   └── SRL_train_with_entities
│       ├── train_2/ …
│       ├── train_3/ …
│       └── train_4/ …
│
└── Results
│   ├── MTL
│   │   ├── GloBERTise/ …
│   │   ├── XLM‑R/ …
│   │   └── mBERT/ …
│   ├── Single-task
│       ├── NERC
│       │   ├── GloBERTise/ …
│       │   ├── XLM‑R/ …
│       │   └── mBERT/ …
│       ├── SRL
│       │   ├── GloBERTise/ …
│       │   ├── XLM‑R/ …
│       │   └── mBERT/ …

LICENSE
README.md
requirements.txt
.gitignore
```

---

## Code

The `code/` directory contains the Python and shell scripts required to reproduce every experiment. It is split into two main pipelines:

| Pipeline                    | Purpose                                   |
| --------------------------- | ----------------------------------------- |
| **Multitask Learning**      | Joint fine‑tuning for SRL **+** NERC      |
| **Single‑task fine‑tuning** | Independent fine‑tuning for SRL *or* NERC |

Each sub‑folder includes a dedicated `README.md` with usage instructions.

---

## Results

The evaluation output are stored in `Results/`, organised first by **framework** (MTL / Single-task (NERC and SRL)) and then by **model** (GloBERTise, XLM‑R, mBERT). Sub‑folder READMEs explain the CSV, JSON and TXT files provided.

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Python 3.10 + is recommended.

---

## Thesis Report

The full thesis report is listed in this repository.

---

## References

The data is taken from [this repository](https://github.com/globalise-huygens/nlp-event-detection/tree/main/annotated_data_processing_for_training).

The multitask code is adapted from [this source](https://medium.com/@shahrukhx01/multi-task-learning-with-transformers-part-1-multi-prediction-heads-b7001cf014bf). 


