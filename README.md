# Multitask Learning of Semantic Role Labeling and Named Entity Recognition for domain‑specific documents from the Dutch East‑India Company archives

## Master's Degree in 'Linguistics: Text‑Mining' (currently Language & AI), VU Amsterdam, 2024‑2025

---

**Overview**

This repository accompanies the master's‑thesis project *Multitask Learning of Semantic Role Labeling and Named Entity Recognition for domain‑specific documents from the Dutch East‑India Company archives* by **Hannah Goossens** (supervisors : Stella Verkijk & Piek Vossen). The work forms part of the **GLOBALISE** project, which is building a search engine over VOC archives.

The thesis investigates whether a Multitask‑Learning (MTL) setup that jointly trains **Semantic Role Labeling (SRL)** and **Named Entity Recognition & Classification (NERC)** can outperform single‑task fine‑tuning on SRL alone when training data are scarce. Three transformer models are evaluated: multilingual BERT, XLM‑RoBERTa, and **GloBERTise** (a domain‑specific Dutch RoBERTa). Sixteen annotated Early‑Modern‑Dutch VOC documents are used in cross‑validation. The methodology, results, and extensive error analysis are described in the thesis report.

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
    ├── MTL
    │   ├── GloBERTise/ …
    │   ├── XLM‑R/ …
    │   └── mBERT/ …
    ├── NERC
    │   ├── GloBERTise/ …
    │   ├── XLM‑R/ …
    │   └── mBERT/ …
    └── SRL
        ├── GloBERTise/ …
        ├── XLM‑R/ …
        └── mBERT/ …

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

All evaluation artefacts live in `Results/`, organised first by **framework** (MTL / SRL / NERC) and then by **model** (GloBERTise, XLM‑R, mBERT). Sub‑folder READMEs explain the CSV, JSON and TXT files provided.

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Python 3.10 + is recommended.**

---

## Thesis Report

The full thesis (PDF) is available in the repository release or via **VU Amsterdam Scripties**.

---

## References



