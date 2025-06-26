#Multitask Learning of Semantic Role Labeling and Named Entity Recognition for domain-specific documents from the Dutch East-India Company archives
-----------------------------------------------------------------------------------------------------------------------------------------
##Master's Degree in 'Linguistics: Text-Mining' (currently Language & AI), VU Amsterdam, 2024-2025
-----------------------------------------------------------------------------------------------------------------------------------------
**Overview**

This repository belongs to the Master's thesis Project 'Multitask Learning of Semantic Role Labeling and Named Entity Recognition for domain-specific documents from the Dutch East-India Company archives' by Hannah Goossens, supervised by Stella Verkijk and Piek Vossen. The project was part of the GLOBALISE research project, which aims to build a search engine for the Dutch East-India Company (VOC) archives. *

The thesis investigated the effect of Multitask Learning (MTL) on Semantic Role Labeling (SRL) using annotated documents from the Dutch East-India Company (VOC) archives, written in Early Modern Dutch. Several Transformer-based models were evaluated to determine whether an MTL framework, jointly training for SRL and Named Entity Recognition and Classification (NERC), improves SRL performance compared to single-task finetuning for SRL. The models that were implemented are multilingual BERT, XLM-RoBERTa, and GloBERTise: a domain-specific pre-trained RoBERTa model. The choice for multitask learning lies in the fact that there was a limited amount of annotated data available. The models were trained and tested using cross-validation of the 16 annotated documents. An evaluation of the results of SRL was carried out for each model, comparing both the different models in the single-task finetuning for SRL and NERC and multitask learning of SRL and NERC, as well as the same models in the different implementations (single-task vs. multitask) for SRL. An extensive error analysis was reported on for best performing model, based on SRL performance.

The complete background, data description, methodology, results, error analysis and discussion can be found in the thesis report..**.

---------------------------------------------------------------------------------------------------------------------------
## Project structure
<details>
<summary>Click to expand</summary>

```text
Thesis Project Structure
------------------------------
Thesis Project Structure 
└───code
│       │   Multitask Learning 
│                           └─── MTL.py
│                           └─── functions_MT.py
│                           └─── label_mapping.json
│                           └─── label_mapping.NER.json
│                           └─── script_MT.sh
│       │ Single-task fine-tuning
│                               │ NERC
│                                   └─── fine_tune_NERC.py
│                                   └─── functins_NER.py
│                                   └─── label_mapping_NER.json
│                                   └─── script_NER.sh
│                               │ SRL
│                                   └─── fine_tune_SRL.py
│                                   └─── functions_SRL.py
│                                   └─── label_mapping.json
│                                   └─── script.sh
│       └─── data_distribution.py
│       └─── error_analysis.py
│
└───data
│       │ SRL_train
│                 │ train_2
│                         └─── ....1626.conllu
│                         └─── ....1647.conllu
│                         └─── ....1777.conllu
│                         └─── ....1720.conllu
│                 │ train_3
│                         │ special_topic_ESTA
│                                           └─── ....1679-notitle.conllu
│                                           └─── ....1686-notitle.conllu
│                                           └─── ....1746-notitle.conllu
│                                           └─── ....1716-notitle.conllu
│                         └─── ....1747-.conllu
│                         └─── ....1736-.conllu
│                         └─── ....1679-.conllu
│                         └─── ....1781-.conllu
│                 │ train_4
│                         └─── ....1618-.conllu
│                         └─── ....1686-.conllu
│                         └─── ....1713-.conllu
│                         └─── ....1707-.conllu
│                
│       │ SRL_train_with_entities
│                 │ train_2
│                         └─── ....1626.conllu
│                         └─── ....1647.conllu
│                         └─── ....1777.conllu
│                         └─── ....1720.conllu
│                 │ train_3
│                         │ special_topic_ESTA
│                                           └─── ....1679-notitle.conllu
│                                           └─── ....1686-notitle.conllu
│                                           └─── ....1746-notitle.conllu
│                                           └─── ....1716-notitle.conllu
│                         └─── ....1747-.conllu
│                         └─── ....1736-.conllu
│                         └─── ....1679-.conllu
│                         └─── ....1781-.conllu
│                 │ train_4
│                         └─── ....1618-.conllu
│                         └─── ....1686-.conllu
│                         └─── ....1713-.conllu
│                         └─── ....1707-.conllu
└───Results
│         │ MTL
│             │ GloBERTise
│                       └─── MT_all_metrics_complete.json
│                       └─── NER_allmatrices.csv
│                       └─── NER_long_format_CM.csv
│                       └─── SRL_all_matrices.csv
│                       └─── SRL_long_format_CM.csv
│                       └─── ner_classification_reports.txt
│                       └─── srl_classification_reports.txt
│             │ XLM-R
│                       └─── MT_all_metrics_complete.json
│                       └─── NER_allmatrices.csv
│                       └─── NER_long_format_CM.csv
│                       └─── SRL_all_matrices.csv
│                       └─── SRL_long_format_CM.csv
│                       └─── ner_classification_reports.txt
│                       └─── srl_classification_reports.txt
│             │ mBERT
│                       └─── MT_all_metrics_complete.json
│                       └─── NER_allmatrices.csv
│                       └─── NER_long_format_CM.csv
│                       └─── SRL_all_matrices.csv
│                       └─── SRL_long_format_CM.csv
│                       └─── ner_classification_reports.txt
│                       └─── srl_classification_reports.txt
│         │ NERC
│             │ GloBERTise
│                       └─── all_matrices.csv
│                       └─── all_metrics_complete.json
│                       └─── classification_reports.txt
│                       └─── long_format_matrix.csv
│             │ XLM-R
│                       └─── all_matrices.csv
│                       └─── all_metrics_complete.json
│                       └─── classification_reports.txt
│                       └─── long_format_matrix.csv
│             │ mBERT
│                       └─── all_matrices.csv
│                       └─── all_metrics_complete.json
│                       └─── classification_reports.txt
│                       └─── long_format_matrix.csv
│         │ SRL
│             │ GloBERTise
│                       └─── all_matrices.csv
│                       └─── all_metrics_complete.json
│                       └─── classification_reports.txt
│                       └─── long_format_matrix.csv
│             │ XLM-R
│                       └─── all_matrices.csv
│                       └─── all_metrics_complete.json
│                       └─── classification_reports.txt
│                       └─── long_format_matrix.csv
│             │ mBERT
│                       └─── all_matrices.csv
│                       └─── all_metrics_complete.json
│                       └─── classification_reports.txt
│                       └─── long_format_matrix.csv
│
│   LICENSE
│   README.md
│   requirements.txt   
│   gitignore.txt
