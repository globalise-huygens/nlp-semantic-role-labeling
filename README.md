**Multitask Learning of Semantic Role Labeling and Named Entity Recognition for domain-specific documents from the Dutch East-India Company archives**
-------------------------------------------------------------------------------------------------------------------------------------------------------
Master's Degree in 'Linguistics: Text-Mining' (currently Language & AI), VU Amsterdam, 2024-2025
-------------------------------------------------------------------------------------------------------------------------------------------------------
**Overview**

This repository belongs to the Master's thesis Project 'Multitask Learning of Semantic Role Labeling and Named Entity Recognition for domain-specific documents from the Dutch East-India Company archives' by Hannah Goossens, supervised by Stella Verkijk and Piek Vossen. The project was part of the GLOBALISE research project, which aims to build a search engine for the Dutch East-India Company (VOC) archives. *

The thesis investigated the effect of Multitask Learning (MTL) on Semantic Role Labeling (SRL) using annotated documents from the Dutch East-India Company (VOC) archives, written in Early Modern Dutch. Several Transformer-based models were evaluated to determine whether an MTL framework, jointly training for SRL and Named Entity Recognition and Classification (NERC), improves SRL performance compared to single-task finetuning for SRL. The models that were implemented are multilingual BERT, XLM-RoBERTa, and GloBERTise: a domain-specific pre-trained RoBERTa model. The choice for multitask learning lies in the fact that there was a limited amount of annotated data available. The models were trained and tested using cross-validation of the 16 annotated documents. An evaluation of the results of SRL was carried out for each model, comparing both the different models in the single-task finetuning for SRL and NERC and multitask learning of SRL and NERC, as well as the same models in the different implementations (single-task vs. multitask) for SRL. An extensive error analysis was reported on for best performing model, based on SRL performance.

The complete background, data description, methodology, results, error analysis and discussion can be found in the thesis report..**.

---------------------------------------------------------------------------------------------------------------------------------------------------------
