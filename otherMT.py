import torch
import torch.nn as nn
import glob
import os
from sklearn.preprocessing import LabelEncoder
import json
from evaluate import load
import numpy as np
from itertools import chain
import itertools
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from datasets import Dataset, concatenate_datasets
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import re
import argparse

import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from functions_MT import list_of_files, process_file_to_dict, augment_sent_with_pred, get_first_non_O_label, post_process, generate_report 
from transformers import (
    AutoConfig, AutoTokenizer, AutoModel,
    BertModel, RobertaModel, XLMRobertaModel,
    PreTrainedModel, BertPreTrainedModel
)
from transformers.modeling_outputs import TokenClassifierOutput

import datasets
import pandas as pd
import transformers

import logging
import torch
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
from tqdm import tqdm as tqdm1

from accelerate import Accelerator
from filelock import FileLock
from transformers import set_seed
from transformers.file_utils import is_offline_mode
from pathlib import Path

import dataclasses
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import DataCollator, InputDataClass
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict


class TaskSpecificDataset(torch.utils.data.Dataset):
    def __init__(self, full_data, task):
        self.data = full_data
        self.task = task  # 'srl' or 'ner'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = int(idx) 
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(item[f"labels_{self.task.upper()}"]),
            "task_name": self.task
        }

class DataCollatorWithTaskName:
    def __call__(self, features):
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
            "task_name": [f["task_name"] for f in features] # keep as list of strings
        }
        return batch
    

class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """

    def to(self, device):
        return self

class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """

    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            yield batch

class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset) for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])


class MultitaskTrainer(Trainer):
    def get_single_dataloader(self, task_name, dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.dataset is None:
            raise ValueError("Trainer: training requires a dataset.")

        train_sampler = (
            RandomSampler(dataset)
            if self.args.local_rank == -1
            else DistributedSampler(dataset)
        )

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
                dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
            ),
        )
        return data_loader

    def get_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        return MultitaskDataloader(
            {
                task_name: self.get_single_dataloader(task_name, task_dataset)
                for task_name, task_dataset in self.dataset.items()
            }
        )
    
class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(transformers.PretrainedConfig())
        self.num_labels = kwargs.get("task_labels_map", {})
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        ## add task specific output heads
        self.classifier1 = nn.Linear(
            config.hidden_size, list(self.num_labels.values())[0]
        )
        self.classifier2 = nn.Linear(
            config.hidden_size, list(self.num_labels.values())[1]
        )

        self.init_weights()


class BertForTokenClassificationMultitask(BertPreTrainedModel):
    def __init__(self, config, task_labels_map):
        super().__init__(config)
        self.num_labels = task_labels_map
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.ModuleDict({
            "srl": nn.Linear(config.hidden_size, task_labels_map["srl"]),
            "ner": nn.Linear(config.hidden_size, task_labels_map["ner"])
        })
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None, task_name=None):
        if isinstance(task_name, list):
            task_name = task_name[0] 
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier[task_name](sequence_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels[task_name]), labels.view(-1))
        return TokenClassifierOutput(loss=loss, logits=logits)

# Step 6: Custom trainer for batch alternation

class MultiTaskTrainer(Trainer):
    def get_train_dataloader(self):
        loaders = {
            task: DataLoader(
                dataset, batch_size=2, shuffle=True, collate_fn=self.data_collator
            ) for task, dataset in self.train_dataset.items()
        }
        return AlternatingTaskLoader(loaders)

class AlternatingTaskLoader:
    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.task_names = list(dataloader_dict.keys())
        self.iters = {k: iter(dl) for k, dl in dataloader_dict.items()}
        self.epoch_length = 100  # number of steps per epoch
        self.total_epochs = num_epochs

    def __len__(self):
        return self.epoch_length * self.total_epochs  # match total number of training steps

    def __iter__(self):
        step = 0
        while step < self.__len__():
            task = np.random.choice(self.task_names)
            try:
                batch = next(self.iters[task])
            except StopIteration:
                self.iters[task] = iter(self.dataloader_dict[task])
                batch = next(self.iters[task])
            step += 1
            yield batch

    #def __iter__(self):
        #while True:
            #task = np.random.choice(self.task_names)
            #try:
                #batch = next(self.iters[task])
            #except StopIteration:
                #self.iters[task] = iter(self.dataloader_dict[task])
                #batch = next(self.iters[task])
            #yield batch
# Metrics computation with seqeval

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

seqeval = load("seqeval")

def to_str_labels(preds, labels, id2label):
    preds_str, labels_str = [], []
    for pred, label in zip(preds, labels):
        p_seq, l_seq = [], []
        for p, l in zip(pred, label):
            if l != -100:
                p_seq.append(id2label[p])
                l_seq.append(id2label[l])
        preds_str.append(p_seq)
        labels_str.append(l_seq)
    return preds_str, labels_str

# Directory and file paths setup
directory = '../Data/SRL_train_with_entities'

file_paths = list_of_files(directory)
print(file_paths)

# Load mappings
with open('label_mapping_NER', 'r', encoding='utf-8') as f:
    label_mapping_NER = json.load(f)

# Get list of unique labels
label_list_NER = [label for label in label_mapping_NER.keys()]
print(label_list_NER)
# Load mapping
with open('label_mapping', 'r', encoding='utf-8') as f:
    label_mapping_SRL = json.load(f)

# Get list of unique labels
label_list_SRL = [label for label in label_mapping_SRL.keys()]
print(label_list_SRL)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--epoch', type=int, required=True)
parser.add_argument('--num_folds', type=int, help="Number of splits for KFold CV")
parser.add_argument('--subsetsize', type=int, help="Subset size for tuning")
parser.add_argument('--model_checkpoint', type=str, required=True)
args = parser.parse_args()

# Inject arguments
learning_rate = args.learning_rate
num_epochs = args.epoch
model_checkpoint = args.model_checkpoint
num_folds = args.num_folds
subsetsize = args.subsetsize

task = "MultiTask SRL and NERC"
batch_size = 16 
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
special_tokens = {'additional_special_tokens': ['[PRED]']}
tokenizer.add_special_tokens(special_tokens)


files_dict = process_file_to_dict(file_paths, label_mapping_SRL, label_mapping_NER)
print(files_dict[0])
augmented_files = augment_sent_with_pred(files_dict, tokenizer)
#augmented_files = augmented_files[:subsetsize]
dataset = Dataset.from_list(augmented_files)
raw_dataset = dataset

srl_dataset = TaskSpecificDataset(dataset, task="srl")
ner_dataset = TaskSpecificDataset(dataset, task="ner")

total_examples = len(srl_dataset) + len(ner_dataset)
batch_size = 16
num_folds = 3
SEED = 42
steps_per_epoch = total_examples // batch_size
max_steps = steps_per_epoch * num_epochs
value_to_key_mapping_SRL = {v: k for k, v in label_mapping_SRL.items()}
value_to_key_mapping_NER = {v: k for k, v in label_mapping_NER.items()}

def get_first_non_O_label(seq, id2label, pad_val=-100):
    for l in seq:
        if l != pad_val:
            label = id2label[l]
            if label != "O":
                return label
    return "O"  

stratify_labels = [get_first_non_O_label(seq, value_to_key_mapping_SRL) for seq in dataset["labels_SRL"]]


all_f1_scores = []
all_precision_scores = []
all_recall_scores = []
all_accuracy_scores = []
all_loss_scores = []
all_metrics = []
metrics_log = []

# K-fold cross-validation
kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset, stratify_labels)):
    print(f"\n📘 Fold {fold+1}/{num_folds}...")

    from torch.utils.data import Subset

    train_data_srl = Subset(srl_dataset, train_idx)
    val_data_srl = Subset(srl_dataset, val_idx)
    train_data_ner = Subset(ner_dataset, train_idx)
    val_data_ner = Subset(ner_dataset, val_idx)
    
    model = BertForTokenClassificationMultitask.from_pretrained(
        model_checkpoint,
        config=AutoConfig.from_pretrained(model_checkpoint),
        task_labels_map={"srl": len(label_mapping_SRL), "ner": len(label_mapping_NER)}
    )
    model.resize_token_embeddings(len(tokenizer)) #adjust for special token
    trainer = MultiTaskTrainer(
        model=model,
        args=TrainingArguments(
            output_dir="./outputs",
            per_device_train_batch_size=16,
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            logging_dir="./logs",
            report_to=[]
        ),
        train_dataset={"srl": train_data_srl, "ner": train_data_ner},
        eval_dataset={'srl': val_data_srl, 'ner': val_data_ner},
        tokenizer=tokenizer,
        data_collator=DataCollatorWithTaskName()
    )

    trainer.train()

    srl_metrics = trainer.evaluate(val_data_srl)
    srl_loss = srl_metrics["eval_loss"]
    
    ner_metrics = trainer.evaluate(val_data_ner)
    ner_loss = ner_metrics["eval_loss"]
    # Get predictions for test set
    # Predict on SRL evaluation set
    # Predict SRL


    srl_predictions, srl_labels, _ = trainer.predict(val_data_srl)

    # Predict NER
    ner_predictions, ner_labels, _ = trainer.predict(val_data_ner)

        # Collect training logs for the fold
    for log in trainer.state.log_history:
        logs = dict(log)
        logs['fold'] = fold + 1
        all_metrics.append(logs)
    
    srl_predictions = np.argmax(srl_predictions, axis=2)
    ner_predictions = np.argmax(ner_predictions, axis=2)
    

    all_loss_scores.append({'srl': ner_loss, 'ner': ner_loss})

    
    # Get the original sentences (test data)
    test_dicts = raw_dataset.select(val_idx).to_list()
    
    
    # Post-process predictions (converting indices to labels)
    srl_word_predictions, srl_word_labels = post_process(srl_predictions, srl_labels, test_dicts)
    ner_word_predictions, ner_word_labels = post_process(ner_predictions, ner_labels, test_dicts)
    
    # Get tokens for each sentence
    test_tokens = [files_dict[i] for i in val_idx]
    all_sents = []
    all_lbls = []
    for dct in test_tokens:
        sent = dct['text']
        all_sents.append(sent)
    
    
    # Flatten predictions and labels for both tasks for metrics
    flat_srl_preds = list(itertools.chain.from_iterable(srl_word_predictions))
    flat_srl_labels = list(itertools.chain.from_iterable(srl_word_labels))
    flat_ner_preds = list(itertools.chain.from_iterable(ner_word_predictions))
    flat_ner_labels = list(itertools.chain.from_iterable(ner_word_labels))
    
    # Convert indices to label strings (e.g., using value_to_key mapping)
    flat_srl_preds = [value_to_key_mapping_SRL[label] for label in flat_srl_preds]
    flat_srl_labels = [value_to_key_mapping_SRL[label] for label in flat_srl_labels]
    
    flat_ner_preds = [value_to_key_mapping_NER[label] for label in flat_ner_preds]
    flat_ner_labels = [value_to_key_mapping_NER[label] for label in flat_ner_labels]

        # Write predictions to file
    outputfile = f'MT_pred_versus_gold_test_fold_{fold+1}.txt'
    with open(outputfile, 'a') as outfile:
        if outfile.tell() == 0:
            outfile.write("Token\tGold\tPrediction\tTask\n")
        for preds, golds, tokens in zip(srl_word_predictions, srl_word_labels, test_tokens):
            for pred, gold, token in zip(preds, golds, tokens):
                pred_label = value_to_key_mapping_SRL[pred]
                gold_label = value_to_key_mapping_SRL[gold]
                outfile.write(f"{token}\t{gold_label}\t{pred_label}\tSRL\n")
        for preds, golds, tokens in zip(ner_word_predictions, ner_word_labels, test_tokens):
            for pred, gold, token in zip(preds, golds, tokens):
                pred_label = value_to_key_mapping_NER[pred]
                gold_label = value_to_key_mapping_NER[gold]
                outfile.write(f"{token}\t{gold_label}\t{pred_label}\tNER\n")
    
    # Calculate metrics for SRL and NER
    srl_precision = precision_score(flat_srl_labels, flat_srl_preds, average='macro')
    srl_recall = recall_score(flat_srl_labels, flat_srl_preds, average='macro')
    srl_f1 = f1_score(flat_srl_labels, flat_srl_preds, average='macro')
    srl_accuracy = accuracy_score(flat_srl_labels, flat_srl_preds)


    ner_precision = precision_score(flat_ner_labels, flat_ner_preds, average='macro')
    ner_recall = recall_score(flat_ner_labels, flat_ner_preds, average='macro')
    ner_f1 = f1_score(flat_ner_labels, flat_ner_preds, average='macro')
    ner_accuracy = accuracy_score(flat_ner_labels, flat_ner_preds)
    
    # Append the results for the current fold
    all_precision_scores.append({'srl': srl_precision, 'ner': ner_precision})
    all_recall_scores.append({'srl': srl_recall, 'ner': ner_recall})
    all_f1_scores.append({'srl': srl_f1, 'ner': ner_f1})
    all_accuracy_scores.append({'srl': srl_accuracy, 'ner': ner_accuracy})
    
    # Save classification reports
    generate_report(flat_srl_labels, flat_srl_preds, value_to_key_mapping_SRL, fold, output_file='srl_classification_reports_test_3.txt')
    generate_report(flat_ner_labels, flat_ner_preds, value_to_key_mapping_NER, fold, output_file='ner_classification_reports_test_3.txt')

# Save all metrics to a JSON file after all folds
metrics_file = 'MT_all_metrics_test_3.json'
run_result = {
    'learning_rate': learning_rate,
    'epochs': num_epochs,
    'model_checkpoint': model_checkpoint,
    'batch_size': batch_size,
    'metrics': {
        'precision': all_precision_scores,
        'recall': all_recall_scores,
        'f1': all_f1_scores,
        'accuracy': all_accuracy_scores,
        'loss': all_loss_scores,
        'epoch_logs': all_metrics
    }
}

# Load previous runs and append the current run's results
if os.path.exists(metrics_file):
    with open(metrics_file, 'r') as f:
        all_runs = json.load(f)
else:
    all_runs = []

all_runs.append(run_result)

# Save all runs
with open(metrics_file, 'w') as f:
    json.dump(all_runs, f)