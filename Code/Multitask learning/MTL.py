import os

base_output_dir = os.path.join(os.environ["TMPDIR"], "outputs")
os.makedirs(base_output_dir, exist_ok=True)
os.environ["HF_HOME"] = os.path.join(os.environ["TMPDIR"], "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ["TMPDIR"], "hf_datasets")
os.environ["HF_METRICS_CACHE"] = os.environ["HF_DATASETS_CACHE"]

import torch
import torch.nn as nn
import glob
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

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support

from functions_MT import list_of_files, process_file_to_dict, augment_sent_with_pred, train_test_split_documents, post_process, generate_report, generate_confusion_matrix, save_confusion_matrix_long_format, compute_filtered_macro_scores
from transformers import (
    AutoConfig, AutoTokenizer,
    BertModel, RobertaModel, XLMRobertaModel, BertPreTrainedModel, RobertaTokenizerFast, RobertaPreTrainedModel, XLMRobertaTokenizerFast
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

#creates dataset for specific task
class TaskSpecificDataset(torch.utils.data.Dataset):
    #wraps the input dataset and selects either SRL or NER labels
    def __init__(self, full_data, task):
        self.data = full_data #The full list of augmented inputs 
        self.task = task  # 'srl' or 'ner'

    def __len__(self):
        return len(self.data) #required by pytorch: number of examples

    def __getitem__(self, idx):
        idx = int(idx) 
        item = self.data[idx] #selects a data point
        return {
            "input_ids": torch.tensor(item["input_ids"]), #token IDs
            "attention_mask": torch.tensor(item["attention_mask"]), #attention mask
            "labels": torch.tensor(item[f"labels_{self.task.upper()}"]), #srl or ner labels
            "task_name": self.task #include task name for routing
        }

class DataCollatorWithTaskName:
    #add task_name to batches and formats tensors properly
    def __call__(self, features):
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
            "task_name": [f["task_name"] for f in features] # keep as list of strings
        }
        return batch
    
#allos task_name strings to safely go through trainer.to(device)
class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """

    def to(self, device):
        return self
    
#wraps a pytorch dataloader and injects task_name into each batch (used for mulitask routing)
class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """

    def __init__(self, task_name, data_loader):
        self.task_name = task_name #store task name
        self.data_loader = data_loader #the underlying pytorch dataloader

        self.batch_size = data_loader.batch_size #for compatibility 
        self.dataset = data_loader.dataset #for trainer compatibility

    def __len__(self): #forward the length
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name) #add task name into each batch
            yield batch

#combines task-specific dataloaders and samples batches proportionally
class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items() #store batch sizes
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset) for dataloader in self.dataloader_dict.values() #fake dataset for compatibility
        )

    def __len__(self):
        return sum(self.num_batches_dict.values()) #total number of batches

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name] #repeat index proportionally
        task_choice_list = np.array(task_choice_list) #convert to numpy array for shuffling
        np.random.shuffle(task_choice_list) #shuffle to randomize task order
        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items() #create iterators 
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name]) #yield batch from selected task

#multitask BERT model with seperate classifiers per task
class BertForTokenClassificationMultitask(BertPreTrainedModel):
    def __init__(self, config, task_labels_map):
        super().__init__(config)
        self.num_labels = task_labels_map #{'srl': num_srl_lables, 'ner': num_ner_labels}
        self.bert = BertModel(config) #shared BERT encoder
        self.dropout = nn.Dropout(config.hidden_dropout_prob) #regularization layer
        self.classifier = nn.ModuleDict({
            "srl": nn.Linear(config.hidden_size, task_labels_map["srl"]),
            "ner": nn.Linear(config.hidden_size, task_labels_map["ner"])
        }) #seperate classification heads per task
        self.init_weights() #initialize weights (BERT + heads)

    def forward(self, input_ids, attention_mask=None, labels=None, task_name=None):
        if isinstance(task_name, list):
            task_name = task_name[0] #use first task if batch is mixed
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0]) #apply dropout to BERT output
        logits = self.classifier[task_name](sequence_output) #use the right classifier
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100) #ignore special/pad tokens
            loss = loss_fct(logits.view(-1, self.num_labels[task_name]), labels.view(-1))
        return TokenClassifierOutput(loss=loss, logits=logits) #huggingface output object

#multitask BERT model with seperate classifiers per task
class RobertaForTokenClassificationMultitask(RobertaPreTrainedModel):
    def __init__(self, config, task_labels_map):
        super().__init__(config)
        self.num_labels = task_labels_map #{'srl': num_srl_lables, 'ner': num_ner_labels}
        self.roberta = RobertaModel(config) #shared BERT encoder
        self.dropout = nn.Dropout(config.hidden_dropout_prob) #regularization layer
        self.classifier = nn.ModuleDict({
            "srl": nn.Linear(config.hidden_size, task_labels_map["srl"]),
            "ner": nn.Linear(config.hidden_size, task_labels_map["ner"])
        }) #seperate classification heads per task
        self.init_weights() #initialize weights (BERT + heads)

    def get_input_embeddings(self):
        return self.roberta.get_input_embeddings()
    
    def set_input_embeddings(self, new_embeddings):
        return self.roberta.set_input_embeddings(new_embeddings)

    def forward(self, input_ids, attention_mask=None, labels=None, task_name=None):
        if isinstance(task_name, list):
            task_name = task_name[0] #use first task if batch is mixed
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0]) #apply dropout to BERT output
        logits = self.classifier[task_name](sequence_output) #use the right classifier
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100) #ignore special/pad tokens
            loss = loss_fct(logits.view(-1, self.num_labels[task_name]), labels.view(-1))
        return TokenClassifierOutput(loss=loss, logits=logits) #huggingface output object

class XLMRobertaForTokenClassificationMultitask(RobertaPreTrainedModel):
    def __init__(self, config, task_labels_map):
        super().__init__(config)
        self.num_labels = task_labels_map
        self.roberta = XLMRobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.ModuleDict({
            "srl": nn.Linear(config.hidden_size, task_labels_map["srl"]),
            "ner": nn.Linear(config.hidden_size, task_labels_map["ner"])
        })
        self.init_weights()

    def get_input_embeddings(self):
        return self.roberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.roberta.set_input_embeddings(new_embeddings)

    def forward(self, input_ids, attention_mask=None, labels=None, task_name=None):
        if isinstance(task_name, list):
            task_name = task_name[0]
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier[task_name](sequence_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels[task_name]), labels.view(-1))
        return TokenClassifierOutput(loss=loss, logits=logits)
# Step 6: Custom trainer for batch alternation

#custom trainer that handles multitask batch construction
class MultitaskTrainer(transformers.Trainer):
    #builds a dataloader for a specific task (srl or ner)
    def get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        #choose appropriate sampler depending on single-GPU or distributed
        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1 or not torch.distributed.is_initialized()
            else DistributedSampler(train_dataset)
        )
        #wraps standard dataloader in a task-named loader
        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
            ),
        )
        return data_loader
    #constructs a multitask dataloader that samples batches from multiple tasks
    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        return MultitaskDataloader(
            {
                task_name: self.get_single_train_dataloader(task_name, task_dataset)
                for task_name, task_dataset in self.train_dataset.items()
            }
        )
        # other trainer methods (evaluate, predict, save_model) are inherited and stay compatible
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


file_paths = list_of_files(directory)

# Load mappings
with open('label_mapping_NER.json', 'r', encoding='utf-8') as f:
    label_mapping_NER = json.load(f)

# Get list of unique labels
label_list_NER = [label for label in label_mapping_NER.keys()]

# Load mapping
with open('label_mapping.json', 'r', encoding='utf-8') as f:
    label_mapping_SRL = json.load(f)

# Get list of unique labels
label_list_SRL = [label for label in label_mapping_SRL.keys()]


parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--epoch', type=int, required=True)
parser.add_argument('--batch_size', type=int, help="Batch size for training")
parser.add_argument('--model_checkpoint', type=str, required=True)
parser.add_argument('--model_type', type=str, required=True, help="Which type of model is used (BERT/RoBERTa/XLM-R)")
parser.add_argument('--directory', type=str, required=True, help="File path to where data is stored")
args = parser.parse_args()

learning_rate = args.learning_rate
epoch = args.epoch
batch_size = args.batch_size
model_checkpoint = args.model_checkpoint
model_type = args.model_type
directory = args.directory

#total_examples = len(srl_dataset) + len(ner_dataset)
SEED = 222
#steps_per_epoch = total_examples // batch_size
#max_steps = steps_per_epoch * num_epochs
value_to_key_mapping_SRL = {v: k for k, v in label_mapping_SRL.items()}
value_to_key_mapping_NER = {v: k for k, v in label_mapping_NER.items()}


all_f1_scores = []
all_precision_scores = []
all_recall_scores = []
all_accuracy_scores = []
all_loss_scores = []
all_metrics = []
metrics_log = []
all_classes = []


#cross validation per document
docs = range(len(file_paths)) 

for index in docs:
    train_files, test_file, test_file_name, fold = train_test_split_documents(file_paths, index)
    test_file_name = os.path.splitext(os.path.basename(test_file_name))[0]

    if model_type == 'BERT':
        model = BertForTokenClassificationMultitask.from_pretrained(
            model_checkpoint,
            config=AutoConfig.from_pretrained(model_checkpoint),
            task_labels_map={"srl": len(label_mapping_SRL), "ner": len(label_mapping_NER)}
        )
    elif model_type == 'RoBERTa':
        model = RobertaForTokenClassificationMultitask.from_pretrained(
            model_checkpoint,
            config=AutoConfig.from_pretrained(model_checkpoint),
            task_labels_map={"srl": len(label_mapping_SRL), "ner": len(label_mapping_NER)}
        )

    if model_type == 'XLM-R':
        model = XLMRobertaForTokenClassificationMultitask.from_pretrained(
        model_checkpoint,
        config=AutoConfig.from_pretrained(model_checkpoint),
        task_labels_map={"srl": len(label_mapping_SRL), "ner": len(label_mapping_NER)}
        )
    if model_type == 'BERT':
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    elif model_type == 'RoBERTa':
        tokenizer = RobertaTokenizerFast.from_pretrained(model_checkpoint, add_prefix_space=True)
    elif model_type == 'XLM-R':
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_checkpoint, add_prefix_space=True)
    special_tokens = {'additional_special_tokens': ['[PRED]']}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer)) #adjust for special token

    files_dict_train = process_file_to_dict(train_files, label_mapping_SRL, label_mapping_NER)
    files_dict_test = process_file_to_dict(test_file, label_mapping_SRL, label_mapping_NER)
    
    augmented_inputs_train = augment_sent_with_pred(files_dict_train, tokenizer)
    augmented_inputs_test = augment_sent_with_pred(files_dict_test, tokenizer)

    dataset_train = Dataset.from_list(augmented_inputs_train)
    dataset_test = Dataset.from_list(augmented_inputs_test)
    print("DEBUG: dataset_test[0] =", dataset_test[0])

    SRL_dataset_train = TaskSpecificDataset(dataset_train, task="srl")
    NER_dataset_train = TaskSpecificDataset(dataset_train, task="ner")

    SRL_dataset_test = TaskSpecificDataset(dataset_test, task="srl")
    NER_dataset_test = TaskSpecificDataset(dataset_test, task="ner")

    SRL_raw_dataset_test = SRL_dataset_test
    NER_raw_dataset_test = NER_dataset_test
    
    task = "MultiTask SRL and NERC"

    trainer = MultitaskTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=base_output_dir,
            per_device_train_batch_size=batch_size,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy = 'epoch',
            save_total_limit = 1,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            logging_dir="./logs", #necessary?
            seed=SEED,
            report_to=None
        ),
        train_dataset={"srl": SRL_dataset_train, "ner": NER_dataset_train},
        eval_dataset={'srl': SRL_dataset_test, 'ner': NER_dataset_test},
        tokenizer=tokenizer,
        data_collator=DataCollatorWithTaskName()
    )

    trainer.train()

    srl_metrics = trainer.evaluate(SRL_dataset_test)
    srl_loss = srl_metrics["eval_loss"]
        
    ner_metrics = trainer.evaluate(NER_dataset_test)
    ner_loss = ner_metrics["eval_loss"]

    # Predict SRL
    srl_predictions, srl_labels, _ = trainer.predict(SRL_dataset_test)

    # Predict NER
    ner_predictions, ner_labels, _ = trainer.predict(NER_dataset_test)

        # Collect training logs for the fold
    for log in trainer.state.log_history:
        logs = dict(log)
        logs['fold'] = fold + 1
        all_metrics.append(logs)
    
    srl_predictions = np.argmax(srl_predictions, axis=2)
    ner_predictions = np.argmax(ner_predictions, axis=2)
    

    all_loss_scores.append({'srl': srl_loss, 'ner': ner_loss})

    
    # Get the original sentences (test data)
    #SRL_test_dicts = SRL_raw_dataset_test.to_list()
    #NER_test_dicts = NER_raw_dataset_test.to_list()
    
    # Post-process predictions (converting indices to labels)
    srl_word_predictions, srl_word_labels = post_process(srl_predictions, srl_labels, augmented_inputs_test)
    ner_word_predictions, ner_word_labels = post_process(ner_predictions, ner_labels, augmented_inputs_test)
    
    # Get tokens for each sentence
    test_tokens = files_dict_test
    all_sents = []
    all_lbls = []
    for dct in test_tokens:
        sent = dct['text']
        all_sents.append(sent)
    
    
    # Flatten predictions and labels for both tasks for metrics
    flat_srl_preds = list(chain.from_iterable(srl_word_predictions))
    flat_srl_labels = list(chain.from_iterable(srl_word_labels))
    flat_ner_preds = list(chain.from_iterable(ner_word_predictions))
    flat_ner_labels = list(chain.from_iterable(ner_word_labels))
    
    # Convert indices to label strings (e.g., using value_to_key mapping)
    flat_srl_preds = [value_to_key_mapping_SRL[label] for label in flat_srl_preds]
    flat_srl_labels = [value_to_key_mapping_SRL[label] for label in flat_srl_labels]
    
    flat_ner_preds = [value_to_key_mapping_NER[label] for label in flat_ner_preds]
    flat_ner_labels = [value_to_key_mapping_NER[label] for label in flat_ner_labels]

        # Write predictions to file
        
    outputfile = os.path.join(base_output_dir, f"complete_pred_versus_gold_fold{fold+1}.txt")
    with open(outputfile, 'w') as outfile:
        outfile.write("Token\tGold\tPrediction\tTask\n")

        outfile.write(f"=== Fold {fold+1} Document {test_file_name} ===\n")
        for preds, golds, tokens in zip(srl_word_predictions, srl_word_labels, all_sents):
            for pred, gold, token in zip(preds, golds, tokens):
                pred_label = value_to_key_mapping_SRL[pred]
                gold_label = value_to_key_mapping_SRL[gold]
                outfile.write(f"{token}\t{gold_label}\t{pred_label}\tSRL\n")
            outfile.write("\n")
        for preds, golds, tokens in zip(ner_word_predictions, ner_word_labels, all_sents):
            for pred, gold, token in zip(preds, golds, tokens):
                pred_label = value_to_key_mapping_NER[pred]
                gold_label = value_to_key_mapping_NER[gold]
                outfile.write(f"{token}\t{gold_label}\t{pred_label}\tNER\n")
            outfile.write("\n")
    
    # Calculate metrics for SRL and NER
    srl_precision, srl_recall, srl_f1, srl_accuracy, srl_per_class = compute_filtered_macro_scores(flat_srl_labels, flat_srl_preds)

    ner_precision, ner_recall, ner_f1, ner_accuracy, ner_per_class = compute_filtered_macro_scores(flat_ner_labels, flat_ner_preds)
    
    # Append the results for the current fold
    all_precision_scores.append({'srl': srl_precision, 'ner': ner_precision})
    all_recall_scores.append({'srl': srl_recall, 'ner': ner_recall})
    all_f1_scores.append({'srl': srl_f1, 'ner': ner_f1})
    all_accuracy_scores.append({'srl': srl_accuracy, 'ner': ner_accuracy})
    all_classes.append({'srl': srl_per_class, 'ner': ner_per_class}) 
    
    # Save classification reports
    generate_report(flat_srl_labels, flat_srl_preds, label_list_SRL, fold, test_file_name, output_file=os.path.join(base_output_dir, 'srl_classification_reports.txt'))
    generate_report(flat_ner_labels, flat_ner_preds, label_list_NER, fold, test_file_name, output_file=os.path.join(base_output_dir, 'ner_classification_reports.txt'))
    #save confusion matrices
    cf_matrix_srl = generate_confusion_matrix(flat_srl_labels, flat_srl_preds, label_list_SRL, fold, test_file_name, output_plot_file=os.path.join(base_output_dir, f'SRL_CM_fold{fold+1}.png'), output_matrix_file=os.path.join(base_output_dir,'SRL_all_matrices.csv'))
    cf_matrix_ner = generate_confusion_matrix(flat_ner_labels, flat_ner_preds, label_list_NER, fold, test_file_name, output_plot_file=os.path.join(base_output_dir, f'NER_CM_fold{fold+1}.png'), output_matrix_file=os.path.join(base_output_dir, 'NER_all_matrices.csv'))
    #save CM as long format
    save_confusion_matrix_long_format(cf_matrix_srl, label_list_SRL, fold, test_file_name, output_long_matrix_file= os.path.join(base_output_dir, 'SRL_long_format_CM.csv'))
    save_confusion_matrix_long_format(cf_matrix_ner, label_list_NER, fold, test_file_name, output_long_matrix_file= os.path.join(base_output_dir, 'NER_long_format_CM.csv'))

# Save all metrics to a JSON file after all folds
metrics_file = os.path.join(base_output_dir, 'MT_all_metrics_complete.json')
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
        'per_class_scores': all_classes,
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


