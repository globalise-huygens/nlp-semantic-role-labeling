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

from functions_MT import list_of_files, process_file_to_dict, augment_sent_with_pred, get_first_non_O_label, post_process, generate_report, log_fold_metrics
from transformers import (
    AutoConfig, AutoTokenizer, AutoModel,
    BertModel, RobertaModel, XLMRobertaModel,
    PreTrainedModel, BertPreTrainedModel
)

# Directory and file paths setup
directory = '../Data/SRL_train_with_entities' #path to files

file_paths = list_of_files(directory) #get list of files

# Load mappings
with open('label_mapping_NER', 'r', encoding='utf-8') as f:
    label_mapping_NER = json.load(f)

# Get list of unique labels NER
label_list_NER = [label for label in label_mapping_NER.keys()]

# Load mapping
with open('label_mapping', 'r', encoding='utf-8') as f:
    label_mapping_SRL = json.load(f)

# Get list of unique labels SRL
label_list_SRL = [label for label in label_mapping_SRL.keys()]
print(label_list_SRL)

#argument parser 
parser = argparse.ArgumentParser(description="Fine-tune SRL model")
parser.add_argument('--learning_rate', type=float, help="Learning rate for training")
parser.add_argument('--epoch', type=int, help="Number of epochs for training")
parser.add_argument('--model_checkpoint', type=str, help="Model checkpoint name")
parser.add_argument('--num_folds', type=int, help="Number of splits for KFold CV")
parser.add_argument('--subsetsize', type=int, help="Subset size for tuning")
args = parser.parse_args() #parse command-line arguments

#arguments to variables 
learning_rate = args.learning_rate
epoch = args.epoch
model_checkpoint = args.model_checkpoint
num_folds = args.num_folds
#subsetsize = args.subsetsize
subsetsize = 300

#initialize task name, batch size, tokenizer and add special token to indicate predicate [PRED]
task = "MultiTask SRL and NERC"
batch_size = 16 
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
special_tokens = {'additional_special_tokens': ['[PRED]']}
tokenizer.add_special_tokens(special_tokens)

#pre process raw data
files_dict = process_file_to_dict(file_paths, label_mapping_SRL, label_mapping_NER)
print(files_dict[0])
#add predicate token and tokenize data (input id's, word id's, attention mask, labels)
augmented_files = augment_sent_with_pred(files_dict, tokenizer)
#convert to Dataset
dataset = Dataset.from_list(augmented_files)
dataset = dataset.select(range(subsetsize))

class TaskSpecificDataset(torch.utils.data.Dataset):
    def __init__(self, full_data, task):
        self.data = full_data
        self.task = task  # 'srl' or 'ner'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx].copy()
        item["labels"] = item[f"labels_{self.task.upper()}"]  # set labels from correct column
        item["task_name"] = self.task
        return item
    
srl_dataset = TaskSpecificDataset(dataset, task="srl")
ner_dataset = TaskSpecificDataset(dataset, task="ner")

#multitask DataCollator
class DataCollatorForMultiTask:
    def __call__(self, features):
        #stack features into batch tensors
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        labels_SRL = torch.stack([f["labels_SRL"] for f in features])
        labels_NER = torch.stack([f["labels_NER"] for f in features])

    
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels_SRL": labels_SRL,
            "labels_NER": labels_NER,
        }
    
#multitask model
class BertForMultitaskTokenClassification(BertPreTrainedModel):
    def __init__(self, config, srl_num_labels, ner_num_labels):
        super().__init__(config)
        self.num_labels_srl = srl_num_labels #number srl labels
        self.num_labels_ner = ner_num_labels #number ner labels

        self.bert = BertModel(config) #shared BERT encoder
        self.dropout = nn.Dropout(config.hidden_dropout_prob) #apply dropout (regularization technique) before classification

        self.srl_classifier = nn.Linear(config.hidden_size, srl_num_labels) #task-specific SRL head
        self.ner_classifier = nn.Linear(config.hidden_size, ner_num_labels) #task-specific NER head

        self.init_weights() #initialize model weights

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                labels_SRL=None, labels_NER=None):

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids if token_type_ids is not None else None) #run input through model

        sequence_output = self.dropout(outputs[0]) #dropout applied to token embeddings 

        srl_logits = self.srl_classifier(sequence_output) #predict SRL labels
        ner_logits = self.ner_classifier(sequence_output) #predict NER labels

        loss = None
        if labels_SRL is not None and labels_NER is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100) #only comput loss if labels exist
            loss_srl = loss_fct(srl_logits.view(-1, self.num_labels_srl), labels_SRL.view(-1)) #SRL loss
            loss_ner = loss_fct(ner_logits.view(-1, self.num_labels_ner), labels_NER.view(-1)) #NER loss
            loss = 0.5*loss_srl + 0.5*loss_ner #average losses

        if not self.training: #if evaluation mode, store individual losses
            self.eval_loss_srl = loss_srl.detach()
            self.eval_loss_ner = loss_ner.detach()

        return (loss, srl_logits, ner_logits) # return combined loss and both predictions 

class MultiTaskTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval() #switch to eval mode
        with torch.no_grad(): #disable gradients for inference
            inputs = self._prepare_inputs(inputs) #prepares input batch
            loss, srl_logits, ner_logits = model(**inputs) #forward pass
        return (loss, (srl_logits, ner_logits), (inputs["labels_SRL"], inputs["labels_NER"])) #return all values
    
# Metrics computation with seqeval, like single task
seqeval = load("seqeval")

def compute_metrics_seqeval(eval_preds):
    srl_logits, ner_logits = eval_preds.predictions #predicted logits for both tasks
    labels_srl, labels_ner = eval_preds.label_ids #true labels for both tasks

    srl_preds = np.argmax(srl_logits, axis=2) #take index of max logit for each token
    ner_preds = np.argmax(ner_logits, axis=2)

    def to_str_labels(preds, labels, id2label): #convert indices to label strings
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

    srl_preds_str, srl_labels_str = to_str_labels(srl_preds, labels_srl, value_to_key_mapping_SRL)
    ner_preds_str, ner_labels_str = to_str_labels(ner_preds, labels_ner, value_to_key_mapping_NER)

    srl_result = seqeval.compute(predictions=srl_preds_str, references=srl_labels_str) #metrics
    ner_result = seqeval.compute(predictions=ner_preds_str, references=ner_labels_str)

    return {
        "srl_f1": srl_result["overall_f1"],
        "nerc_f1": ner_result["overall_f1"],
        "srl_precision": srl_result["overall_precision"],
        "nerc_precision": ner_result["overall_precision"],
        "srl_recall": srl_result["overall_recall"],
        "nerc_recall": ner_result["overall_recall"],
    }

# Ensure num_folds and SEED are defined
num_folds = 3 # Or the desired number of folds
SEED = 42  # For reproducibility
batch_size = 16

raw_dataset = dataset
columns_to_remove = ["token_type_ids", "word_ids"]
# Check if columns exist before removing
existing_columns = [col for col in columns_to_remove if col in dataset.column_names]
dataset = dataset.remove_columns(existing_columns)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels_SRL", "labels_NER"])
# Prepare for stratified splitting
value_to_key_mapping_SRL = {v: k for k, v in label_mapping_SRL.items()}
value_to_key_mapping_NER = {v: k for k, v in label_mapping_NER.items()}
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

    train_data = dataset.select(train_idx) #training subset
    val_data = dataset.select(val_idx) #eval subset

    data_collator = DataCollatorForMultiTask() #multitask batch collator
    config = AutoConfig.from_pretrained(model_checkpoint) #load model config
    model = BertForMultitaskTokenClassification.from_pretrained(
        model_checkpoint,
        config=config,
        srl_num_labels=len(label_list_SRL),
        ner_num_labels=len(label_list_NER)
    )
    model.resize_token_embeddings(len(tokenizer)) #adjust for special token

    training_args = TrainingArguments(
        output_dir="./multitask-checkpoints",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=4,
        num_train_epochs=epoch,
        weight_decay=0.01,
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_seqeval,
    )

    trainer.train()
    trainer.evaluate()

    # Get predictions for test set
    predictions, labels, metrics = trainer.predict(val_data)

    loss_srl_val = model.eval_loss_srl.item() #get loss
    loss_ner_val = model.eval_loss_ner.item()

        # Collect training logs for the fold
    for log in trainer.state.log_history:
        logs = dict(log)
        logs['fold'] = fold + 1
        all_metrics.append(logs)
    
    # Apply argmax to predictions
    srl_predictions, ner_predictions = predictions # task predictions

    labels_srl, labels_ner = labels  #task labels
    srl_predictions = np.argmax(srl_predictions, axis=2)
    ner_predictions = np.argmax(ner_predictions, axis=2)
    
    # Collect the loss score
    loss = metrics["test_loss"]
    all_loss_scores.append(loss)
    
    # Get the original sentences (test data)
    test_dicts = raw_dataset.select(val_idx).to_list()
    
    
    # Post-process predictions (converting indices to labels)
    srl_word_predictions, srl_word_labels = post_process(srl_predictions, labels[0], test_dicts)
    ner_word_predictions, ner_word_labels = post_process(ner_predictions, labels[1], test_dicts)
    
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
    
    
    log_fold_metrics(
    fold=fold,
    alpha=0.5,
    beta=0.5,
    loss_srl=loss_srl_val,
    loss_ner=loss_ner_val,
    f1_srl=srl_f1,
    f1_ner=ner_f1,
    log_list=metrics_log
    	)


    # Append the results for the current fold
    all_precision_scores.append({'srl': srl_precision, 'ner': ner_precision})
    all_recall_scores.append({'srl': srl_recall, 'ner': ner_recall})
    all_f1_scores.append({'srl': srl_f1, 'ner': ner_f1})
    all_accuracy_scores.append({'srl': srl_accuracy, 'ner': ner_accuracy})
    
    # Save classification reports
    generate_report(flat_srl_labels, flat_srl_preds, value_to_key_mapping_SRL, fold, output_file='srl_classification_reports_test.txt')
    generate_report(flat_ner_labels, flat_ner_preds, value_to_key_mapping_NER, fold, output_file='ner_classification_reports_test.txt')

with open("MT_fold_weight_log.json", "w") as f:
    json.dump(metrics_log, f, indent=4)

# Save all metrics to a JSON file after all folds
metrics_file = 'MT_all_metrics_test.json'
run_result = {
    'learning_rate': learning_rate,
    'epochs': epoch,
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
