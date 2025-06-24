import glob
import os

base_output_dir = os.path.join(os.environ["TMPDIR"], "outputs")
os.makedirs(base_output_dir, exist_ok=True)
os.environ["HF_HOME"] = os.path.join(os.environ["TMPDIR"], "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ["TMPDIR"], "hf_datasets")
os.environ["HF_METRICS_CACHE"] = os.environ["HF_DATASETS_CACHE"]

from sklearn.preprocessing import LabelEncoder
import json
from evaluate import load
import numpy as np
from itertools import chain
import itertools
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from datasets import Dataset
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, RobertaTokenizerFast, XLMRobertaTokenizerFast
import re
from statistics import Counter
import argparse

import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from functions_SRL import process_file_to_dict, augment_sent_with_pred, train_test_split_documents, post_process, generate_report, generate_confusion_matrix, save_confusion_matrix_long_format, compute_filtered_macro_scores

parser = argparse.ArgumentParser(description="Fine-tune SRL model")
parser.add_argument('--learning_rate', type=float, help="Learning rate for training")
parser.add_argument('--epoch', type=int, help="Number of epochs for training")
parser.add_argument('--batch_size', type=int, help="Batch size for training")
parser.add_argument('--model_checkpoint', type=str, help="Model checkpoint name")
parser.add_argument('--model_type', type=str, required=True)
args = parser.parse_args()

learning_rate = args.learning_rate
epoch = args.epoch
batch_size = args.batch_size
model_checkpoint = args.model_checkpoint
model_type = args.model_type

directory = '../Data/SRL_train_with_entities'

def list_of_files(directory):
    file_paths = []
    for path in glob.glob(os.path.join(directory, '**', '*.conllu'), recursive = True):
        if os.path.isfile(path):
            file_paths.append(path)
    return file_paths

file_paths = list_of_files(directory)

#load mapping
with open('label_mapping.json', 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)

#get list of unique labels
label_list = []
for label in label_mapping.keys():
    label_list.append(label)

task = "semantic-role-labeling"
SEED = 222

metric = load("seqeval")

def compute_metrics(p):
    """
    Computes the performance metrics (precision, recall, F1, accuracy) for token classification tasks.
    
    Args:
        p (tuple): A tuple containing two elements:
            - predictions: The predicted labels from the model.
            - labels: The true labels (ground truth) for the tokens.
    
    Returns:
        dict: A dictionary containing the evaluation metrics (precision, recall, F1, accuracy).
    """
    predictions, labels = p 

    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens) (like padding tokens, denoted by -100)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]  # Use the label list to map predicted indices to labels
        for prediction, label in zip(predictions, labels)  # Iterate through each batch (prediction, label pair)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]  # Same for true labels
        for prediction, label in zip(predictions, labels)  # Iterate through each batch (prediction, label pair)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    # Return the computed metrics in a dictionary
    return {
        "precision": results["overall_precision"],  # Precision metric
        "recall": results["overall_recall"],        # Recall metric
        "f1": results["overall_f1"],                # F1 score (harmonic mean of precision and recall)
        "accuracy": results["overall_accuracy"],    # Accuracy metric
    }

all_f1_scores = []
all_precision_scores = []
all_recall_scores = []
all_accuracy_scores = []
all_loss_scores = []
all_metrics = []
all_classes = []

docs = range(len(file_paths)) 

for index in docs:

    train_files, test_file, test_file_name, fold = train_test_split_documents(file_paths, index)
    test_file_name = os.path.splitext(os.path.basename(test_file_name))[0]

    if model_type == 'BERT':
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    elif model_type == 'RoBERTa':
        tokenizer = RobertaTokenizerFast.from_pretrained(model_checkpoint, add_prefix_space=True)
    elif model_type == 'XLM-R':
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_checkpoint, add_prefix_space=True)
    special_tokens = {'additional_special_tokens': ['[PRED]']}
    tokenizer.add_special_tokens(special_tokens)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
    # to include the added special tokens cue[0],cue[1],cue[2], model is resized
    model.resize_token_embeddings(len(tokenizer))
    data_collator = DataCollatorForTokenClassification(tokenizer)


    files_dict_train = process_file_to_dict(train_files, label_mapping)
    files_dict_test = process_file_to_dict(test_file, label_mapping)
    augmented_inputs_train = augment_sent_with_pred(files_dict_train, tokenizer)
    augmented_inputs_test = augment_sent_with_pred(files_dict_test, tokenizer)
    dataset_train = Dataset.from_list(augmented_inputs_train)
    dataset_test = Dataset.from_list(augmented_inputs_test)
    raw_dataset_test = dataset_test

    columns_to_remove = ["token_type_ids", "word_ids"]
    # Check if columns exist before removing
    existing_columns = [col for col in columns_to_remove if col in dataset_train.column_names]
    dataset_train = dataset_train.remove_columns(existing_columns)
    dataset_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dataset_test = dataset_test.remove_columns(existing_columns)
    dataset_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    #initialize model

    model_name = model_checkpoint.split("/")[-1]
    args = TrainingArguments(
        output_dir = base_output_dir,
        eval_strategy = "epoch",
        logging_strategy = 'epoch', 
        save_strategy = 'epoch',
        save_total_limit = 1,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epoch,
        weight_decay=0.01,
        seed=SEED,
        report_to=None,
    )

    #initilize model trainer 
    trainer = Trainer(
        model,
        args, 
        train_dataset=dataset_train,   
        eval_dataset=dataset_test,
        data_collator=data_collator, 
        tokenizer=tokenizer,
        compute_metrics=compute_metrics 
        )

    trainer.train()
    trainer.evaluate()

    for log in trainer.state.log_history:
        logs= dict(log)
        logs['fold'] = fold + 1
        all_metrics.append(logs)

    predictions, labels, metrics = trainer.predict(dataset_test)
    predictions = np.argmax(predictions, axis=2)
    loss = metrics["test_loss"] 
    all_loss_scores.append(loss)

    word_predictions, word_labels = post_process(predictions, labels, augmented_inputs_test)

    test_tokens = files_dict_test
    all_sents = []
    all_lbls = []
    for dct in test_tokens:
        sent = dct['text']
        all_sents.append(sent)

    value_to_key_mapping = {v: k for k, v in label_mapping.items()}

    outputfile = os.path.join(base_output_dir, f"complete_pred_versus_gold_fold{fold+1}.txt") 
    with open(outputfile, 'w') as outfile:
        # Write the column headers to the TSV file
        if outfile.tell() == 0:
            outfile.write("Token\tGold\tPrediction\n")  # Column headers for TSV

        outfile.write(f"=== Fold {fold+1} Document {test_file_name} ===\n")
        for preds, labels, tokens in zip(word_predictions, word_labels, all_sents):
            for pred, label, token in zip(preds, labels, tokens):
                pred = value_to_key_mapping[pred]
                label = value_to_key_mapping[label]
                outfile.write(f"{token}\t{label}\t{pred}\n")
            outfile.write("\n")
                    

    # Convert indices to label strings
    flat_predictions = list(itertools.chain.from_iterable(word_predictions))
    flat_labels = list(itertools.chain.from_iterable(word_labels))
    flat_predictions = [value_to_key_mapping[label] for label in flat_predictions]
    flat_labels = [value_to_key_mapping[label] for label in flat_labels]

    precision, recall, f1, accuracy, per_class = compute_filtered_macro_scores(flat_labels, flat_predictions)

    all_precision_scores.append(precision)
    all_recall_scores.append(recall)
    all_f1_scores.append(f1)
    all_accuracy_scores.append(accuracy)
    all_classes.append(per_class)
    generate_report(flat_labels, flat_predictions, label_list, fold, test_file_name, output_file = os.path.join(base_output_dir, "classification_reports.txt"))
    cf_matrix = generate_confusion_matrix(flat_labels, flat_predictions, label_list, fold, test_file_name, output_plot_file= os.path.join(base_output_dir, f"SRL_CM_fold{fold+1}.png"), output_matrix_file= os.path.join(base_output_dir,'all_matrices.csv'))
    save_confusion_matrix_long_format(cf_matrix, label_list, fold, test_file_name, output_long_matrix_file=os.path.join(base_output_dir, "long_format_matrix.csv"))

# Save to a JSON file
metrics_file = os.path.join(base_output_dir, "all_metrics_complete.json")
run_result = {'learning_rate': learning_rate, 'epochs': epoch, 'model_checkpoint': model_checkpoint, 'batch_size':batch_size,
            'metrics': {'precision': all_precision_scores, 'recall': all_recall_scores, 'f1': all_f1_scores, 'accuracy': all_accuracy_scores, 'per_class_scores': all_classes, 'loss': all_loss_scores, 'epoch_logs': all_metrics}}

if os.path.exists(metrics_file):
    with open(metrics_file, 'r') as f: 
        all_runs = json.load(f)
else:
    all_runs = []

all_runs.append(run_result)

with open(metrics_file, 'w') as f:
    json.dump(all_runs, f, indent=2)
