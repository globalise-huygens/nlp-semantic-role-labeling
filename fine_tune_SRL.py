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
from datasets import Dataset
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import re
from statistics import Counter
import argparse

import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from functions_SRL import process_file_to_dict, augment_sent_with_pred, post_process, generate_report, get_first_non_O_label

parser = argparse.ArgumentParser(description="Fine-tune SRL model")
parser.add_argument('--learning_rate', type=float, help="Learning rate for training")
parser.add_argument('--epoch', type=int, help="Number of epochs for training")
parser.add_argument('--model_checkpoint', type=str, help="Model checkpoint name")
parser.add_argument('--num_folds', type=int, help="Number of splits for KFold CV")
parser.add_argument('--subsetsize', type=int, help="Subset size for tuning")

args = parser.parse_args()

learning_rate = args.learning_rate
epoch = args.epoch
model_checkpoint = args.model_checkpoint
num_folds = args.num_folds
subsetsize = args.subsetsize

directory = '../Data/SRL_train_with_entities'

def list_of_files(directory):
    file_paths = []
    for path in glob.glob(os.path.join(directory, '**', '*.conllu'), recursive = True):
        if os.path.isfile(path):
            file_paths.append(path)
    return file_paths

file_paths = list_of_files(directory)
print(file_paths)
#load mapping
with open('label_mapping', 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)

#get list of unique labels
label_list = []
for label in label_mapping.keys():
    label_list.append(label)
print(label_list)

task = "semantic-role-labeling"
batch_size = 16 #determines the number of examples that are processed in one forward pass (when data is passed through the model
SEED = 222

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
special_tokens = {'additional_special_tokens': ['[PRED]']}
tokenizer.add_special_tokens(special_tokens)
print(tokenizer.all_special_tokens)

#The DataCollatorForTokenClassification handles dynamic padding of input sequences during training.
#It ensures that all sequences in a batch are padded to the longest sequence length, maintaining consistency while keeping memory usage optimized.
data_collator = DataCollatorForTokenClassification(tokenizer)

files_dict = process_file_to_dict(file_paths, label_mapping)

augmented_inputs = augment_sent_with_pred(files_dict, tokenizer)
dataset_files = Dataset.from_list(augmented_inputs)
dataset_files = dataset_files.select(range(subsetsize))
print(dataset_files)


# We use the predefined metric from seqeval to calculate the model's performance metric = load("seqeval") 

metric = load("seqeval")
#metric = evaluate.load("./seqeval/seqeval.py")

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
    predictions, labels = p  # Unpack the tuple 'p' into predictions and labels

    # Convert the model's predictions (raw probabilities) to the predicted class labels
    # np.argmax(predictions, axis=2) gives the index of the highest probability for each token (token classification task)
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens) (like padding tokens, denoted by -100)
    # We filter out predictions and labels where the label is -100 (special token, padding)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]  # Use the label list to map predicted indices to labels
        for prediction, label in zip(predictions, labels)  # Iterate through each batch (prediction, label pair)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]  # Same for true labels
        for prediction, label in zip(predictions, labels)  # Iterate through each batch (prediction, label pair)
    ]

    # Compute the evaluation metrics using the seqeval library
    # This calculates precision, recall, F1-score, and accuracy based on the filtered predictions and true labels
    results = metric.compute(predictions=true_predictions, references=true_labels)

    # Return the computed metrics in a dictionary
    return {
        "precision": results["overall_precision"],  # Precision metric
        "recall": results["overall_recall"],        # Recall metric
        "f1": results["overall_f1"],                # F1 score (harmonic mean of precision and recall)
        "accuracy": results["overall_accuracy"],    # Accuracy metric
    }

raw_dataset = dataset_files
dataset = dataset_files 
dataset = dataset.remove_columns(["token_type_ids", "word_ids"])
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

id2label = {v: k for k, v in label_mapping.items()}
# Generate stratify labels for all sequences
stratify_labels = [get_first_non_O_label(seq, id2label) for seq in dataset["labels"]]

#stratified KFold
#KFold

all_f1_scores = []
all_precision_scores = []
all_recall_scores = []
all_accuracy_scores = []
all_loss_scores = []
all_metrics = []


#stratified: 
folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state = 42)
for fold, (train_index, test_index) in enumerate(folds.split(dataset, stratify_labels)):
    print(f"\nFold {fold + 1} Training and Evaluation:")
    #initialize model
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

    # to include the added special tokens cue[0],cue[1],cue[2], model is resized
    model.resize_token_embeddings(len(tokenizer))

    model_name = model_checkpoint.split("/")[-1]
    args = TrainingArguments(
        f"{model_name}-finetuned-{task}",
        eval_strategy = "epoch",
        logging_strategy = 'epoch', 
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epoch,
        weight_decay=0.01,
        seed=SEED,
        report_to=None,
    )
    # Split dataset into training and validation sets for the fold
    train_data = dataset.select(train_index)
    test_data = dataset.select(test_index)
    #initilize model trainer 
    trainer = Trainer(
        model, # BERT model
        args, #training arguments that define training parameters like learning rate, batch size as specified above
        train_dataset=train_data,   # dataset used for training
        eval_dataset=test_data, #dataset used for evaluating the model during training
        #Data collator to batch and preprocess the data (handles padding, tokenization, etc.)
        data_collator=data_collator, 
        #tokenizer as initiated in the beginning
        tokenizer=tokenizer,
        #function to compute evaluation metrics (precision, recall, f1-score)
        compute_metrics=compute_metrics #evaluation metrics based on model's predictions

        )

    trainer.train()
    trainer.evaluate()

    for log in trainer.state.log_history:
        logs= dict(log)
        logs['fold'] = fold + 1
        all_metrics.append(logs)
    
    test_dicts = raw_dataset.select(test_index).to_list()

    predictions, labels, metrics = trainer.predict(test_data)
    predictions = np.argmax(predictions, axis=2)
    loss = metrics["test_loss"] 
    all_loss_scores.append(loss)
    
    test_dict = test_data.to_list()
    word_predictions, word_labels = post_process(predictions, labels, test_dicts)

    test_tokens = [files_dict[i] for i in test_index]
    all_sents = []
    all_lbls = []
    for dct in test_tokens:
        sent = dct['text']
        all_sents.append(sent)

    value_to_key_mapping = {v: k for k, v in label_mapping.items()}

    outputfile = 'pred_versus_gold_test.txt' 
    with open(outputfile, 'a') as outfile:
        # Write the column headers to the TSV file
        if outfile.tell() == 0:
            outfile.write("Token\tGold\tPrediction\n")  # Column headers for TSV
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

    
    precision = precision_score(flat_labels, flat_predictions, average = 'macro')  # Precision metric
    recall = recall_score(flat_labels, flat_predictions, average= 'macro')        # Recall metric
    f1 = f1_score(flat_labels, flat_predictions, average = 'macro')               # F1 score (harmonic mean of precision and recall)
    accuracy = accuracy_score(flat_labels, flat_predictions)  # Accuracy metric
    all_precision_scores.append(precision)
    all_recall_scores.append(recall)
    all_f1_scores.append(f1)
    all_accuracy_scores.append(accuracy)
    generate_report(flat_labels, flat_predictions, label_list, fold, output_file = 'classification_reports_test.txt')
    
data_scores = {'precision': all_precision_scores, 'recall': all_recall_scores, 'f1': all_f1_scores, 'accuracy': all_accuracy_scores, 'loss': all_loss_scores}
# Save to a JSON file
metrics_file = 'all_metrics_test.json'
run_result = {'learning_rate': learning_rate, 'epochs': epoch, 'model_checkpoint': model_checkpoint, 
              'metrics': {'precision': all_precision_scores, 'recall': all_recall_scores, 'f1': all_f1_scores, 'accuracy': all_accuracy_scores, 'loss': all_loss_scores, 'epoch_logs': all_metrics}}

#with open('agg_metrics_test.json', 'a') as f:
    #json.dump(data_scores, f)

if os.path.exists(metrics_file):
    with open(metrics_file, 'r') as f: 
        all_runs = json.load(f)
else:
    all_runs = []

all_runs.append(run_result)

with open(metrics_file, 'w') as f:
    json.dump(all_runs, f)

