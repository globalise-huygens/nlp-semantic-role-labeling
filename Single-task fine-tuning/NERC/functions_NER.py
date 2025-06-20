import glob
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import os

def list_of_files(directory):
    file_paths = []
    for path in glob.glob(os.path.join(directory, '**', '*.conllu'), recursive = True):
        if os.path.isfile(path):
            file_paths.append(path)
    return file_paths

def process_file_to_dict(file_paths, label_mapping):
    """
    Processes a file containing tokenized sentences with corresponding argument labels and predicates.
    Converts the data into a structured list of dictionaries, where each dictionary represents a sentence
    with its tokens, predicate status (True/False), and numeric argument labels.

    This function assumes the input file is in a tab-separated format where:
    - The second column contains the token (word),
    - The 11th column indicates whether the token is a predicate ('_' means no, any other value means yes),
    - The 12th column contains the argument label (or '_' for no label).
    
    The function returns a list of dictionaries, where each dictionary represents a sentence.

    Parameters:
        file_path (str): The path to the file to be processed.
        
        label_mapping (dict): A dictionary that maps argument labels (e.g., "ARG1", "ARG2", "O") to their corresponding numeric values.
            This mapping is used to convert string labels into numeric labels. 

    Output:
        list: A list of dictionaries, where each dictionary represents a sentence. Each dictionary contains:
            - 'text' (list): A list of tokens (words) in the sentence.
            - 'predicate' (list): A list of Boolean values indicating whether each token is a predicate (True or False).
            - 'label' (list): A list of numeric values corresponding to the argument label for each token. 
                The numeric labels are based on the `label_mapping` provided.
    

    """
    
    file_sentences = []
    for path in file_paths:
        with open(path, encoding = 'utf-8') as infile:
            sentence = []
            sentence_labels = []

            for line in infile:
                if line != '\n':
                    #remove the newline at the end of the line, then split the line based on the tab separator
                    line_split = line.strip('\n').split('\t')
                    #extract the token from the line in the file
                    token = line_split[4]



                    #check whether label for argument is in label_mapping (train_labels), else 'O'
                    argument = line_split[7]
                    #get numeric label
                    if argument in label_mapping.keys():
                        numeric_arg = label_mapping[argument]
                    else:
                        numeric_arg = label_mapping['O']

                    
                    #per sentence
                    sentence.append(token)    
                    sentence_labels.append(numeric_arg)
                else:
                    if sentence and sentence_labels:
                        sentence_dict = {'text':sentence, 'label':sentence_labels}
                    #get list of dicts
                        file_sentences.append(sentence_dict)
                        sentence_dict = {}
                        sentence = []

                    else:
                        sentence_dict = {}
                        sentence = []
                        sentence_labels = []


    return file_sentences

def augment_sent_with_pred(dict_sentences, tokenizer):
    """
    Augments a list of sentences with special tokens for predicates, tokenizes the augmented sentences, 
    and generates token-level information including token ids, word ids, and labels.

    This function takes a list of sentence dictionaries containing 'text', 'predicate', and 'label' keys. 
    It processes each sentence, appending the '[PRED]' token to the token identified as predicate. 
    The sentences are then tokenized, and word-level ids and labels are mapped to the tokenized words.

    Paramters:
        dict_sentences (list of dict): A list of dictionaries, each containing:
            - 'text': A list of tokens representing the sentence.
            - 'predicate': A list of boolean values indicating whether a token is a predicate.
            - 'label': A list of labels associated with each token.

    Output:
        list of dict: A list of dictionaries where each dictionary contains:
            - 'input_ids': The tokenized input IDs corresponding to the sentence.
            - 'attention_mask': 1 for token, 0 for padding, 0 should be ignored.
            - 'word_ids': The word IDs for each token in the sentence.
            - 'labels': The labels for each token in the tokenized sentence.
    """
    

    all_inputs = []
    #we iterate over the list of dictionaries
    for i,sentence_dict in enumerate(dict_sentences):
        list_of_sent=sentence_dict['text']
        labels=sentence_dict['label']
 
        temp_label = []
        temp_word_ids = []
        

        if list_of_sent:
            #the augmented sentence is tokenized, only if it contains all the necessary elements
            tokenized_sentence = tokenizer(list_of_sent, padding = 'max_length', truncation=True, max_length=512, is_split_into_words=True)
    

            tokenized_tokens = tokenizer.convert_ids_to_tokens(tokenized_sentence['input_ids'])

            # A subword tokenizer means that that even if your inputs have been split into words already, each of those words could be split again by the tokenizer. 
            #This can result in more input_ids than the list of labels for each word. To account for this, by using word.ids() maps special tokens (like [CLS] and [SEP]) 
            #to -100 and all tokenized input (input_ids) to their respective word. The storing of the special tokens as -100 is used for ignoring padding during loss computation. 

            word_ids = tokenized_sentence.word_ids()
            for id in word_ids:
                if id is not None:
                    temp_label.append(labels[id])
                    temp_word_ids.append(id)
                else:
                    temp_label.append(-100)
                    temp_word_ids.append(-100)

            tokenized_sentence['word_ids'] = temp_word_ids
            tokenized_sentence['labels'] = temp_label
            #dictionary with input_ids, attention_mask, word_ids, and labels as keys
            #list of dicts
            all_inputs.append(tokenized_sentence)

            #empty lists for new sentence dict
            temp_word_ids=[]
            temp_label=[]

    
    
                

    return all_inputs

def train_test_split_documents(file_paths, index):
    test_file = []
    train_files = []
    for i, path in enumerate(file_paths):
        if i == index:
            test_file_name = path 
            test_file.append(path)
        else:
            train_files.append(path)

    return train_files, test_file, test_file_name, index

def post_process(predictions, labels, augmented):
    
    """
    Post-processes the token-level predictions and labels to produce word-level predictions and labels.

    This function aggregates the token-level predictions and labels into word-level predictions and labels.
    It groups tokens that belong to the same word, takes a majority vote for each word, and produces word-level
    predictions. It handles special cases such as subword tokenization where a word may be split into multiple tokens.

    Parameters:
        predictions (list of list of int): A list of lists where each inner list contains the token-level 
                                            predictions (e.g., predicted semantic roles or labels) for a sentence.
        labels (list of list of int): A list of lists where each inner list contains the token-level true labels 
                                      for a sentence (with -100 used for padding tokens).
        augmented (list of dict): A list of dictionaries where each dictionary contains tokenized information 
                                  for a sentence, including the 'word_ids' which maps tokens back to their word-level IDs.

    Output:
            - word_level_preds (list of list of int): A list of lists where each inner list contains the word-level 
                                                      predictions for a sentence, with each word represented by a single 
                                                      prediction (majority vote from subwords).
            - word_level_labels (list of list of int): A list of lists where each inner list contains the word-level 
                                                      true labels for a sentence, with each word represented by a single 
                                                      label (majority vote from subwords).
    """

    word_level_preds = []  # This will store the final word-level predictions
    word_level_labels = []
    all_preds = []
    all_labels = []

    for lst_pred, lst_label in zip(predictions, labels):
        sent_preds = []
        sent_labels = []
        for pred, label in zip(lst_pred, lst_label):
            if label != -100:
                sent_preds.append(pred)
                sent_labels.append(label)

        all_preds.append(sent_preds)
        all_labels.append(sent_labels)

    word_ids = []
    for dct in augmented:
        sent_ids = []
        word_id = dct['word_ids']
        for idx in word_id:
            if idx != -100:
                sent_ids.append(idx)
        word_ids.append(sent_ids)
    
    sent_preds = []  # To store predictions for the current sentence
    sent_labels = []
    word_level = []  # To collect predictions for tokens of the same word
    word_label = []
    prev_word_id = None
    
    # Iterate over word_ids, predictions, and true_labels to group by word
    for ids, preds, lbls in zip(word_ids, all_preds, all_labels):
        sent_preds = []  # To store predictions for the current sentence
        sent_labels = []
        word_level = []  # To collect predictions for tokens of the same word
        word_label = []
        prev_word_id = None
        
        for word_id, pred, label in zip(ids, preds, lbls):
    
            if len(word_level) == 0:  # Handling the first token in a new word group
                word_level.append(pred)
                word_label.append(label)
                prev_word_id = word_id  # Assign word_id to prev_word_id
            elif word_id == prev_word_id:  # Same word group, append prediction
                word_level.append(pred)
                word_label.append(label)
            else: # calculate sentence-level prediction
                one_pred = np.argmax(np.bincount(word_level))  ## Gets the majority vote of the subtokens
                one_label = np.argmax(np.bincount(word_label))
                sent_preds.append(one_pred)
                sent_labels.append(one_label)

                word_level = [pred]  # Start a new group with the current prediction
                word_label = [label]
                prev_word_id = word_id  # Update previous word ID

        # Handle the last word group after iterating through all tokens
        if word_level:
            one_pred = np.argmax(np.bincount(word_level))  ## Gets the majority vote of the subtokens
            one_label = np.argmax(np.bincount(word_label))
            sent_preds.append(one_pred)
            sent_labels.append(one_label)
        # Store sentence-level predictions for the current sample
        word_level_preds.append(sent_preds)
        word_level_labels.append(sent_labels)
    return word_level_preds, word_level_labels
            

def generate_report(all_labels, all_preds, label_list, fold, test_file_name, output_file):
    """
    Displays confusion matrix and classification report of predictions against the gold labels.

    Parameters:
    all_preds: word_level predictions
    all_labels: gold labels
   

    Output:
    Display of confusion matrix and print of classification report.

    """

    report = classification_report(all_labels, all_preds, labels=label_list)
    print("Classification Report -----------------------------------------------------------")
    print(report)


    output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'a') as f:
        if fold + 1 == 1:
            f.write('Classification Report all Folds\n')
            f.write('-------------------------------------------------\n')
        f.write(f"Classification Report for Fold {fold + 1}\n")
        f.write(f"Classification Report for File: {test_file_name}\n")
        f.write(report)
        f.write("\n-------------------------------------------------\n")


def generate_confusion_matrix(all_labels, all_preds, label_list, fold, test_file_name, output_plot_file=None, output_matrix_file=None):
    cf_matrix = confusion_matrix(all_labels, all_preds, labels=label_list)

     # Create a heatmap
    plt.figure(figsize=(30, 24))
    ax = sns.heatmap(cf_matrix, annot=True, fmt='d', cmap="Blues",
                     xticklabels=label_list, yticklabels=label_list,
                     cbar=False, annot_kws={"size":14})

    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.title('Confusion Matrix', fontsize=18)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.tight_layout()

    if output_plot_file:
        plt.savefig(output_plot_file, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    if output_matrix_file:
        df_cm = pd.DataFrame(cf_matrix, index=label_list, columns=label_list)

        file_exists = os.path.exists(output_matrix_file)

        separator = '-'*50
        header_text = f"{separator}\nConfusion matrix for Fold {fold + 1}, Test File: {test_file_name}\n{separator}\n\n"

        # Open file once
        with open(output_matrix_file, 'a') as f:
            if file_exists:
                f.write("\n")  # Blank line before new fold if not first
            f.write(header_text)

            # Now append the DataFrame manually
            df_cm.to_csv(f, index=True, header=True)

            f.write('\n\n')  # Blank lines after matrix

    return cf_matrix

def save_confusion_matrix_long_format(cf_matrix, label_list, fold, test_file_name, output_long_matrix_file):
    rows = []
    for i, true_label in enumerate(label_list):
        for j, pred_label in enumerate(label_list):
            count = cf_matrix[i, j]
            rows.append({
                "Fold": fold + 1,
                "Test_File": test_file_name,
                "True_Label": true_label,
                "Predicted_Label": pred_label,
                "Count": count
            })
    
    df_long = pd.DataFrame(rows)

    # Check if file exists
    file_exists = os.path.exists(output_long_matrix_file)

    # Save
    df_long.to_csv(output_long_matrix_file, mode='a', index=False, header=not file_exists)

def compute_filtered_macro_scores(y_true, y_pred):
    """
    Computes macro-averaged precision, recall, and F1, ignoring labels with support=0.
    Returns: macro scores, accuracy, and per-class metrics for included labels.
    """
    labels = sorted(list(set(y_true + y_pred)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    mask = np.array(support) > 0
    filtered_labels = np.array(labels)[mask]
    filtered_precision = precision[mask]
    filtered_recall = recall[mask]
    filtered_f1 = f1[mask]
    filtered_support = np.array(support)[mask]

    macro_precision = np.mean(filtered_precision)
    macro_recall = np.mean(filtered_recall)
    macro_f1 = np.mean(filtered_f1)
    accuracy = accuracy_score(y_true, y_pred)

    per_class = {
        label: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s)
        }
        for label, p, r, f, s in zip(filtered_labels, filtered_precision, filtered_recall, filtered_f1, filtered_support)
    }

    return macro_precision, macro_recall, macro_f1, accuracy, per_class
