import glob
def list_of_files(directory):
    file_paths = []
    for path in glob.glob(os.path.join(directory, '**', '*.conllu'), recursive = True):
        if os.path.isfile(path):
            file_paths.append(path)
    return file_paths

def process_file_to_dict(file_paths, label_mapping, label_mapping_NER):
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
    

    import re
    file_sentences = []
    for path in file_paths:
        with open(path, encoding = 'utf-8') as infile:
            sentence = []
            labels_SRL = []
            sentence_predicate = []
            labels_NER = []

            for line in infile:
                if line != '\n':
                    #remove the newline at the end of the line, then split the line based on the tab separator
                    line_split = line.strip('\n').split('\t')
                    #extract the token from the line in the file
                    token = line_split[4]
                    
                    #check whether the token is the predicate
                    if line_split[5] =='O':
                        predicate = False
                    else:
                        predicate = True


                    #check whether label for argument is in label_mapping (train_labels), else 'O'
                    argument = line_split[6]
                    #get numeric label
                    if argument in label_mapping.keys():
                        numeric_arg = label_mapping[argument]
                    else:
                        numeric_arg = label_mapping['O']

                    NER = line_split[7]
                    #get numeric label
                    if NER in label_mapping_NER.keys():
                        numeric_NER = label_mapping_NER[NER] 
                    else:
                        numeric_NER = label_mapping_NER['O']

                    
                    #per sentence
                    sentence.append(token)    
                    sentence_predicate.append(predicate)
                    labels_SRL.append(numeric_arg)
                    labels_NER.append(numeric_NER)
                else:
                    if sentence and labels_SRL and labels_NER and sentence_predicate:
                        sentence_dict = {'text':sentence,'predicate':sentence_predicate,'labels_SRL': labels_SRL, 'labels_NER': labels_NER}

                    #get list of dicts
                        file_sentences.append(sentence_dict)
                        sentence_dict = {}
                        sentence = []
                        labels_SRL = []
                        sentence_predicate =[]
                        labels_NER = []
                    else:
                        sentence_dict = {}
                        sentence = []
                        labels_SRL = []
                        sentence_predicate =[]
                        labels_NER = []

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
        sent=sentence_dict['text']
        predicate=sentence_dict['predicate']
        SRL=sentence_dict['labels_SRL']
        NER=sentence_dict['labels_NER']
 
        sent_pred = []
        word_ids_aug = []
        labels_SRL = []
        labels_NER = []
        

        #for each token, it is checked whether this is a negation cue token
        for token,pred in zip(sent, predicate):
            #if yes it is decided which special token is added to the cue
            if pred == True:
                sent_pred.append('[PRED] '+token)
                
            else:
                sent_pred.append(token)

        #SRL
        if len(sent_pred)==len(sent):
            #the augmented sentence is tokenized, only if it contains all the necessary elements
            tokenized = tokenizer(sent_pred, padding = 'max_length', truncation=True, max_length=512, is_split_into_words=True, return_tensors='pt')
    

            #tokenized_tokens = tokenizer.convert_ids_to_tokens(tokenized_sentence['input_ids'])

            # A subword tokenizer means that that even if your inputs have been split into words already, each of those words could be split again by the tokenizer. 
            #This can result in more input_ids than the list of labels for each word. To account for this, by using word.ids() maps special tokens (like [CLS] and [SEP]) 
            #to -100 and all tokenized input (input_ids) to their respective word. The storing of the special tokens as -100 is used for ignoring padding during loss computation. 

            word_ids = tokenized.word_ids()
            for id in word_ids:
                if id is not None:
                    labels_SRL.append(SRL[id])
                    labels_NER.append(NER[id])
                    word_ids_aug.append(id)
                else:
                    labels_SRL.append(-100)
                    labels_NER.append(-100)
                    word_ids_aug.append(-100)
                        #tokenized_sentence['word_ids'] = temp_word_ids
            #tokenized_sentence['labels'] = temp_label
            input_ids = tokenized['input_ids'].squeeze(0)
            attention_mask = tokenized['attention_mask'].squeeze(0)


            #dictionary with input_ids, attention_mask, word_ids, and labels as keys
            #list of dicts
            input_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'word_ids': word_ids_aug, 'labels_SRL': labels_SRL, 'labels_NER': labels_NER}
            all_inputs.append(input_dict)
        

    return all_inputs

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
    import numpy as np
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
                print(word_level)
                print(word_label)
                one_pred = np.argmax(np.bincount(word_level))  ## Gets the majority vote of the subtokens
                one_label = np.argmax(np.bincount(word_label))
                print(one_pred)
                print(one_label)
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

# Function to get informative label per sequence
def get_first_non_O_label(seq, id2label, pad_val=-100):
    for l in seq:
        if l != pad_val:
            label = id2label[l.item()]
            if label != "O":
                return label
    return "O"  

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
def generate_report(all_labels, all_preds, label_list, fold, output_file):
    """
    Displays confusion matrix and classification report of predictions against the gold labels.

    Parameters:
    all_preds: word_level predictions
    all_labels: gold labels
   

    Output:
    Display of confusion matrix and print of classification report.

    """

    report = classification_report(all_labels, all_preds)
    print("Classification Report -----------------------------------------------------------")
    print(report)


    output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'a') as f:
        if fold + 1 == 1:
            f.write('Classification Report all Folds\n')
            f.write('-------------------------------------------------\n')
        f.write(f"Classification Report for Fold {fold + 1}\n")
        f.write(report)
        f.write("\n-------------------------------------------------\n")

def log_fold_metrics(fold, alpha, beta, loss_srl, loss_ner, f1_srl, f1_ner, log_list=None):
    log_entry = {
        "fold": fold + 1,
        "alpha": alpha,
        "beta": beta,
        "loss_srl": round(loss_srl, 4),
        "loss_ner": round(loss_ner, 4),
        "f1_srl": round(f1_srl, 4),
        "f1_ner": round(f1_ner, 4),
    }
    print(f"📊 Fold {fold+1} | α={alpha:.2f} β={beta:.2f} | "
          f"SRL Loss: {loss_srl:.4f}, NER Loss: {loss_ner:.4f} | "
          f"SRL F1: {f1_srl:.4f}, NER F1: {f1_ner:.4f}")
    
    if log_list is not None:
        log_list.append(log_entry)
    
    return log_entry