import pandas as pd
import seaborn as sns
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel, wilcoxon
import numpy as np

def create_combined_cm(file_path):
    """
    Creates a combined confusion matrix from multiple fold-specific matrices
    stored in a text file and saves the result as a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the file containing confusion matrices per fold.

    Returns
    -------
    None
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw_lines = f.read().splitlines()

    # Containers
    label_set = []
    matrices = []
    inside_matrix = False
    current_labels = []
    current_matrix = []

    for line in raw_lines:
        if line.startswith("Confusion matrix for Fold"):
            if current_matrix:
                df = pd.DataFrame(current_matrix, columns=current_labels, index=row_labels)
                matrices.append(df)
                current_matrix = []
            inside_matrix = False
            continue
        if line.startswith(",") and not inside_matrix:
            current_labels = line.strip().split(",")[1:]
            label_set = current_labels
            row_labels = []
            inside_matrix = True
            continue
        if inside_matrix and line.strip() and not line.startswith("-"):
            parts = line.strip().split(",")
            row_labels.append(parts[0])
            row_values = list(map(int, parts[1:]))
            current_matrix.append(row_values)

    if current_matrix:
        df = pd.DataFrame(current_matrix, columns=current_labels, index=row_labels)
        matrices.append(df)

    # Combine all matrices
    combined_matrix = sum(matrices)
    combined_matrix.to_csv("SRL_combined_confusion_matrix.csv")
    print("Combined confusion matrix saved as 'SRL_combined_confusion_matrix.csv'")

def load_combined_matrix(file_name):
    """
    Loads a saved combined confusion matrix from CSV, normalizes it row-wise,
    and plots it as a heatmap.

    Parameters
    ----------
    file_name : str
        Path to the CSV file containing the confusion matrix.

    Returns
    -------
    None
    """
    # Load the combined confusion matrix
    combined_matrix = pd.read_csv(file_name, index_col=0)

    # Normalize by row (true class)
    normalized_matrix = combined_matrix.div(combined_matrix.sum(axis=1), axis=0)

    # Plot heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        normalized_matrix,
        annot=False,
        cmap="Blues",
        xticklabels=True,
        yticklabels=True
    )

    plt.title("Normalized Confusion Matrix (multitask)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def averages_multitask(path_metrics_mt):
    """
    Computes and prints average precision, recall, and F1 scores for both SRL and NER
    from a multitask metrics JSON file.

    Parameters
    ----------
    path_metrics_mt : str
        Path to the JSON file containing multitask evaluation metrics.

    Returns
    -------
    tuple of lists
        (srl_prec, srl_rec, srl_f1, ner_prec, ner_rec, ner_f1)
    """
    srl_prec = []
    srl_rec = []
    srl_f1 = []
    ner_prec = []
    ner_rec = []
    ner_f1 = []

    with open(path_metrics_mt) as f:
        metrics = json.load(f)

    for dct in metrics: 
        precision = dct['metrics']['precision']
        for p in precision:
            srl_prec.append(p['srl'])
            ner_prec.append(p['ner'])
        recall = dct['metrics']['recall']
        for r in recall:
            srl_rec.append(r['srl'])
            ner_rec.append(r['ner'])
        f1 = dct['metrics']['f1']
        for f in f1:
            srl_f1.append(f['srl'])
            ner_f1.append(f['ner'])
        average_precision = sum(srl_prec)/len(srl_prec)
        average_recall = sum(srl_rec)/len(srl_rec)
        average_f1 = sum(srl_f1)/len(srl_f1)
        avg_ner_prec = sum(ner_prec)/len(ner_prec)
        avg_ner_rec = sum(ner_rec)/len(ner_rec)
        avg_ner_f1 = sum(ner_f1)/len(ner_f1)
    print('SRL scores')
    print(f'Average precision = {average_precision:.2f}')
    print(f'Average recall = {average_recall:.2f}')
    print(f'Average F1 = {average_f1:.2f}')
    print()
    print('NER scores')
    print(f'Average precision = {avg_ner_prec:.2f}')
    print(f'Average recall = {avg_ner_rec:.2f}')
    print(f'Average F1 = {avg_ner_f1:.2f}')

    return srl_prec, srl_rec, srl_f1, ner_prec, ner_rec, ner_f1

def averages(path_metrics):
    """
    Computes and prints average precision, recall, and F1 scores from a JSON file
    for single-task evaluation.

    Parameters
    ----------
    path_metrics : str
        Path to the JSON file containing single-task evaluation metrics.

    Returns
    -------
    tuple of lists
        (precision, recall, f1)
    """
    with open(path_metrics) as f:
        metrics = json.load(f)

    for dct in metrics: 
        precision = dct['metrics']['precision']
        recall = dct['metrics']['recall']
        f1 = dct['metrics']['f1']
        average_precision = sum(precision)/len(precision)
        average_recall = sum(recall)/len(recall)
        average_f1 = sum(f1)/len(f1)
    
    print(f'Average precision = {average_precision:.2f}')
    print(f'Average recall = {average_recall:.2f}')
    print(f'Average F1 = {average_f1:.2f}')

    return precision, recall, f1

def plot_f1_comparison(precision, srl_prec):
    """
    Plots a comparison between F1 scores of two models (e.g. single-task and multitask)
    across multiple folds.

    Parameters
    ----------
    precision : list of float
        F1 scores of the first model (e.g., single-task) across folds.
    srl_prec : list of float
        F1 scores of the second model (e.g., multitask) across folds.

    Returns
    -------
    None
    """
    # Plotting
    x = range(1, len(precision)+1)
    plt.plot(x, precision, label='Single task gloBERTise', marker='o', color='blue')
    plt.plot(x, srl_prec, label='Multitask gloBERTise', marker='s', color='red')
    plt.xlabel('Folds')
    plt.ylabel('F1-score')
    plt.title('F1-score Comparison Between Two Models')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_significance(scores_1, scores_2):
    """
    Performs a paired t-test to assess the statistical significance between
    two sets of model scores.

    Parameters
    ----------
    scores_1 : list of float
        Scores from the first model.
    scores_2 : list of float
        Scores from the second model.

    Returns
    -------
    None
    """
#    Convert to numpy arrays for convenience
    a = np.array(scores_1)
    b = np.array(scores_2)

    # Mean and std
    mean_a, std_a = np.mean(a), np.std(a)
    mean_b, std_b = np.mean(b), np.std(b)

    print(f"Model SRL F1: {mean_a:.4f} ± {std_a:.4f}")
    print(f"Model Multi SRL F1: {mean_b:.4f} ± {std_b:.4f}")

    # Paired t-test
    t_stat, p_ttest = ttest_rel(a, b)
    print(f"Paired t-test p-value: {p_ttest:.4f}")