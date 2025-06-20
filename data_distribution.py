import glob
import os
import pandas as pd
import re
from statistics import Counter
import numpy as np
import matplotlib.pyplot as plt

def list_of_files(directory):
    """
    Recursively collects all .conllu files from a given directory and its subdirectories.

    Parameters:
    directory (str): The root directory to search for .conllu files.

    Returns:
    list of str: A list of full file paths to all found .conllu files.
    """

    file_paths = []
    for path in glob.glob(os.path.join(directory, '**', '*.conllu'), recursive = True):
        if os.path.isfile(path):
            file_paths.append(path)
    return file_paths

def count_sents(path):
    """
    Counts the number of sentences present in a file. Each sentence is constructed by adding the lines until an empty line is present (indication of end of sentence).
    For each finished construction of lines (a sentence), the sentence count goes up by 1.

    Parameters:
    path: path to file to be counted

    Output:
    count_sents: integer representing number of sentences
        int: The total number of tokens in the file.
    """
    with open(path, encoding = 'utf-8') as infile:
            current_sentence = []
            count_sents = 0
            for line in infile:
                line = line.strip()
                if line == "":  # Empty line marks the end of the sentence
                    if current_sentence:
                        count_sents += 1
                        current_sentence = []
                else:
                    current_sentence.append(line)
    
            if current_sentence:
                count_sents += 1
                
    return count_sents

def count_role_and_ne_tokens(file_paths):
    """
    Extracts and counts all non-'O' semantic role and named entity labels 
    from a list of .conllu files.

    Parameters:
    file_paths (list of str): A list of file paths pointing to .conllu-formatted files.

    Returns:
    tuple:
        - roles (list of str): All semantic role labels (column 7) that are not 'O'.
        - ner (list of str): All named entity labels (column 8) that are not 'O'.

    Notes:
    - Each file is expected to be tab-separated and follow a structure where:
        column 6 = SRL label,
        column 7 = NER label.
    - Empty lines and malformed lines (with fewer than 8 columns) are skipped.
    """
    roles = []
    ner = []
    for path in file_paths:
        with open(path, 'r', encoding = 'utf-8') as file:
            for line in file:
                line = line.strip()
                if line == '\n':
                    continue
                else:
                    line_split = line.split('\t')
                    if len(line_split) < 8:
                        continue
                    else:
                        role = line_split[6]
                        NER = line_split[7]
                        if role != 'O':
                            roles.append(role)
                        if NER != 'O':
                            ner.append(NER)
                    


    return roles, ner

def count_tokens(path):
    """
    Counts the number of tokens present in a file. Each sentence is constructed by adding the lines in the file until an empty line is present (indication of end of sentence).
    For each finished construction of lines (a sentence), the function iterates through the rows present in the sentence, where each row represents a token,
    so per row the count goes up by 1.

    Parameters:
    path: path to file to be counted

    Output:
    count_tokens: integer representing number of tokens
    """
    with open(path, encoding = 'utf-8') as infile:
            current_sentence = []
            count_tokens = 0
            for line in infile:
                line = line.strip()
                if line == "":  # Empty line marks the end of the sentence
                    if current_sentence:
                        #each row represent a token
                        for row in current_sentence:
                            count_tokens += 1
                        current_sentence = []
                else:
                    current_sentence.append(line)
    
            if current_sentence:
                for row in current_sentence:
                    count_tokens += 1
                
    return count_tokens

def document_statistics(file_paths):
    """
    Computes summary statistics for a list of .conllu-formatted documents, including sentence and token counts,
    as well as the number of non-'O' SRL and NER tags. Extracts the year from the filename if present.

    Parameters:
    file_paths (list of str): A list of file paths to .conllu files.

    Returns:
    dict: A dictionary where each key is a document index (starting from 1), and each value is a dictionary with:
        - 'year' (str): The 4-digit year extracted from the filename (if found).
        - 'num_sents' (int): Number of sentences in the document.
        - 'num_tokens' (int): Total number of tokens.
        - 'srl_tags' (int): Number of semantic role labels (not 'O').
        - 'ner_tags' (int): Number of named entity labels (not 'O').

    Notes:
    - Assumes helper functions `count_sents`, `count_tokens`, and `count_role_and_ne_tokens` are defined.
    - Filenames should contain a 4-digit year (between 1500 and 2099) for year extraction.
    """
    dct_count = {}
    for i, path in enumerate(file_paths):
        sents = count_sents(path)
        tokens = count_tokens(path)
        filename = os.path.basename(path) 
        roles, ner = count_role_and_ne_tokens(path) 
        num_roles = len(roles)
        num_ner = len(ner)
        match = re.search(r'\b(1[5-9]\d{2}|20\d{2})\b', filename)
        if match:
            year = match.group(0)
        dct_count[i+1] = {'year': year, 'num_sents': sents, 'num_tokens': tokens,'srl_tags': num_roles, 'ner_tags': num_ner}

    return dct_count

def distribution(ner, roles, type='Mention'):
    """
    Processes lists of NER and SRL tags to extract role and entity type distributions
    based on a specified tag representation format.

    Parameters:
    ner (list of str): List of named entity labels (BIO-tagged).
    roles (list of str): List of semantic role labels (BIO-tagged).
    type (str): The processing mode to apply. One of:
        - 'Mention': Extracts only the beginning of each entity/role mention (i.e., B- labels) to indicate a role mention,
                     skips 'I-' and 'None', and strips the BIO prefix.
        - 'Token': Includes all role/entity tokens, removes BIO prefixes ('B-' and 'I-').
        - 'BIO': Keeps the full original BIO tags (no stripping or filtering).

    Returns:
    tuple:
        - role_list (list of str): Processed list of role labels based on the selected type.
        - ner_list (list of str): Processed list of named entity labels based on the selected type.

    Notes:
    - In 'Mention' mode, only B- prefixed roles/entities are retained for counting distinct mentions.
    - In 'Token' mode, both B- and I- tags are included but stripped of their prefix.
    - In 'BIO' mode, all labels are preserved as-is, including the BIO prefixes.
    - Entries labeled as 'None' in the NER list are excluded in all modes.
    """
    role_list = []
    ner_list = []
    if type == 'Mention':
        for role in roles:
            if role.startswith('I'):
                continue
            else:
                role = re.sub(r'^[B]-', '', role)
                role_list.append(role)

        for ne in ner:
            if 'None' in ne:
                continue
            if ne.startswith('I'):
                continue
            else:
                ne = re.sub(r'^[B]-', '', ne)
                ner_list.append(ne)

    elif type == 'Token':
        for role in roles:
            role = re.sub(r'^[B|I]-', '', role)
            role_list.append(role)

        for ne in ner:
            if 'None' in ne:
                continue
            ne = re.sub(r'^[B|I]-', '', ne)
            ner_list.append(ne)
    
    elif type == 'BIO':
        for role in roles:
            role_list.append(role)

        for ne in ner:
            if 'None' in ne:
                continue
            else:
                ner_list.append(ne)
                
    return role_list, ner_list

def barplot(list, xlabel='value', ylabel='count', rotation=0):
    """
    Creates a bar plot visualizing the frequency distribution of items in a given list.

    Parameters:
    list (list): A list of categorical items to count and visualize.
    xlabel (str): Label for the x-axis. Default is 'value'.
    ylabel (str): Label for the y-axis. Default is 'count'.
    rotation (int): Degree of rotation for x-axis tick labels. Default is 0 (horizontal).

    Returns:
    None

    Notes:
    - Uses `collections.Counter` to compute frequencies.
    - Displays the plot using `matplotlib.pyplot.show()`.
    """
    count = Counter(list)

    labels, values = zip(*count.items())
    plt.bar(labels, values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation) 
    plt.show()

def overlap_role_and_ne_tokens(file_paths):
    """
    Extracts semantic role and named entity labels from a list of .conllu-formatted files,
    and identifies tokens that are annotated with both a semantic role and a named entity.

    Parameters:
    file_paths (list of str): A list of file paths pointing to .conllu-formatted files.

    Returns:
    tuple:
        - roles (list of str): All non-'O' semantic role labels (column 7).
        - ner (list of str): All non-'O' named entity labels (column 8).
        - overlap_ner (list of str): Named entity labels for tokens that are also assigned a semantic role.
        - overlap_role (list of str): Semantic role labels for tokens that are also assigned a named entity.

    Notes:
    - Each file is assumed to be tab-separated, with:
        column 6 = SRL label,
        column 7 = NER label.
    - Lines with fewer than 8 columns or empty lines are skipped.
    - 'O' labels are considered non-annotated and excluded from overlap.
    """
    roles = []
    ner = []
    overlap_ner = []
    overlap_role = []
    for path in file_paths:
        with open(path, 'r', encoding = 'utf-8') as file:
            for line in file:
                line = line.strip()
                if line == '\n':
                    continue
                else:
                    line_split = line.split('\t')
                    if len(line_split) < 8:
                        continue
                    else:
                        role = line_split[6]
                        NER = line_split[7]
                        if role != 'O':
                            roles.append(role)
                        if NER != 'O':
                            ner.append(NER)
                        if role != 'O' and NER != 'O':
                            overlap_ner.append(NER)
                            overlap_role.append(role)

    return roles, ner, overlap_ner, overlap_role

def percentages_overlap(roles, ner, overlap_ner):
    """
    Calculates and prints the overlap between semantic role labels and named entity labels.

    Parameters:
    roles (list of str): All non-'O' semantic role labels.
    ner (list of str): All non-'O' named entity labels.
    overlap_ner (list of str): Named entity labels that also have a corresponding semantic role.

    Returns:
    None

    Prints:
    - Total number of semantic role tokens.
    - Total number of named entity tokens.
    - Number of tokens annotated as both a semantic role and a named entity.
    - Percentage of semantic roles that are also named entities.
    - Percentage of named entities that are also semantic roles.

    Notes:
    - Assumes `overlap_ner` contains only those NE tokens that overlap with SRL tokens.
    - Rounds percentage values to two decimal places.
    """
    num_roles = len(roles)
    num_ne = len(ner)
    num_ne_overlap = len(overlap_ner)
    percentage = num_ne_overlap/num_roles * 100
    rounded_per = round(percentage, 2)
    percentage_ne = num_ne_overlap/num_ne * 100
    rounded_per_ne = round(percentage_ne, 2)

    print(f'Total amount of semantic role tokens: {num_roles}')
    print(f'Total amount of named entity tokens: {num_ne}')
    print(f'Total amount of semantic role tokens that are also a named entitiy: {num_ne_overlap}')
    print(f'Percentage of all semantic role tokens that are both an semantic role and named entity: {rounded_per} %')
    print(f'Percentage of all NE tokens that are also a semantic role: {rounded_per_ne} %')

def table_overlap(roles, overlap_role, ner, overlap_ner, file_name):
    """
    Creates a detailed table showing the overlap between semantic role labels and named entity labels,
    including their frequencies and relative percentages. Saves the table as a TSV file.

    Parameters:
    roles (list of str): All non-'O' semantic role labels.
    overlap_role (list of str): Semantic role labels that also overlap with NER labels.
    ner (list of str): All non-'O' named entity labels.
    overlap_ner (list of str): Named entity labels that overlap with semantic role labels.
    file_name (str): Path to the output file (.tsv) where the resulting table will be saved.

    Returns:
    pd.DataFrame: A DataFrame with the following columns:
        - Role: Semantic role label (BIO prefix removed).
        - NE: Named entity label (BIO prefix removed).
        - Count: Frequency of the NE occurring within the given role.
        - Role Total: Total occurrences of the role in the full dataset.
        - NE Total: Total occurrences of the named entity type.
        - Percentage NE: NE count as a percentage of its NE total.
        - Percentage NE of Role: NE count as a percentage of its Role total.

    Notes:
    - BIO prefixes (B- and I-) are removed before aggregation.
    - Output file is saved as a tab-separated values (TSV) file.
    """
    role_dict = {}
    all_roles = []
    all_ner = []
    table_data = []
    
    for role, ne in zip(overlap_role, overlap_ner):
        role = re.sub(r'^[B|I]-', '', role)
        ne = re.sub(r'^[B|I]-', '', ne)
        if role not in role_dict.keys():
            role_dict[role] = [ne]
        else:
            role_dict[role].append(ne)

    for arg in roles:
        arg = re.sub(r'^[B|I]-', '', arg)
        all_roles.append(arg)
    counter = Counter(all_roles)
    dict_counter = dict(counter)

        
    for ne in ner:
        ne = re.sub(r'^[B|I]-', '', ne)
        all_ner.append(ne)
    counter_ne = Counter(all_ner)
    dict_counter_ne = dict(counter_ne)

    for role in role_dict.keys():
        entity_counts = Counter(role_dict[role])
        for entity, count in entity_counts.items():
            table_data.append([role, entity, count])

    # Create a pandas DataFrame
    df_counts = pd.DataFrame(table_data, columns=["Role", "NE", "Count"])
    role_totals_df = pd.DataFrame(list(dict_counter.items()), columns=["Role", "Role Total"])

    # Step 4: Convert NE totals (count of each NE across all roles) into a DataFrame
    NE_totals_df = pd.DataFrame(list(dict_counter_ne.items()), columns=["NE", "NE Total"])

    # Step 5: Merge the role totals and NE totals into the df_counts DataFrame
    df_counts = pd.merge(df_counts, role_totals_df, on="Role", how="left")
    df_counts = pd.merge(df_counts, NE_totals_df, on="NE", how="left")

    # Step 6: Calculate the percentage for each NE relative to its NE total and the percentage of each NE relative to its role total
    df_counts['Percentage NE'] = round((df_counts['Count'] / df_counts['NE Total']) * 100, 2)
    df_counts['Percentage NE of Role'] = round((df_counts['Count'] / df_counts['Role Total']) * 100, 2)

    # Step 7: Save the DataFrame to a TSV file (with tab separator)
    df_counts.to_csv(file_name, sep='\t', index=False)

    return df_counts

def top_10_plot(df_counts, new_file):
    """
    Generates a horizontal bar plot showing the top 10 Role–Named Entity combinations 
    by percentage of the role taken by a named entity. Saves both the plot and 
    the top 10 rows to a TSV file.

    Parameters:
    df_counts (pd.DataFrame): DataFrame containing at least the columns:
        - 'Role': Semantic role label.
        - 'NE': Named entity label.
        - 'Percentage NE of Role': Percentage of the role covered by the named entity.
    new_file (str): File path to save the top 10 rows as a TSV file.

    Returns:
    None

    Notes:
    - Adds a new column 'Role_NE' combining the Role and NE labels for display.
    - Colors bars using a Viridis colormap scaled to the 'Percentage NE of Role'.
    - Sorts by 'Percentage NE of Role' and visualizes the top 10 highest values.
    - Saves the top 10 rows to a TSV file for inspection or reporting.
    """
    sorted_perc = df_counts.sort_values(by='Percentage NE of Role', ascending=False)
    top10sorted = sorted_perc.head(10)
    # Stap 2: Zorg dat de kolommen 'Role' en 'NE' strings zijn
    df_counts["Role"] = df_counts["Role"].astype(str)
    df_counts["NE"] = df_counts["NE"].astype(str)

    # Stap 3: Voeg een gecombineerde labelkolom toe
    df_counts["Role_NE"] = df_counts["Role"] + " + " + df_counts["NE"]
    colors = plt.cm.viridis(df_counts["Percentage NE of Role"] / df_counts["Percentage NE of Role"].max())
    # Stap 4: Visualiseer met een horizontale staafdiagram
    plt.figure(figsize=(12, 6))
    plt.barh(
        df_counts["Role_NE"],
        df_counts["Percentage NE of Role"], color=colors
    )
    plt.xlabel("Percentage of Role Taken by Named Entity")
    plt.gca().invert_yaxis()  # Zet hoogste percentage bovenaan
    plt.tight_layout()
    plt.show()

    top10sorted.to_csv(new_file, sep='\t', index=True)