def categorize_error(gold, pred):
    """
    Categorizes the type of prediction error between a gold and predicted SRL label.

    Parameters:
    gold (str): The gold (true) SRL label (e.g., 'B-AGENT', 'O').
    pred (str): The predicted SRL label.

    Returns:
    str or None: The type of error or outcome:
        - 'Correct': The prediction matches the gold label (not 'O').
        - 'False Positive': The model predicted a label when the gold label is 'O'.
        - 'False Negative': The model missed a label (predicted 'O' when gold has a role).
        - 'Boundary Error': The predicted and gold labels have the same role type but differ in BIO tag (e.g., 'B-' vs. 'I-').
        - 'Label Confusion': The predicted and gold labels are both non-'O' but of different role types.
        - None: Both gold and predicted labels are 'O', indicating no SRL role.

    """
    if gold == pred and gold == 'O':
        return None  # not an SRL label
    if gold == pred and gold != 'O':
        return 'Correct'
    if gold == "O" and pred != "O":
        return "False Positive"
    elif gold != "O" and pred == "O":
        return "False Negative"
    elif gold != pred and gold != 'O' and pred != 'O':
        gold_type = gold.split("-")[-1]
        pred_type = pred.split("-")[-1]
        if gold_type == pred_type:
            return "Boundary Error"
        else:
            return "Label Confusion"
        
def error_analysis(df, train_setup = 'single-task', task = 'SRL', fold = 1):
    """
    Performs error analysis on SRL (Semantic Role Labeling) predictions, categorizing errors
    and summarizing their distribution. Saves both summary statistics and detailed token-level errors.

    Parameters:
    df (pd.DataFrame): The dataframe containing token-level predictions and gold labels. 
                       Required columns: 'Token', 'Gold', 'Prediction'. For multi-task: also 'Task'.
    train_setup (str): Indicates whether the model was trained in 'single-task' or 'multi-task' mode. Default is 'single-task'.
    task (str): The task to filter for in multi-task setups (e.g., 'SRL'). Ignored for single-task. Default is 'SRL'.
    fold (int): Fold number used for naming output files. Default is 1.

    Operations:
    - Filters and cleans the input dataframe.
    - Applies `categorize_error()` to label each prediction.
    - Summarizes the counts and percentages of four main error types:
        * False Positive
        * False Negative
        * Boundary Error
        * Label Confusion
    - Saves:
        * Error summary CSV (`{train_setup}_srl_error_summary_fold{fold}.csv`)
        * Detailed token-level error list (`{train_setup}_srl_token_errors_fold{fold}.csv`)
        * Correct predictions on actual roles (`{train_setup}_srl_correct_predictions_fold{fold}.csv`)
    - Prints summary statistics and unique correct SRL predictions.

    Returns:
        None
    """
    if train_setup == 'single-task':
        df = df[~df["Token"].astype(str).str.startswith("===")]
        df = df.dropna(subset=["Token", "Gold", "Prediction"])
    elif train_setup == 'multi-task':
        df = df[~df["Token"].astype(str).str.startswith("===")]
        df = df.dropna(subset=["Token", "Gold", "Prediction", "Task"])

        # Filter for SRL only
        df = df[df["Task"] == task]
        
    #apply categorization
    df["Error Type"] = df.apply(lambda row: categorize_error(row["Gold"], row["Prediction"]), axis=1)
    # All labeled predictions (excluding non-SRL)
    df_labeled = df[df["Error Type"].notna()]

    # Filter alleen de echte fouten (zonder 'Correct')
    df_errors_only = df_labeled[df_labeled["Error Type"].isin([
        "False Positive", "False Negative", "Label Confusion", "Boundary Error"
    ])]
    num_errors = len(df_errors_only)

    # Samenvatting van fouten
    summary_errors = df_errors_only["Error Type"].value_counts().reset_index()
    summary_errors.columns = ["Error Type", "Count"]
    summary_errors["Percentage"] = (summary_errors["Count"] / num_errors * 100).round(2)

    # Voeg total toe
    summary_errors.loc[len(summary_errors)] = ["Total Errors", num_errors, 100.00]

        # Save summary
    summary_errors.to_csv(f"{train_setup}_srl_error_summary_fold{fold}.csv", index=False)

    # Save detailed error list
    df_errors_only.to_csv(f"{train_setup}_srl_token_errors_fold{fold}.csv", index=False)

    print(f"\nSummary of 4 error types (as % of total errors) {train_setup} {task}:")
    print(summary_errors)

    # Filter correcte voorspellingen van echte SRL-rollen
    df_correct = df[(df["Gold"] != "O") & (df["Gold"] == df["Prediction"])]
    num_correct = len(df_correct)

    # Aantal echte SRL-rollen (gold ≠ O)
    num_actual_roles = len(df[df["Gold"] != "O"])

    accuracy = round((num_correct / num_actual_roles) * 100, 2)

    # Save correct predictions
    df_correct.to_csv(f"{train_setup}_srl_correct_predictions_fold{fold}.csv", index=False)

    print(f"\nCorrect predictions on actual semantic roles: {num_correct} / {num_actual_roles} → {accuracy:.2f}%")

    print("\nUnique correct SRL predictions:")
    print(sorted(df_correct["Prediction"].unique()))
        
        
