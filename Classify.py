#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import time
import yaml
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, top_k_accuracy_score


def load_yaml_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def save_csv_for_iteration(temp_dir, model_dir, split_percentage, iteration, df):
    csv_file = os.path.join(temp_dir, f'{model_dir}_{split_percentage}_{iteration}.csv')
    df.to_csv(csv_file, index=False)

def find_last_iteration(temp_dir, model_dir, split_percentage, num_iterations):
    # Iterate backward from the total number of iterations
    for iteration in range(num_iterations, 0, -1):
        csv_file = os.path.join(temp_dir, f'{model_dir}_{split_percentage}_{iteration}.csv')
        if os.path.exists(csv_file):
            return iteration, pd.read_csv(csv_file)
    return 0, pd.DataFrame()


def calculate_mean_std(df):
    metric_columns = ['Accuracy', 'Top-1%', 'Top-3%', 'Top-5%', 'Precision', 'Recall', 'F1 Score', 'Time Taken (s)']
    aggregated_results = {}
    for col in metric_columns:
        if col in df:
            aggregated_results[f'Mean {col}'] = round(df[col].mean(), 2)
            aggregated_results[f'Std {col}'] = round(df[col].std(), 2)
    return pd.DataFrame([aggregated_results])

def save_final_csv_with_aggregates(results_dir, model_dir, split_percentage, final_results_df):
    aggregated_results_df = calculate_mean_std(final_results_df)
    final_csv_file = os.path.join(results_dir, f'{model_dir}_{split_percentage}_{100-split_percentage}.csv')
    aggregated_results_df.to_csv(final_csv_file, index=False)
    print(f"Final aggregated results saved to {final_csv_file}")

def combine_csv_files_for_model(results_dir, model_dir):
    combined_df = pd.DataFrame()
    for split_percentage in [70, 80, 90]:
        csv_file = os.path.join(results_dir, f'{model_dir}_{split_percentage}_{100-split_percentage}.csv')
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df.insert(0, 'Split', split_percentage)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    combined_csv_file = os.path.join(results_dir, f'{model_dir}.csv')
    combined_df.to_csv(combined_csv_file, index=False)
    print(f"Combined CSV saved as {combined_csv_file}")

def classify_and_save_results(encodings_dir, temp_dir, results_dir, model_dir, split_percentage, kernel_type, n_splits):

    # Reading features and labels 
    features_path = os.path.join(encodings_dir, model_dir, 'encoded_images.npy')
    labels_path = os.path.join(encodings_dir, model_dir, 'labels.npy')
    features = np.load(features_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)

    # Scaling and SVM
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    svm = SVC(kernel=kernel_type, probability=True)

    # Finding it there are already some iterations done
    start_iteration, final_results_df = find_last_iteration(temp_dir, model_dir, split_percentage,n_splits)
    updated_splits = n_splits - start_iteration # you must look if some splits are done already

    # StratifiedShuffleSplit with the "updated_splits"
    stratified_split = StratifiedShuffleSplit(n_splits=updated_splits, test_size=(1 - split_percentage / 100), random_state=42)

    for iteration, (train_idx, test_idx) in enumerate(stratified_split.split(features, labels), start=start_iteration):
        print(f"Iteration {iteration + 1}/{n_splits}")
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        start_time = time.time()
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        y_prob = svm.predict_proba(X_test)

        n_classes = len(np.unique(labels))
        results = {
            'Iteration': iteration + 1,
            'Model': model_dir,
            'Split Percentage': split_percentage,
            'Accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
            'Precision': round(precision_score(y_test, y_pred, average='weighted') * 100, 2),
            'Recall': round(recall_score(y_test, y_pred, average='weighted') * 100, 2),
            'F1 Score': round(f1_score(y_test, y_pred, average='weighted') * 100, 2),
            'Time Taken (s)': time.time() - start_time
        }

        if n_classes > 2:
            results['Top-1%'] = round(top_k_accuracy_score(y_test, y_prob, k=1, labels=np.unique(labels)) * 100, 2)
            results['Top-3%'] = round(top_k_accuracy_score(y_test, y_prob, k=3, labels=np.unique(labels)) * 100, 2) if n_classes > 3 else None
            results['Top-5%'] = round(top_k_accuracy_score(y_test, y_prob, k=5, labels=np.unique(labels)) * 100, 2) if n_classes > 5 else None

        final_results_df = pd.concat([final_results_df, pd.DataFrame([results])], ignore_index=True)
        save_csv_for_iteration(temp_dir, model_dir, split_percentage, iteration + 1, final_results_df)
    
    save_final_csv_with_aggregates(results_dir, model_dir, split_percentage, final_results_df)

def cleanup_temp_folder(temp_dir):
    for file in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, file)
        if file.endswith('.csv'):
            os.remove(file_path)
    print("All temporary CSV files deleted from temp folder.")

def combine_all_results_into_excel(results_dir, model_dirs, output_file):
    """
    Combines all individual model CSV files into a single Excel file,
    splitting the rows into separate sheets based on the 'Split' column.

    Parameters:
    - results_dir (str): Directory containing the combined CSV files for each model.
    - model_dirs (list): List of model directory names used to look for corresponding CSV files.
    - output_file (str): Name of the output Excel file.
    """
    combined_df = pd.DataFrame()
    for model_dir in model_dirs:
        csv_file = os.path.join(results_dir, f"{model_dir}.csv")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df.insert(0, 'Model', model_dir)  # Add a 'Model' column for differentiation
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        else:
            print(f"CSV file for model {model_dir} not found in {results_dir}. Skipping.")

    output_path = os.path.join(results_dir, output_file)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Split the data by the 'Split' column
        for split_value in combined_df['Split'].unique():
            split_df = combined_df[combined_df['Split'] == split_value]
            sheet_name = f"split-{int(split_value)}"
            split_df.to_excel(writer, index=False, sheet_name=sheet_name)
    print(f"All results combined and saved as {output_path}")

def main(config_path):
    config = load_yaml_config(config_path)
    proj_dir = config['project_directory']
    encodings_dir = config['encodings_dir']
    results_dir = config['results_path']
    kernel_type = config['svm_kernel']
    n_splits = config['n_splits']

    temp_dir = os.path.join(proj_dir, 'temp_results')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    model_dirs = [d for d in os.listdir(encodings_dir) if os.path.isdir(os.path.join(encodings_dir, d))]
    for model_dir in model_dirs:
        print(f"Processing model: {model_dir}")
        for split_percentage in [90, 80, 70]:
            final_csv_file = os.path.join(results_dir, f'{model_dir}_{split_percentage}_{100-split_percentage}.csv')
            if os.path.exists(final_csv_file):
                print(f"Results for {model_dir} with {split_percentage}% split already exist. Skipping.")
                continue

            classify_and_save_results(
                encodings_dir, temp_dir, results_dir, model_dir, split_percentage, kernel_type, n_splits
            )

    cleanup_temp_folder(temp_dir)
    
    # Combine results for individual models
    for model_dir in model_dirs:
        combine_csv_files_for_model(results_dir, model_dir)
    print("Processing completed.")
    
    # Combine all model results into a single Excel file
    combine_all_results_into_excel(results_dir, model_dirs, "final_results.xlsx")
    print("All processing completed.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Classify images and save results using SVM.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    args = parser.parse_args()
    
    main(args.config)

