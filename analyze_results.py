import pandas as pd
import json
import os

results_dir = "results"

def load_results(filename):
    filepath = os.path.join(results_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            data = json.load(file)
        return data
    return []

# Load fit, evaluation, and global evaluation results
fit_results = load_results("fit_results.json")
eval_results = load_results("eval_results.json")
global_eval_results = load_results("global_eval_results.json")

# Convert results to DataFrames based on their structure
def fit_results_to_dataframe(results):
    if not results:
        return pd.DataFrame()

    processed_results = []
    for result in results:
        processed_result = {
            'round': result.get('round', None),
            'client_id': result.get('client_id', None),
            'i1i_score': result.get('i1i_score', None)
        }
        processed_results.append(processed_result)

    return pd.DataFrame(processed_results)

def eval_results_to_dataframe(results):
    if not results:
        return pd.DataFrame()

    processed_results = []
    for result in results:
        processed_result = {
            'round': result.get('round', None),
            'client_id': result.get('client_id', None),
            'global_accuracy': result.get('global_accuracy', None),
            'l1o_score': result.get('l1o_score', None)
        }
        processed_results.append(processed_result)

    return pd.DataFrame(processed_results)

def global_eval_results_to_dataframe(results):
    if not results:
        return pd.DataFrame()

    processed_results = []
    for result in results:
        processed_result = {
            'round': result.get('round', None),
            'loss': result.get('loss', None),
            'accuracy': result.get('accuracy', None),
            'num_clients': result.get('num_clients', None)
        }
        processed_results.append(processed_result)

    return pd.DataFrame(processed_results)

fit_df = fit_results_to_dataframe(fit_results)
eval_df = eval_results_to_dataframe(eval_results)
global_eval_df = global_eval_results_to_dataframe(global_eval_results)

# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)

# Save the data to CSV files
fit_df.to_csv(os.path.join(results_dir, "fit_results.csv"), index=False)
eval_df.to_csv(os.path.join(results_dir, "eval_results.csv"), index=False)
global_eval_df.to_csv(os.path.join(results_dir, "global_eval_results.csv"), index=False)

# Displaying the tables as CSV
print("Fit Results:")
print(fit_df.to_string(index=False))
print("\nEvaluation Results:")
print(eval_df.to_string(index=False))
print("\nGlobal Evaluation Results:")
print(global_eval_df.to_string(index=False))

# Save the tables as text files
fit_df.to_csv(os.path.join(results_dir, "fit_results_table.txt"), sep='\t', index=False)
eval_df.to_csv(os.path.join(results_dir, "eval_results_table.txt"), sep='\t', index=False)
global_eval_df.to_csv(os.path.join(results_dir, "global_eval_results_table.txt"), sep='\t', index=False)
