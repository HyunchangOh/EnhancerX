import numpy as np
import os
import pickle
import time
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

time_plot = time.time()

# Define the save directory for metrics
save_dir = '../../Subsampled_Final/NN_results'

# Define the chromosomes and features directories
chromosomes = ['chr1', 'chr21', 'all']
features_list = ['feature_mix', 'feature_dist', 'feature_mem']

# Recursively create the metrics directory structure
for chromosome in chromosomes:
    for feature in features_list:
        metrics_save_dir = os.path.join(save_dir, 'metrics', chromosome, feature)
        os.makedirs(metrics_save_dir, exist_ok = True)

# Define the root directory
root_dir = '../../Subsampled_Final/'

# Function to calculate metrics
def calculate_metrics(y_test, y_pred, y_pred_prob):
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_prob)
    }
    return metrics

# Iterate over each chromosome
for chromosome in chromosomes:
    for feature in features_list:
        # Define paths
        data_dir = os.path.join(root_dir, chromosome, feature)
        metrics_save_dir = os.path.join(save_dir, 'metrics', chromosome, feature)
        os.makedirs(metrics_save_dir, exist_ok=True)
        
        # Dictionary to store metrics for each structure
        metrics_dict = {}

        # Iterate over the .npy files in the directory
        for file in os.listdir(data_dir):
            if file.startswith('testvpred') and file.endswith('.npy'):
                # Extract hyperparameters from the file name
                _, structure, batch_size, lrn_rate, num_epochs = file.split('_')
                num_epochs = num_epochs.split('.')[0]
                label = (int(batch_size), float(lrn_rate), int(num_epochs))

                if structure not in metrics_dict:
                    metrics_dict[structure] = {
                        "accuracy": [],
                        "precision": [],
                        "recall": [],
                        "f1": [],
                        "roc_auc": [],
                        "labels": []
                    }
                
                # Load the data
                data = np.load(os.path.join(data_dir, file))
                y_test = data[:, 0]
                y_pred = data[:, 1]
                y_pred_prob = data[:, 1]  # Assuming binary predictions already represent probabilities
                
                # Calculate metrics
                metrics = calculate_metrics(y_test, y_pred, y_pred_prob)
                
                # Store the metrics
                for key in metrics_dict[structure].keys():
                    if key != "labels":
                        metrics_dict[structure][key].append((label, metrics[key]))
                metrics_dict[structure]["labels"].append(label)
                
                print(f"Metrics for {chromosome} - {feature} - structure: {structure}, batch_size: {batch_size}, lrn_rate: {lrn_rate}, num_epochs: {num_epochs}:")
                print(classification_report(y_test, y_pred))
                print(f"ROC AUC: {metrics['roc_auc']}\n")

        # Save metrics
        with open(os.path.join(metrics_save_dir, "metrics.pkl"), "wb") as f:
            pickle.dump(metrics_dict, f)

        # Sort labels and metrics
        for structure in metrics_dict:
            sorted_indices = sorted(range(len(metrics_dict[structure]["labels"])), key=lambda i: metrics_dict[structure]["labels"][i])
            for key in metrics_dict[structure].keys():
                if key != "labels":
                    metrics_dict[structure][key] = [metrics_dict[structure][key][i][1] for i in sorted_indices]
            metrics_dict[structure]["labels"] = [f"{x[0]}_{x[1]}_{x[2]}" for x in sorted(metrics_dict[structure]["labels"])]

        # Plot and save each metric individually with different hyperparameters in different colors
        metrics_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        
        for structure, metric_data in metrics_dict.items():
            plt.figure(figsize=(12, 8))
            for metric_key in metrics_keys:
                plt.plot(metric_data[metric_key], marker='o', label=metric_key.capitalize())
            
            plt.xticks(ticks=range(len(metric_data["labels"])), labels=metric_data["labels"], rotation=45, ha='right')
            plt.title(f'{chromosome} - {feature} - {structure}')
            plt.xlabel('Hyperparameter Combinations (batch_size_lr_rate_num_epochs)')
            plt.ylabel('Metrics')
            plt.legend()
            plt.tight_layout()  # Adjust the layout to ensure the labels fit
            plot_path = os.path.join(metrics_save_dir, f"combined_metrics_plot_{structure}.png")
            plt.savefig(plot_path)
            plt.close()

print("Metrics calculation and plotting complete for all chromosomes and features. Done in %s seconds." % round(time.time() - time_plot))
