# 

import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pickle

# Define paths
input_dir = '../../../../../scratch/ohh98/LGT_subsampled_reboot/'
all_sample_input_dir = '../../../../../scratch/ohh98/la_grande_table/'
chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX']
model_save_dir = '../../../../../scratch/ohh98/models/logistic_regression/each_chromosome_subsampled/'
os.makedirs(model_save_dir, exist_ok=True)

# Define the Gradient Boosting model
def create_gb_model():
    model = LogisticRegression(max_iter=200, random_state=42)
    return model

# Initialize metrics storage
accuracy = np.zeros((23, 23))
precision = np.zeros((23, 23))
recall = np.zeros((23, 23))
f1 = np.zeros((23, 23))
roc_auc = np.zeros((23, 23))

# Function to load data
def load_data(input_dir, chr_folder):
    data_dir = os.path.join(input_dir, chr_folder)
    response_file = os.path.join(data_dir, 'BIN50_enhancer_atlas.npy')
    
    # Load response variable
    response = np.load(response_file)
    
    # Load feature data
    features = {}
    for file in os.listdir(data_dir):
        if file.startswith('BIN50_') and file.endswith('.npy') and (file != 'BIN50_enhancer_atlas.npy'  and not file.startswith("BIN50_enhancer_atlas")):
            feature_name = file.split('.')[0]
            features[feature_name] = np.load(os.path.join(data_dir, file))

    # Handle inf values in promoter_forward and promoter_reverse
    for key in ['promoter_forward', 'promoter_reverse']:
        if key in features:
            max_value = np.max(features[key][np.isfinite(features[key])])  # Find the max finite value
            features[key][np.isinf(features[key])] = 10 * max_value       # Replace inf with 10 times the max value

    # Stack features
    X = np.hstack([features[key].reshape(len(features[key]), -1) for key in features.keys()])
    y = (response != 0).astype(int)  # Convert response variable to binary

    # Check and handle NaNs, infinities, and large values in X
    X = np.nan_to_num(X, nan=np.nanmean(X, axis=0), posinf=10*np.nanmax(X[np.isfinite(X)]), neginf=-10*np.nanmax(X[np.isfinite(X)]))

    # Verify that there are no NaNs or infinities in the final dataset
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("The dataset still contains NaNs or infinities after preprocessing.")
    
    return X, y

# Loop through each chromosome for training
for train_idx, train_chr in enumerate(chromosomes):
    print(f"Training on {train_chr}")
    X_train, y_train = load_data(input_dir, train_chr)
    
    # Create and train the model
    model = create_gb_model()
    model.fit(X_train, y_train)
    
    # Save the model
    model_path = os.path.join(model_save_dir, f"model_{train_chr}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Test the model on each chromosome
    for test_idx, test_chr in enumerate(chromosomes):
        print(f"Testing on {test_chr}")
        X_test, y_test = load_data(input_dir,test_chr)
        
        # Evaluate the model
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        accuracy[train_idx, test_idx] = accuracy_score(y_test, y_pred)
        precision[train_idx, test_idx] = precision_score(y_test, y_pred)
        recall[train_idx, test_idx] = recall_score(y_test, y_pred)
        f1[train_idx, test_idx] = f1_score(y_test, y_pred)
        roc_auc[train_idx, test_idx] = roc_auc_score(y_test, y_pred_prob)

# Save metrics
np.save(os.path.join(model_save_dir, "accuracy.npy"), accuracy)
np.save(os.path.join(model_save_dir, "precision.npy"), precision)
np.save(os.path.join(model_save_dir, "recall.npy"), recall)
np.save(os.path.join(model_save_dir, "f1.npy"), f1)
np.save(os.path.join(model_save_dir, "roc_auc.npy"), roc_auc)

print("Training and testing complete.")
