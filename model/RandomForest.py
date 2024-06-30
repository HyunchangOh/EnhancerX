import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pickle

# Define paths
data_dir = "../../../../scratch/ohh98/la_grande_table/chr1"
response_file = os.path.join(data_dir, "enhancer_atlas.npy")
model_save_dir = "../../../../scratch/ohh98/models/chr1_random_forest"
os.makedirs(model_save_dir, exist_ok=True)

# Load data
response = np.load(response_file)
features = {}
for file in os.listdir(data_dir):
    if file != "enhancer_atlas.npy" and file.endswith(".npy"):
        feature_name = file.split(".")[0]
        features[feature_name] = np.load(os.path.join(data_dir, file))

# Handle inf values in promoter_forward and promoter_reverse
for key in ['promoter_forward', 'promoter_reverse']:
    if key in features:
        max_value = np.max(features[key][np.isfinite(features[key])])  # Find the max finite value
        features[key][np.isinf(features[key])] = 10 * max_value       # Replace inf with 10 times the max value

# Convert categorical variables using pandas
categorical_feature = 'seq'
categorical_data = features[categorical_feature].reshape(-1, 1)
categorical_data_one_hot = pd.get_dummies(categorical_data[:, 0]).values
features[categorical_feature] = categorical_data_one_hot

# Stack features
X = np.hstack([features[key].reshape(len(features[key]), -1) for key in features.keys()])
y = (response != 0).astype(int)  # Convert response variable to binary

# Check and handle NaNs, infinities, and large values in X
X = np.nan_to_num(X, nan=np.nanmean(X, axis=0), posinf=10*np.nanmax(X[np.isfinite(X)]), neginf=-10*np.nanmax(X[np.isfinite(X)]))

# Verify that there are no NaNs or infinities in the final dataset
if np.isnan(X).any() or np.isinf(X).any():
    raise ValueError("The dataset still contains NaNs or infinities after preprocessing.")

# Define the Random Forest model
def create_rf_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model

# Leave-One-Out Cross-Validation
chromosome_length = len(y)
num_splits = 5
split_size = chromosome_length // num_splits

metrics = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "roc_auc": []
}

kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

for i, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Training fold {i + 1}/{num_splits}")
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Create and train the model
    model = create_rf_model()
    model.fit(X_train, y_train)

    # Save the model
    model_path = os.path.join(model_save_dir, f"model_{i + 1}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Evaluate the model
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics["accuracy"].append(accuracy_score(y_test, y_pred))
    metrics["precision"].append(precision_score(y_test, y_pred))
    metrics["recall"].append(recall_score(y_test, y_pred))
    metrics["f1"].append(f1_score(y_test, y_pred))
    metrics["roc_auc"].append(roc_auc_score(y_test, y_pred_prob))

    print(f"Fold {i + 1} results:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_prob)}\n")

# Save metrics
with open(os.path.join(model_save_dir, "metrics.pkl"), "wb") as f:
    pickle.dump(metrics, f)

print("Training complete.")
