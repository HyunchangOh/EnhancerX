import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pickle

# Define paths
data_dir = "../../../../../scratch/ohh98/la_grande_table/chr22"
response_file = os.path.join(data_dir, "enhancer_atlas.npy")
model_save_dir = "../../../../../scratch/ohh98/models/chr22_random_forest"
os.makedirs(model_save_dir, exist_ok=True)

# Load response data
response = np.load(response_file)
selected_features = ['CTCF', 'h3k27ac']
features = {}

# Load features
for feature in selected_features:
    feature_file = os.path.join(data_dir, f"{feature}.npy")
    if os.path.exists(feature_file):
        features[feature] = np.load(feature_file)
    else:
        raise FileNotFoundError(f"Feature file for {feature} not found at {feature_file}")

# Ensure that 'promoter_1D_Dist' is treated as binary if necessary
if 'promoter_1D_Dist' in features:
    features['promoter_1D_Dist'] = features['promoter_1D_Dist'].astype(int)

# Stack features
X = np.hstack([features[key].reshape(len(features[key]), -1) for key in selected_features])
y = (response != 0).astype(int)  # Convert response variable to binary

# Define the Random Forest model with class weight adjustment
def create_rf_model():
    model = RandomForestClassifier(n_estimators=10, class_weight='balanced', random_state=42)
    return model

# Cross-Validation
num_splits = 5

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
