import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pickle

# Define paths
data_dir = "../../../../scratch/ohh98/la_grande_table/chr1"
response_file = os.path.join(data_dir, "atl.npy")
model_save_dir = "../../../../scratch/ohh98/models/chr1"
os.makedirs(model_save_dir, exist_ok=True)

# Load data
response = np.load(response_file)
features = {}
for file in os.listdir(data_dir):
    if file != "atl.npy" and file.endswith(".npy"):
        feature_name = file.split(".")[0]
        features[feature_name] = np.load(os.path.join(data_dir, file))

# Convert categorical variables
categorical_feature = 'seq'
categorical_data = features[categorical_feature]
categorical_data_one_hot = tf.keras.utils.to_categorical(categorical_data)
features[categorical_feature] = categorical_data_one_hot

# Stack features
X = np.hstack([features[key].reshape(len(features[key]), -1) for key in features.keys()])
y = (response != 0).astype(int)  # Convert response variable to binary

# Reshape X for Conv1D (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Define the CNN model
def create_model(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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

for i in range(num_splits):
    print(f"Training fold {i + 1}/{num_splits}")
    
    test_indices = range(i * split_size, (i + 1) * split_size)
    train_indices = [x for x in range(chromosome_length) if x not in test_indices]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # Create and train the model
    model = create_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

    # Save the model
    model_path = os.path.join(model_save_dir, f"model_{i + 1}.h5")
    model.save(model_path)

    # Evaluate the model
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

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