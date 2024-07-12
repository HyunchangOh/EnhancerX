import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import pickle

def load_data(data_dir, feature_prefix='BIN50_', target_filename='BIN50_enhancer_atlas.npy'):
    """ Load features and target data from specified directory. """
    chr_folders = [f"chr{i}" for i in range(1, 23)] + ['chrX']
    features = []
    targets = []

    for chr_folder in chr_folders:
        chr_path = os.path.join(data_dir, chr_folder)
        if not os.path.exists(chr_path):
            print(f"Skipping {chr_folder}: path does not exist.")
            continue

        target_path = os.path.join(chr_path, target_filename)
        if not os.path.exists(target_path):
            print(f"Skipping {chr_folder}: target file does not exist.")
            continue

        y = np.load(target_path)
        targets.append(y)

        feature_files = [f for f in os.listdir(chr_path) if f.startswith(feature_prefix) 
                         and f.endswith('.npy') and f != 'BIN50_enhancer_atlas_1D_Dist.npy'
                         and f != 'BIN50_enhancer_atlas_3D_Dist.npy']  # Exclude specific files
        chr_features = [np.load(os.path.join(chr_path, f)) for f in feature_files]
        if len(chr_features) > 0:
            features.append(np.vstack(chr_features).T)
    
    if len(features) == 0 or len(targets) == 0:
        raise ValueError("No valid data found in the specified directory.")
    
    X = np.vstack(features)
    y = np.concatenate(targets)
    
    return X, y, feature_files

def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def plot_roc_curve(y_true, y_pred_prob, output_file):
    """ Plot and save the ROC curve. """
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(output_file)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_file):
    """ Plot and save the confusion matrix as a heatmap. """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(output_file)
    plt.close()



def main(data_dir, output_dir, feature_prefix='BIN50_', target_filename='BIN50_enhancer_atlas.npy'):
    os.makedirs(output_dir, exist_ok=True)
    X, y, _ = load_data(data_dir, feature_prefix, target_filename)
    input_shape = (X.shape[1], 1)  # Assuming input features are 1D
    
    # Reshape X to 3D array (samples, features, channels)
    X = np.expand_dims(X, axis=-1)
    
    # Split data into train and test sets (example: 80% train, 20% test)
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Build CNN model
    model = build_cnn_model(input_shape)
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    # Save the model
    model_path = os.path.join(output_dir, 'cnn_model.h5')
    model.save(model_path)
    
    # Evaluate the model
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Save ROC curve
    plot_roc_curve(y_test, y_pred_prob, os.path.join(output_dir, 'roc_curve.png'))
    
    # Save confusion matrix as heatmap
    plot_confusion_matrix(y_test, y_pred, os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Save classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))

    print(f"Model, plots, and reports saved to {output_dir}")

if __name__ == "__main__":
    data_dir = "../../../../../scratch/ohh98/la_grande_table_subsampled/subsampled_data"
    output_dir = "../../../../../scratch/ohh98/la_grande_table_subsampled/subsampled_data/cnn_results/"
    main(data_dir, output_dir)