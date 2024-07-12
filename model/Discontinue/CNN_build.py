import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

def load_data(data_dir, feature_prefix='BIN50_', target_filename='BIN50_enhancer_atlas.npy'):
    """ Load features and target data from specified directory. """
    chr_folders = [f"chr{i}" for i in range(1, 23)] + ['chrX']
    features = []
    targets = []
    expected_num_features = None

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
            chr_features_stacked = np.vstack(chr_features).T
            if expected_num_features is None:
                expected_num_features = chr_features_stacked.shape[1]
            if chr_features_stacked.shape[1] == expected_num_features:
                features.append(chr_features_stacked)
            else:
                print(f"Skipping {chr_folder}: feature dimension mismatch.")
    
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
    
    # Save the test data for visualization
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    y_pred_prob = model.predict(X_test).flatten()
    np.save(os.path.join(output_dir, 'y_pred_prob.npy'), y_pred_prob)

if __name__ == "__main__":
    data_dir = "../../../../../scratch/ohh98/la_grande_table_subsampled/subsampled_data"
    output_dir = "../../../../../scratch/ohh98/la_grande_table_subsampled/subsampled_data/cnn_results/"
    main(data_dir, output_dir)
