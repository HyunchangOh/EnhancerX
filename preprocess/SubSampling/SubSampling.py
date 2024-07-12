import numpy as np
import os
from sklearn.utils import resample

def subsample_data(data_dir, target_filename='BIN50_enhancer_atlas.npy', output_dir='subsampled_data', random_state=42):
    """
    Subsample data to address class imbalance for each chromosome folder.

    Parameters:
    - data_dir: str, path to the main data directory containing chromosome folders
    - target_filename: str, filename of the target variable file (default: 'BIN50_enhancer_atlas.npy')
    - output_dir: str, directory where subsampled data will be saved (default: 'subsampled_data')
    - random_state: int, random state for reproducibility (default: 42)
    """
    output_dir = "/scratch/ohh98/Subsampled_Final/"
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List all chromosome folders
    chr_folders = [f"chr{i}" for i in range(1, 23)] + ['chrX']

    for chr_folder in chr_folders:
        chr_path = os.path.join(data_dir, chr_folder)
        if not os.path.exists(chr_path):
            print(f"Skipping {chr_path}: path does not exist.")
            continue

        # Load the target variable
        target_path = os.path.join(chr_path, target_filename)
        if not os.path.exists(target_path):
            print(f"Skipping {chr_path}: target file does not exist.")
            continue
        
        y = np.load(target_path)
        minority_class_indices = np.where(y != 0)[0]
        majority_class_indices = np.where(y == 0)[0]

        if len(minority_class_indices) == 0 or len(majority_class_indices) == 0:
            print(f"Skipping {chr_path}: insufficient class instances for subsampling.")
            continue

        # Subsample the majority class to match the minority class size
        subsample_size = len(minority_class_indices)
        majority_class_indices_subsampled = resample(majority_class_indices, 
                                                     replace=False, 
                                                     n_samples=subsample_size, 
                                                     random_state=random_state)
        
        # Combine subsampled majority class with the minority class
        subsampled_indices = np.concatenate([majority_class_indices_subsampled, minority_class_indices])
        np.random.shuffle(subsampled_indices)

        # Load and subsample features
        feature_files = [f for f in os.listdir(chr_path) if "BIN50" in f and f.endswith('.npy') and not "enhancer" in f and f != target_filename]
        for feature_file in feature_files:
            print(feature_file)
            feature_path = os.path.join(chr_path, feature_file)
            feature_data = np.load(feature_path)
            
            # Subsample the feature data
            feature_data_subsampled = feature_data[subsampled_indices]
            
            # Save the subsampled feature data
            subsampled_feature_path = os.path.join(output_dir, chr_folder)
            os.makedirs(subsampled_feature_path, exist_ok=True)
            np.save(os.path.join(subsampled_feature_path, feature_file), feature_data_subsampled)

        # Save the subsampled target variable
        y_subsampled = y[subsampled_indices]
        np.save(os.path.join(output_dir, chr_folder, target_filename), y_subsampled)
        print(f"Subsampled data for {chr_folder} saved successfully.")

if __name__ == "__main__":
    data_dir = "/scratch/ohh98/la_grande_table/"
    subsample_data(data_dir)
