import pickle
import os
import numpy as np
import sklearn.ensemble 
import matplotlib.pyplot as plt
model_path = "/scratch/ohh98/LGT_subsampled_reboot/random_forest_each/all/model_1.pkl"

# Load the model from the pickle file
with open(model_path, 'rb') as file:
    model = pickle.load(file)

input_dir = '/scratch/ohh98/LGT_subsampled_reboot/'
chr_folder = 'all'
data_dir = os.path.join(input_dir, chr_folder)

# Load feature data
feature_names = []

for file in os.listdir(data_dir):
    if file.startswith('BIN50_') and file.endswith('.npy') and (file != 'BIN50_enhancer_atlas.npy'  and not file.startswith("BIN50_enhancer_atlas")):
        feature_name = file.split('.')[0]
        feature_names.append(feature_name)
print(feature_names)
# Extract feature importances
feature_importances = model.feature_importances_

# Sort feature importances in descending order for better visualization
indices = np.argsort(feature_importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(14, 14))
plt.bar(range(len(feature_importances)), feature_importances[indices], align='center')
plt.xticks(range(len(feature_importances)), [feature_names[i] for i in indices], rotation=90)
plt.title('Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.savefig("important_features.png")