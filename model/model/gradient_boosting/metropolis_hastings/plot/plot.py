import numpy as np
import matplotlib.pyplot as plt

# Sample data
arrays = [
    np.load("f1_scores.npy"),
    np.load("precisions.npy"),
    np.load("recalls.npy"),
    np.load("roc_auc.npy"),
    np.load("accuracies.npy")
]

# Names of the arrays
array_names = ["f1_score", "precision", "recall", "roc_auc", "accuracy"]

# Sample "tomato" data
tomato = np.load("parameters.npy")

# Number of arrays
n_arrays = len(arrays)

# Calculate means and standard deviations
means = [np.mean(arr, axis=1) for arr in arrays]
stds = [np.std(arr, axis=1) for arr in arrays]

# Convert "tomato" arrays to concatenated strings
x_labels = ["_".join([f"{num:.5f}" for num in arr]) for arr in tomato]

# Colors for each line
colors = ['b', 'g', 'r', 'c', 'm']

# Plotting
plt.figure(figsize=(18, 12))

# Plot each array
for i in range(n_arrays):
    x = np.arange(len(means[i]))
    plt.errorbar(x, means[i], yerr=stds[i], label=array_names[i], color=colors[i], capsize=5)

# Set custom x-axis labels
plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, rotation=90, ha='right')

plt.xlabel('Custom Labels')
plt.ylabel('Value')
plt.title('Line Graphs with Error Bars and Custom X-axis Labels')
plt.legend()
plt.grid(True)
plt.tight_layout()  # Adjust layout to make room for the x-axis labels
plt.savefig("plot.png")