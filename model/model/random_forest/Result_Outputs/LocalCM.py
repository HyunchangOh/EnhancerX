import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(ax, conf_matrix, title):
    # Convert all values to integers
    conf_matrix = conf_matrix.astype(int)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'],
                ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)

# These metrics are taken from the results individually, as the scikit confusion matrix is not working on the HPC cluster
# The code was run locally 
metrics = [
    # (precision_0, recall_0, f1_0, support_0, precision_1, recall_1, f1_1, support_1)
    (0.95, 0.93, 0.94, 33672, 0.94, 0.96, 0.95, 33588), #chr1
    (0.97, 0.95, 0.96, 29109, 0.95, 0.97, 0.96, 29079), #chr2
    (0.96, 0.94, 0.95, 22757, 0.94, 0.96, 0.95, 22313), #chr3
    (0.97, 0.96, 0.96, 14135, 0.96, 0.97, 0.96, 14132), #chr4
    (0.97, 0.96, 0.96, 18487, 0.96, 0.97, 0.96, 18444), #chr5
    (0.96, 0.95, 0.96, 22534, 0.95, 0.96, 0.96, 22506), #chr6
    (0.91, 0.89, 0.90, 17391, 0.89, 0.92, 0.90, 17131), #chr7
    ( 0.74, 0.97, 0.84, 15745, 0.96, 0.66, 0.78, 15800), #chr8
    (0.75, 0.97, 0.84, 13711, 0.95, 0.68, 0.79, 13855), #chr9
    (0.73, 0.97, 0.84, 17834, 0.96, 0.65, 0.77, 17948), #chr10
    (0.73, 0.97, 0.83, 16808, 0.95, 0.65, 0.77, 16958), #chr11
    (0.72, 0.96, 0.83, 18832, 0.95, 0.63, 0.76, 18740), #chr12
    (0.75, 0.97, 0.85, 8154, 0.96, 0.67, 0.79, 8139), #chr13
    (0.73, 0.97, 0.83, 12998, 0.96, 0.65, 0.77, 13204), #chr14
    (0.74, 0.96, 0.83, 12933, 0.95, 0.66, 0.78, 13191), #chr15
    (0.72, 0.95, 0.82, 13212, 0.93, 0.63, 0.75, 13082), #chr16
    (0.71, 0.94, 0.81, 16849, 0.92, 0.61, 0.74, 16892), #chr17
    (0.76, 0.97, 0.86, 7536, 0.96, 0.70, 0.81, 7610), #chr18
    (0.70, 0.94, 0.80, 13095, 0.90, 0.60, 0.72, 12969), #chr19
    (0.74, 0.96, 0.84, 9847, 0.95, 0.66, 0.78, 9953), #chr20
    (0.74, 0.97, 0.84, 4599, 0.95, 0.65, 0.77, 4499), #chr21
    (0.97, 0.94, 0.95, 8407, 0.94, 0.97, 0.95, 8220), #chr22
    (0.78, 0.90, 0.83, 7367, 0.88, 0.74, 0.81, 7491) #chrX
]

# Create subplots
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
axes = axes.flatten()

# Define chromosome titles
titles = [f'Chromosome {i+1}' for i in range(22)] + ['Chromosome X']

# Plot each confusion matrix heatmap
for i, (precision_0, recall_0, f1_0, support_0, precision_1, recall_1, f1_1, support_1) in enumerate(metrics):
    # Calculate confusion matrix elements
    TN = recall_0 * support_0
    FP = (1 - recall_0) * support_0
    FN = (1 - recall_1) * support_1
    TP = recall_1 * support_1

    # Construct the confusion matrix
    conf_matrix = np.array([[TN, FP], [FN, TP]])
    
    # Plot the heatmap
    plot_heatmap(axes[i], conf_matrix, titles[i])

# Hide any unused subplots
for j in range(len(metrics), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
