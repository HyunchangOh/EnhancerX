import re
import matplotlib.pyplot as plt

# Provided log content as a string
log_content = #my path here locally on .txt files 

# Extract accuracy and ROC AUC for each fold
folds = []
accuracies = []
roc_aucs = []

for match in re.finditer(r'Training fold (\d+)/\d+.*?accuracy\s+([\d.]+).*?ROC AUC:\s+([\d.]+)', log_content, re.DOTALL):
    fold = int(match.group(1))
    accuracy = float(match.group(2))
    roc_auc = float(match.group(3))
    folds.append(fold)
    accuracies.append(accuracy)
    roc_aucs.append(roc_auc)

# Plot the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(folds, accuracies, marker='o', label='Accuracy')
plt.title('Accuracy per Fold')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Start y-axis from 0
plt.xticks(folds)  # Set x-ticks to 1, 2, 3, 4, 5
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(folds, roc_aucs, marker='o', color='orange', label='ROC AUC')
plt.title('ROC AUC per Fold')
plt.xlabel('Fold')
plt.ylabel('ROC AUC')
plt.ylim(0, 1)  # Start y-axis from 0
plt.xticks(folds)  # Set x-ticks to 1, 2, 3, 4, 5
plt.grid(True)

# Add a main title
plt.suptitle('Chromosome ', fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the main title
plt.show()
