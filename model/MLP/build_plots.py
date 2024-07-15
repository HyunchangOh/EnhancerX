import matplotlib.pyplot as plt

# Sample data: you should replace these with your actual results
structures = [
    [64, 64],
    [128, 128],
    [128, 256, 128],
    [64, 64, 64],
    [128, 128, 128],
    [256, 128, 64],
    [64, 32, 16]
]

# For the sake of the example, these lists should be populated with actual results
accuracies = [0.85, 0.88, 0.90, 0.87, 0.89, 0.86, 0.84]
precisions = [0.82, 0.84, 0.88, 0.83, 0.85, 0.81, 0.80]
runtimes = [100, 120, 150, 110, 130, 140, 90]

# Create a function to plot and save the bar plots
def plot_and_save(data, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    x = range(len(structures))
    labels = ['-'.join(map(str, structure)) for structure in structures]
    
    plt.bar(x, data, tick_label=labels, color='skyblue')
    plt.xlabel('NN Structures')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Plot and save accuracy, precision, and runtime
plot_and_save(accuracies, 'Model Accuracy by NN Structure', 'Accuracy', 'accuracy.png')
plot_and_save(precisions, 'Model Precision by NN Structure', 'Precision', 'precision.png')
plot_and_save(runtimes, 'Model Runtime by NN Structure', 'Runtime (seconds)', 'runtime.png')
