import matplotlib.pyplot as plt
import numpy as np

# Define the data for each chromosome
results = {
    'chr1': {'accuracy': 0.94, 'roc_auc': 0.9841260139780827},
    'chr2': {'accuracy': 0.96, 'roc_auc': 0.9891196697397182},
    'chr3': {'accuracy': 0.95, 'roc_auc': 0.9863957508854266},
    'chr4': {'accuracy': 0.96, 'roc_auc': 0.9912032275204796},
    'chr5': {'accuracy': 0.96, 'roc_auc': 0.9906798395918649},
    'chr6': {'accuracy': 0.96, 'roc_auc': 0.9890313797448457},
    'chr7': {'accuracy': 0.90, 'roc_auc': 0.9587611181129241},
    'chr8': {'accuracy': 0.82, 'roc_auc': 0.8456286625048738},
    'chr9': {'accuracy': 0.82, 'roc_auc': 0.8592980408773879},
    'chr10': {'accuracy': 0.81, 'roc_auc': 0.8394729913181213},
    'chr11': {'accuracy': 0.81, 'roc_auc': 0.8419801621347565},
    'chr12': {'accuracy': 0.80, 'roc_auc': 0.8286298614429537},
    'chr13': {'accuracy': 0.82, 'roc_auc': 0.8592199149659387},
    'chr14': {'accuracy': 0.81, 'roc_auc': 0.8458464166579538},
    'chr15': {'accuracy': 0.81, 'roc_auc': 0.8441989116443879},
    'chr16': {'accuracy': 0.79, 'roc_auc': 0.828488014051242},
    'chr17': {'accuracy': 0.78, 'roc_auc': 0.8109885325530879},
    'chr18': {'accuracy': 0.84, 'roc_auc': 0.8636031586274625},
    'chr19': {'accuracy': 0.77, 'roc_auc': 0.798419083825203},
    'chr20': {'accuracy': 0.81, 'roc_auc': 0.8348633979316885},
    'chr21': {'accuracy': 0.81, 'roc_auc': 0.846984937968627},
    'chr22': {'accuracy': 0.95, 'roc_auc': 0.9881495608022164},
    'chrX': {'accuracy': 0.82, 'roc_auc': 0.8526520771126881}
}

# Extract chromosome names and corresponding metrics
chromosomes = list(results.keys())
accuracies = [results[chr]['accuracy'] for chr in chromosomes]
roc_aucs = [results[chr]['roc_auc'] for chr in chromosomes]

# Function to create accuracy bar plot
def plot_accuracy(chromosomes, accuracies):
    plt.figure(figsize=(12, 6))
    plt.bar(chromosomes, accuracies, color='lightblue', alpha=0.7, label='Accuracy')
    plt.xlabel('Chromosome')
    plt.ylabel('Accuracy')
    plt.title('Accuracy across Chromosomes')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to create ROC AUC bar plot
def plot_roc_auc(chromosomes, roc_aucs):
    plt.figure(figsize=(12, 6))
    plt.bar(chromosomes, roc_aucs, color='lightblue', alpha=0.7, label='ROC AUC')
    plt.xlabel('Chromosome')
    plt.ylabel('ROC AUC')
    plt.title('ROC AUC across Chromosomes')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Generate the plots
plot_accuracy(chromosomes, accuracies)
plot_roc_auc(chromosomes, roc_aucs)
