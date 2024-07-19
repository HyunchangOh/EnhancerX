import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt 
import seaborn as sns

chromosomes = [
        'chr1',
        'chr2',
        'chr3',
        'chr4',
        'chr5',
        'chr6',
        'chr7',
        'chr8',
        'chr9',
        'chr10',
        'chr11',
        'chr12',
        'chr13',
        'chr14',
        'chr15',
        'chr16',
        'chr17',
        'chr18',
        'chr19',
        'chr20',
        'chr21',
        'chr22',
        'chrX'
    ]

features = ["CTCF","cod","DHS","EP300Conservative","h3k4me1","h3k4me2","h3k9me3","h3k27ac","h3k27me3","h3k36me3","promoter_any","promoter_forward","promoter_reverse"]

data_path = "../../../../scratch/ohh98/le_grande_table/"
save_path = "../../../../scratch/ohh98/CV_correlation/"

# Define a function to compute Cramér's V
def cramers_v(x, y):
    confusion_matrix = np.histogram2d(x, y, bins=2)[0]
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = np.sum(confusion_matrix)
    return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))

# Number of rows and columns for the subplot grid
nrows = 4
ncols = 4

fig, axs = plt.subplots(nrows, ncols, figsize=(24, 24))
axs = axs.flatten()

for i in range(len(features)):
    f = features[i]
    correlation_array = np.zeros((len(chromosomes), len(chromosomes)))
    
    for j in range(len(chromosomes)):
        c1 = chromosomes[j]
        d1 = np.load(data_path + c1 + "/" + f + "2000.npy")
        
        for k in range(len(chromosomes)):
            if k < j:
                continue
            else:
                c2 = chromosomes[k]
                d2 = np.load(data_path + c2 + "/" + f + "2000.npy")
                # Calculate Cramér's V for binary data
                correlation = cramers_v(d1, d2)
                correlation_array[j, k] = correlation
                correlation_array[k, j] = correlation
    
    sns.heatmap(correlation_array, annot=False, cmap="coolwarm", cbar=True, square=True, 
                vmin=0, vmax=1, linewidths=0.5, linecolor='black', ax=axs[i])
    
    axs[i].set_title(f)
    axs[i].set_xlabel('Chromosomes')
    axs[i].set_ylabel('Chromosomes')

# Adjust layout
plt.tight_layout()
plt.savefig(save_path + "combined_correlation_heatmaps.png")
plt.show()
