import numpy as np
from scipy import stats
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


p_value_array = np.zeros((len(chromosomes),len(chromosomes)))

data_path = "../../../../../scratch/ohh98/vingron/"
save_path = "../../../../../scratch/ohh98/vingron/mann_whitney/"
for i in range(len(features)):
    f = features[i]

    p_value_array=np.load(save_path+f+".npy")
    plt.figure(figsize=(16, 12))
    sns.heatmap(p_value_array, annot=True, cmap="coolwarm", cbar=True, square=True, vmin=0, vmax=0.5, linewidths=0.5, linecolor='black',fmt=".2f")

    # Set labels and title
    plt.xlabel('Chromosomes')
    plt.ylabel('Chromosomes')
    plt.title(f)
    plt.savefig(save_path+f+"_redrawn.png")
    plt.close()
    
