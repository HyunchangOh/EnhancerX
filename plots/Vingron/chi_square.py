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
save_path = "../../../../../scratch/ohh98/vingron/chi/"
for i in range(len(features)):
    f = features[i]
    for j in range(len(chromosomes)):
        c1 = chromosomes[j]
        d1 = np.load(data_path+c1+"/"+f+"2000.npy")
        for k in range(len(chromosomes)):
            if k<j:
                continue
            else:
                c2 = chromosomes[k]
                d2 = np.load(data_path+c2+"/"+f+"2000.npy")
                h1, _ = np.histogram(d1, bins=np.arange(min(d1), max(d2)+2))
                h2, _ = np.histogram(d2, bins=np.arange(min(d1), max(d2)+2))
                statistic, p_value = stats.chisquare(h1, h2)
                p_value_array[j,k] = p_value
                p_value_array[k,j] = p_value

    np.save(save_path+f+".npy",p_value_array)
    p_value_array=np.load(save_path+f+".npy")
    plt.figure(figsize=(16, 12))
    sns.heatmap(p_value_array, annot=True, cmap="coolwarm", cbar=True, square=True, vmin=0, vmax=1, linewidths=0.5, linecolor='black',fmt=".2f")

    # Set labels and title
    plt.xlabel('Chromosomes')
    plt.ylabel('Chromosomes')
    plt.title(f)
    plt.savefig(save_path+f+".png")
    plt.close()
    
