import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt 

path = "/scratch/ohh98/vingron3D/"

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
        'chrX',
        'overall'
    ]

features = ["CTCF","cod","DHS","EP300Conservative","h3k4me1","h3k4me2","h3k9me3","h3k27ac","h3k27me3","h3k36me3","promoter_any","promoter_forward","promoter_reverse"]

chromosomes=["overall"]
# chromosomes = ["chr1"]
# features = ["CTCF"]

for c in chromosomes:
    for f in features:
        e = np.load("/scratch/ohh98/vingron/"+c+"/"+f+"2000.npy")
        p = np.load("/scratch/ohh98/vingron3D/"+c+"/"+f+"2000.npy")
        
        hist, bin_edges = np.histogram(e, bins=np.arange(0, 2000 + 1))
        hist2, bin_edges2 = np.histogram(p, bins=np.arange(0, 2000 + 1))

        # Normalize the histogram
        hist_normalized = hist / np.max(hist[:-1])
        hist_normalized2 = hist2 / np.max(hist2[:-1])
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(bin_edges[:-3], hist_normalized[:-2], marker='', linestyle='-', color='green', markeredgecolor='black',label="1D Distances")
        plt.plot(bin_edges2[:-3], hist_normalized2[:-2], marker='', linestyle='-', color='purple', markeredgecolor='black', label="3D Distances")
        plt.xlim(0, 2000)
        plt.ylim(0, 1)
        plt.legend()
        plt.xlabel('Distance')
        plt.ylabel('Normalized Frequency')
        plt.title(c+"_"+f)

        plt.savefig("1D3D/"+f+".png")
        plt.close()