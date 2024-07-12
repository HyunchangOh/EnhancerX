import numpy as np
import os

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

path = "/scratch/ohh98/Subsampled_Final/"

for filename in os.listdir(path+chromosomes[0]):
    
    if os.path.isfile(os.path.join(path+chromosomes[0], filename)):
        a= np.array([])
        for c in chromosomes:
            print(filename)
            a = np.concatenate((a,np.load(path+"/"+c+"/"+filename)))
        np.save(path+"all/"+filename,a)