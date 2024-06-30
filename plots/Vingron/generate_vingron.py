import numpy as np
import matplotlib.pyplot as plt

frame_size = 2000

def generate_distribution(enhancer_boolean_array, enhancer_distances_array, feature_boolean_array,frame_size):
    distances = []
    for i in range(len(feature_boolean_array)):
        if feature_boolean_array[i]:
            if i>0:
                if enhancer_boolean_array[i-1] and enhancer_boolean_array[i]:
                    continue
                else:
                    d = enhancer_distances_array[i]
                    if d<=frame_size:
                        distances.append(d)
            else:
                d = enhancer_distances_array[i]
                if d<=frame_size:
                    distances.append(d)
    return distances

ex_enhancer_boolean_array = [False,False,False,True,True,True,False,False]
ex_enhancer_1D_array = [3,2,1,0,0,0,1,2]
ex_feature_boolean_array = [True,True,True,True,True,True,True,True]

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
        'chrY'
    ]

features = ["CTCF","cod","DHS","EP300Conservative","h3k4me1","h3k4me2","h3k9me3","h3k27ac","h3k27me3","h3k36me3","promoter_any","promoter_forward","promoter_reverse"]

chromosomes=["chrY"]
data_path = "../../../../../scratch/ohh98/la_grande_table/"
save_path = "../../../../../scratch/ohh98/vingron/"
for c in chromosomes:
    chr_path = data_path+c+"/"
    enhancer_boolean_array = np.load(chr_path+"enhancer_atlas.npy")
    enhancer_distances_array = np.load(chr_path+"enhancer_atlas_1D_Dist.npy")
    for f in features:
        feature_boolean_array = np.load(chr_path+f+".npy")
        distances = generate_distribution(enhancer_boolean_array, enhancer_distances_array, feature_boolean_array,frame_size)
        
        hist, bin_edges = np.histogram(distances, bins=np.arange(0, frame_size + 1))
        hist_normalized = hist / np.max(hist[:-1])

        plt.figure(figsize=(10, 6))
        plt.plot(bin_edges[:-3], hist_normalized[:-2], marker='', linestyle='-', color='black', markeredgecolor='black')
        plt.xlim(0, frame_size)
        plt.ylim(0, 1)
        plt.xlabel('Distance')
        plt.ylabel('Normalized Frequency')
        plt.title(c+"_"+f)

        plt.savefig(save_path+c+"/"+f+str(frame_size)+".png")
        plt.close()
        np.save(save_path+c+"/"+f+str(frame_size)+".npy",distances)



for f in features:
    overall_distribution = np.array([])
    for c in chromosomes:
        dist = np.load(save_path+c+"/"+f+str(frame_size)+".npy")
        overall_distribution = np.concatenate((overall_distribution,dist))
    np.save(save_path+"overall"+"/"+f+str(frame_size)+".npy", overall_distribution)

    hist, bin_edges = np.histogram(overall_distribution, bins=np.arange(0, frame_size + 1))

    # Normalize the histogram
    hist_normalized = hist / np.max(hist[:-1])
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(bin_edges[:-3], hist_normalized[:-2], marker='', linestyle='-', color='black', markeredgecolor='black')
    plt.xlim(0, frame_size)
    plt.ylim(0, 1)
    plt.xlabel('Distance')
    plt.ylabel('Normalized Frequency')
    plt.title(c+"_"+f)

    plt.savefig(save_path+"overall"+"/"+f+str(frame_size)+".png")
    plt.close()
