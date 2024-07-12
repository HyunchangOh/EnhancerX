import os
import numpy as np
import time
from collections import Counter

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

structures = [
    [32, 32],
    [64, 64],
    [32, 32, 32],
    [64, 64, 64],
    [32, 32, 32, 32]
]



omni_time = time.time()

print("\n" + "#"*90)
print("------------------------------------------------------------------------------------------")
print("#"*90)
print("\n---------------- End of program, ran feature_mix for chr22 in %s seconds ----------------\n" % round(time.time() - omni_time))
print("#"*90)
print("------------------------------------------------------------------------------------------")
print(("#"*90) + "\n")

print("\n--------------------------------------------------------------")
print("-------------------  Trying learning rates -------------------")
print("--------------------------------------------------------------\n")

# print("\n--------------------------------------------------------------")
# print("---------------------  Trying batch size ---------------------")
# print("--------------------------------------------------------------\n")

# print("\n--------------------------------------------------------------")
# print("---------------------  Trying num epochs ---------------------")
# print("--------------------------------------------------------------\n")

print("\n-------------------------------- Running chr22: 5 epochs, 32 batch size, 0.001 learning rate  --------------------------------")

print("\n------------------------ Ran NN on chr22: 5 epochs, 32 batch size, 0.001 learning rate in 155 seconds ------------------------\n")

def mucho_load(chromosome_number, validation = "BIN50_enhancer_atlas"):

    file_path = "../../../../../scratch/ohh98/LGT_subsampled_reboot/" + chromosome_number + "/"
    
    mucho_time = time.time()

    # Get feature list
    feature_list = []
    for file in os.listdir(file_path):
        if file.startswith('BIN50_') and file.endswith('.npy') and (file != 'BIN50_enhancer_atlas.npy'  or not (file.startswith('BIN50_enhancer_atlas') and file.endswith('_1D_Dist.npy') or file.endswith('_3D_Dist.npy'))):
            feature_list.append(file.split('.')[0])
            #print("FEATURE", file, " IS LOADED")
    
    data_length = np.load(file_path + validation + ".npy", mmap_mode='r').size
    no_col = len(feature_list) + 1

    # Check for size comparing all columns to validation data length
    check = [np.load(file_path + feature + ".npy", mmap_mode='r').size for feature in feature_list]
    if any(size != data_length for size in check):
        print(next(feature for feature, size in zip(feature_list, check) if size != check[0]) + " is not of data length.")

    # Initialize the la_grande_table with the correct shape
    la_grande_table = np.empty((no_col, data_length))

    # Load validation as the first column and then fill the rest
    la_grande_table[0] = np.load(file_path + validation + ".npy")
    for idx, feature in enumerate(feature_list, start = 1):
        la_grande_table[idx] = np.load(file_path + feature + ".npy")

    print("\n--- Loaded la grande table for " + chromosome_number + " with " + str(la_grande_table.shape[0]) + " columns, and " + str(la_grande_table.shape[1]) + " items in %s seconds ---\n" % (time.time() - mucho_time))

    return la_grande_table


######################################################### BIG DEMO ##################################################
# Checkeos varios

big_chungus = []
big_feature = []

# Get all feature lists
# for chr in chromosomes[:-1]:
#     big_feature.append(get_feature_list(chr))
"""
# Get all la grande tables on subsampled data
j=0
for chr in chromosomes[:-1]:
    #big_chungus.append(mucho_load(chr, [validation] + big_feature[j]))
    big_chungus.append(mucho_load(chr))
    #j += 1

#Print all unique values for each chromsoome
j=0
for chr in chromosomes[:-1]:
    print(("UNIQUE VALUES OF ENH_ATL at " + chr + ":"))
    print(np.unique(big_chungus[j][0]))
    print(("FIRST 15 VALUES OF ENH_ATL at " + chr + ":"))
    print(big_chungus[j][0][:15])
    print("\n")
    j += 1
"""

def check_unique_n_first_n(feature, chromosome_number, path = "../../../../../scratch/ohh98/la_grande_table/", num_values = 15):
    #testo = np.load("../../../../../scratch/ohh98/la_grande_table/chr1/promoter_any.npy")
    testo = np.load(path + chromosome_number + "/" + feature + ".npy")
    print("\n")
    print(("UNIQUE VALUES OF " + feature + " at " + chromosome_number + ":"))
    print(np.unique(testo))
    print(("FIRST 15 VALUES OF " + feature + " at " + chromosome_number + ":"))
    print(testo[:15])
    print("\n")


check_unique_n_first_n("BIN50_enhancer_atlas", "chr1", path = "../../../../../scratch/ohh98/Subsampled_Final/")
#check_unique_n_first_n("BIN50_Interaction_1D_Dist", "chr1", path = "../../../../../scratch/ohh98/Subsampled_Final/")

save_res = "../../../../../scratch/ohh98/Subsampled_Final/NN_results/"

features = [
    'BIN50_cod',
    'BIN50_cod_1D_Dist',

    'BIN50_promoter_any',
    'BIN50_promoter_1D_Dist_any',
    'BIN50_promoter_3D_Dist_any',
    'BIN50_promoter_forward',
    'BIN50_promoter_1D_Dist_forward',
    'BIN50_promoter_reverse',
    'BIN50_promoter_1D_Dist_reverse',

    'BIN50_h3k4me1',
    'BIN50_h3k4me1_1D_Dist',
    'BIN50_h3k4me1_3D_Dist',

    'BIN50_h3k4me2',
    'BIN50_h3k4me2_1D_Dist',
    'BIN50_h3k4me2_3D_Dist',

    'BIN50_h3k9me3',
    'BIN50_h3k9me3_1D_Dist',
    'BIN50_h3k9me3_3D_Dist',

    'BIN50_h3k27ac',
    'BIN50_h3k27ac_1D_Dist',
    'BIN50_h3k27ac_3D_Dist',

    'BIN50_h3k27me3',
    'BIN50_h3k27me3_3D_Dist',
    'BIN50_h3k27me3_1D_Dist',

    'BIN50_h3k36me3',
    'BIN50_h3k36me3_1D_Dist',
    'BIN50_h3k36me3_3D_Dist',

    'BIN50_CTCF',
    'BIN50_CTCF_3D_Dist',
    'BIN50_CTCF_1D_Dist',

    'BIN50_EP300Conservative_1D_Dist',
    
    'BIN50_DHS',
    'BIN50_DHS_1D_Dist',
    'BIN50_DHS_3D_Dist',

    'BIN50_EP300Conservative',
    'BIN50_EP300Conservative_3D_Dist',
    'BIN50_Interaction_1D_Dist'
    ]

#features_subset = features[0,1,2,4,9,12]
#print("FEATURES SUBSET:", features_subset)
#new_list = a[0:2] + [a[4]] + a[6:].

#########################################################################################################


#########################################################################################################

print("-" * 100)
print("-" * 100)
print("-" * 100)


def list_npy_files_info(directory_path, chromosome_number, f_list):
    """
    Lists all .npy files in the given directory and displays their shape, unique values, and counts.

    Args:
        directory_path (str): The path to the directory to list .npy files from.
    """
    directory_path = directory_path + chromosome_number + "/" + f_list

    print("\n-------------------------------------------------------------------")
    print("-------------------  CHECK " + f_list + " at " + chromosome_number + " -------------------")
    print("-------------------------------------------------------------------\n")

    if not os.path.exists(directory_path):
        print("The directory {} does not exist.".format(directory_path))
        return

    npy_files = [f for f in os.listdir(directory_path) if f.endswith('.npy')]

    if not npy_files:
        print("No .npy files found in the directory {}.".format(directory_path))
        return

    for npy_file in npy_files:
        file_path = os.path.join(directory_path, npy_file)
        try:
            data = np.load(file_path, mmap_mode='r')
            unique_values, counts = np.unique(data, return_counts=True)
            print("File: {}".format(npy_file))
            print("  Shape: {}".format(data.shape))
            print("  Unique Values and Counts: {}".format(dict(zip(unique_values, counts))))
        except Exception as e:
            print("Could not process file {}: {}".format(npy_file, e))
    
    print("\n-------------------------------------------------------------------")
    print("-------------------  FINISHED " + f_list + " at " + chromosome_number + " -------------------")
    print("-------------------------------------------------------------------\n")

# Example usage

# for i in chromosomes[:-1]:
#     for j in ["feature_mix", "feature_mem", "feature_dist"]:
#         list_npy_files_info(save_res, i, j)

# for i in ["feature_mix", "feature_mem", "feature_dist"]:
#     for j in chromosomes[:-1]:
#         list_npy_files_info(save_res, j, i)

list_npy_files_info("temp/", "all", "feature_mix")

list_npy_files_info("temp/", "all", "feature_mem")

list_npy_files_info("temp/", "all", "feature_dist")

