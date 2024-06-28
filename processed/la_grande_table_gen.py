# GENERATE LA GRANDE TABLE
## Some comments:
'''
Check whether there are already sequence and coding annotation npy ingrande table: CHECK
Time and progress for fasta reading: CHECK

Ranges are only implemented for seq and cod, not for enhancer atlas annotation, don't know wheter to apply for both or none.
Reference genome is defaulted to hg19, don't know whether we're always gonna work with it and take it out, or leave it as it is.
Cell line enhacner atlas is set to "GM12878". Should be simple adding more
read_enh_atlas is kinda generic, was originally intended to be, but fit to do enhancer atlas.

Format is:
#########################################################################################################
## Description of function/section

#########################################################################################################

'''

#########################################################################################################
## Imports
### The Python Standard Library
import os
import time
### Others
import numpy as np

#########################################################################################################

## Globals?
## Ranges to analyze
# chr_no = "chr1"
# demo_start = 0
# demo_end = 20000000 # 20M
# #demo_end = 500000 # 500k
# demo_len = demo_end-demo_start
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

#########################################################################################################
## Create folder structure. If some folder already exists, then nothing happens.

def ensure_folder_structure():
    base_path = "../la_grande_table/"
    
    # Create all folders, exist=ok then ignores already created folders
    for folder in chromosomes:
        os.makedirs(base_path + folder, exist_ok=True)

ensure_folder_structure()

#########################################################################################################


#########################################################################################################
## Check whether fasta file exists for given chromosome. Read fasta, print progress every 5%, then return string of base pairs.

def read_chromosome(chromosome_number, ref_genome = "hg19", range_start=None, range_end=None):
    file_path = "../data/" + ref_genome + "/" + chromosome_number + ".fa"
    if not os.path.exists(file_path):
        print("Fasta file not found for " + chromosome_number + ".")
        return
    with open(file_path, "r") as g:
        g.readline()  # Skip the header line
        lines = g.readlines()
        print(chromosome_number + "'s lines are read.")
    
    # Combine all lines into a single string and strip newlines
    full_chromosome_data = ''.join(line.strip() for line in lines)

    # Determine the length of the chromosome data
    chromosome_length = len(full_chromosome_data)
    
    # Set default values for range_start and range_end, and validate range
    if range_start is None:
        range_start = 0
    if range_end is None:
        range_end = chromosome_length

    if range_end > chromosome_length:
        range_end = chromosome_length

    # Print progress every 5%
    step = (range_end - range_start) // 20 

    # Extract the desired range of chromosome data
    chromosome_data = ""
    for i in range(range_start, range_end):
        chromosome_data += full_chromosome_data[i]
        
        # Print out progress
        if (i - range_start) % step == 0:
            print(round(((i - range_start) / (range_end - range_start)) * 100), "% read")

    print("Chromosome data extracted")

    return chromosome_data

#########################################################################################################

#########################################################################################################
## Check whether fasta and npy files exist. 
## Then use read_chromosome function and translate to numeric. 
## Then "divide" into sequence and coding/non-coding annotation numpy arrays.
## Save npy files, and in case one already exists, only save the other one

def read_translate_save(chromosome_number, ref_genome = "hg19", range_start = None, range_end = None):

    file_path = "../la_grande_table/" + chromosome_number + "/"
    file_path_single = None
    # Check whether the fasta file is there, and whether there is already an npy file
    if os.path.exists(file_path + "seq.npy") and os.path.exists(file_path + "cod.npy"):
        print("Sequence and coding files already exist for " + chromosome_number + ".")
        return
    elif not os.path.exists("../data/" + ref_genome + "/" + chromosome_number +  ".fa"):
        print("Fasta for " + chromosome_number + " not found.")
        return
    else:
        if os.path.exists(file_path + "cod.npy"):
            file_path_single = file_path + "seq"
            seq = True
            print("Ahead with loading (coding file already exists for " + chromosome_number + ").")
        elif os.path.exists(file_path + "seq.npy"):
            file_path_single = file_path + "cod"
            seq = False
            print("Ahead with loading (sequence file already exists for " + chromosome_number + ").")
        else: 
            print("Ahead with loading both sequence and coding annotation for " + chromosome_number + ".")

    start_time = time.time()
    Genome_l = list(read_chromosome(chromosome_number, ref_genome=ref_genome, range_start=range_start, range_end=range_end))
    print("\n--- Reading " + chromosome_number + ", with " + str(len(Genome_l)) + "bp, took %s seconds ---\n" % (time.time() - start_time))

    # Transform letters to numbers
    dDNA = {"a": 0 , "c": 1 , "g": 2, "t": 3, "n": -1,
            "A": 0 , "C": 1 , "T": 2, "G": 3, "N": -1}
    
    # Create genome sequence
    genome = np.array([dDNA[bp] for bp in Genome_l])

    # Create coding annotation
    coding = np.array([1 if bp.isupper() else 0 for bp in Genome_l])

    # Save sequence and/or coding/non-coding annotation to la grande table, depending on whether one of them already exists
    if file_path_single is not None:
        if seq==True:
            np.save(file_path_single, genome, allow_pickle=False, fix_imports=False)
        else:
            np.save(file_path_single, coding, allow_pickle=False, fix_imports=False)
    else: 
        np.save(file_path + "seq", genome, allow_pickle=False, fix_imports=False)
        np.save(file_path + "cod", coding, allow_pickle=False, fix_imports=False)

#########################################################################################################


#########################################################################################################
## Read all available fasta files stored in your data/

for i in chromosomes:
    read_translate_save(i)

#########################################################################################################


#########################################################################################################
## Quick test for loading sequence and coding annotation

# time_seq1 = time.time()
# testo1 = np.load("../la_grande_table/chr1/seq.npy", allow_pickle=False, fix_imports=False)
# print("\n--- Loading chr1 seq with " + str(len(testo1)) + "bp, took %s seconds ---\n" % (time.time() - time_seq1))
# print("Seq chr1 demo: ", testo1[11470:11480])

# time_cod1 = time.time()
# testo2 = np.load("../la_grande_table/chr1/cod.npy", allow_pickle=False, fix_imports=False)
# print("\n--- Loading chr1 cod with " + str(len(testo2)) + " elements, took %s seconds ---\n" % (time.time() - time_cod1))
# print("Cod chr1 demo: ", testo2[11470:11480])

#########################################################################################################


#########################################################################################################
## Read enhancer atlas for all cell line files stored in data/enhancer_atlas/.
## Could be extended to other databases in principle.
## Outputs a dictionary, where the key is the string of cell line name, value is a list (all lines) of lists (each line) of items (columns).

def read_enh_atlas(database_name, file_ext=".txt"):
    
    database_path = "../data/" + database_name + "/"
    database = {}
    if not os.path.exists(database_path):
        print(database_name + " not found, could not be read.")
        return

    # Iterate over all files in the directory
    for filename in os.listdir(database_path):
        if filename.endswith(file_ext):
            cell_line = filename.split(".")[0]  # Extract the cell line name from the filename

            file_path = database_path + filename

            with open(file_path, "r") as f:
                data_entries = []
                for line in f.readlines():
                    line = line.strip().split("\t")
                    data_entries.append([line[0], int(line[1]), int(line[2]), float(line[3])])

                # Store the data in the dictionary
                database[cell_line] = data_entries

    return database
    # Output format: database["cell_line"][line_no][item_no], where item_no are 0:chromosome_no, 1:start, 2: end, 3:enrichment_score

#########################################################################################################


#########################################################################################################
## Load enhancer atlas, save to la grande table.

def load_annotate_save(chromosome_number):

    file_path = "../la_grande_table/" + chromosome_number + "/"

    # Check whether enhancer atlas npy file is already there
    if os.path.exists(file_path + "atl.npy"):
        print("Enhancer atlas annotation already exists for " + chromosome_number + ".")
        return
    
    if not os.path.exists(file_path + "seq.npy"):
        print("Seq file not found for " + chromosome_number + ", will not load enhacer atlas annotation.")
        return
    
    # Read enhacner atlas.
    enhancer_atlas_db = read_enh_atlas("enhancer_atlas")

    # Keep only enhancers inside chromosome
    ranged_cell_line = []
    cur_cell_line = enhancer_atlas_db["GM12878"]
    for i in range(len(cur_cell_line)):
    # Save only enhancers inside chromosome
        if (cur_cell_line[i][0] == chromosome_number):
            ranged_cell_line.append(cur_cell_line[i])
    
    Annotation = []
    # We don't wanna load it all, we just want to get the length, so mmap_mode r.
    seq = np.load(file_path + "seq.npy", mmap_mode='r', allow_pickle=False, fix_imports=False)
    # Set all annotations to 0
    for i in range(seq.size):
        Annotation.append(0)
    # Set enhancer regions to 1
    for i in range(len(ranged_cell_line)):
        cur_enha = ranged_cell_line[i]
        for j in range(cur_enha[1], cur_enha[2]):
            Annotation[j] = 1
    
    np.save(file_path + "atl", Annotation, allow_pickle=False, fix_imports=False)

#########################################################################################################


#########################################################################################################
## Read and store all enhancers of "GM12878" into each chromosome into la grande table

for i in chromosomes:
    load_annotate_save(i)

#########################################################################################################


#########################################################################################################
## Test for loading enhancer atlas annotation

# testo5 = np.load("../la_grande_table/chr2/seq.npy", allow_pickle=False, fix_imports=False)
# print("Seq chr2 demo: ", testo5[193025:193040])

# time_atl2 = time.time()
# testo6 = np.load("../la_grande_table/chr2/atl.npy", allow_pickle=False, fix_imports=False)
# print("\n--- Loading chr2 atl with " + str(len(testo6)) + " elements, took %s seconds ---\n" % (time.time() - time_atl2))
# print("Enhancer atlas chr2 demo: ", testo6[193025:193040])

#########################################################################################################


#########################################################################################################
# Demo for loading, but also for applying loading function:

columns = ["seq", "cod", "atl"]
def mucho_load(chromosome_number, list_of_features: list):

    file_path = "../la_grande_table/" + chromosome_number + "/"
    
    # Check for size comparing all columns to sequence file length
    check = [np.load(file_path + feature + ".npy", mmap_mode='r', allow_pickle=False, fix_imports=False).size for feature in list_of_features]
    if any(size != check[0] for size in check):
        print(next(feature for feature, size in zip(list_of_features, check) if size != check[0]) + " is not of chromosome length.")

    mucho_time = time.time()
    
    first_feature = np.load(file_path + list_of_features[0] + ".npy", allow_pickle=False, fix_imports=False) # Assuming column 1 is seq
    chromosome_length = first_feature.size
    no_columns = len(list_of_features)

    # Initialize the la_grande_table with the correct shape
    la_grande_table = np.empty((no_columns, chromosome_length))

    # Fill in the first column and then the rest
    la_grande_table[0] = first_feature
    for idx, column in enumerate(list_of_features[1:], start=1):
        la_grande_table[idx] = np.load(file_path + column + ".npy", allow_pickle=False, fix_imports=False)

    print("\n--- Loaded la grande table for " + chromosome_number + " with " + str(la_grande_table.shape[0]) + " columns, and " + str(la_grande_table.shape[1]) + "bp in %s seconds ---\n" % (time.time() - mucho_time))

    return la_grande_table

# Use, before any model, would be:
# np_array = mucho_load("chr1", columns)
# Where columns is a list of features to use: ["seq", "cod", "atl"]. Those are the ones created by this py file.

#########################################################################################################


#########################################################################################################
## mucho_load demo

# la_demo_table = mucho_load("chr1", columns)

# print("Positions 10.000 to 10.010 of all features in la grande table for chromosome 1:")
# j = 0
# for col in columns:
#     print("Column " + col + ":", la_demo_table[j][10000:10010])
#     j += 1

#########################################################################################################
