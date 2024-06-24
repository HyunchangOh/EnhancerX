# GENERATE LA GRANDE TABLE

# Imports
import os
import numpy as np
import time

# Globals?
# Ranges to analyze
chr_no = "chr1"
demo_start = 0
demo_end = 20000000 # 20M
#demo_end = 500000 # 500k
demo_len = demo_end-demo_start


# READ BASE PAIR SEQUENCE FROM DATA FOLDER, CHOOSING CHROMOSOME NUMBER, RANGE (ALL BY DEFAULT) AND REF GENOME (HG19 BY DEFAULT)
def read_chromosome(chromosome_number, ref_genome="hg19", range_start=None, range_end=None):
    file_path = "../data/" + ref_genome + "/" + chromosome_number + ".fa"
    if not os.path.exists(file_path):
        print("Fasta file not found for " + chromosome_number + ".")
        return
    with open(file_path, "r") as g:
        g.readline()  # Skip the header line
        lines = g.readlines()
        print(chromosome_number + "'s lines are read.")

    #length = len(lines)
    if range_start==None:
        range_start=0
    if range_end==None:
        range_start=len(lines)-1
    
    # Combine all lines into a single string and strip newlines
    full_chromosome_data = ''.join(line.strip() for line in lines)

    # Determine the length of the chromosome data
    chromosome_length = len(full_chromosome_data)
    
    # Set default values for range_start and range_end
    if range_start is None:
        range_start = 0
    if range_end is None:
        range_end = chromosome_length

    # Validate and adjust range_end if it exceeds the chromosome length
    if range_end > chromosome_length:
        range_end = chromosome_length

    step = (range_end - range_start) // 20 # Print progress every 5%

    # Extract the desired range of chromosome data
    chromosome_data = ""
    for i in range(range_start, range_end):
        chromosome_data += full_chromosome_data[i]
        
        # Print out progress
        if (i - range_start) % step == 0:
            print(round(((i - range_start) / (range_end - range_start)) * 100), "% read")

    print("Chromosome data extracted")

    return chromosome_data


##################################################################
# LOAD GENOME DATA from data folder into variable Genome, 
# which is a list of numbers from the dictionary dDNA


def read_translate_save(chromosome_number, ref_genome="hg19", range_start=None, range_end=None):

    file_path = "../la_grande_table/" + chromosome_number + "/"

    # Check whether the fasta file is there, and whether there is already an npy file
    if os.path.exists(file_path + "seq.npy") and os.path.exists(file_path + "cod.npy"):
        print("Sequence and coding files already exist.")
        return
    elif not os.path.exists("../data/" + ref_genome + "/" + chromosome_number +  ".fa"):
        print("Chromosome data could not be read.")
        return
    else:
        if not os.path.exists(file_path + "seq.npy"):
            file_path_single = file_path + "seq"
            seq = True
            print("Ahead with loading (coding file already exists).")
        elif not os.path.exists(file_path + "cod.npy"):
            file_path_single = file_path + "cod"
            seq = False
            print("Ahead with loading (sequence file already exists).")
        else: 
            print("Ahead with loading.")

    start_time = time.time()
    Genome_l = list(read_chromosome(chromosome_number, ref_genome=ref_genome, range_start=range_start, range_end=range_end))
    # Genome_l = list(Genome_l)
    print("\n--- Reading " + chromosome_number + ", with " + str(len(Genome_l)) + "bp, took %s seconds ---\n" % (time.time() - start_time))

    # Transform letters to numbers
    dDNA = {"a": 0 , "c": 1 , "g": 2, "t": 3, "n": -1,
            "A": 0 , "C": 1 , "T": 2, "G": 3, "N": -1}
    
    # Create genome sequence
    genome = np.array([dDNA[bp] for bp in Genome_l])

    # Create coding annotation
    coding = np.array([1 if bp.isupper() else 0 for bp in Genome_l])

    # Save sequence and/or coding/non-coding annotation to la grande table, depending on whether one of them already exists
    if file_path_single:
        if seq==True:
            np.save(file_path_single, genome, allow_pickle=False, fix_imports=False)
        else:
            np.save(file_path_single, coding, allow_pickle=False, fix_imports=False)
    else: 
        np.save(file_path + "seq", genome, allow_pickle=False, fix_imports=False)
        np.save(file_path + "cod", coding, allow_pickle=False, fix_imports=False)




##################################################################

for i in ["chr1", "chr2", "chr3"]:
    read_translate_save(i)

testo=np.load("../la_grande_table/chr2/seq.npy", allow_pickle=False, fix_imports=False)
print("Testo: ", testo[10000:10020])

# Demo for loading, but also for applying loading function:
# def mucho_load(chromosome_number, ):
#     la_grande_table = []

#     return la_grande_table

'''
##################################################################
# READ ANNOTATION DATA
def load_database(database_name, file_ext=".txt"):
    database_path = "../data/" + database_name
    database = {}

    # Iterate over all files in the directory
    for filename in os.listdir(database_path):
        if filename.endswith(file_ext):
            cell_line = filename.split(".")[0]  # Extract the cell line name from the filename
            file_path = os.path.join(database_path, filename)

            with open(file_path, "r") as f:
                data_entries = []
                for line in f.readlines():
                    line = line.strip().split("\t")
                    data_entries.append([line[0], int(line[1]), int(line[2]), float(line[3])])

                # Store the data in the dictionary
                database[cell_line] = data_entries

    return database
# OUTPUT: database["cell_line"][line_no][item_no], where item_no are 0:chromosome_no, 1:start, 2: end, 3:enrichment_score

# We load enhancer atlas, for example, and specifically GM12878
enhancer_atlas_db = load_database("enhancer_atlas")

# Keep only enhancers inside chromosome and range
ranged_cell_line = []
cur_cell_line=enhancer_atlas_db["GM12878"]
for i in range(len(cur_cell_line)):
    # Check chromosome number, and range of enhancer
    if (cur_cell_line[i][0] == chr_no) and (demo_start <= cur_cell_line[i][1] <= demo_end and demo_start <= cur_cell_line[i][2] <= demo_end):
        ranged_cell_line.append(cur_cell_line[i])


# Multidimensional annotation
Annotation=[[]]

# FOR NOW WE WILL DO 0-1
# Set all annotations to 0
for i in range(len(Genome)):
    Annotation[0].append(0)
# Set enhancer regions to 1
for i in range(len(ranged_cell_line)):
    cur_enha=ranged_cell_line[i]
    for j in range(cur_enha[1], cur_enha[2]):
        Annotation[0][j] = 1

# Adding an annotation layer would be:
#Annotation.append([])
# Accessing new annotation layer:
#Annotation[1]

cur_annotation = Annotation[0]
##################################################################

# Here we would save to npy


##################################################################
## Split into halves for train and test

genome_train = Genome[:(demo_len//2)]
genome_test = Genome[(demo_len//2):]
annotation_train = cur_annotation[:(demo_len//2)]
annotation_test = cur_annotation[(demo_len//2):]

##################################################################

##################################################################
## Just some interesting data:
print("Random, very deletable data:\n length genome:", len(Genome),
       "\n length 1st annotation:", len(Annotation[0]), # Just to check they're the same length
        "\n nº enhancers in range:", len(ranged_cell_line), 
        "\n nº total enhancers this cell line:", len(cur_cell_line))
##################################################################
'''