# Proto HMM
# UNDER CONSTRUCTION

# Load .npy files for the genome sequences and 
# its annotation 
import os
import numpy as np

# Globals?
chr_no = "chr1"
demo_start = 0
demo_end = 1000000
demo_len = demo_end-demo_start

# READ BASE PAIR SEQUENCE FROM DATA FOLDER, CHOOSING CHROMOSOME NUMBER, RANGE (ALL BY DEFAULT) AND REF GENOME (HG19 BY DEFAULT)
def read_chromosome(chromosome_number, ref_genome="hg19", range_start=None, range_end=None):
    file_path = f"../data/{ref_genome}/{chromosome_number}.fa"
    with open(file_path, "r") as g:
        g.readline()  # Skip the header line
        lines = g.readlines()
        print("Lines are read")

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

    # Extract the desired range of chromosome data
    chromosome_data = ""
    for i in range(range_start, range_end):
        chromosome_data += full_chromosome_data[i]
        if (i - range_start) % 100000 == 0:
            print(f"{((i - range_start) / (range_end - range_start)) * 100:.2f}% read")

    print("Chromosome data extracted")

    return chromosome_data

#Genome=np.load('genome.npy') # the first Mio bp of E. coli

Genome_letters=read_chromosome(chr_no, range_start=demo_start, range_end=demo_end)
Genome_l=list(Genome_letters)

# Transform letters to numbers
dDNA = {"a": 0 , "c": 1 , "g": 2, "t": 3, "n": -1,
         "A": 0 , "C": 1 , "T": 2, "G": 3, "N": -1}
Genome = np.array([dDNA[bp] for bp in Genome_l])


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

#Annotation=np.load('annotation.npy') # gene annotation 
# We load enhancer atlas, for example, and specifically GM12878
enhancer_atlas_db = load_database("enhancer_atlas")
#cur_cell_line=enhancer_atlas_db["GM12878"]

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

# Add an annotation layer would be:
#Annotation.append([])
# Access new annotation layer:
#Annotation[1]

## Split into halves for train and test
genome_train=Genome[:(demo_len//2)]
genome_test=Genome[(demo_len//2):]
annotation_train=Annotation[:(demo_len//2)]
annotation_test=Annotation[(demo_len//2):]



def learnHMM(obs, allq, N, K):
    """ Learn an HMM given a pair of observation and states 
    np.array[int] * np.array[int] * int * int -> 
            (np.array[double,double], np.array[double,double])
    return transition matrices A and B"""
    A = np.zeros((N, N)) 
    B = np.zeros((N, K))

    # Estimate the transition probabilities
    for i in range(len(allq) - 1):
        A[allq[i], allq[i+1]] += 1
    
    # Normalize the transition matrix
    A = A / A.sum(axis=1, keepdims=True)

    # Estimate the emission probabilities
    for i in range(len(obs)):
        B[allq[i], obs[i]] += 1
    
    # Normalize the emission matrix
    B = B / B.sum(axis=1, keepdims=True)

    return A, B

Pi = np.array([1, 0])
nb_states= 2 ## (non-enhancer, enhancer)
nb_obs = 4 ## (A,C,G,T)

A,B = learnHMM(genome_train, annotation_train, nb_states, nb_obs)
print("Transition matrix A:")
print(A)
print("\nEmission matrix B:")
print(B)


# VITERBI. OUTPUT: "ANNOTATION OF STATES"
# obs was originally nparray, though I think it works with lists, gotta try
def viterbi(obs,Pi,A,B):
    """
    Parameters
    ----------------------------------------
    obs : Sequence of observations [array (T,)]
    Pi: Distribution of initial probabilities [array, (K,)]
    A : Transition matrix [array (K, K)]
    B : Emission matrix [array (K, N)]
    Where K is the number of possible states, and N number of states. 
    (transition matrix K columns, N rows)
    """
    T = len(obs)
    N = len(Pi)

    ## Initialisation
    psi = np.zeros((N, T))
    psi[:,0]= -1
    delta = np.zeros((N, T)) #Initializing delta

    # Initialize the first column of delta
    delta[:, 0] = np.log(Pi + 1e-10) + np.log(B[:, obs[0]] + 1e-10)
    
    # Recursion
    for t in range(1, T):
        for j in range(N):
            # Compute delta for state j at time t
            probabilities = delta[:, t-1] + np.log(A[:, j] + 1e-10)
            psi[j, t] = np.argmax(probabilities)
            delta[j, t] = np.max(probabilities) + np.log(B[j, obs[t]] + 1e-10)
    
    # Termination
    last_state = np.argmax(delta[:, -1])
    
    # Path backtracking
    path = np.zeros(T, dtype=int)
    path[-1] = last_state
    for t in range(T-2, -1, -1):
        path[t] = psi[path[t+1], t+1]

    return path

pred = viterbi(genome_test, Pi, A, B)

print("Most probable path:\n", pred)