import time

dDNA = {"a": 0, "c": 1, "g": 2, "t": 3, "n": -1,
        "A": 0, "C": 1, "T": 2, "G": 3, "N": -1}

def load_reference_genome(chromosome_number, ref_genome="hg19"):
    file_path = f"../data/{ref_genome}/{chromosome_number}.fa"
    with open(file_path, "r") as g:
        g.readline()  # Skip the header line
        lines = g.readlines()
    full_chromosome_data = ''.join(line.strip() for line in lines)
    return full_chromosome_data

def is_enhancer(position, enhancer_regions):
    for start, end in enhancer_regions:
        if start <= position < end:
            return 1
    return 0

def create_table(chromosome_data, enhancer_regions, chunk_start, chunk_end):
    # Create the output file for the current chunk
    output_file_name = f"la_grande_table_{chunk_start + 1}_{chunk_end}.tsv"
    with open(output_file_name, "w") as h:
        h.write("Position\tBase_Pair\tAnnotation\t1D_Distance\t3D_Distance\n")

        for position in range(chunk_start, chunk_end):
            base_pair = chromosome_data[position]
            base_pair_translated = dDNA.get(base_pair, -1)
            annotation = is_enhancer(position, enhancer_regions)
            row = f"{position + 1}\t{base_pair_translated}\t{annotation}\t\t\n"
            h.write(row)

            # Print percentage progress every 5%
            if (position - chunk_start) % ((chunk_end - chunk_start) // 20) == 0:
                print(f"{((position - chunk_start) / (chunk_end - chunk_start)) * 100:.0f}% completed for chunk {chunk_start + 1} to {chunk_end}")

def main(ref_genome="hg19", chromosome_number="chr1", start_chunk=0, end_chunk=5):
    start_time = time.time()

    # Load the reference genome data
    chromosome_data = load_reference_genome(chromosome_number)

    # Load the enhancer regions
    enhancer_regions = []
    with open("../data/enhancer_atlas/GM12878.txt", "r") as f:
        for line in f:
            chrom, start, end, _ = line.strip().split("\t")
            if chrom == chromosome_number:
                enhancer_regions.append((int(start), int(end)))

    # Define the chunk size
    chunk_size = 20000000
    total_positions = len(chromosome_data)

    # Loop over the specified range of chunks
    for chunk_index in range(start_chunk, end_chunk):
        chunk_start = chunk_index * chunk_size
        chunk_end = min(chunk_start + chunk_size, total_positions)
        
        create_table(chromosome_data, enhancer_regions, chunk_start, chunk_end)

    end_time = time.time()
    print(f"Tables have been created successfully in {end_time - start_time:.2f} seconds.")

# Example usage:
# Create tables for the first two chunks
#main(start_chunk=0, end_chunk=1)import time

dDNA = {"a": 0, "c": 1, "g": 2, "t": 3, "n": -1,
        "A": 0, "C": 1, "T": 2, "G": 3, "N": -1}

def load_reference_genome(chromosome_number, ref_genome="hg19"):
    file_path = f"../data/{ref_genome}/{chromosome_number}.fa"
    with open(file_path, "r") as g:
        g.readline()  # Skip the header line
        lines = g.readlines()
    full_chromosome_data = ''.join(line.strip() for line in lines)
    return full_chromosome_data

def is_enhancer(position, enhancer_regions):
    for start, end in enhancer_regions:
        if start <= position < end:
            return 1
    return 0

def create_table(chromosome_data, enhancer_regions, chunk_start, chunk_end):
    # Create the output file for the current chunk
    output_file_name = f"la_grande_table_{chunk_start + 1}_{chunk_end}.tsv"
    with open(output_file_name, "w") as h:
        h.write("Position\tBase_Pair\tAnnotation\t1D_Distance\t3D_Distance\n")

        for position in range(chunk_start, chunk_end):
            base_pair = chromosome_data[position]
            base_pair_translated = dDNA.get(base_pair, -1)
            annotation = is_enhancer(position, enhancer_regions)
            row = f"{position + 1}\t{base_pair_translated}\t{annotation}\t\t\n"
            h.write(row)

            # Print percentage progress every 5%
            if (position - chunk_start) % ((chunk_end - chunk_start) // 20) == 0:
                print(f"{((position - chunk_start) / (chunk_end - chunk_start)) * 100:.0f}% completed for chunk {chunk_start + 1} to {chunk_end}")

def main(ref_genome="hg19", chromosome_number="chr1", start_chunk=0, end_chunk=5):
    start_time = time.time()

    # Load the reference genome data
    chromosome_data = load_reference_genome(chromosome_number, ref_genome)

    # Load the enhancer regions
    enhancer_regions = []
    with open("../data/enhancer_atlas/GM12878.txt", "r") as f:
        for line in f:
            chrom, start, end, _ = line.strip().split("\t")
            if chrom == chromosome_number:
                enhancer_regions.append((int(start), int(end)))

    # Define the chunk size
    chunk_size = 20000000
    total_positions = len(chromosome_data)

    # Loop over the specified range of chunks
    for chunk_index in range(start_chunk, end_chunk):
        chunk_start = chunk_index * chunk_size
        chunk_end = min(chunk_start + chunk_size, total_positions)
        
        create_table(chromosome_data, enhancer_regions, chunk_start, chunk_end)

    end_time = time.time()
    print(f"Tables have been created successfully in {end_time - start_time:.2f} seconds.")

# Example usage:
# Create tables for the first two chunks
#main(start_chunk=0, end_chunk=2)
main(start_chunk=0, end_chunk=1) # just one chunk today

# Continue with the next chunks
# main(start_chunk=2, end_chunk=4)


# Continue with the next chunks
# main(start_chunk=2, end_chunk=4)
