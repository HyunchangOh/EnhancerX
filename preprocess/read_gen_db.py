# Basically, input for a generic database in the data folder.

import os

# INPUT: Name of the database (path is always in data folder)
#        Type of file extension (.txt by double default)
# OUTPUT: Dictionary of list of lists. key is cell line (name of file), lists of lines of text

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

# Predefined database names
databases = {
    'A': "enhancer_atlas",
    'B': "VISTA",
    'C': "yet_another_database"
}

# Ask the user to choose a database
print("Please choose a database:")
print("A: Enhancer Atlas")
print("B: VISTA")
print("C: Another Database")

choice = input("Enter your choice (A, B, or C): ").strip().upper()
file_extension = input("Please enter file extension (.txt, .bed, etc.): ").strip()
if not file_extension:
    file_extension = ".txt"  # Default to .txt if no input is provided


if choice in databases:
    database_name = databases[choice]
    # Load database using choice
    database = load_database(database_name, file_ext=file_extension)
    print(f"Database {choice} ({database_name}) loaded successfully.")
    if choice == 'A':
        print("Example, GM12878, line 69:\n")
        print(database["GM12878"][68])
else:
    print("Invalid choice. Please re-run script and choose a valid option.")


###############################################################################
#Continue code here, using database[cell_line][text_line] 
#Enhancer atlas files don't have a header.

# Now would read hg19 maybe chromosome by chromosome

def read_chromosome(chromosome_number):
    file_path = f"../data/hg19/{chromosome_number}.fa"
    with open(file_path, "r") as g:
        g.readline()  # Skip the header line
        lines = g.readlines()
        print("Lines are read")

    length = len(lines)
    chromosome_data = ""
    for i in range(length):
        chromosome_data += lines[i].strip()
        if i % 100000 == 0:
            print(f"{(i / length) * 100:.2f}% read")

    return chromosome_data

chromosome_number = input("Please enter chromosome number as chrX, where X is the number (chr1 by default):").strip()
if not chromosome_number:
    chromosome_number = "chr1"  # Default to .txt if no input is provided

chromosome_data = read_chromosome(chromosome_number)
print("Chromosome data length:", len(chromosome_data))


