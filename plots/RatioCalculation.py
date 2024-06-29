import numpy as np
import pandas as pd
import glob
import os

# Define the base directory - we moved the data to this path
base_directory = '/scratch/ohh98/la_grande_table'

# List of chromosomes to process - edit here in batches of chr as it is expensive computationally 
chromosomes = ['chrX', 'chrY']

# Create an empty list to store the results
results = []

# Loop through each chromosome
for chr in chromosomes:
    # Define the directory and pattern to match files for the current chromosome
    directory = os.path.join(base_directory, chr)
    pattern = '*_1D_Dist.npy'
    full_pattern = os.path.join(directory, pattern)
    
    # Find all files matching the pattern
    file_list = glob.glob(full_pattern)
    
    if not file_list:
        print(f"No files found for chromosome {chr}")

    # Loop through each file and calculate the ratio
    for file_path in file_list:
        try:
            # Load the .npy file
            data = np.load(file_path)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            continue
        
        # Count the number of zeros
        num_zeros = np.count_nonzero(data == 0)
        
        # Count the number of non-zero elements
        num_non_zeros = np.count_nonzero(data != 0)
        
        # Calculate the ratio of zeros to non-zero elements
        ratio = num_zeros / num_non_zeros if num_non_zeros != 0 else float('inf')  # Handle division by zero
        
        # Append the results to the list
        results.append({
            'Chromosome': chr,
            'File': file_path,
            'Number of zeros': num_zeros,
            'Number of non-zero elements': num_non_zeros,
            'Ratio': ratio
        })

# Create a DataFrame from the results
if results:
    df = pd.DataFrame(results)
    
    # Save the DataFrame to a CSV file - edit the name as the batch name you put in
    output_file = '/home/ohh98/enhancerX/plots/Ratio_Results-X-and-Y.csv'
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
else:
    print("No results to save.")
