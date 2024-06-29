#This code is to merge all the batches we created and create just one main table for plotting $ analysis

import pandas as pd

# Define the file paths
file_paths = [
    'Ratio_Results-X-and-Y.csv',
    'Ratio_Results-1-to-3.csv',
    'Ratio_Results-4-to-7.csv',
    'Ratio_Results-8-to-22.csv'
]

# Initialize an empty list to hold the DataFrames
dfs = []

# Loop through the file paths and read each CSV file into a DataFrame
for file_path in file_paths:
    df = pd.read_csv(file_path)
    dfs.append(df)

# Concatenate all the DataFrames into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('Ratio_Results-Total.csv', index=False)
