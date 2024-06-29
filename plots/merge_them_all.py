# Merge the batches all in cleaner way

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

# Extract the feature from the File column and create a new column for it
merged_df['Feature'] = merged_df['File'].apply(lambda x: x.split('/')[-1].replace('_1D_Dist.npy', ''))

# Drop the original 'File' column
merged_df = merged_df.drop(columns=['File'])

# Re-arrange the columns
merged_df = merged_df[['Chromosome', 'Feature', 'Number of zeros', 'Number of non-zero elements', 'Ratio']]

# Save the cleaned DataFrame to a new CSV file
merged_df.to_csv('Cleaned_Merged_Ratio_Results.csv', index=False)

# Display the first few rows of the cleaned DataFrame to verify
print(merged_df.head())
