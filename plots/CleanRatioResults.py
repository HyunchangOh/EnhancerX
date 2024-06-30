import pandas as pd
<<<<<<< HEAD
=======
import numpy as np
>>>>>>> 478a73cfd03c9e2ea657b3a8fc8e2a34bfd4adc1

# Define the file path
file_path = 'Ratio_Results-Total.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Extract the feature from the File column and create a new column for it
df['Feature'] = df['File'].apply(lambda x: x.split('/')[-1].replace('_1D_Dist.npy', ''))

<<<<<<< HEAD
# Replace 'enhancer_atlas' with 'atl' in the Feature column
df['Feature'] = df['Feature'].replace('enhancer_atlas', 'atl')

=======
>>>>>>> 478a73cfd03c9e2ea657b3a8fc8e2a34bfd4adc1
# Drop the original 'File' column
df.drop(columns=['File'], inplace=True)

# Re-arrange the columns
df = df[['Chromosome', 'Feature', 'Number of zeros', 'Number of non-zero elements', 'Ratio']]

# Save the cleaned DataFrame to a new CSV file
df.to_csv('Cleaned_Merged_Ratio_Results.csv', index=False)

# Display the first few rows of the cleaned DataFrame to verify
print(df.head())
<<<<<<< HEAD
=======

>>>>>>> 478a73cfd03c9e2ea657b3a8fc8e2a34bfd4adc1
