import numpy as np

# Load the .npy file
file_path = 'subsampled_data/chr1/BIN50_CTCF.npy'
data = np.load(file_path)

# Display the type of the data
print(type(data))

# Display the shape of the data (if it's an array)
print(data.shape)

# Display the data itself
print(data)
