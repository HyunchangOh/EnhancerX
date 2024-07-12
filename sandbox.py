import numpy as np

input_dir = '/scratch/ohh98/LGT_subsampled_reboot/chr1/BIN50_CTCF.npy'

legacy_dir =  '/scratch/ohh98/la_grande_table_subsampled/subsampled_data/chr1/BIN50_CTCF.npy'

a = np.load(input_dir)
b = np.load(legacy_dir)
print(np.unique(b))