import numpy as np

import numpy as np

import os

p = "../../../../../scratch/ohh98/la_grande_table/chr22/"
files_and_directories = os.listdir("../../../../../scratch/ohh98/la_grande_table/chr22/")
        
# Filter out directories, keeping only the files
files = [f for f in files_and_directories if os.path.isfile(os.path.join("../../../../../scratch/ohh98/la_grande_table/chr22/", f))]

for f in files:
    if "BIN" in f:
        a = np.load(p+f)
        print(len(a))