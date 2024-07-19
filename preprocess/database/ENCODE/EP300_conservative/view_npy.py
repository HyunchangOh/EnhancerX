import numpy as np

a = np.load("../../../la_grande_table/chr1/DHS.npy")
print(a[235623:235633+100])
# a = np.load("../../../la_grande_table/chr1/DHS.npy")
print(len(a))