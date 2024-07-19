import numpy as np

# a = np.load("../../../../../../scratch/ohh98/la_grande_table/chr1/cod.npy")
# print(a[228795000:228795000+10])
a = np.load("../../../../../../scratch/ohh98/la_grande_table/chr1/enhancer_atlas.npy")

b = np.load("../../../../../../scratch/ohh98/la_grande_table/chr1/CTCF.npy")

print(a[:100])
print(b[:100])
# print(match_,mismatch)
# a = np.load("../../../la_grande_table/chr1/DHS.npy")
# print(len(a))