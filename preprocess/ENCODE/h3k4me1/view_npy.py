import numpy as np

a = np.load("../../../la_grande_table/chr1/h3k27ac_3D_Dist.npy")
print(a[228795000:228795000+10])
a = np.load("../../../la_grande_table/chr1/h3k27ac_1D_Dist.npy")
print(a[228795000:228795000+10])