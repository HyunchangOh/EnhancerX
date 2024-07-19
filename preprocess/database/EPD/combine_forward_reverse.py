import numpy as np
import os
names = ["chr" + str(i) for i in range(1, 23)]
names += ["chrX","chrY"]

root_folder = "../../la_grande_table/"

for name in names:
    d = root_folder+name+"/"
    f = np.load(d+"promoter_forward.npy")
    r = np.load(d+"promoter_reverse.npy")
    np.save(d+"promoter_any.npy",np.logical_or(f,r))