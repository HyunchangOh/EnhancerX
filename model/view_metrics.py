import numpy as np
import pickle

path = "/scratch/ohh98/LGT_subsampled_reboot/logistic_regression/all/metrics.pkl"

a = pickle.load(open(path))
print(a)