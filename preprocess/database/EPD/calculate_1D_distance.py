# arr = a boolean array
# direction = will the distance only be calculated from one direction? or both direction?
# check with the example code below if you are unsure.
import numpy as np


def closest_true_distances(arr,direction="both"):
    n = len(arr)
    result = [float('inf')] * n

    # Traverse from left to right
    if direction == "both" or direction == "reverse":
        closest_true_index = -1
        for i in range(n):
            if arr[i]:
                closest_true_index = i
            if closest_true_index != -1:
                result[i] = i - closest_true_index

    # Traverse from right to left
    if direction == "both" or direction =="forward":
        closest_true_index = -1
        for i in range(n - 1, -1, -1):
            if arr[i]:
                closest_true_index = i
            if closest_true_index != -1:
                result[i] = min(result[i], closest_true_index - i)
    
    return result

# Example usage:
# input_list = [False, False, True, True, False, False, True, False]
# output_list = closest_true_distances(input_list,"forward")
# print(output_list)

names = ["chr" + str(i) for i in [2,3,7,9]]
# names += ["chrX","chrY"]

root_folder = "/scratch/ohh98/la_grande_table/"

#7 and 9 any not done
for name in names:
    print(name)
    d = root_folder+name+"/"
    # a = np.load(d+"promoter_any.npy")
    # f = np.load(d+"promoter_forward.npy")
    r = np.load(d+"promoter_reverse.npy")

    # f_d =closest_true_distances(f,"forward")
    # np.save(d+"promoter_1D_Dist_forward.npy",f_d)
    # print("forward done")

    r_d = closest_true_distances(r,"reverse")
    np.save(d+"promoter_1D_Dist_reverse.npy",r_d)
    print("reverse done")

    # a_d = closest_true_distances(a,"both")
    # np.save(d+"promoter_1D_Dist_any.npy",a_d)
    # print("both direction done")