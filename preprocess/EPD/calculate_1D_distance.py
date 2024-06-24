# arr = a boolean array
# direction = will the distance only be calculated from one direction? or both direction?
# check with the example code below if you are unsure.

def closest_true_distances(arr,direction="both"):
    n = len(arr)
    result = [float('inf')] * n

    # Traverse from left to right
    if direction == "both" or direction == "forward":
        closest_true_index = -1
        for i in range(n):
            if arr[i]:
                closest_true_index = i
            if closest_true_index != -1:
                result[i] = i - closest_true_index

    # Traverse from right to left
    if direction == "both" or direction =="reverse":
        closest_true_index = -1
        for i in range(n - 1, -1, -1):
            if arr[i]:
                closest_true_index = i
            if closest_true_index != -1:
                result[i] = min(result[i], closest_true_index - i)
    
    return result

# Example usage:
input_list = [False, False, True, True, False, False, True, False]
output_list = closest_true_distances(input_list,"forward")
print(output_list)