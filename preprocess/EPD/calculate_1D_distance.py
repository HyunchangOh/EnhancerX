def closest_true_distances(arr):
    n = len(arr)
    result = [float('inf')] * n

    # Traverse from left to right
    closest_true_index = -1
    for i in range(n):
        if arr[i]:
            closest_true_index = i
        if closest_true_index != -1:
            result[i] = i - closest_true_index

    # Traverse from right to left
    closest_true_index = -1
    for i in range(n - 1, -1, -1):
        if arr[i]:
            closest_true_index = i
        if closest_true_index != -1:
            result[i] = min(result[i], closest_true_index - i)
    
    return result

# Example usage:
# input_list = [False, False, True, True, False, False, True, False]
# output_list = closest_true_distances(input_list)
# print(output_list)