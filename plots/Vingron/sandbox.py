import numpy as np
distances = [10,-10,1000,12,13,15,13,11,140,21]
frame_size=20
filtered_distances = [d for d in distances if 0 <= d <= frame_size]
print(np.max(filtered_distances))
print(np.histogram(filtered_distances))
print(filtered_distances)