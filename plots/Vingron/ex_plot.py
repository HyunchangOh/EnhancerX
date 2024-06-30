import numpy as np
import matplotlib.pyplot as plt

# Sample data
distances = np.random.randint(-200, 200, size=1000)  # Example array of integers
frame_size = 150

# Filter distances to be within the range [-frame_size, frame_size]
filtered_distances = [d for d in distances if -frame_size <= d <= frame_size]

# Compute the histogram
hist, bin_edges = np.histogram(filtered_distances, bins=np.arange(-frame_size, frame_size + 1))

# Normalize the histogram
hist_normalized = hist / np.max(hist)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(bin_edges[:-1], hist_normalized, marker='', linestyle='-', color='skyblue', markeredgecolor='black')
plt.xlim(-frame_size, frame_size)
plt.ylim(0, 1)
plt.xlabel('Distance')
plt.ylabel('Normalized Frequency')
plt.title('Distribution of Distances')

# Save the plot as a PNG file
plt.savefig('distance_distribution.png')

# Show the plot
plt.show()