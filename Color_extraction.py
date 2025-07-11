# Day 1

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image
image_path = '/Users/vaibhav/Desktop/0_9H74uSINU-qGHWz5.jpg' # Replace with your image file
image = cv2.imread(image_path)

# Step 2: Convert image from BGR (OpenCV default) to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 3: Display the image
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()

# Step 4: Reshape image into a 2D array of RGB pixels
pixels = image_rgb.reshape((-1, 3))
print(f"Pixel data shape: {pixels.shape}")

# Day 2

from sklearn.cluster import KMeans

# Step 5: Apply KMeans clustering
num_colors = 5  # You can change this to however many dominant colors you want
kmeans = KMeans(n_clusters=num_colors, random_state=42)
kmeans.fit(pixels)

# Step 6: Get the cluster centers (i.e., dominant colors)
colors = kmeans.cluster_centers_.astype(int)

print("Dominant color clusters (RGB):")
for idx, color in enumerate(colors):
    print(f"Color {idx + 1}: {color}")

# Day 3

    from collections import Counter

    # Step 7: Get labels for each pixel (which cluster it belongs to)
    labels = kmeans.labels_

    # Step 8: Count how many pixels are in each cluster
    counts = Counter(labels)

    # Step 9: Sort clusters by pixel count (most to least dominant)
    sorted_counts = counts.most_common()

    print("\nDominant colors ranked by pixel count:")
    for idx, (cluster_idx, count) in enumerate(sorted_counts):
        color = colors[cluster_idx]
        print(f"{idx + 1}. RGB: {color}, Pixels: {count}")

# Day 4

# Step 10: Create a color palette bar using matplotlib

def plot_color_palette(sorted_counts, colors):
    palette = []
    total_pixels = sum([count for _, count in sorted_counts])

    for cluster_idx, count in sorted_counts:
        color = colors[cluster_idx]
        percentage = count / total_pixels
        palette.append((color, percentage))

    # Create a figure
    plt.figure(figsize=(8, 2))
    start = 0
    for color, percentage in palette:
        end = start + percentage
        plt.fill_between([start, end], 0, 1, color=color.astype(int) / 255)
        start = end
    plt.axis('off')
    plt.title("Dominant Color Palette")
    plt.show()


plot_color_palette(sorted_counts, colors)
