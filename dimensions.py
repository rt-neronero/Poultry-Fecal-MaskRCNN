import cv2
import os

# Path to your image dataset
dataset_dir = "annotated"

# Initialize min and max dimensions
min_width, min_height = float('inf'), float('inf')
max_width, max_height = 0, 0

# Loop through all files in the directory
for filename in os.listdir(dataset_dir):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(dataset_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            height, width, _ = img.shape
            
            # Update min and max dimensions
            min_width = min(min_width, width)
            min_height = min(min_height, height)
            max_width = max(max_width, width)
            max_height = max(max_height, height)

# Print results
print(f"Minimum dimensions: {min_width}x{min_height}")
print(f"Maximum dimensions: {max_width}x{max_height}")