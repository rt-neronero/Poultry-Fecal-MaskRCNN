import os
import shutil
import random
import json

# Input and output paths
input_folder = "images"  # Folder where images are stored
annotation_file = "annotations_coco.json"  # Path to original COCO annotations
output_folder = "output"  # Main output folder

# Create train and val folders
train_folder = os.path.join(output_folder, "train")
val_folder = os.path.join(output_folder, "val")
annotations_folder = os.path.join(output_folder, "annotations")

# Ensure output folders exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(annotations_folder, exist_ok=True)

# Classes to be handled
classes = ["cocci", "healthy", "ncd", "salmo"]

# Dictionary to store class-specific images
class_images = {cls: [] for cls in classes}

# Get all image files and group by class
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
        for cls in classes:
            if filename.startswith(cls + "."):
                class_images[cls].append(filename)
                break

# Split and copy images for each class
train_images_set, val_images_set = set(), set()

for cls, images in class_images.items():
    random.shuffle(images)  # Randomize order

    # Split 80:20 for train and val
    split_index = int(len(images) * 0.8)
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Add to sets for annotation filtering
    train_images_set.update(train_images)
    val_images_set.update(val_images)

    # Function to copy files to respective folders
    def copy_files(image_list, dest_folder):
        for image in image_list:
            src_path = os.path.join(input_folder, image)
            dest_path = os.path.join(dest_folder, image)
            shutil.copy2(src_path, dest_path)

    # Copy files for each class
    copy_files(train_images, train_folder)
    copy_files(val_images, val_folder)

    print(f"✅ {cls}: {len(train_images)} train images, {len(val_images)} val images")

# Load original COCO annotations
with open(annotation_file, "r") as f:
    coco_data = json.load(f)

# Get image IDs corresponding to the file names
image_id_map = {img["file_name"]: img["id"] for img in coco_data["images"]}

# Split annotations based on image sets
train_annotations = {"info": coco_data["info"], "images": [], "annotations": [], "licenses": coco_data["licenses"], "categories": coco_data["categories"]}
val_annotations = {"info": coco_data["info"], "images": [], "annotations": [], "licenses": coco_data["licenses"], "categories": coco_data["categories"]}

# Separate images and annotations for train and val
for img in coco_data["images"]:
    if img["file_name"] in train_images_set:
        train_annotations["images"].append(img)
    elif img["file_name"] in val_images_set:
        val_annotations["images"].append(img)

# Split annotations corresponding to the images
train_image_ids = {img["id"] for img in train_annotations["images"]}
val_image_ids = {img["id"] for img in val_annotations["images"]}

for ann in coco_data["annotations"]:
    if ann["image_id"] in train_image_ids:
        train_annotations["annotations"].append(ann)
    elif ann["image_id"] in val_image_ids:
        val_annotations["annotations"].append(ann)

# Save new train and val annotation files
train_annotations_path = os.path.join(annotations_folder, "train_annotations.json")
val_annotations_path = os.path.join(annotations_folder, "val_annotations.json")

with open(train_annotations_path, "w") as f:
    json.dump(train_annotations, f, indent=4)

with open(val_annotations_path, "w") as f:
    json.dump(val_annotations, f, indent=4)

print(f"\n🎉 Split complete!")
print(f"👉 Train images and annotations saved in 'output/train' and 'annotations/train_annotations.json'")
print(f"👉 Val images and annotations saved in 'output/val' and 'annotations/val_annotations.json'")
