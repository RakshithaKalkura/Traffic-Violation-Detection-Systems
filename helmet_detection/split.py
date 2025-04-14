import os
import shutil
import random

# Paths
base_dir = r"C:\Users\raksh\Downloads\helmet_data"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

# Output folders
splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

# Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Collect image files
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(image_files)

# Split
total = len(image_files)
train_end = int(train_ratio * total)
val_end = train_end + int(val_ratio * total)

train_files = image_files[:train_end]
val_files = image_files[train_end:val_end]
test_files = image_files[val_end:]

# Function to copy image and its label
def move_files(file_list, split):
    for file in file_list:
        img_src = os.path.join(images_dir, file)
        img_dst = os.path.join(images_dir, split, file)

        label_file = os.path.splitext(file)[0] + ".txt"
        label_src = os.path.join(labels_dir, label_file)
        label_dst = os.path.join(labels_dir, split, label_file)

        if os.path.exists(img_src) and os.path.exists(label_src):
            shutil.move(img_src, img_dst)
            shutil.move(label_src, label_dst)

# Move files
move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

print(f"Dataset split into: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test.")
