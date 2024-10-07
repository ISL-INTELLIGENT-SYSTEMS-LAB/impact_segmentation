import os
import shutil
import random

random.seed(20)

# Set paths
rootdir = os.path.join("ai4mars-dataset")
    
train_images_dir = os.path.join(rootdir, "train","images")
train_labels_dir = os.path.join(rootdir, "train","labels")

val_images_dir = os.path.join(rootdir, "val","images")
val_labels_dir = os.path.join(rootdir, "val","labels")

os.makedirs(val_images_dir, exist_ok = True)
os.makedirs(val_labels_dir, exist_ok = True)

# Get list of images and labels
image_files = [f for f in os.listdir(train_images_dir) if os.path.isfile(os.path.join(train_images_dir, f))]
label_files = [f for f in os.listdir(train_labels_dir) if os.path.isfile(os.path.join(train_labels_dir, f))]


# make sure they match
data_pairs = []
for label in os.listdir(train_labels_dir):
    basename = label.split(".")[0]
    imagename = f"{basename}.JPG"
    #print(basename)
    if os.path.exists(os.path.join(train_images_dir, imagename)):
        data_pairs.append((imagename, label))


# Calculate the number of samples for validation
num_val_samples = int(len(data_pairs) * 0.20)


# Randomly select 20% of the data for validation
validation_pairs = random.sample(data_pairs, num_val_samples)


# Move files to the validation 
for image_file, label_file in validation_pairs:
   
   shutil.move(os.path.join(train_images_dir, image_file), os.path.join(val_images_dir, image_file))

   shutil.move(os.path.join(train_labels_dir, label_file), os.path.join(val_labels_dir, label_file))


print(f"Moved {num_val_samples} image-label pairs to the validation set.")