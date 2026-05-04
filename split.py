import os
import shutil
import random

input_dir = "dataset/full/input"
target_dir = "dataset/full/target"

files = os.listdir(input_dir)
files.sort()
random.shuffle(files)

# splits
train_split = int(0.8 * len(files))
val_split = int(0.9 * len(files))

splits = {
    "train": files[:train_split],
    "val": files[train_split:val_split],
    "test": files[val_split:]
}

# create folders
for split in splits:
    os.makedirs(f"dataset/{split}/input", exist_ok=True)
    os.makedirs(f"dataset/{split}/target", exist_ok=True)

# copy files
for split in splits:
    for file in splits[split]:
        shutil.copy(os.path.join(input_dir, file), f"dataset/{split}/input/{file}")
        shutil.copy(os.path.join(target_dir, file), f"dataset/{split}/target/{file}")

print("Done splitting dataset!")
print("Train:", len(os.listdir("dataset/train/input")), len(os.listdir("dataset/train/target")))
print("Val:", len(os.listdir("dataset/val/input")), len(os.listdir("dataset/val/target")))
print("Test:", len(os.listdir("dataset/test/input")), len(os.listdir("dataset/test/target")))