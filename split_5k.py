import os
import shutil
import random

def categorize_file(filename):
    fname = filename.lower()
    if "rain" in fname:
        return "rain"
    elif "snow" in fname:
        return "snow"
    elif "winter" in fname:
        return "winter"
    else:
        return "outdoor_other"

def main():
    input_dir = "dataset/full/input"
    target_dir = "dataset/full/target"

    if not os.path.exists(input_dir) or not os.path.exists(target_dir):
        raise FileNotFoundError("dataset/full/input or dataset/full/target missing.")

    all_files = sorted(os.listdir(input_dir))
    print(f"Total available full files: {len(all_files)}")

    categories = {
        "rain": [],
        "snow": [],
        "winter": [],
        "outdoor_other": []
    }

    for f in all_files:
        cat = categorize_file(f)
        categories[cat].append(f)

    for cat, files in categories.items():
        print(f"Category '{cat}': {len(files)} files")

    random.seed(42)
    selected_files = []
    target_per_cat = 1250

    for cat, files in categories.items():
        random.shuffle(files)
        taken = files[:target_per_cat]
        selected_files.extend(taken)
        print(f"Selected {len(taken)} files from '{cat}'")

    if len(selected_files) < 5000:
        remaining = [f for f in all_files if f not in set(selected_files)]
        random.shuffle(remaining)
        needed = 5000 - len(selected_files)
        selected_files.extend(remaining[:needed])

    random.shuffle(selected_files)
    selected_files = selected_files[:5000]
    print(f"\nTotal selected dataset size: {len(selected_files)}")

    train_count = 4000
    val_count = 500

    train_files = selected_files[:train_count]
    val_files = selected_files[train_count:train_count + val_count]
    test_files = selected_files[train_count + val_count:]

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    for split in splits:
        in_path = f"dataset/{split}/input"
        tar_path = f"dataset/{split}/target"
        
        if os.path.exists(in_path):
            shutil.rmtree(in_path)
        if os.path.exists(tar_path):
            shutil.rmtree(tar_path)
            
        os.makedirs(in_path, exist_ok=True)
        os.makedirs(tar_path, exist_ok=True)

        for f in splits[split]:
            shutil.copy(os.path.join(input_dir, f), os.path.join(in_path, f))
            shutil.copy(os.path.join(target_dir, f), os.path.join(tar_path, f))

        print(f"Split '{split}': {len(os.listdir(in_path))} input files, {len(os.listdir(tar_path))} target files")

    print("\nDataset balancing and splitting completed successfully!")

if __name__ == "__main__":
    main()
