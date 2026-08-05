import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset

class WeatherDataset(Dataset):
    def __init__(self, input_dir, target_dir, image_size=128, is_train=True):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]

        inp_path = os.path.join(self.input_dir, img_name)
        tar_path = os.path.join(self.target_dir, img_name)

        inp = cv2.imread(inp_path)
        tar = cv2.imread(tar_path)

        if inp is None or tar is None:
            raise FileNotFoundError(f"Missing file pair for {img_name}")

        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)

        h, w, _ = inp.shape
        crop_size = self.image_size

        if self.is_train and h >= crop_size and w >= crop_size:
            # Random Patch Cropping to preserve high-frequency details
            top = random.randint(0, h - crop_size)
            left = random.randint(0, w - crop_size)
            inp = inp[top:top + crop_size, left:left + crop_size]
            tar = tar[top:top + crop_size, left:left + crop_size]
        else:
            # Standard resize fallback if image dimensions are smaller than crop_size or for validation
            eval_size = 256 if not self.is_train else crop_size
            inp = cv2.resize(inp, (eval_size, eval_size), interpolation=cv2.INTER_LINEAR)
            tar = cv2.resize(tar, (eval_size, eval_size), interpolation=cv2.INTER_LINEAR)

        # Data Augmentations during Training
        if self.is_train:
            # Random Horizontal Flip
            if random.random() > 0.5:
                inp = cv2.flip(inp, 1)
                tar = cv2.flip(tar, 1)
            # Random Vertical Flip
            if random.random() > 0.5:
                inp = cv2.flip(inp, 0)
                tar = cv2.flip(tar, 0)

        # Normalize to [0.0, 1.0]
        inp_tensor = torch.tensor(inp, dtype=torch.float32).permute(2, 0, 1) / 255.0
        tar_tensor = torch.tensor(tar, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return inp_tensor, tar_tensor