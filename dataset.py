import os
import cv2
import torch
from torch.utils.data import Dataset

class WeatherDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.files = sorted(os.listdir(input_dir))
        self.input_dir = input_dir
        self.target_dir = target_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]

        inp = cv2.imread(os.path.join(self.input_dir, img_name))
        tar = cv2.imread(os.path.join(self.target_dir, img_name))

        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)

        # IMPORTANT LINE (resize)
        inp = cv2.resize(inp, (256, 256))
        tar = cv2.resize(tar, (256, 256))

        # normalize
        inp = torch.tensor(inp).permute(2,0,1).float() / 255.0
        tar = torch.tensor(tar).permute(2,0,1).float() / 255.0

        return inp, tar