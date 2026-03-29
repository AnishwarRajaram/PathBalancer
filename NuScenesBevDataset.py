import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import torchvision.transforms.functional as TF
import random

class NuScenesBevDataset(Dataset):
    def __init__(self, input_dir, gt_dir, split = "train"):
        self.input_path = Path(input_dir)
        self.gt_path = Path(gt_dir)
        # Find all input files and sort them so they match the GT files
        self.input_files = sorted(list(self.input_path.glob("*_x.npy")))
        self.split = split

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        x_file = self.input_files[idx]
        token = x_file.name.replace("_x.npy", "")
        y_file = self.gt_path / f"{token}_y.npy"

        # Load the numpy data
        x_np = np.load(x_file) # (200, 200, 3)
        y_np = np.load(y_file) # (200, 200)

        # IMPORTANT: PyTorch wants (Channels, Height, Width)
        # We move the 3 channels from the end to the front
        x_tensor = torch.from_numpy(x_np).permute(2, 0, 1).float()
        
        # Ground truth needs to be a Long tensor for the Loss function
        y_tensor = torch.from_numpy(y_np).long()

        # 3. Data Augmentation (Only for training)
        if self.split == 'train':
            # 50% chance to flip horizontally
            if random.random() > 0.5:
                x_tensor = TF.hflip(x_tensor)
                y_tensor = TF.hflip(y_tensor)
                
            # Optional: Add 50% chance to flip vertically 
            # (Careful: only if your "monster truck" can drive backwards!)
            if random.random() > 0.5:
                x_tensor = TF.vflip(x_tensor)
                y_tensor = TF.vflip(y_tensor)

        return x_tensor, y_tensor