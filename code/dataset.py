import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset

class OCTDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Get all .npy files
        self.npy_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        # Get file name without extension
        file_name = os.path.splitext(self.npy_files[idx])[0]

        # Load mean A-scan array (.npy)
        npy_path = os.path.join(self.data_dir, file_name + '.npy')
        mean_a_scan = np.load(npy_path, allow_pickle=True)
        mean_a_scan = np.log(mean_a_scan + 1e-6)  # Apply log transform for better range

        # Normalize to [-1,1] and convert to tensor
        mean_a_scan = cv2.resize(mean_a_scan, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        mean_a_scan = (mean_a_scan - mean_a_scan.min()) / (mean_a_scan.max() - mean_a_scan.min())*2 - 1
        mean_a_scan = torch.from_numpy(mean_a_scan).float().unsqueeze(0)  # Add channel dim

        # Load target OCT image (.tiff)
        tiff_path = os.path.join(self.data_dir, file_name + '.tiff')
        target_image = Image.open(tiff_path)  # Convert to grayscale
        target_image = target_image.resize((1024,1024), Image.BILINEAR)
        target_image = np.array(target_image, dtype=np.float32)
        target_image = (target_image - target_image.min()) / (target_image.max() - target_image.min()) * 2 - 1

        # Convert B-scan to tensor and add channel dimension
        target_image = torch.from_numpy(target_image).float().unsqueeze(0)

        return mean_a_scan, target_image