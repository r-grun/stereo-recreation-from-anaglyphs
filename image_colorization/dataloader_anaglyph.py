from torchvision.transforms import InterpolationMode

import config as c
import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class AnaglyphDataset(Dataset):
    def __init__(self, path_anaglyph, path_left, path_right):
        self.transforms = transforms.Compose([
            transforms.Resize((c.IMAGE_SIZE, c.IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),  # Resize images to a fixed size
            transforms.ToTensor(),         # Convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

        self.size = c.IMAGE_SIZE
        self.path_anaglyph = path_anaglyph
        self.path_left = path_left
        self.path_right = path_right

    def __getitem__(self, idx):
        img_anaglyph = Image.open(self.path_anaglyph[idx], mode='r').convert("RGB")
        img_anaglyph = self.transforms(img_anaglyph)

        img_left = Image.open(self.path_left[idx]).convert("RGB")
        img_left = self.transforms(img_left)

        img_right = Image.open(self.path_right[idx]).convert("RGB")
        img_right = self.transforms(img_right)


        return {'a': img_anaglyph, 'l': img_left, 'r': img_right}

    def __len__(self):
        return len(self.path_anaglyph)


def make_dataloaders(batch_size=16, n_workers=4, pin_memory=True, **kwargs):  # A handy function to make our dataloaders
    """Create dataloaders for the given dataset. **kwargs should be path_anaglyph, path_left, path_right"""
    dataset = AnaglyphDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader
