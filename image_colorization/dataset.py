import config as c
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class ColorizationDataset(Dataset):
    def __init__(self, path_anaglyph, path_reversed, split='train'):
        if split == 'train':
            # Do data augmentation here (e.g. flipping image, but not in this case because anaglyphs need to keep the left/right orientation)
            self.transforms = transforms.Compose([
                transforms.Resize((c.IMAGE_SIZE, c.IMAGE_SIZE), Image.BICUBIC),
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((c.IMAGE_SIZE, c.IMAGE_SIZE), Image.BICUBIC)

        self.split = split
        self.size = c.IMAGE_SIZE
        self.path_anaglyph = path_anaglyph
        self.path_reversed = path_reversed

    def __getitem__(self, idx):
        img_anaglyph = Image.open(self.path_anaglyph[idx]).convert("RGB")
        img_anaglyph = self.transforms(img_anaglyph)
        img_anaglyph = np.array(img_anaglyph)
        img_anagyph_lab = rgb2lab(img_anaglyph).astype("float32")  # Converting RGB to L*a*b
        img_anagyph_lab = transforms.ToTensor()(img_anagyph_lab)

        L_a = img_anagyph_lab[[0], ...] / 50. - 1.  # Between -1 and 1
        ab_a = img_anagyph_lab[[1, 2], ...] / 110.  # Between -1 and 1

        img_reversed = Image.open(self.path_reversed[idx]).convert("RGB")
        img_reversed = self.transforms(img_reversed)
        img_reversed = np.array(img_reversed)
        img_reversed_lab = rgb2lab(img_reversed).astype("float32")  # Converting RGB to L*a*b
        img_reversed_lab = transforms.ToTensor()(img_reversed_lab)

        L_r = img_reversed_lab[[0], ...] / 50. - 1.
        ab_r = img_reversed_lab[[1, 2], ...] / 110.

        return {'L_a': L_a, 'ab_a': ab_a, 'L_r': L_r, 'ab_r': ab_r}

    def __len__(self):
        return len(self.paths)


def make_dataloaders(batch_size=16, n_workers=4, pin_memory=True, **kwargs):  # A handy function to make our dataloaders
    """Create dataloaders for the given dataset. **kwargs should be path_anaglyph, path_reversed, split"""
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader
