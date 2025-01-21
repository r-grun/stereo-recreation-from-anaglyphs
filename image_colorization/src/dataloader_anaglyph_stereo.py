from torchvision.transforms import InterpolationMode
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class AnaglyphStereoDataset(Dataset):
    def __init__(self, path_anaglyph, path_left, path_right, image_size=256, files_limit=0):
        self.transforms = transforms.Compose([
            # transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),  # Resize images to a fixed size
            transforms.ToTensor(),         # Convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

        self.size = image_size
        self.path_anaglyph = path_anaglyph
        self.anaglyphs_paths = open(self.path_anaglyph, 'r').read().splitlines()
        if files_limit > 0:
            self.anaglyphs_paths = self.anaglyphs_paths[:files_limit]

        self.path_left = path_left
        self.left_paths = open(self.path_left, 'r').read().splitlines()
        if files_limit > 0:
            self.left_paths = self.left_paths[:files_limit]

        self.path_right = path_right
        self.right_paths = open(self.path_right, 'r').read().splitlines()
        if files_limit > 0:
            self.right_paths = self.right_paths[:files_limit]

    def __getitem__(self, idx):
        img_anaglyph = Image.open(self.anaglyphs_paths[idx]).convert("RGB")
        img_anaglyph = self.transforms(img_anaglyph)

        img_left = Image.open(self.left_paths[idx]).convert("RGB")
        img_left = self.transforms(img_left)

        img_right = Image.open(self.right_paths[idx]).convert("RGB")
        img_right = self.transforms(img_right)


        return {'a': img_anaglyph, 'l': img_left, 'r': img_right}

    def __len__(self):
        return len(self.anaglyphs_paths)


def make_dataloaders(batch_size=16, n_workers=4, pin_memory=True, **kwargs):  # A handy function to make our dataloaders
    """Create dataloaders for the given dataset. **kwargs should be path_anaglyph, path_left, path_right, image_size=256, files_limit=0"""
    dataset = AnaglyphStereoDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader
