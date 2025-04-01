import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class MonetPhotoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.photo_paths = sorted(os.listdir(os.path.join(root_dir, "trainA")))
        self.monet_paths = sorted(os.listdir(os.path.join(root_dir, "trainB")))

    def __len__(self):
        return max(len(self.photo_paths), len(self.monet_paths))

    def __getitem__(self, idx):
        photo_path = os.path.join(self.root_dir, "trainA", self.photo_paths[idx % len(self.photo_paths)])
        monet_path = os.path.join(self.root_dir, "trainB", self.monet_paths[idx % len(self.monet_paths)])
        photo = Image.open(photo_path).convert("RGB")
        monet = Image.open(monet_path).convert("RGB")

        if self.transform:
            photo = self.transform(photo)
            monet = self.transform(monet)

        return photo, monet

