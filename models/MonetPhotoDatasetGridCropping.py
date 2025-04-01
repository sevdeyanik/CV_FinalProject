# Model 4: Grid-based Cropping
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Define Grid Crop Function
def grid_crop(image, crop_size=128, stride=128):
    """
    Splits a 256x256 image into multiple overlapping 128x128 crops.
    Args:
        image (PIL.Image): Input image (expected 256x256).
        crop_size (int): Size of each crop.
        stride (int): Overlap between crops.
    Returns:
        List of cropped images.
    """
    crops = []
    for i in range(0, image.size[0] - crop_size + 1, stride):
        for j in range(0, image.size[1] - crop_size + 1, stride):
            crop = image.crop((i, j, i + crop_size, j + crop_size))
            crops.append(crop)
    return crops

# Define Dataset Class with Cropping
class MonetPhotoDatasetGridCropping(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.photo_paths = sorted(os.listdir(os.path.join(root_dir, "trainA")))
        self.monet_paths = sorted(os.listdir(os.path.join(root_dir, "trainB")))

    def __len__(self):
        return max(len(self.photo_paths), len(self.monet_paths)) * 4  # Each image gives 4 crops

    def __getitem__(self, idx):
        # Get image paths
        photo_path = os.path.join(self.root_dir, "trainA", self.photo_paths[idx % len(self.photo_paths)])
        monet_path = os.path.join(self.root_dir, "trainB", self.monet_paths[idx % len(self.monet_paths)])

        # Open images
        photo = Image.open(photo_path).convert("RGB")
        monet = Image.open(monet_path).convert("RGB")

        # Generate grid crops
        photo_crops = grid_crop(photo)  # List of 128x128 crops
        monet_crops = grid_crop(monet)

        # Select one crop (cycle through based on index)
        selected_crop = idx % 4  # Picks 0,1,2,3 for different crops
        photo = photo_crops[selected_crop]
        monet = monet_crops[selected_crop]

        # Apply transforms
        if self.transform:
            photo = self.transform(photo)
            monet = self.transform(monet)

        return photo, monet

