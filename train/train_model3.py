# Model 3: Random cropping
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.MonetPhotoDataset import MonetPhotoDataset
from train import train

dataset_path = "../datasets/monet2photo"
checkpoint_path_model3 = "../checkpoints/checkpoints_model3_randomcropping"

transform_model3 = transforms.Compose([
    transforms.RandomCrop(128), # Crop patches of 128x128
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create DataLoader with Cropped Images
dataloader_model3 = DataLoader(
    MonetPhotoDataset(root_dir=dataset_path, transform=transform_model3),
    batch_size=8, shuffle=True
)

train(dataloader_model3, checkpoint_path_model3, 50)