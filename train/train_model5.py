# Model 5: Downscaling + Cropping
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.MonetPhotoDataset import MonetPhotoDataset
from train import train

dataset_path = "../datasets/monet2photo"
checkpoint_path_model5 = "../checkpoints/checkpoints_model5_downsampling_randomcropping"

# Define Transformations: Downscaling + Random Cropping
transform_model5 = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),  # Randomly resizes & crops in one go
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize [-1,1]
])

dataloader_model5 = DataLoader(MonetPhotoDataset(root_dir=dataset_path, transform=transform_model5), batch_size=8, shuffle=True)

train(dataloader_model5, checkpoint_path_model5, 50)