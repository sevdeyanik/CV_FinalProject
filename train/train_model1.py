# Model 1: 256 x 256
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.MonetPhotoDataset import MonetPhotoDataset
from train import train

dataset_path = "../datasets/monet2photo"
checkpoint_path_model1 = "../checkpoints/checkpoints_256_4"

# Define Image Transformations for Model 1
transform_model1 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize [-1,1]
])


# Create DataLoader
dataloader_model1 = DataLoader(MonetPhotoDataset(root_dir=dataset_path, transform=transform_model1), batch_size=4, shuffle=True)


# Train 30 epochs
train(dataloader_model1, checkpoint_path_model1, 30)