# Model 2: 128 x 128
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.MonetPhotoDataset import MonetPhotoDataset
from train import train

dataset_path = "../datasets/monet2photo"
checkpoint_path_model2 = "../checkpoints/checkpoints_128_4"

# Define Image Transformations for Model 2 (Downsizing)
transform_model2 = transforms.Compose([
    transforms.Resize((128, 128)),  # Reduce size for faster training
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
# Create DataLoader
dataloader_model2 = DataLoader(MonetPhotoDataset(root_dir=dataset_path, transform=transform_model2), batch_size=4, shuffle=True)

# Train 30 epochs
train(dataloader_model2, checkpoint_path_model2, 30)
