# Model 4 training: Grid-based Cropping
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.MonetPhotoDatasetGridCropping import MonetPhotoDatasetGridCropping
from train import train

dataset_path = "../datasets/monet2photo"
checkpoint_path_model4 = "../checkpoints/checkpoints_model4_gridcropping"

transform_model4 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize [-1,1]
])

dataloader_model4 = DataLoader(MonetPhotoDatasetGridCropping(root_dir=dataset_path, transform=transform_model4), batch_size=16, shuffle=True)

train(dataloader_model4, checkpoint_path_model4, 30) #afterwards trained for 20 more so in total 50 epochs