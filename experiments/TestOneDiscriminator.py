
import os
import matplotlib
import torch
import random
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")  # Use Agg backend (non-GUI)
from models.Model6 import ResnetGeneratorOneDiscriminator
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model 6
checkpoint_path_model6 = "../checkpoints/checkpoints_model6_one_discriminator"

G_AB = ResnetGeneratorOneDiscriminator().to(device)  # Photo → Monet
G_BA = ResnetGeneratorOneDiscriminator().to(device)  # Monet → Photo

G_AB.load_state_dict(torch.load(f"{checkpoint_path_model6}/G_AB_final.pth", map_location=device))
G_BA.load_state_dict(torch.load(f"{checkpoint_path_model6}/G_BA_final.pth", map_location=device))
G_AB.eval()
G_BA.eval()

# Define Transformations (Keep 256x256)
transform_model6 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load Test Images
test_photo_path = "../datasets/monet2photo/testA"  # Real photos
test_monet_path = "../datasets/monet2photo/testB"  # Monet paintings

test_photos = random.sample(os.listdir(test_photo_path), 5)  # Pick 5 random photos
test_monets = random.sample(os.listdir(test_monet_path), 5)  # Pick 5 random Monet paintings

# Generate and Display Results
fig, axs = plt.subplots(len(test_photos), 4, figsize=(10, 15))

for i in range(len(test_photos)):
    # Load and preprocess images
    photo_img = Image.open(os.path.join(test_photo_path, test_photos[i])).convert("RGB")
    monet_img = Image.open(os.path.join(test_monet_path, test_monets[i])).convert("RGB")

    photo_tensor = transform_model6(photo_img).unsqueeze(0).to(device)
    monet_tensor = transform_model6(monet_img).unsqueeze(0).to(device)

    # Generate Monet-style and Photo-style images
    with torch.no_grad():
        fake_monet = G_AB(photo_tensor).squeeze(0).cpu()
        fake_photo = G_BA(monet_tensor).squeeze(0).cpu()

    fake_monet = transforms.ToPILImage()(fake_monet * 0.5 + 0.5)  # Denormalize
    fake_photo = transforms.ToPILImage()(fake_photo * 0.5 + 0.5)  # Denormalize

    # Display
    axs[i, 0].imshow(photo_img)
    axs[i, 0].set_title("Original Painting")
    axs[i, 0].axis("off")

    axs[i, 1].imshow(fake_monet)
    axs[i, 1].set_title("Generated Photo (1 Discriminator)")
    axs[i, 1].axis("off")

    axs[i, 2].imshow(monet_img)
    axs[i, 2].set_title("Original Photo")
    axs[i, 2].axis("off")

    axs[i, 3].imshow(fake_photo)
    axs[i, 3].set_title("Generated Painting (1 Discriminator)")
    axs[i, 3].axis("off")

plt.tight_layout()
plt.show()
