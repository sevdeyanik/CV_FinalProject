import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import os

# ✅ Load Generator (Photo → Monet)
from models.Model import ResnetGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_BA = ResnetGenerator().to(device)
checkpoint_path = "../checkpoints/checkpoints_256_4/G_BA_epoch20.pth"
G_BA.load_state_dict(torch.load(checkpoint_path, map_location=device))
G_BA.eval()
print("Monet Generator Loaded!")

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Folder Paths
testA_folder = "datasets/monet2photo/testB"
output_folder = "monetized_images_v2"
os.makedirs(output_folder, exist_ok=True)

# Process Multiple Images
for img_name in os.listdir(testA_folder):
    img_path = os.path.join(testA_folder, img_name)
    original_image = Image.open(img_path).convert("RGB")
    input_image = transform(original_image).unsqueeze(0).to(device)

    # Generate Monet-style image
    with torch.no_grad():
        output_image = G_BA(input_image).squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_image = (output_image + 1) / 2  # Rescale

    # Save Output
    output_path = os.path.join(output_folder, f"monet_{img_name}")
    plt.imsave(output_path, output_image)

print(f"Processed {len(os.listdir(testA_folder))} images. Monetized images saved in {output_folder}")
