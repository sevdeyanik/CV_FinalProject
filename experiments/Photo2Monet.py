import torch
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("TkAgg")  # Use Agg backend (non-GUI)
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load Generator (Photo → Monet)
from models.Model import ResnetGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_BA = ResnetGenerator().to(device)
checkpoint_path = "../checkpoints/checkpoints_model5_downsampling_randomcropping/G_BA_final.pth"
G_BA.load_state_dict(torch.load(checkpoint_path, map_location=device))
G_BA.eval()
print("Monet Generator Loaded!")

# Image Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Folder Paths
testB_folder = "../datasets/monet2photo/testB"  # Real photos
output_folder = "../results/monetized_images_subset_model5_epoch70"
os.makedirs(output_folder, exist_ok=True)

# Process & Display 5 Sample Images
sample_images = os.listdir(testB_folder)[:15]  # Use first 5 images

for img_name in sample_images:
    img_path = os.path.join(testB_folder, img_name)
    original_image = Image.open(img_path).convert("RGB")
    input_image = transform(original_image).unsqueeze(0).to(device)

    # Generate Monet-style image
    with torch.no_grad():
        output_image = G_BA(input_image).squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_image = (output_image + 1) / 2  # Rescale

    # Save Output
    output_path = os.path.join(output_folder, f"monet_{img_name}")
    plt.imsave(output_path, output_image)

    # Show Side-by-Side Comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Original Photo")
    axes[0].axis("off")

    axes[1].imshow(output_image)
    axes[1].set_title("Generated Monet Painting")
    axes[1].axis("off")

    plt.show()

print(f"Converted {len(sample_images)} real photos into Monet-style paintings! Saved in {output_folder}")
