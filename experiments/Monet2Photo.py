# Load Generator (Monet → Photo)
from models.Model import ResnetGenerator
import torch
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("TkAgg")  # Use Agg backend (non-GUI)
import matplotlib.pyplot as plt
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G_AB = ResnetGenerator().to(device)
reverse_checkpoint = "../checkpoints/checkpoints_model5_downsampling_randomcropping/G_AB_final.pth"

G_AB.load_state_dict(torch.load(reverse_checkpoint, map_location=device))
G_AB.eval()
print("Reverse Model Loaded (Monet → Photo)")

# Image Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Folder Paths
testA_folder = "../datasets/monet2photo/testA"  # Monet paintings
reverse_output_folder = "../results/photo_restored_model5"
os.makedirs(reverse_output_folder, exist_ok=True)

# Process & Display 5 Sample Images
sample_images = os.listdir(testA_folder)[:15]  # Use first 5 images

for img_name in sample_images:
    img_path = os.path.join(testA_folder, img_name)
    original_image = Image.open(img_path).convert("RGB")
    input_image = transform(original_image).unsqueeze(0).to(device)

    # Generate Photo-style image
    with torch.no_grad():
        output_image = G_AB(input_image).squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_image = (output_image + 1) / 2  # Rescale

    # Save Output
    output_path = os.path.join(reverse_output_folder, f"photo_{img_name}")
    plt.imsave(output_path, output_image)

    # Show Side-by-Side Comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Original Monet Painting 🎨")
    axes[0].axis("off")

    axes[1].imshow(output_image)
    axes[1].set_title("Generated Real Photo 🏞️")
    axes[1].axis("off")

    plt.show()

print(f" Converted {len(sample_images)} Monet paintings back to real photos! Saved in {reverse_output_folder}")
