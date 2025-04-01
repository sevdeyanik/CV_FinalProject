import os
import matplotlib
import torch
from models.Model import ResnetGenerator
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")  # Use Agg backend (non-GUI)
from models.Model6 import ResnetGeneratorOneDiscriminator
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Model Paths
model_checkpoints_photo_to_monet = {
    "Model 1 (256x256)": "../checkpoints/checkpoints_256_4/G_BA_final.pth",
    "Model 2 (128x128)": "../checkpoints/checkpoints_128_4/G_BA_final.pth",
    "Model 3 (Random Crop)": "../checkpoints/checkpoints_model3_randomcropping/G_BA_final.pth",
    "Model 4 (Grid Crop)": "../checkpoints/checkpoints_model4_gridcropping/G_BA_final.pth",
    "Model 5 (D + Crop)": "../checkpoints/checkpoints_model5_downsampling_randomcropping/G_BA_final.pth",
    "Model 6 (1 Discriminator)": "../checkpoints/checkpoints_model6_one_discriminator/G_BA_final.pth",
}
transform_model1 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize [-1,1]
])


# Function to Load a Model
def load_model(checkpoint_path):
    model = ResnetGenerator().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # Set to evaluation mode
    return model


def load_model_model6(checkpoint_path):
    model = ResnetGeneratorOneDiscriminator().to(device)  # Match the exact model used in Model 6
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


# Select Test Images (Photos)
test_images_path = "../datasets/monet2photo/testB"
test_images = sorted(os.listdir(test_images_path))[10:15]  # Load a few images

# Load All Models
models = {}
for name, path in model_checkpoints_photo_to_monet.items():
    if "1 Discriminator" in name:
        models[name] = load_model_model6(path)
    else:
        models[name] = load_model(path)

# Generate Outputs for Each Model
fig, axes = plt.subplots(len(test_images), len(models) + 1, figsize=(15, 12))
for row, img_name in enumerate(test_images):
    img_path = os.path.join(test_images_path, img_name)
    original_img = Image.open(img_path).convert("RGB")
    input_tensor = transform_model1(original_img).unsqueeze(0)  # Add batch dimension

    # First Column: Original Image
    axes[row, 0].imshow(original_img)
    axes[row, 0].set_title("Original Photo")
    axes[row, 0].axis("off")

    # Generate and Display for Each Model
    for col, (model_name, model) in enumerate(models.items()):
        with torch.no_grad():
            generated = model(input_tensor).squeeze(0).cpu()
            generated = (generated * 0.5) + 0.5  # Denormalize to [0,1]
            generated = transforms.ToPILImage()(generated)

        axes[row, col + 1].imshow(generated)
        axes[row, col + 1].set_title(model_name)
        axes[row, col + 1].axis("off")

plt.tight_layout()
plt.show()
