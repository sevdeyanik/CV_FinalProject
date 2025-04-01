import os
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from cleanfid import fid
from PIL import Image

from models.Model import ResnetGenerator
from models.Model6 import ResnetGeneratorOneDiscriminator

# Paths to real images
real_monet_dir = "../datasets/monet2photo/trainB"
real_photo_dir = "../datasets/monet2photo/trainA"

# Paths to trained models
model_checkpoints = {
    "Model 1 (256x256)": {
        "G_AB": "../checkpoints/checkpoints_256_4/G_AB_final.pth",
        "G_BA": "../checkpoints/checkpoints_256_4/G_BA_final.pth",
    },
    "Model 2 (128x128)": {
        "G_AB": "../checkpoints/checkpoints_128_4/G_AB_final.pth",
        "G_BA": "../checkpoints/checkpoints_128_4/G_BA_final.pth",
    },
    "Model 3 (Random Crop)": {
        "G_AB": "../checkpoints/checkpoints_model3_randomcropping/G_AB_final.pth",
        "G_BA": "../checkpoints/checkpoints_model3_randomcropping/G_BA_final.pth",
    },
    "Model 4 (Grid Crop)": {
        "G_AB": "../checkpoints/checkpoints_model4_gridcropping/G_AB_final.pth",
        "G_BA": "../checkpoints/checkpoints_model4_gridcropping/G_BA_final.pth",
    },
    "Model 5 (Downsampling + Random Crop)": {
        "G_AB": "../checkpoints/checkpoints_model5_downsampling_randomcropping/G_AB_final.pth",
        "G_BA": "../checkpoints/checkpoints_model5_downsampling_randomcropping/G_BA_final.pth",
    },
    "Model 6 (1 Discriminator)": {
        "G_AB": "../checkpoints/checkpoints_model6_one_discriminator/G_AB_final.pth",
        "G_BA": "../checkpoints/checkpoints_model6_one_discriminator/G_BA_final.pth",
    }

}

# Output directories for generated images
output_dirs = {
    name: {
        "Photo_to_Monet": f"../generated_images/{name.replace(' ', '_')}_Photo_to_Monet",
        "Monet_to_Photo": f"../generated_images/{name.replace(' ', '_')}_Monet_to_Photo"
    }
    for name in model_checkpoints.keys()
}

transform_model1 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize [-1,1]
])

# Create necessary directories
for paths in output_dirs.values():
    os.makedirs(paths["Photo_to_Monet"], exist_ok=True)
    os.makedirs(paths["Monet_to_Photo"], exist_ok=True)


# Function to generate images
def generate_images(model_path, input_dir, output_dir, use_model6=False):
    print(f"Generating images with model: {model_path}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_model6:
        model = ResnetGeneratorOneDiscriminator().to(device)
    else:
        model = ResnetGenerator().to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image_paths = sorted(os.listdir(input_dir))[:200]  # Use 200 images
    for img_name in image_paths:
        img_path = os.path.join(input_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = transform_model1(image).unsqueeze(0).to(device)

        with torch.no_grad():
            generated = model(image)
            generated = (generated.squeeze().cpu().numpy().transpose(1, 2, 0) + 1) / 2  # Denormalize

        # Save generated image
        save_path = os.path.join(output_dir, img_name)
        cv2.imwrite(save_path, (generated * 255).astype(np.uint8))


# Generate images for each model
for model_name, paths in model_checkpoints.items():
    is_model6 = "1 Discriminator" in model_name
    generate_images(paths["G_AB"], real_photo_dir, output_dirs[model_name]["Photo_to_Monet"], use_model6=is_model6)
    generate_images(paths["G_BA"], real_monet_dir, output_dirs[model_name]["Monet_to_Photo"], use_model6=is_model6)


# Function to compute SSIM
def compute_ssim(real_dir, gen_dir):
    real_images = sorted(os.listdir(real_dir))[:200]
    gen_images = sorted(os.listdir(gen_dir))[:200]

    ssim_scores = []
    for r_img, g_img in zip(real_images, gen_images):
        real = cv2.imread(os.path.join(real_dir, r_img))
        gen = cv2.imread(os.path.join(gen_dir, g_img))

        real_gray = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)
        gen_gray = cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY)

        score, _ = ssim(real_gray, gen_gray, full=True)
        ssim_scores.append(score)

    return np.mean(ssim_scores)


# Compute FID & SSIM for each model
fid_scores = {}
ssim_scores = {}

for model_name, dirs in output_dirs.items():
    print(f"Processing {model_name}...")

    fid_photo_monet = fid.compute_fid(dirs["Photo_to_Monet"], real_monet_dir)
    fid_monet_photo = fid.compute_fid(dirs["Monet_to_Photo"], real_photo_dir)

    fid_scores[model_name] = (fid_photo_monet, fid_monet_photo)

    ssim_photo_monet = compute_ssim(real_monet_dir, dirs["Photo_to_Monet"])
    ssim_monet_photo = compute_ssim(real_photo_dir, dirs["Monet_to_Photo"])

    ssim_scores[model_name] = (ssim_photo_monet, ssim_monet_photo)

# Print numerical values for FID and SSIM scores for better interpretation
print("\n=== FID Scores (Lower is Better) ===")
for model, (fid_photo_monet, fid_monet_photo) in fid_scores.items():
    print(f"{model}: Photo → Monet: {fid_photo_monet:.2f}, Monet → Photo: {fid_monet_photo:.2f}")

print("\n=== SSIM Scores (Higher is Better) ===")
for model, (ssim_photo_monet, ssim_monet_photo) in ssim_scores.items():
    print(f"{model}: Photo → Monet: {ssim_photo_monet:.4f}, Monet → Photo: {ssim_monet_photo:.4f}")


# Plot FID & SSIM results
def plot_metric(metric_dict, title, ylabel):
    models = list(metric_dict.keys())
    monet_values = [metric_dict[m][0] for m in models]
    photo_values = [metric_dict[m][1] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, monet_values, width, label="FID (Photo → Monet)")
    ax.bar(x + width / 2, photo_values, width, label="FID (Monet → Photo)")

    ax.set_xlabel("Models")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.legend()
    plt.show()


plot_metric(fid_scores, "FID Scores Across Models", "FID (Lower is Better)")
plot_metric(ssim_scores, "SSIM Scores Across Models", "SSIM (Higher is Better)")
