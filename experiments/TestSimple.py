import torch
import matplotlib
matplotlib.use("TkAgg")  # Use Agg backend (non-GUI)
import matplotlib.pyplot as plt
from models.Model import ResnetGenerator
import cv2

# Load trained generator (Photo → Monet)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_BA = ResnetGenerator().to(device)
G_BA.load_state_dict(torch.load("../checkpoints/checkpoints_256_4/G_AB_final.pth", map_location=torch.device('cpu'), weights_only=True))

G_BA.eval()

# Load and preprocess a tests image
def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (256, 256))  # Resize to match training size
    img = img / 255.0  # Normalize to [0,1]
    img = (img - 0.5) * 2  # Normalize to [-1,1] (CycleGAN standard)
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return img

# Select a tests image
test_image_path = "../datasets/monet2photo/testB/2014-08-04 08_01_08.jpg"  # Change this to your real tests image
input_image = load_image(test_image_path)

# Generate image
with torch.no_grad():
    output_image = G_BA(input_image).squeeze(0).permute(1, 2, 0).cpu().numpy()

# Rescale output to [0,1] for displaying
output_image = (output_image + 1) / 2

# Display input & output images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv2.imread(test_image_path)[:, :, ::-1])  # Original Image
axes[0].set_title("Original Photo")
axes[1].imshow(output_image)  # Monet-stylized Image
axes[1].set_title("Generated Monet Painting")
plt.show()
