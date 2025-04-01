# Model 6: One Discriminator
import torch
import torch.nn as nn
import torch.optim as optim
from models.Model import Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration: Choose which discriminator to remove
remove_D_A = False  # Set to True to remove Discriminator A (real photos)
remove_D_B = True  # Set to True to remove Discriminator B (Monet paintings)


# Define Generator
class ResnetGeneratorOneDiscriminator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, num_residual_blocks=6):
        super(ResnetGeneratorOneDiscriminator, self).__init__()
        model = [
            nn.Conv2d(input_channels, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_residual_blocks):
            model += [nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        model += [
            nn.Conv2d(64, output_channels, kernel_size=7, padding=3),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# Initialize Models
G_AB = ResnetGeneratorOneDiscriminator().to(device)  # Monet → Photo
G_BA = ResnetGeneratorOneDiscriminator().to(device)  # Photo → Monet

if not remove_D_A:
    D_A = Discriminator().to(device)
if not remove_D_B:
    D_B = Discriminator().to(device)

# Define Loss Functions
cycle_loss = nn.L1Loss()
adversarial_loss = nn.MSELoss()

# Define Optimizers
optimizer_G = optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=0.0002, betas=(0.5, 0.999))
if not remove_D_A:
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
if not remove_D_B:
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))
