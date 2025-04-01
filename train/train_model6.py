# Model 6: Training Loop
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.Model import D_A, D_B, optimizer_D_A, optimizer_D_B
from models.MonetPhotoDataset import MonetPhotoDataset
from models.Model6 import optimizer_G, G_AB, G_BA, cycle_loss, remove_D_A, adversarial_loss, remove_D_B

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = "../datasets/monet2photo"
checkpoint_path_model6 = "../checkpoints/checkpoints_model6_one_discriminator"

# Define Transformations (Keep 256x256)
transform_model6 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Initialize DataLoader
dataloader_model6 = DataLoader(MonetPhotoDataset(root_dir=dataset_path, transform=transform_model6), batch_size=8, shuffle=True)

print(f"DataLoader Ready: {len(dataloader_model6)} batches per epoch")

num_epochs = 50

for epoch in range(num_epochs):
    for batch_idx, (real_A, real_B) in enumerate(dataloader_model6):
        real_A, real_B = real_A.to(device), real_B.to(device)

        # Train Generators
        optimizer_G.zero_grad()
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)
        cycle_A = G_BA(fake_B)
        cycle_B = G_AB(fake_A)

        loss_G = cycle_loss(real_A, cycle_A) + cycle_loss(real_B, cycle_B)

        if not remove_D_A:
            loss_G += adversarial_loss(D_A(fake_A), torch.ones_like(D_A(fake_A)))
        if not remove_D_B:
            loss_G += adversarial_loss(D_B(fake_B), torch.ones_like(D_B(fake_B)))

        loss_G.backward(retain_graph=True)
        optimizer_G.step()

        # Train Discriminators
        if not remove_D_A:
            optimizer_D_A.zero_grad()
            loss_D_A = adversarial_loss(D_A(real_A), torch.ones_like(D_A(real_A))) + \
                       adversarial_loss(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A)))
            loss_D_A.backward()
            optimizer_D_A.step()
        else:
            loss_D_A = torch.tensor(0)

        if not remove_D_B:
            optimizer_D_B.zero_grad()
            loss_D_B = adversarial_loss(D_B(real_B), torch.ones_like(D_B(real_B))) + \
                       adversarial_loss(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B)))
            loss_D_B.backward()
            optimizer_D_B.step()
        else:
            loss_D_B = torch.tensor(0)

        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader_model6)}], Loss_G: {loss_G.item()}, Loss_D_A: {loss_D_A.item()}, Loss_D_B: {loss_D_B.item()}")

    # Save Checkpoints every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(G_AB.state_dict(), f"{checkpoint_path_model6}/G_AB_epoch{epoch+1}_one_discriminator.pth")
        torch.save(G_BA.state_dict(), f"{checkpoint_path_model6}/G_BA_epoch{epoch+1}_one_discriminator.pth")
        print(f"Saved checkpoint at epoch {epoch+1}")


# Save Final Model
torch.save(G_AB.state_dict(), f"{checkpoint_path_model6}/G_AB_final.pth")
torch.save(G_BA.state_dict(), f"{checkpoint_path_model6}/G_BA_final.pth")

print("Training Completed!")
print("Final Model Saved!")
