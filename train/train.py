import torch
from models.Model import G_AB, G_BA, cycle_loss, D_A, D_B
from models.Model import adversarial_loss, optimizer_G, optimizer_D_A, optimizer_D_B

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(dataloader, checkpoint_path, num_epochs=20):

  print(f"DataLoader Ready: {len(dataloader)} batches per epoch")

  for epoch in range(num_epochs):
      for batch_idx, (real_A, real_B) in enumerate(dataloader):
          real_A, real_B = real_A.to(device), real_B.to(device)

          # Train Generators
          optimizer_G.zero_grad()
          fake_B = G_AB(real_A)
          fake_A = G_BA(real_B)
          cycle_A = G_BA(fake_B)
          cycle_B = G_AB(fake_A)
          loss_G = cycle_loss(real_A, cycle_A) + cycle_loss(real_B, cycle_B) + \
                  adversarial_loss(D_A(fake_A), torch.ones_like(D_A(fake_A))) + \
                  adversarial_loss(D_B(fake_B), torch.ones_like(D_B(fake_B)))
          loss_G.backward(retain_graph=True)
          optimizer_G.step()

          # Train Discriminators
          optimizer_D_A.zero_grad()
          loss_D_A = adversarial_loss(D_A(real_A), torch.ones_like(D_A(real_A))) + \
                    adversarial_loss(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A)))
          loss_D_A.backward()
          optimizer_D_A.step()

          optimizer_D_B.zero_grad()
          loss_D_B = adversarial_loss(D_B(real_B), torch.ones_like(D_B(real_B))) + \
                    adversarial_loss(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B)))
          loss_D_B.backward()
          optimizer_D_B.step()

          print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss_G: {loss_G.item()}, Loss_D_A: {loss_D_A.item()}, Loss_D_B: {loss_D_B.item()}")

      if (epoch + 1) % 10 == 0:
              torch.save(G_AB.state_dict(), f"{checkpoint_path}/G_AB_epoch{epoch+1}.pth")
              torch.save(G_BA.state_dict(), f"{checkpoint_path}/G_BA_epoch{epoch+1}.pth")
              torch.save(optimizer_G.state_dict(), f"{checkpoint_path}/optim_G_epoch{epoch+1}.pth")  # Save Optimizer
              torch.save(optimizer_D_A.state_dict(), f"{checkpoint_path}/optim_D_A_epoch{epoch+1}.pth")  # Save Optimizer
              torch.save(optimizer_D_B.state_dict(), f"{checkpoint_path}/optim_D_B_epoch{epoch+1}.pth")  # Save Optimizer
              print(f"Saved checkpoint at epoch {epoch+1}")

  # Save Final Model
  torch.save(G_AB.state_dict(), f"{checkpoint_path}/G_AB_final.pth")
  torch.save(G_BA.state_dict(), f"{checkpoint_path}/G_BA_final.pth")
  torch.save(optimizer_G.state_dict(), f"{checkpoint_path}/optim_G_final.pth")
  torch.save(optimizer_D_A.state_dict(), f"{checkpoint_path}/optim_D_A_final.pth")
  torch.save(optimizer_D_B.state_dict(), f"{checkpoint_path}/optim_D_B_final.pth")
  print("Final Model Saved!")