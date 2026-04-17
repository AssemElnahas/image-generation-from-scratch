import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# ==========================
# Hyperparameters
# ==========================
batch_size = 128
image_size = 64
nz = 100                 # latent vector size
ngf = 64
ndf = 64
num_epochs = 10          # Increase for better results
lr = 0.0002
beta1 = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================
# Dataset
# ==========================
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==========================
# Generator (unchanged)
# ==========================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# ==========================
# Discriminator - FIXED: Proper shape collapse
# ==========================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Final layer: output a single value per image
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)           # shape: [batch, 1, 1, 1]
        return output.view(-1)              # ← FIXED: Flatten to [batch]

# ==========================
# Initialize models
# ==========================
netG = Generator().to(device)
netD = Discriminator().to(device)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG.apply(weights_init)
netD.apply(weights_init)

# ==========================
# Loss and Optimizers
# ==========================
criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

print("Starting training...")

# ==========================
# Training Loop
# ==========================
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        real_images = data[0].to(device)
        b_size = real_images.size(0)

        # === Train Discriminator ===
        netD.zero_grad()

        # Real images
        label_real = torch.full((b_size,), 1.0, dtype=torch.float, device=device)
        output_real = netD(real_images)
        loss_real = criterion(output_real, label_real)

        # Fake images
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_images = netG(noise)
        label_fake = torch.full((b_size,), 0.0, dtype=torch.float, device=device)
        output_fake = netD(fake_images.detach())
        loss_fake = criterion(output_fake, label_fake)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizerD.step()

        # === Train Generator ===
        netG.zero_grad()
        output = netD(fake_images)
        loss_G = criterion(output, label_real)   # Try to fool discriminator
        loss_G.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(dataloader)}] "
                  f"Loss_D: {loss_D.item():.4f}  Loss_G: {loss_G.item():.4f}")

    # Save generated images after each epoch
    with torch.no_grad():
        fake = netG(torch.randn(64, nz, 1, 1, device=device)).detach().cpu()
        vutils.save_image(fake, f"generated_epoch_{epoch+1}.png", normalize=True, nrow=8)

print("Training finished! Check the generated_epoch_*.png files.")