import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 128
IMAGE_SIZE = 28
EPOCHS = 50
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

images, labels = next(iter(train_loader))
print(f"Image shape: {images.shape}")

# Beta Scheduler(linear scheduler)
T = 200 # scheduler steps
betas = torch.linspace(0.0001, 0.02, T).to(device) # Move to device
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0).to(device) # Move to device
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device) # Move to device
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device) # Move to device

# Noise addition
def q_sample(x_start, t, noise=None):
    """
    Add noise to x_start at timestep t.
    x_start: original image
    t: timestep
    noise: Gaussian noise
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alpha_prod = sqrt_alphas_cumprod[t][:, None, None, None]
    sqrt_one_minus = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
    return sqrt_alpha_prod * x_start + sqrt_one_minus * noise

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_embedding_dim=128, num_classes=10, cond_embedding_dim=128):
        super().__init__()

        # Time embedding should take a single scalar timestep and output a vector
        self.time_embedding = nn.Sequential(
            nn.Linear(1, cond_embedding_dim), # Input is a single timestep value
            nn.ReLU(),
            nn.Linear(cond_embedding_dim, cond_embedding_dim),
        )

        self.label_embedding = nn.Embedding(num_classes, cond_embedding_dim)

        # Linear layer to project conditional embedding to match the first convolution's output channels
        self.cond_proj1 = nn.Linear(cond_embedding_dim, 32)
        self.cond_proj2 = nn.Linear(cond_embedding_dim, 64)
        self.cond_proj3 = nn.Linear(cond_embedding_dim, 128)


        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling path
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Upsampling path
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1) # After skip connection

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1) # After skip connection

        # Output layer
        self.conv6 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)


    def forward(self, x, t, labels):
        # Time and label embeddings
        # Reshape t to (BATCH_SIZE, 1) and cast to float
        t_reshaped = t.float().unsqueeze(-1)
        t_emb = self.time_embedding(t_reshaped)
        label_emb = self.label_embedding(labels)
        cond = t_emb + label_emb # Combine time and label embeddings

        # Project conditional embedding to match the channels of the convolutions
        cond_proj1 = self.cond_proj1(cond)[:, :, None, None].repeat(1, 1, x.size(2), x.size(3))
        cond_proj2 = self.cond_proj2(cond)[:, :, None, None].repeat(1, 1, x.size(2)//2, x.size(3)//2)
        cond_proj3 = self.cond_proj3(cond)[:, :, None, None].repeat(1, 1, x.size(2)//4, x.size(3)//4)


        # Downsampling
        x = self.conv1(x)
        x = self.act(x + cond_proj1) # Apply conditioning after first convolution
        x1 = x # Store for skip connection
        x = self.pool(x)

        x = self.conv2(x)
        x = self.act(x + cond_proj2)
        x2 = x # Store for skip connection
        x = self.pool(x)

        x = self.conv3(x)
        x = self.act(x + cond_proj3)


        # Upsampling
        x = self.upconv1(x)
        # Skip connection
        x = torch.cat([x, x2], dim=1) # Concatenate along channel dimension
        x = self.conv4(x) # Convolution after skip connection
        x = self.act(x)

        x = self.upconv2(x)
        # Skip connection
        x = torch.cat([x, x1], dim=1) # Concatenate along channel dimension
        x = self.conv5(x) # Convolution after skip connection
        x = self.act(x)

        # Output
        x = self.conv6(x)

        return x

# Function to calculate the loss based on:
def get_loss(model, x_0, t, y):
    noise = torch.randn_like(x_0)
    x_noisy = q_sample(x_0, t, noise)
    noise_pred = model(x_noisy, t, y)
    return F.mse_loss(noise_pred, noise)


model = ConditionalUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_model(Epochs=EPOCHS):
  for epoch in range(Epochs):
    for batch in train_loader:
      x, y = batch
      x = x.to(device)
      y = y.to(device)

      t = torch.randint(0, T, (x.size(0),), device=device).long()
      loss = get_loss(model, x, t, y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")

