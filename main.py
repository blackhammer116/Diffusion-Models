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

