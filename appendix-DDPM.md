---
layout: home
title: Appendix - Code for DDPM
nav_order: 7
---

<head>
  <title>A Preliminary Mathematical Exegesis of Diffusion Models</title>
  <meta name="description" content="An in-depth mathematical analysis of diffusion models in machine learning.">
  <meta name="keywords" content="diffusion models, mathematics, machine learning, exegesis">
  <meta name="author" content="Shuyue Wang">
  <!-- Open Graph (for social sharing) -->
  <meta property="og:title" content="A Preliminary Mathematical Exegesis of Diffusion Models">
  <meta property="og:description" content="An in-depth mathematical analysis of diffusion models in machine learning.">
  <meta property="og:url" content="https://shuyuew1991.github.io/A-Preliminary-Mathematical-Exegesis-of-Diffusion-Models/">
  <meta property="og:type" content="website">
</head>

<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>

<div style="text-align: left; font-size: 1.3em;">
Appendix - Code for DDPM.
</div>
<br>


DDPM is a fundamental diffuison model, which is often used as the demo for the algorithm, because it is the simplest of all its descendant variants.
To train a diffusion model such as DDPM, it requires four things: **forward process**, **noise prediction model (U-Net)**, **loss function**, and **sampling loop (reverse process)**. 


---


**1. Forward Process**

The forward process gradually adds Gaussian noise to the data.
We already have:  

$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \varepsilon,
$$

where $$\alpha_t = 1 - \beta_t$$, $$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$$, and $$\varepsilon \sim \mathcal{N}(0, I)$$.


As for original DDPM the linear schedule is:

```python
import torch

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

timesteps = 1000
betas = linear_beta_schedule(timesteps)
alphas = 1. - betas
alpha_bars = torch.cumprod(alphas, dim=0)  # ᾱ_t = ∏(1 - β_s) from s=1 to t
```


As for \\(q(x_t | x_0)\\):
```python
def forward_diffusion(x0, t, alpha_bars, device="cpu"):
    """
    x0: Original clean image (B, C, H, W)
    t: Timestep (B,)
    alpha_bars: Precomputed ᾱ_t
    Returns: Noisy x_t and noise ε
    """
    noise = torch.randn_like(x0)  # ε ~ N(0, I)
    sqrt_alpha_bar = torch.sqrt(alpha_bars[t])[:, None, None, None]  # (B, 1, 1, 1)
    sqrt_one_minus_alpha_bar = torch.sqrt(1. - alpha_bars[t])[:, None, None, None]
    
    x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
    return x_t, noise
```


---


**2. Noise Prediction Model and Loss Function**

A **U-Net** is typically used to predict the noise \\( \epsilon \\) at each timestep.  
Here’s a simplified U-Net:

```python
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, timesteps=1000):
        super().__init__()
        self.time_embed = nn.Embedding(timesteps, 32)  # Embed timestep
        self.conv1 = nn.Conv2d(in_channels + 32, 64, kernel_size=3, padding=1)  # Concatenate time embedding
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        # Time embedding
        t_embed = self.time_embed(t).unsqueeze(-1).unsqueeze(-1)  # (B, 32, 1, 1)
        t_embed = t_embed.expand(-1, -1, x.shape[2], x.shape[3])  # (B, 32, H, W)
        
        # Concatenate time embedding to input
        x = torch.cat([x, t_embed], dim=1)  # (B, C+32, H, W)
        
        # U-Net (simplified)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x  # Predicted noise ε_θ(x_t, t)
```


Now we can calculate the loss function, which is basically noise prediction.
The loss is **mean squared error (MSE)** between the true noise \\( \epsilon \\) and predicted noise \\( \epsilon_\theta \\):  

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

```python
def loss_fn(model, x0, t, alpha_bars, device="cpu"):
    x_t, noise = forward_diffusion(x0, t, alpha_bars, device)
    pred_noise = model(x_t, t)
    return F.mse_loss(pred_noise, noise)  # MSE between true and predicted noise
```

---

**3. Sampling Loop (Reverse Process)**

The reverse process gradually denoises \\( x_T \sim \mathcal{N}(0, I) \\) back to \\( x_0 \\).  
At each step, we compute:  

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I),
$$  

where \\( \sigma_t = \sqrt{\beta_t} \\) as the choice for DDPM.

```python
@torch.no_grad()
def sample(model, img_size, timesteps, alpha_bars, betas, device="cpu"):
    # Start from pure noise
    x_t = torch.randn((1, 3, img_size, img_size), device=device)
    
    for t in reversed(range(timesteps)):
        # Predict noise
        t_tensor = torch.full((1,), t, dtype=torch.long, device=device)
        pred_noise = model(x_t, t_tensor)
        
        # Compute coefficients
        alpha_t = 1. - betas[t]
        alpha_bar_t = alpha_bars[t]
        beta_t = betas[t]
        
        # Update x_{t-1}
        x_t = (1. / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1. - alpha_bar_t)) * pred_noise
        )
        
        # Add noise (except at t=0)
        if t > 0:
            z = torch.randn_like(x_t)
            x_t += torch.sqrt(beta_t) * z
    
    return x_t.clamp(-1, 1)  # Clamp to [-1, 1]
```

---

**4. Training and Inference Script**

Here is a training code:
```python
import os
import torch
from torchvision.utils import save_image

# Hyperparameters
num_epochs = 1000
batch_size = 64
img_size = 64
timesteps = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model and optimizer
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Noise scheduler
betas = linear_beta_schedule(timesteps)
alphas = 1. - betas
alpha_bars = torch.cumprod(alphas, dim=0)

# Create save directory
os.makedirs("saved_models", exist_ok=True)
os.makedirs("samples", exist_ok=True)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (x0, _) in enumerate(dataloader):  # Assume dataloader yields (images, _)
        x0 = x0.to(device)
        t = torch.randint(0, timesteps, (x0.shape[0],), device=device)
        
        # Compute loss and update
        loss = loss_fn(model, x0, t, alpha_bars, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log training progress
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # Save model checkpoint every N epochs
    if epoch % 50 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
        }, f"saved_models/ddpm_epoch_{epoch}.pt")
    
    # Generate and save samples
    if epoch % 20 == 0:
        generated_img = sample(model, img_size, timesteps, alpha_bars, betas, device)
        save_image(generated_img, f"samples/sample_epoch_{epoch}.png")

# Save final model
torch.save(model.state_dict(), "saved_models/ddpm_final.pt")
```

And here is a standalone inference script that loads a trained model and generates new images:
```python
import torch
from torchvision.utils import save_image

def load_model(model_path, device="cuda"):
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    return model

def generate_images(
    model_path, 
    num_images=4, 
    img_size=64, 
    timesteps=1000, 
    output_dir="generated_samples",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_model(model_path, device)
    
    # Load noise schedule (must match training)
    betas = linear_beta_schedule(timesteps)
    alphas = 1. - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    # Generate images
    for i in range(num_images):
        with torch.no_grad():
            img = sample(model, img_size, timesteps, alpha_bars, betas, device)
        save_image(img, f"{output_dir}/sample_{i}.png")
        print(f"Saved sample_{i}.png")

if __name__ == "__main__":
    generate_images(
        model_path="saved_models/ddpm_final.pt",
        num_images=8,
        img_size=64,
    )
```


