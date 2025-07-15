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


DDPM is a fundamental diffusion model, which is often used as the demo for the algorithm, because it is the simplest of all its descendant variants.
To train a diffusion model such as DDPM, you can vaguely think of some blocks to define.
It is not necessary to memorize anything: thinking as the script will do.
We should easily envision that some common packages that should be imported, the details of which can be determined while coding the subsequent.
We should make something to load the dataset, with some cleaning. A class can help us do this. 
Then we come up with idea that the equivalent loss is the noise itself. 
We can use UNet to predict that noise, while the reverse process that is **heavily** discussed in Chapter 4.
With this, training paradigm and schedulers can be prepared.
And that's all.
Let's see the code coded by me.

The code is not written linearly piece by piece, a non-linear adaptation and change is unavoidable. The following is just one final look of the code.

---

Let's decide the gpu that we use in a server. Suppose we have 8.

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
```

Some hyper params should be assigned in configuration.

```python
class Config:
    data_dir      = "/image_folder"
    save_path     = "ddpm_best.pt"
    epochs        = 1000
    batch_size    = 4096 
    lr            = 2e-4
    timesteps     = 1000
    image_size    = 64
cfg = Config()
```
Some codes requires the images are located in an intermediate class_tagged folder, we decided to eliminate this requirement.
The batch_size can be set to maximize the use of gpu.
You can install `nvitop` to check instantly the status of the gpus.
The image_size is the real size that is transformed into for the images.
timesteps being 1000 is a common practice.
The final line is the instantiation of the class.


Let's define how to load the images first.

```python
class FlatFolder(VisionDataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        
        # Supported image extensions
        self.extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
        
        # Get all image files
        self.samples = [
            str(p) for p in Path(root).iterdir() 
            if p.is_file() and p.suffix.lower() in self.extensions
        ]
        
        if not self.samples:
            raise FileNotFoundError(f"No images found in {root} with extensions {self.extensions}!")

    def __getitem__(self, index):
        path = self.samples[index]
        img = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        # Return a dummy label (0) for compatibility
        return img, 0

    def __len__(self):
        return len(self.samples)
```
and we equipe it with 

```python
from torchvision.datasets import VisionDataset
from pathlib import Path
from PIL import Image
```

`__init__` and `__getitem__` are natual components of dataloader.



We now define a class to add noise:
```python
class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        betas  = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1. - betas
        alpha_hats = torch.cumprod(alphas, 0)

        # register as buffers → move with .to(device) automatically
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_hats", alpha_hats)
        self.timesteps = timesteps

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_hat = self.alpha_hats[t].sqrt()      .view(-1,1,1,1)
        sqrt_one_minus = (1 - self.alpha_hats[t]).sqrt().view(-1,1,1,1)
        return sqrt_alpha_hat * x0 + sqrt_one_minus * noise

```

and of course we have to equip it with:
```python
import torch
import torch.nn as nn

```

Inherit `nn.Module` is a common practice.
So is `__init__` for the class.
The class is basically for determining the noise at all `t` for given `x0`.
They're the target for the noise prediction, for sure.
The `q_sample` is basically `q(x_t | x_0)`that defines the noising process:

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})
$$

where:
- \\(\alpha_t = 1 - \beta_t \\), with \\( \beta_t \\) being the noise schedule at step \\( t \\),
- \\( \bar{\alpha}_t = \prod_{s=1}^t \alpha_s \\),
- \\( \mathcal{N} \\) denotes a Gaussian distribution with mean \\( \sqrt{\bar{\alpha}_t} x_0 \\) and variance \\( (1 - \bar{\alpha}_t) \mathbf{I} \\).

This can also be written in terms of the reparameterization trick as:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$


By reshaping with `.view(-1,1,1,1)`, the coefficients are broadcast correctly during multiplication across all dimensions of the image tensors. The -1 preserves the `batch` dimension, while the three 1s create singleton dimensions for `channels`, `height`, and `width`, allowing the coefficients to be applied uniformly across the spatial dimensions of each image in the batch.





So we can fulfill the preparation of the training in the main function:
```python
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    tfm = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor()
    ])
    
    dataset = FlatFolder(cfg.data_dir, transform=tfm)

    train_len = int(0.9 * len(dataset))
    val_len   = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    dl_train = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_val   = DataLoader(val_set,   batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

```

Ok, now we come to design the module to predict the noise, i.e. the UNet.


```python
class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base=64):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.enc1 = block(in_channels, base)
        self.enc2 = block(base, base * 2)
        self.enc3 = block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # These project upsampled tensors to match skip connection dims
        self.reduce_e3 = nn.Conv2d(base * 4, base * 2, 1)
        self.reduce_d2 = nn.Conv2d(base * 2, base, 1)

        self.dec2 = block(base * 2, base * 2)
        self.dec1 = block(base, base)

        self.outc = nn.Conv2d(base, out_channels, 1)

    def forward(self, x, t_embed):
        t_embed = t_embed[:, None, None, None].expand(-1, 1, x.shape[2], x.shape[3])
        x = torch.cat([x, t_embed], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Project upsampled layers before addition
        u3 = self.up(e3)
        u3 = self.reduce_e3(u3)
        d2 = self.dec2(u3 + e2)

        u2 = self.up(d2)
        u2 = self.reduce_d2(u2)
        d1 = self.dec1(u2 + e1)

        return self.outc(d1)

```


The `nn.MaxPool2d(2)` function creates a layer that downsamples the input by taking the maximum value across each 2x2 window.

The `nn.Upsample` function creates a module that upsamples the input data by a factor of 2 in both the height and width dimensions, using nearest-neighbor interpolation.

Here’s a merged and LaTeX-formatted version of parts 2 through 5, describing how UNet is used to predict noise in diffusion models:


The core idea is to train a UNet to predict the noise \\(\epsilon\\) added to a noisy image \\(x_t\\) at timestep \\(t\\), enabling iterative denoising. 
The goal of denoising in the reverse process is to recover the clean image \\(x_0\\) from pure noise \\(x_T\\) by reversing the diffusion process. 
At each timestep \\(t\\), the UNet predicts the noise \\(\epsilon_\theta(x_t, t)\\) in the noisy image \\(x_t\\). 
The denoising step should be analytically given by:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z
$$

where:
- \\(\epsilon_\theta(x_t, t)\\) is the UNet’s predicted noise,
- \\(\alpha_t\\) and \\(\bar{\alpha}_t\\) are noise scheduling terms,

but the UNet is  directly trained to minimize the mean squared error (MSE) between the predicted and true noise:

$$
\mathcal{L} = \mathbb{E}_{t,x_0,\epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$


The UNet is chosen for noise prediction due to its ability to capture multi-scale features. Key components include:

- **Encoder-Decoder Structure**:  
  - The encoder downsamples the noisy input \\(x_t\)\ to extract high-level features.  
  - The decoder upsamples while refining features to reconstruct the noise map.  

- **Skip Connections**:  
  Preserve spatial details lost during downsampling, crucial for accurate noise estimation.

- **Timestep Conditioning**:  
  The current timestep \\(t\\) is embedded (e.g., as a sinusoidal or learned embedding) and injected into the UNet to guide denoising.


To generate an image from noise:
1. Start with pure Gaussian noise \\(x_T \sim \mathcal{N}(0, I)\\).  
2. For each timestep \\(t = T, \dots, 1\\):  
   - Predict noise: \\(\epsilon_\theta(x_t, t)\\).  
   - Apply the reverse update rule (above) to compute \\(x_{t-1}\\).  
3. Final output \\(x_0\\) is the generated image.  

Why UNet is deemed to excel at noise prediction？
- **Multi-Scale Processing**: Noise exists at varying frequencies; UNet’s hierarchical structure captures both coarse and fine noise patterns.  
- **Efficiency**: Skip connections mitigate information loss, unlike plain CNNs.  
- **Flexibility**: Can be conditioned on timesteps, text, or other inputs.  



Now, we can write the rest of the main function to realize the training stuff:
```python
def main
    ####### data prep is already up there.

    ####### model & diffusion
    # net = UNet().to(device) # single-GPU
    net = UNet()  # create bare model
    net = nn.DataParallel(net)  # for multi-GPU
    net = net.to(device)  # move model to GPU

    diff = GaussianDiffusion(timesteps=cfg.timesteps).to(device)  # create diffusion model
    opt  = torch.optim.Adam(net.parameters(), lr=cfg.lr)  # create optimizer
    mse  = nn.MSELoss()  # define loss function

    best = math.inf  # keep track of best validation loss
    for epoch in range(cfg.epochs):  # loop over epochs
        net.train()  # switch to training mode
        pbar = tqdm(dl_train, desc=f"Epoch {epoch+1}/{cfg.epochs}")  # create tqdm progress bar
        for imgs, _ in pbar:  # loop over mini-batches
            imgs = imgs.to(device)  # move data to GPU
            t    = torch.randint(0, cfg.timesteps, (imgs.size(0),), device=device)  # sample random timesteps
            noise = torch.randn_like(imgs)  # generate random noise
            x_t   = diff.q_sample(imgs, t, noise)  # sample x_t according to q(x_t | x_0)
            pred  = net(x_t, t.float()/cfg.timesteps)  # predict noise
            loss  = mse(pred, noise)  # compute loss
            opt.zero_grad(); loss.backward(); opt.step()  # backprop and update params
            pbar.set_postfix(train_loss=loss.item())  # update progress bar with training loss

        # ------ validation ------
        net.eval()  # switch to eval mode
        val_loss = 0.0  # keep track of validation loss
        with torch.no_grad():  # disable gradients
            for imgs, _ in dl_val:  # loop over validation set
                imgs = imgs.to(device)  # move data to GPU
                t    = torch.randint(0, cfg.timesteps, (imgs.size(0),), device=device)  # sample random timesteps
                noise = torch.randn_like(imgs)  # generate random noise
                x_t   = diff.q_sample(imgs, t, noise)  # sample x_t according to q(x_t | x_0)
                pred  = net(x_t, t.float()/cfg.timesteps)  # predict noise
                val_loss += mse(pred, noise).item()  # compute loss
        val_loss /= len(dl_val)  # average over validation set
        print(f"➜  Val loss: {val_loss:.5f}")  # print validation loss

        # checkpoint
        if val_loss < best:  # if validation loss improved
            torch.save(net.state_dict(), cfg.save_path)  # save model
            best = val_loss  # update best validation loss
            print(f"✅  New best model saved ({best:.5f})")  # print message

```


alright！ with checking again the imported packages
```python
import math, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
```

we're finished with the code of training script of DDPM.

And now, I provide you with the inference script:
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

"""
ddpm_infer.py  –  sample images with the trained DDPM
No argparse / argv; edit InferCfg below.
"""

import math, torch, torch.nn as nn
from pathlib import Path
from torchvision.utils import save_image

# ──────────────────────────────────────────
# 1.  Shared classes, config from training
# ──────────────────────────────────────────
from ddpm_train import UNet, GaussianDiffusion, cfg   # keep name of training file

# ──────────────────────────────────────────
# 2.  Inference configuration (EDIT HERE)
# ──────────────────────────────────────────
class InferCfg:
    n_samples = 64               # total images to generate
    out_dir   = "samples"        # folder for the grid PNG
    ckpt_path = cfg.save_path    # uses same path as training
infer = InferCfg()

# ──────────────────────────────────────────
# 3.  Device & model loading
# ──────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net   = UNet()                                # create bare model
state = torch.load(infer.ckpt_path, map_location=device)

# Strip "module." if checkpoint came from DataParallel training
if any(k.startswith("module.") for k in state.keys()):
    state = {k.replace("module.", ""): v for k, v in state.items()}

net.load_state_dict(state)
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)                # multi‑GPU wrapper
net = net.to(device).eval()

diff = GaussianDiffusion(timesteps=cfg.timesteps).to(device)

# ──────────────────────────────────────────
# 4.  Sampling loop
# ──────────────────────────────────────────
@torch.no_grad()
def sample(batch: int):
    """Returns a batch of images in [-1,1]."""
    x = torch.randn(batch, 3, cfg.image_size, cfg.image_size, device=device)
    for t in reversed(range(diff.timesteps)):
        t_batch = torch.full((batch,), t, device=device)
        eps     = net(x, t_batch.float() / diff.timesteps)
        alpha   = diff.alphas[t]
        a_hat   = diff.alpha_hats[t]
        beta    = diff.betas[t]
        noise   = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = (1. / alpha.sqrt()) * (x - (1 - alpha) / a_hat.sqrt() * eps) + beta.sqrt() * noise
    return x.clamp(-1, 1)

# ──────────────────────────────────────────
# 5.  Generate & save
# ──────────────────────────────────────────
Path(infer.out_dir).mkdir(exist_ok=True)
images = sample(infer.n_samples)
save_image((images + 1) / 2,
           f"{infer.out_dir}/sample.png",
           nrow=int(math.sqrt(infer.n_samples)))
print(f"✅  Saved {infer.n_samples} images to {infer.out_dir}/sample.png")

```
