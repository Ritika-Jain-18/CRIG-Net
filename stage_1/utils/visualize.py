import os
import torch
import torchvision.utils as vutils

from utils.misc import unpatchify


def denorm_rgb(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
    return x * std + mean


def denorm_nir(x):
    return x * 0.5 + 0.5


def save_visualization(
    save_dir,
    epoch,
    rgb_input,
    nir_input,
    rgb_rec_tokens,
    nir_rec_tokens,
):
    os.makedirs(save_dir, exist_ok=True)

    # ========================
    # UNPATCHIFY
    # ========================
    rgb_rec = unpatchify(rgb_rec_tokens, 16, 3)
    nir_rec = unpatchify(nir_rec_tokens, 16, 1)

    # ========================
    # DENORMALIZE
    # ========================
    rgb_input = denorm_rgb(rgb_input)
    nir_input = denorm_nir(nir_input)

    rgb_rec = torch.clamp(rgb_rec, 0, 1)
    nir_rec = torch.clamp(nir_rec, 0, 1)

    # repeat nir to 3 channels for display
    nir_input = nir_input.repeat(1,3,1,1)
    nir_rec = nir_rec.repeat(1,3,1,1)

    # ========================
    # BUILD 2×2 GRID
    # ========================
    row1 = torch.cat([rgb_input, rgb_rec], dim=3)
    row2 = torch.cat([nir_input, nir_rec], dim=3)

    grid = torch.cat([row1, row2], dim=2)

    # ========================
    # SAVE
    # ========================
    latest = f"{save_dir}/latest.png"
    vutils.save_image(grid, latest)

    # Save permanent every 50 epochs
    if epoch % 50 == 0:
        vutils.save_image(grid, f"{save_dir}/epoch_{epoch}.png")