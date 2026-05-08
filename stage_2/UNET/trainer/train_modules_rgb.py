import os
import sys
import yaml
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torchvision.utils as vutils
from datetime import datetime
from torchvision.transforms import Compose, Normalize
from PIL import Image
import kornia

# ==========================================================
# PERFORMANCE BOOST (SAFE)
# ==========================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from common.data_utils import *


def denormalize(tensor):
    return tensor * 0.5 + 0.5


# ==========================================================
# IMAGE GENERATION FUNCTION
# ==========================================================
def generate_images(base_path, model, device, epoch, num_images_per_class, rgb_nir_pairs):
    model.eval()
    latents = []
    original_rgbs = []
    original_nirs = []

    for rgb_img, nir_img in rgb_nir_pairs:
        rgb_img = rgb_img.unsqueeze(0).to(device)
        nir_img = nir_img.unsqueeze(0).to(device)

        with torch.no_grad():
            generated_nir = model(rgb_img)

        latents.append(generated_nir)
        original_rgbs.append(rgb_img)
        original_nirs.append(nir_img)

    gathered_latents = [latent.to("cpu") for latent in latents]
    gathered_original_rgbs = [rgb.to("cpu") for rgb in original_rgbs]
    gathered_original_nirs = [nir.to("cpu") for nir in original_nirs]

    concatenated_latents = torch.cat(gathered_latents, dim=0)
    concatenated_original_rgbs = torch.cat(gathered_original_rgbs, dim=0)
    concatenated_original_nirs = torch.cat(gathered_original_nirs, dim=0)

    reverse_transform_tensor = Compose([Normalize(mean=(-1, -1, -1), std=(2, 2, 2))])

    sample_images = reverse_transform_tensor(concatenated_latents)
    original_images = reverse_transform_tensor(concatenated_original_rgbs)
    original_images2 = reverse_transform_tensor(concatenated_original_nirs)

    comparison_images = torch.cat((original_images, original_images2, sample_images), dim=0)

    vutils.save_image(
        comparison_images,
        f"{base_path}/image_results/epoch_{epoch+1}.png",
        nrow=num_images_per_class
    )


# ==========================================================
# MAIN TRAIN FUNCTION
# ==========================================================
def train(config, G, D, optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D,
          train_loader, test_loader, device, scaler, resume_path=None):

    print(count_parameters(G))

    # ==========================================================
    # RESUME LOGIC (ADDED)
    # ==========================================================
    start_epoch = 0

    # scaler = GradScaler()

    if resume_path is not None and os.path.exists(resume_path):
        print(f"\n🔄 Resuming from checkpoint: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=device)

        # NEW FULL CHECKPOINT
        if "G_state_dict" in checkpoint:
            G.load_state_dict(checkpoint['G_state_dict'])
            D.load_state_dict(checkpoint['D_state_dict'])

            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

            lr_scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
            lr_scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])

            scaler.load_state_dict(checkpoint['scaler_state_dict'])

            start_epoch = checkpoint['epoch'] + 1
            print(f"✅ Fully resumed from epoch {start_epoch}")

        # OLD CHECKPOINT (only G weights)
        else:
            G.load_state_dict(checkpoint)
            start_epoch = config.get("resume_epoch", 0)
            print(f"⚠ Old checkpoint — continuing from epoch {start_epoch}")

    # ==========================================================
    # BASE PATH HANDLING
    # ==========================================================
    # # OLD CODE (always created new folder)
    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # base_path = "train_" + current_time

    # NEW: reuse folder when resuming
    if resume_path is not None and os.path.exists(resume_path):
        base_path = os.path.dirname(os.path.dirname(resume_path))
        print("Using existing base path:", base_path)
    else:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = "train_" + current_time
    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # base_path = "train_" + current_time

    os.makedirs(base_path, exist_ok=True)
    os.makedirs(base_path + "/image_results", exist_ok=True)
    os.makedirs(base_path + "/checkpoints", exist_ok=True)

    with open(base_path + '/train_config.yaml', 'w') as f:
        f.write(f"# {count_parameters(G)}\n")
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    print('base path:', base_path)

    train_epochs = config['loader']['train']['epoch']
    num_images_per_class = 5

    with open(f"{base_path}/epoch_losses.txt", "a") as file:

        # ==========================================================
        # FIXED LOOP (START FROM RESUME EPOCH)
        # ==========================================================
        for epoch in range(start_epoch, train_epochs):

            G.train()
            D.train()

            progress_bar = tqdm(train_loader,
                                desc=f"Epoch [{epoch+1}/{train_epochs}]",
                                unit="batch")

            epoch_loss = 0
            epoch_ssim = 0

            for rgb, nir in progress_bar:
                rgb = rgb.to(device)
                nir = nir.to(device)

                with autocast():
                    generated_nir = G(rgb)
                    losses = D(rgb, nir, generated_nir)

                    loss_D = (losses['loss_D_fake'] + losses['loss_D_real']) * 0.5
                    loss_G = (
                        losses['loss_G_GAN']
                        + losses.get('loss_G_GAN_Feat', 0)
                        + losses.get('loss_G_VGG', 0)
                        + losses.get('loss_G_SSIM', 0)
                    )

                optimizer_G.zero_grad()
                scaler.scale(loss_G).backward()
                scaler.step(optimizer_G)

                optimizer_D.zero_grad()
                scaler.scale(loss_D).backward()
                scaler.step(optimizer_D)

                scaler.update()

                progress_bar.set_postfix({
                    "Loss G": loss_G.item(),
                    "Loss D": loss_D.item(),
                })

                epoch_loss += loss_G.item()
                epoch_ssim += losses.get('loss_G_SSIM', 0).item()

            avg_loss = epoch_loss / len(train_loader)
            avg_ssim = epoch_ssim / len(train_loader)

            file.write(f"Epoch {epoch+1}/{train_epochs}, Loss: {avg_loss}, SSIM: {avg_ssim}\n")
            file.flush()

            # ==========================================================
            # SAVE CHECKPOINT
            # ==========================================================
            checkpoint = {
                'epoch': epoch,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'scheduler_G_state_dict': lr_scheduler_G.state_dict(),
                'scheduler_D_state_dict': lr_scheduler_D.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }

            if (epoch + 1) % 20 == 0:
                torch.save(checkpoint,
                           f"{base_path}/checkpoints/checkpoint_epoch_{epoch+1}.pt")
                print(f"✅ Full checkpoint saved at epoch {epoch+1}")

            torch.save(checkpoint, f"{base_path}/checkpoints/latest.pt")

            # ======================================================
            # SAVE SAMPLE IMAGES
            # ======================================================
            if (epoch + 1) % 20 == 0:
                num_images = min(num_images_per_class, len(test_loader.dataset))
                samples = random.sample(list(test_loader.dataset), num_images)
                generate_images(base_path, G, device, epoch, num_images, samples)

            lr_scheduler_G.step()
            lr_scheduler_D.step()