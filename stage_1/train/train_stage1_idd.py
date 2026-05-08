import os
import time
import glob
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from models.stage1_model import Stage1AVFF
from models.discriminators.rgb_disc import RGBPatchDiscriminator
from models.discriminators.nir_disc import NIRPatchDiscriminator

from utils.misc import patchify, gather_masked_patches

from losses.reconstruction import masked_reconstruction_loss
from losses.contrastive import contrastive_loss
from losses.adversarial import (
    wgan_generator_loss,
    wgan_discriminator_loss,
)

# =====================================================
# SPEED FLAGS
# =====================================================
torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# DATASET
# =====================================================
class PairedFolderDataset(Dataset):
    def __init__(self, root):
        self.rgb_paths = sorted(glob.glob(os.path.join(root, "train_A", "*")))
        self.nir_paths = sorted(glob.glob(os.path.join(root, "train_B", "*")))

        assert len(self.rgb_paths) == len(self.nir_paths), "Mismatch pairs"

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_paths[idx]).convert("RGB")
        nir = Image.open(self.nir_paths[idx]).convert("L")


        rgb = self.transform(rgb)
        nir = self.transform(nir)

        return rgb, nir


# =====================================================
# DATA LOADING
# =====================================================
DATA_ROOT = "Enter path here"

dataset = PairedFolderDataset(DATA_ROOT)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

print(f"Loaded {len(dataset)} image pairs.")

# =====================================================
# MODELS
# =====================================================
model = Stage1AVFF().to(device)
rgb_disc = RGBPatchDiscriminator().to(device)
nir_disc = NIRPatchDiscriminator().to(device)

# =====================================================
# OPTIMIZERS
# =====================================================
lr = 1e-4

opt_G = torch.optim.AdamW(model.parameters(), lr=lr)
opt_D = torch.optim.AdamW(
    list(rgb_disc.parameters()) + list(nir_disc.parameters()),
    lr=lr,
)

# =====================================================
# COSINE SCHEDULER WITH WARMUP
# =====================================================
def get_lr_lambda(warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926)))
    return lr_lambda

num_epochs = 1000
warmup_epochs = 20

scheduler = torch.optim.lr_scheduler.LambdaLR(
    opt_G,
    lr_lambda=get_lr_lambda(warmup_epochs, num_epochs)
)

# =====================================================
# LOSS WEIGHTS
# =====================================================
lambda_rec = 1.0
lambda_c = 0.01
lambda_adv = 0.1

# =====================================================
# LOGGING
# =====================================================
os.makedirs("checkpoints_idd", exist_ok=True)
writer = SummaryWriter("logs/stage1_idd")
scaler = GradScaler()

# =====================================================
# TRAINING LOOP
# =====================================================
best_loss = float("inf")
global_step = 0

for epoch in range(num_epochs):

    epoch_start = time.time()
    running_loss = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch}", dynamic_ncols=True)

    for rgb, nir in progress_bar:

        rgb = rgb.to(device, non_blocking=True)
        nir = nir.to(device, non_blocking=True)

        # ================= GENERATOR =================
        with autocast():
            out = model(rgb, nir)

            rgb_target = patchify(rgb)
            nir_target = patchify(nir)

            loss_rgb = masked_reconstruction_loss(
                out["rgb_rec"], rgb_target, out["rgb_mask"]
            )
            loss_nir = masked_reconstruction_loss(
                out["nir_rec"], nir_target, out["nir_mask"]
            )

            loss_c = contrastive_loss(
                out["rgb_feat"], out["nir_feat"]
            )

            rgb_fake = gather_masked_patches(
                out["rgb_rec"], out["rgb_mask"]
            )
            nir_fake = gather_masked_patches(
                out["nir_rec"], out["nir_mask"]
            )

            loss_adv_G = (
                wgan_generator_loss(rgb_disc(rgb_fake))
                + wgan_generator_loss(nir_disc(nir_fake))
            )

            loss_G = (
                lambda_rec * (loss_rgb + loss_nir)
                + lambda_c * loss_c
                + lambda_adv * loss_adv_G
            )

        opt_G.zero_grad()
        scaler.scale(loss_G).backward()
        scaler.step(opt_G)
        scaler.update()

        # ================= DISCRIMINATOR =================
        with torch.no_grad():
            rgb_real = gather_masked_patches(rgb_target, out["rgb_mask"])
            nir_real = gather_masked_patches(nir_target, out["nir_mask"])

        rgb_real_score = rgb_disc(rgb_real.float())
        nir_real_score = nir_disc(nir_real.float())

        rgb_fake_score = rgb_disc(rgb_fake.detach().float())
        nir_fake_score = nir_disc(nir_fake.detach().float())

        loss_D = (
            wgan_discriminator_loss(rgb_real_score, rgb_fake_score)
            + wgan_discriminator_loss(nir_real_score, nir_fake_score)
        )

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ================= LOGGING =================
        writer.add_scalar("loss/G", loss_G.item(), global_step)
        writer.add_scalar("loss/D", loss_D.item(), global_step)

        running_loss += loss_G.item()
        global_step += 1

        progress_bar.set_postfix(loss=f"{loss_G.item():.3f}")

    scheduler.step()

    avg_loss = running_loss / len(loader)
    epoch_time = (time.time() - epoch_start) / 60

    print(f"\nEpoch {epoch} | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f} min")

    # Save best
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "checkpoints_idd/best.pt")

    # periodic save
    if epoch % 50 == 0:
        torch.save(model.state_dict(), f"checkpoints_idd/epoch_{epoch}.pt")

writer.close()
