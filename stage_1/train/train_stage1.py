import os
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from data.dataset import NIRRGBDataset
from data.transforms import PairedTransform
from utils.build_sampler import build_stage1_sampler
from utils.visualize import save_visualization

from models.stage1_model import Stage1AVFF
from models.discriminators.rgb_disc import RGBPatchDiscriminator
from models.discriminators.nir_disc import NIRPatchDiscriminator

from utils.misc import patchify, gather_masked_patches

from losses.reconstruction import masked_reconstruction_loss
from losses.contrastive import contrastive_loss
from losses.adversarial import wgan_generator_loss, wgan_discriminator_loss


# =====================================================
# SPEED SETTINGS
# =====================================================
torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# PATHS
# =====================================================
CHECKPOINT_DIR = "Enter path here"
LOG_DIR = "logs/stage1"
VIS_DIR = "visuals"
DATA_ROOT = "Enter path here"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

writer = SummaryWriter(LOG_DIR)
scaler = GradScaler()

# =====================================================
# DATA
# =====================================================
dataset = NIRRGBDataset(
    root_dir=DATA_ROOT,
    transform=PairedTransform(),
)

sampler = build_stage1_sampler(
    f"{DATA_ROOT}/metadata.csv",
    alpha=0.5,
)

loader = DataLoader(
    dataset,
    batch_size=64,
    sampler=sampler,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

print(f"Loaded {len(dataset)} pairs.")

# =====================================================
# FIXED SAMPLE FOR VISUALIZATION
# =====================================================
vis_rgb, vis_nir = dataset[0]
vis_rgb = vis_rgb.unsqueeze(0).to(device)
vis_nir = vis_nir.unsqueeze(0).to(device)

# =====================================================
# MODELS
# =====================================================
model = Stage1AVFF().to(device)
rgb_disc = RGBPatchDiscriminator().to(device)
nir_disc = NIRPatchDiscriminator().to(device)

# =====================================================
# OPTIMIZERS
# =====================================================
opt_G = torch.optim.AdamW(model.parameters(), lr=1.5e-4)
opt_D = torch.optim.AdamW(
    list(rgb_disc.parameters()) + list(nir_disc.parameters()),
    lr=1.5e-4,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=500)

# =====================================================
# RESUME TRAINING
# =====================================================
resume_path = f"{CHECKPOINT_DIR}/latest.pt"

start_epoch = 0
best_loss = float("inf")

if os.path.exists(resume_path):
    print(f"\nResuming from {resume_path}")

    checkpoint = torch.load(resume_path, map_location=device)

    model.load_state_dict(checkpoint["model"])
    opt_G.load_state_dict(checkpoint["opt_G"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    best_loss = checkpoint["best_loss"]
    start_epoch = checkpoint["epoch"] + 1

    print(f"Resumed from epoch {start_epoch}\n")
else:
    print("Starting fresh training.")

# =====================================================
# LOSS WEIGHTS
# =====================================================
lambda_rec = 1.0
lambda_c = 0.01
lambda_adv = 0.1

# =====================================================
# TRAINING LOOP
# =====================================================
num_epochs = 500
save_interval = 50
global_step = 0

for epoch in range(start_epoch, num_epochs):

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

            loss_rgb = masked_reconstruction_loss(out["rgb_rec"], rgb_target, out["rgb_mask"])
            loss_nir = masked_reconstruction_loss(out["nir_rec"], nir_target, out["nir_mask"])

            loss_c = contrastive_loss(out["rgb_feat"], out["nir_feat"])

            rgb_fake = gather_masked_patches(out["rgb_rec"], out["rgb_mask"])
            nir_fake = gather_masked_patches(out["nir_rec"], out["nir_mask"])

            rgb_fake_score = rgb_disc(rgb_fake)
            nir_fake_score = nir_disc(nir_fake)

            loss_adv_G = wgan_generator_loss(rgb_fake_score) + wgan_generator_loss(nir_fake_score)

            loss_G = lambda_rec*(loss_rgb+loss_nir) + lambda_c*loss_c + lambda_adv*loss_adv_G

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

        loss_D = wgan_discriminator_loss(rgb_real_score, rgb_fake_score) + \
                 wgan_discriminator_loss(nir_real_score, nir_fake_score)

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        running_loss += loss_G.item()
        global_step += 1

        progress_bar.set_postfix(loss=f"{loss_G.item():.4f}")

    scheduler.step()

    epoch_time = (time.time() - epoch_start) / 60
    avg_loss = running_loss / len(loader)

    print(f"\nEpoch {epoch} | Avg Loss: {avg_loss:.5f} | Time: {epoch_time:.2f} min\n")

    # ================= VISUALIZATION =================
    model.eval()
    with torch.no_grad():
        vis_out = model(vis_rgb, vis_nir)

    save_visualization(VIS_DIR, epoch, vis_rgb, vis_nir,
                       vis_out["rgb_rec"], vis_out["nir_rec"])
    model.train()

    # ================= BEST MODEL =================
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pt")

    # ================= PERIODIC CHECKPOINT =================
    if epoch % save_interval == 0:
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "opt_G": opt_G.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
        }, f"{CHECKPOINT_DIR}/epoch_{epoch}.pt")

    # ================= LATEST CHECKPOINT =================
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "opt_G": opt_G.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_loss": best_loss,
    }, f"{CHECKPOINT_DIR}/latest.pt")

writer.close()