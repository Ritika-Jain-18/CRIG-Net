import os
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from models.stage1_model import Stage1AVFF

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

import lpips
from DISTS_pytorch import DISTS
from pytorch_fid import fid_score


# =====================================================
# CONFIG
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = "Enter path here"
CHECKPOINT = "Enter path here"

SAVE_GEN = "generated_nir_idd"
os.makedirs(SAVE_GEN, exist_ok=True)

RESULTS_DIR = "results_idd"
os.makedirs(RESULTS_DIR, exist_ok=True)


# =====================================================
# UNPATCHIFY FUNCTION
# =====================================================
def unpatchify(x, patch_size=16):
    B, N, D = x.shape
    C = D // (patch_size * patch_size)
    h = w = int(N ** 0.5)

    x = x.reshape(B, h, w, patch_size, patch_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4)
    x = x.reshape(B, C, h * patch_size, w * patch_size)

    return x


# =====================================================
# TRANSFORMS
# =====================================================
rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

nir_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# =====================================================
# LOAD MODEL
# =====================================================
print("Loading model...")

model = Stage1AVFF().to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()


# =====================================================
# METRIC MODELS
# =====================================================
print("Loading metric networks...")

lpips_model = lpips.LPIPS(net="alex").to(device)
dists_model = DISTS().to(device)


# =====================================================
# FILE LISTS
# =====================================================
rgb_paths = sorted(glob.glob(f"{DATA_ROOT}/test_A/*"))
nir_paths = sorted(glob.glob(f"{DATA_ROOT}/test_B/*"))

assert len(rgb_paths) == len(nir_paths)

print(f"Found {len(rgb_paths)} test pairs")


# =====================================================
# METRIC STORAGE
# =====================================================
psnr_vals, ssim_vals, rmse_vals = [], [], []
lpips_vals, dists_vals = [], []

# OPTIONAL per-image storage
records = []


# =====================================================
# GENERATION LOOP
# =====================================================
print("\nGenerating NIR images...\n")

with torch.no_grad():
    for rgb_p, nir_p in tqdm(zip(rgb_paths, nir_paths), total=len(rgb_paths)):

        rgb_img = Image.open(rgb_p).convert("RGB")
        nir_gt = Image.open(nir_p).convert("L")

        rgb = rgb_transform(rgb_img).unsqueeze(0).to(device)
        gt = nir_transform(nir_gt).unsqueeze(0).to(device)

        dummy_nir = torch.zeros_like(gt).to(device)

        out = model(rgb, dummy_nir)

        nir_pred = unpatchify(out["nir_rec"]).clamp(0, 1)

        pred_np = nir_pred.squeeze().cpu().numpy()
        gt_np = gt.squeeze().cpu().numpy()

        # Save generated image
        save_img = (pred_np * 255).astype(np.uint8)
        save_path = os.path.join(SAVE_GEN, os.path.basename(nir_p))
        Image.fromarray(save_img).save(save_path)

        # ======================
        # CLASSICAL METRICS
        # ======================
        p = psnr(gt_np, pred_np, data_range=1)
        s = ssim(gt_np, pred_np, data_range=1)
        r = np.sqrt(mean_squared_error(gt_np.flatten(), pred_np.flatten()))

        psnr_vals.append(p)
        ssim_vals.append(s)
        rmse_vals.append(r)

        # ======================
        # LPIPS & DISTS
        # ======================
        gt_3 = gt.repeat(1, 3, 1, 1)
        pred_3 = nir_pred.repeat(1, 3, 1, 1)

        lp = lpips_model(gt_3, pred_3).item()
        ds = dists_model(gt_3, pred_3).item()

        lpips_vals.append(lp)
        dists_vals.append(ds)

        # Save per-image record
        records.append([os.path.basename(rgb_p), p, s, r, lp, ds])


# =====================================================
# FID COMPUTATION
# =====================================================
print("\nComputing FID...\n")

fid = fid_score.calculate_fid_given_paths(
    [f"{DATA_ROOT}/test_B", SAVE_GEN],
    batch_size=16,
    device=device,
    dims=2048
)


# =====================================================
# FINAL RESULTS
# =====================================================
print("\n================ FINAL RESULTS ================\n")

results = {
    "PSNR": np.mean(psnr_vals),
    "SSIM": np.mean(ssim_vals),
    "RMSE": np.mean(rmse_vals),
    "LPIPS": np.mean(lpips_vals),
    "DISTS": np.mean(dists_vals),
    "FID": fid,
    "STD": np.std(psnr_vals)
}

for k, v in results.items():
    print(f"{k}: {v:.4f}")


# =====================================================
# SAVE RESULTS
# =====================================================
print("\nSaving results...\n")

txt_path = os.path.join(RESULTS_DIR, "generation_results.txt")
csv_path = os.path.join(RESULTS_DIR, "generation_results.csv")
per_img_path = os.path.join(RESULTS_DIR, "per_image_metrics.csv")

# Save summary TXT
with open(txt_path, "w") as f:
    for k, v in results.items():
        f.write(f"{k}: {v:.4f}\n")

# Save summary CSV
pd.DataFrame([results]).to_csv(csv_path, index=False)

# Save per-image metrics
df = pd.DataFrame(records, columns=["filename", "PSNR", "SSIM", "RMSE", "LPIPS", "DISTS"])
df.to_csv(per_img_path, index=False)

print("Saved:")
print(" •", txt_path)
print(" •", csv_path)
print(" •", per_img_path)
print("\nGenerated images saved in:", SAVE_GEN)
