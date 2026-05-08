import os
import csv
import shutil
import re
from pathlib import Path
from PIL import Image
from tqdm import tqdm

SRC_ROOT = Path("Enter path here")
OUT_ROOT = Path("Enter path here")

train_A = OUT_ROOT / "train_A"
train_B = OUT_ROOT / "train_B"
test_A = OUT_ROOT / "test_A"
test_B = OUT_ROOT / "test_B"

for d in [train_A, train_B, test_A, test_B]:
    d.mkdir(parents=True, exist_ok=True)

metadata_file = OUT_ROOT / "metadata.csv"
WEATHERS = ["FOG", "LOWLIGHT", "RAIN", "SNOW"]

def get_id(fname):
    match = re.search(r'(\d+)', fname)
    return match.group(1) if match else None

counter = 0
skipped = 0
metadata_rows = []

print("Preparing IDD-AW for Pix2Next...")

for split in ["train", "val"]:

    split_root = SRC_ROOT / split

    for weather in WEATHERS:

        rgb_root = split_root / weather / "rgb"
        nir_root = split_root / weather / "nir"

        if not rgb_root.exists():
            continue

        for seq in sorted(rgb_root.iterdir()):

            nir_seq = nir_root / seq.name

            # ---------- Build NIR dictionary ----------
            nir_dict = {}
            for f in nir_seq.iterdir():
                if f.suffix.lower() != ".png":
                    continue
                idx = get_id(f.name)
                if idx:
                    nir_dict[idx] = f

            # ---------- Process RGB ----------
            for rgb_file in tqdm(list(seq.iterdir()), desc=f"{split}-{weather}-{seq.name}"):

                if rgb_file.suffix.lower() != ".png":
                    continue

                idx = get_id(rgb_file.name)

                if not idx or idx not in nir_dict:
                    skipped += 1
                    continue

                nir_file = nir_dict[idx]

                try:
                    rgb_img = Image.open(rgb_file)
                    nir_img = Image.open(nir_file)
                except:
                    skipped += 1
                    continue

                if rgb_img.size != nir_img.size:
                    skipped += 1
                    continue

                new_name = f"{counter:06d}.png"

                if split == "train":
                    out_rgb = train_A / new_name
                    out_nir = train_B / new_name
                else:
                    out_rgb = test_A / new_name
                    out_nir = test_B / new_name

                shutil.copy2(rgb_file, out_rgb)
                shutil.copy2(nir_file, out_nir)

                metadata_rows.append([
                    new_name,
                    "IDD_AW",
                    "train" if split == "train" else "test",
                    weather,
                    seq.name,
                    rgb_file.name,
                    rgb_img.size[0],
                    rgb_img.size[1]
                ])

                counter += 1

# ---------- Save metadata ----------
with open(metadata_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename","dataset","split","weather",
        "sequence","original_name","width","height"
    ])
    writer.writerows(metadata_rows)

print("\n====================")
print(f"Valid pairs saved: {counter}")
print(f"Skipped pairs: {skipped}")
print("====================")
