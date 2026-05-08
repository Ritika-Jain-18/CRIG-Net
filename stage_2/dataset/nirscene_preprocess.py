import csv
import shutil
import random
import re
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ==============================
# PATHS
# ==============================
SRC_ROOT = Path("Enter path here")
OUT_ROOT = Path("Enter path here")

train_A = OUT_ROOT / "train_A"
train_B = OUT_ROOT / "train_B"
test_A  = OUT_ROOT / "test_A"
test_B  = OUT_ROOT / "test_B"

for d in [train_A, train_B, test_A, test_B]:
    d.mkdir(parents=True, exist_ok=True)

metadata_file = OUT_ROOT / "metadata.csv"

# ==============================
# REPRODUCIBLE SCENE SPLIT
# ==============================
scene_folders = sorted([d for d in SRC_ROOT.iterdir() if d.is_dir()])

random.seed(42)
random.shuffle(scene_folders)

split_idx = int(0.8 * len(scene_folders))

train_scenes = scene_folders[:split_idx]
test_scenes  = scene_folders[split_idx:]

print("\nTrain scenes:", [s.name for s in train_scenes])
print("Test scenes:", [s.name for s in test_scenes])

# ==============================
# HELPER: extract numeric ID
# ==============================
def get_id(fname):
    match = re.search(r'(\d+)', fname)
    return match.group(1) if match else None

# ==============================
# PROCESSING FUNCTION
# ==============================
counter = 0
skipped = 0
metadata_rows = []

def process_scene(scene_list, out_A, out_B, split_name):
    global counter, skipped

    for scene in scene_list:
        files = list(scene.iterdir())

        # Build NIR dictionary
        nir_dict = {}
        for f in files:
            if "_nir" in f.name.lower():
                idx = get_id(f.name)
                if idx:
                    nir_dict[idx] = f

        # Process RGB
        for f in tqdm(files, desc=f"Processing {scene.name}"):

            if "_rgb" not in f.name.lower():
                continue

            idx = get_id(f.name)

            if not idx or idx not in nir_dict:
                skipped += 1
                continue

            nir_file = nir_dict[idx]

            try:
                rgb_img = Image.open(f).convert("RGB")
                nir_img = Image.open(nir_file).convert("L")
            except:
                skipped += 1
                continue

            if rgb_img.size != nir_img.size:
                skipped += 1
                continue

            # Convert TIFF → PNG automatically via PIL save
            new_name = f"{counter:06d}.png"

            shutil.copy(f, out_A / new_name)
            shutil.copy(nir_file, out_B / new_name)

            metadata_rows.append([
                new_name,
                "NIRScene",
                split_name,
                scene.name,
                f.name,
                rgb_img.size[0],
                rgb_img.size[1]
            ])

            counter += 1

# ==============================
# RUN PROCESSING
# ==============================
process_scene(train_scenes, train_A, train_B, "train")
process_scene(test_scenes,  test_A,  test_B,  "test")

# ==============================
# SAVE METADATA
# ==============================
with open(metadata_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename","dataset","split","scene",
        "original_name","width","height"
    ])
    writer.writerows(metadata_rows)

# ==============================
# SUMMARY
# ==============================
print("\n====================")
print(f"Valid pairs saved: {counter}")
print(f"Skipped pairs: {skipped}")
print("====================")
