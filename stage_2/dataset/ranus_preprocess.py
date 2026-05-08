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
test_A  = OUT_ROOT / "test_A"
test_B  = OUT_ROOT / "test_B"

for d in [train_A, train_B, test_A, test_B]:
    d.mkdir(parents=True, exist_ok=True)

metadata_file = OUT_ROOT / "metadata.csv"
skipped_file  = OUT_ROOT / "skipped.csv"

train_seqs = [f"{i:02d}" for i in range(1, 41)]
test_seqs  = [f"{i:02d}" for i in range(41, 51)]

def get_id(fname):
    match = re.search(r'(\d+)', fname)
    return match.group(1) if match else None

counter = 0
skipped_rows = []
metadata_rows = []

def process_sequences(seq_list, out_A, out_B, split_name):
    global counter

    for seq in seq_list:

        rgb_dir = SRC_ROOT / "RGB" / seq
        nir_dir = SRC_ROOT / "NIR" / seq

        # Build NIR lookup
        nir_dict = {}
        for f in nir_dir.iterdir():
            idx = get_id(f.name)
            if idx:
                nir_dict[idx] = f

        # Process RGB files
        for rgb_file in tqdm(list(rgb_dir.iterdir()), desc=f"Seq {seq}"):

            idx = get_id(rgb_file.name)

            # ONLY skip if pair missing
            if not idx or idx not in nir_dict:
                skipped_rows.append([
                    rgb_file.name,
                    "missing_nir_pair",
                    seq,
                    split_name
                ])
                continue

            nir_file = nir_dict[idx]

            new_name = f"{counter:06d}.png"

            # Direct copy — no validation checks
            shutil.copy2(rgb_file, out_A / new_name)
            shutil.copy2(nir_file, out_B / new_name)

            # Try to read size (optional, never skip)
            try:
                w, h = Image.open(rgb_file).size
            except:
                w, h = -1, -1

            metadata_rows.append([
                new_name,
                "RANUS",
                split_name,
                seq,
                rgb_file.name,
                w,
                h
            ])

            counter += 1

process_sequences(train_seqs, train_A, train_B, "train")
process_sequences(test_seqs,  test_A,  test_B,  "test")

# Save metadata
with open(metadata_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename","dataset","split","sequence",
        "original_name","width","height"
    ])
    writer.writerows(metadata_rows)

# Save skipped
with open(skipped_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename","reason","sequence","split"])
    writer.writerows(skipped_rows)

print("\n====================")
print(f"Valid pairs saved: {counter}")
print(f"Skipped (missing pairs only): {len(skipped_rows)}")
print("====================")
