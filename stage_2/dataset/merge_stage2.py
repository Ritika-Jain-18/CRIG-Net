import csv
import shutil
from pathlib import Path
from PIL import Image

# =====================================
# DATASETS TO MERGE (Pix2Next format)
# =====================================
DATASETS = {
    "IDD_AW": Path("Enter path here"),
    "NIRScene": Path("Enter path here"),
    "RANUS": Path("Enter path here"),
}

# =====================================
# OUTPUT MASTER DATASET
# =====================================
OUT_ROOT = Path("Enter path here")

OUT_TRAIN_A = OUT_ROOT / "train_A"
OUT_TRAIN_B = OUT_ROOT / "train_B"
OUT_TEST_A  = OUT_ROOT / "test_A"
OUT_TEST_B  = OUT_ROOT / "test_B"

for d in [OUT_TRAIN_A, OUT_TRAIN_B, OUT_TEST_A, OUT_TEST_B]:
    d.mkdir(parents=True, exist_ok=True)

metadata_file = OUT_ROOT / "metadata.csv"

# =====================================
# MERGING
# =====================================
counter = 0

with open(metadata_file, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "filename",
        "dataset",
        "split",
        "original_file",
        "width",
        "height"
    ])

    # iterate datasets
    for dataset_name, root in DATASETS.items():

        print(f"\n=== Merging {dataset_name} ===")

        # handle both splits
        splits = [
            ("train", "train_A", "train_B", OUT_TRAIN_A, OUT_TRAIN_B),
            ("test",  "test_A",  "test_B",  OUT_TEST_A,  OUT_TEST_B)
        ]

        for split_name, A_dir, B_dir, out_A, out_B in splits:

            src_A = root / A_dir
            src_B = root / B_dir

            files = sorted(src_A.glob("*.png"))

            print(f"{dataset_name} {split_name}: {len(files)} pairs")

            for rgb_path in files:

                fname = rgb_path.name
                nir_path = src_B / fname

                if not nir_path.exists():
                    continue

                # verify alignment
                try:
                    rgb_img = Image.open(rgb_path)
                    nir_img = Image.open(nir_path)
                except:
                    continue

                if rgb_img.size != nir_img.size:
                    continue

                new_name = f"{counter:07d}.png"

                shutil.copy2(rgb_path, out_A / new_name)
                shutil.copy2(nir_path, out_B / new_name)

                writer.writerow([
                    new_name,
                    dataset_name,
                    split_name,
                    fname,
                    rgb_img.size[0],
                    rgb_img.size[1]
                ])

                counter += 1

                if counter % 5000 == 0:
                    print(f"Total merged: {counter}")

# =====================================
# SUMMARY
# =====================================
print("\n====================")
print(f"FINAL TOTAL PAIRS: {counter}")
print("====================")
