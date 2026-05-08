import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.cross_modal.r2n import R2N

# Path to Stage-1 checkpoint
CKPT_PATH = "Enter path here"

# Where we will save clean R2N weights
SAVE_PATH = "Enter path here"

print("Loading checkpoint...")
ckpt = torch.load(CKPT_PATH, map_location="cpu")

# Extract full state dict
state_dict = ckpt["model"] if "model" in ckpt else ckpt

# Keep only R2N weights
r2n_state = {
    k.replace("r2n.", ""): v
    for k, v in state_dict.items()
    if k.startswith("r2n.")
}

print("Number of R2N params:", len(r2n_state))

# Save clean R2N weights
torch.save(r2n_state, SAVE_PATH)

print(f"✅ Saved clean R2N weights to: {SAVE_PATH}")
