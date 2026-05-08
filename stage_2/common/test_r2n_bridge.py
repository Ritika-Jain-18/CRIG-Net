import torch
import sys

# Add AVFF repo path to import R2N
sys.path.append("Enter path here")

from models.cross_modal.r2n import R2N
from r2n_bridge import R2NBridge


# Path to clean R2N weights
WEIGHT_PATH = "Enter path here"


def main():
    print("Initializing R2N...")

    r2n = R2N()
    state = torch.load(WEIGHT_PATH, map_location="cpu")
    r2n.load_state_dict(state)

    print("Creating R2NBridge...")
    bridge = R2NBridge(r2n)

    # Simulate Pix2Next encoder output
    dummy_input = torch.randn(2, 512, 32, 32)

    print("Input shape:", dummy_input.shape)

    output = bridge(dummy_input)

    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()
