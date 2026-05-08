import sys
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from common.data_utils import load_config
from UNET.models.UNET import *
from UNET.tester.unet_test_modules_imagenet import *


# =====================================================
# CONFIG
# =====================================================

CONFIG_PATH = '../../config_gan_base_internimage.yaml'

CHECKPOINT_PATH = 'Enter path here'

IMAGENET_DIR = 'Enter path here'

BATCH_SIZE = 16
NUM_WORKERS = 8

device = 'cuda:0'
torch.cuda.set_device(device)


# =====================================================
# DATASET
# =====================================================
class ImageNetDataset(Dataset):

    def __init__(self, root_dir):

        self.files = sorted(
            glob.glob(os.path.join(root_dir, "*/*.JPEG"))
        )

        self.transform = Compose([
            Resize((256, 256)),
            ToTensor(),
            Normalize([0.5]*3, [0.5]*3)
        ])

        print("Total images found:", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, path


# =====================================================
# MAIN
# =====================================================
def main():

    config = load_config(CONFIG_PATH)

    G = UNET(config).to(device)

    print("Loading checkpoint...")
    # state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    # G.load_state_dict(state_dict)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    if "G_state_dict" in checkpoint:
       G.load_state_dict(checkpoint["G_state_dict"])
    else:
       G.load_state_dict(checkpoint)
    print("Checkpoint loaded")

    dataset = ImageNetDataset(IMAGENET_DIR)

    test_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_imagenet(config, G, test_loader, device)


if __name__ == "__main__":
    main()