import sys
import os
import numpy as np
import torch

from safetensors.torch import load_file

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from common.data_utils import *
from UNET.models.UNET import *
from UNET.tester.unet_test_modules_rgb import *
from accelerate import Accelerator, DistributedDataParallelKwargs


config_path = '../../config_gan_base_internimage.yaml'
config = load_config(config_path)
loaders = build_loader(config)
print('test_loader len:', len(loaders['test']))


# ==========================================================
# DEVICE SETUP
# ==========================================================
device = 'cuda:0'
torch.cuda.set_device(device)


# ==========================================================
# BUILD MODEL
# ==========================================================
G = UNET(config).to(device)
print(G)


# ==========================================================
# LOAD CHECKPOINT (UPDATED TO SUPPORT OLD + NEW FORMAT)
# ==========================================================
checkpoint_path = 'Enter path here'

print(f"\n🔄 Loading checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device)

# ----------------------------------------------------------
# NEW FULL CHECKPOINT FORMAT
# ----------------------------------------------------------
if isinstance(checkpoint, dict) and "G_state_dict" in checkpoint:
    print("✅ Detected NEW full checkpoint format")
    G.load_state_dict(checkpoint["G_state_dict"])

# ----------------------------------------------------------
# OLD CHECKPOINT FORMAT (only G weights)
# ----------------------------------------------------------
else:
    print("⚠ Detected OLD checkpoint format")
    G.load_state_dict(checkpoint)


print("✅ Model weights loaded successfully!")


# ==========================================================
# RUN TEST
# ==========================================================
test_loader = loaders['test']
test(config, G, test_loader, device)