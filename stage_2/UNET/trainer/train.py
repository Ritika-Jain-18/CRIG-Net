import sys
import os
import numpy as np

import torch
import torch.nn as nn
from safetensors.torch import load_file

# =========================================
# AMP IMPORTS (ADDED FOR SPEEDUP)
# =========================================
from torch.cuda.amp import GradScaler  # ← AMP scaler

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from common.data_utils import *
from UNET.models.UNET import *
from UNET.models.discriminator import *
from UNET.trainer.train_modules_rgb import *


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


config_path = '../../config_gan_base_internimage.yaml'
config = load_config(config_path)
epochs = config['loader']['train']['epoch']
loaders = build_loader(config)

print('train_loader len:', len(loaders['train']))
print('test_loader len:', len(loaders['test']))


# =========================================
# RESUME SETTINGS
# =========================================
resume_path = "Enter path here"

# manual_start_epoch = 820
# resume_path = None
manual_start_epoch = 1

# =========================================
# ADDED: pass resume epoch to trainer
# =========================================
config["resume_epoch"] = manual_start_epoch


# =========================================
# DEVICE SETUP
# =========================================
device = 'cuda:0'
torch.cuda.set_device(device)


# =========================================
# BUILD MODELS
# =========================================
G = UNET(config).to(device)
D = Discriminator().to(device)

print(G)


# =========================================
# WEIGHT INITIALIZATION
# =========================================
# OLD CODE (would overwrite loaded weights)
# G.apply(weights_init)

if not os.path.exists(resume_path):
    G.apply(weights_init)

# # Always initialize from scratch
# G.apply(weights_init)
# D.apply(weights_init)


# =========================================
# OPTIMIZERS
# =========================================
trainable_params = filter(lambda p: p.requires_grad, G.parameters())

optimizer_G = torch.optim.Adam(
    trainable_params,
    lr=1e-4,
    betas=(0.5, 0.999)
)

optimizer_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))


# =========================================
# LR SCHEDULERS
# =========================================
from diffusers.optimization import get_cosine_schedule_with_warmup

lr_scheduler_G = get_cosine_schedule_with_warmup(
    optimizer=optimizer_G,
    num_warmup_steps=1000,
    num_training_steps=(len(loaders['train']) * epochs),
)

lr_scheduler_D = get_cosine_schedule_with_warmup(
    optimizer=optimizer_D,
    num_warmup_steps=1000,
    num_training_steps=(len(loaders['train']) * epochs),
)


# =========================================
# AMP SCALER
# =========================================
scaler = GradScaler()


# =========================================
# LOAD CHECKPOINT
# =========================================
start_epoch = 0

if os.path.exists(resume_path):
    print(f"\n🔄 Loading checkpoint from: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device)

    # OLD CHECKPOINT
    if isinstance(checkpoint, dict) and "G_state_dict" not in checkpoint:
        print("⚠ OLD checkpoint detected (only Generator weights).")

        G.load_state_dict(checkpoint)

        start_epoch = manual_start_epoch
        print(f"⚠ Optimizer & scheduler reset.")
        print(f"⚠ Continuing training from epoch {start_epoch}")

    # NEW FULL CHECKPOINT
    else:
        print("✅ FULL checkpoint detected.")

        G.load_state_dict(checkpoint['G_state_dict'])
        D.load_state_dict(checkpoint['D_state_dict'])

        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

        lr_scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        lr_scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])

        scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Resuming from epoch {start_epoch}")

else:
    print("\n🚀 No checkpoint found — starting fresh training.")


# =========================================
# START TRAINING
# =========================================
train(
    config,
    G,
    D,
    optimizer_G,
    optimizer_D,
    lr_scheduler_G,
    lr_scheduler_D,
    loaders['train'],
    loaders['test'],
    device,
    scaler,
    resume_path=resume_path
)