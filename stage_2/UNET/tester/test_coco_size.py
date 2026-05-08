import sys
import os
import torch
from PIL import Image
from torchvision import transforms

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from common.data_utils import load_config
from UNET.models.UNET import UNET

config_path = "../../config_gan_base_internimage.yaml"
checkpoint_path = "Enter path here"

device = "cuda:0"
torch.cuda.set_device(device)

config = load_config(config_path)

G = UNET(config).to(device)

checkpoint = torch.load(checkpoint_path, map_location=device)

if isinstance(checkpoint, dict) and "G_state_dict" in checkpoint:
    G.load_state_dict(checkpoint["G_state_dict"])
else:
    G.load_state_dict(checkpoint)

G.eval()

img_path = "Enter path here"

rgb = Image.open(img_path).convert("RGB")
print("Original image size:", rgb.size)

rgb_tensor = transforms.ToTensor()(rgb).unsqueeze(0).to(device)
print("Input tensor shape:", rgb_tensor.shape)

with torch.no_grad():
    fake_nir = G(rgb_tensor)

print("Output tensor shape:", fake_nir.shape)
print("Arbitrary size test successful.")