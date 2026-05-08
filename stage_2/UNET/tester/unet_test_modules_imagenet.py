import os
import sys
import torch
import torchvision.utils as vutils
from torchvision.transforms import Compose, Normalize
from datetime import datetime
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)


# =====================================================
# GENERATE + SAVE (PRESERVE CLASS FOLDERS)
# =====================================================
def generate_image_imagenet(base_path, model, device, rgb_img, img_paths):

    model.eval()
    rgb_img = rgb_img.to(device)

    with torch.no_grad():
        generated_nir = model(rgb_img)

    reverse_transform = Compose([
        Normalize(mean=(-1, -1, -1), std=(2, 2, 2))
    ])

    rgb_vis = reverse_transform(rgb_img)
    nir_vis = reverse_transform(generated_nir)

    for i in range(rgb_vis.size(0)):

        full_path = img_paths[i]

        class_name = os.path.basename(os.path.dirname(full_path))
        filename = os.path.basename(full_path).replace(".JPEG", "")

        save_folder = os.path.join(base_path, "image_results", class_name)
        os.makedirs(save_folder, exist_ok=True)

        vutils.save_image(
            rgb_vis[i],
            os.path.join(save_folder, f"{filename}_rgb.png"),
            nrow=1
        )

        vutils.save_image(
            nir_vis[i],
            os.path.join(save_folder, f"{filename}_nir.png"),
            nrow=1
        )


# =====================================================
# TEST FUNCTION
# =====================================================
def test_imagenet(config, G, test_loader, device):

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = "test_imagenet-1k-train-sub200_" + current_time

    os.makedirs(base_path, exist_ok=True)
    os.makedirs(base_path + "/image_results", exist_ok=True)

    print("Saving results to:", base_path)

    G.eval()

    for rgb, paths in tqdm(test_loader, desc="Generating NIR"):
        generate_image_imagenet(base_path, G, device, rgb, paths)

    print("✅ DONE — NIR ImageNet generated")