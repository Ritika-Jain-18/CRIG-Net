import os
import sys
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
from torchvision.transforms import Compose, Normalize
import torchvision.utils as vutils

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from common.data_utils import load_config
from UNET.models.UNET import UNET


def load_generator(config_path, checkpoint_path, device):
    config = load_config(config_path)

    G = UNET(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "G_state_dict" in checkpoint:
        print("Detected full checkpoint format")
        G.load_state_dict(checkpoint["G_state_dict"])
    else:
        print("Detected old checkpoint format")
        G.load_state_dict(checkpoint)

    G.eval()
    print("Model loaded successfully")
    return G


def generate_folder(model, input_dir, output_dir, device, image_size):
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5)),
    ])

    reverse_transform = Compose([
        Normalize(mean=(-1, -1, -1), std=(2, 2, 2))
    ])

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"Found {len(image_files)} images in {input_dir}")

    with torch.no_grad():
        for fname in tqdm(image_files, desc=f"Generating NIR for {os.path.basename(input_dir)}"):
            img_path = os.path.join(input_dir, fname)

            rgb = Image.open(img_path).convert("RGB")
            rgb_tensor = transform(rgb).unsqueeze(0).to(device)

            fake_nir = model(rgb_tensor)
            fake_nir = reverse_transform(fake_nir)

            save_path = os.path.join(output_dir, fname)

            # Save using same filename so COCO annotations remain valid
            vutils.save_image(fake_nir, save_path, nrow=1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
        default="Enter path here"
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="Enter path here"
    )

    parser.add_argument(
        "--input_root",
        type=str,
        default="Enter path here"
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="Enter path here"
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val2017"],
        help="Example: val2017 train2017"
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Use the same image size used during generator training"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0"
    )

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.set_device(device)

    model = load_generator(args.config_path, args.checkpoint_path, device)

    for split in args.splits:
        input_dir = os.path.join(args.input_root, split)
        output_dir = os.path.join(args.output_root, split)

        generate_folder(
            model=model,
            input_dir=input_dir,
            output_dir=output_dir,
            device=device,
            image_size=args.image_size
        )

    print("COCO-NIR generation completed.")


if __name__ == "__main__":
    main()