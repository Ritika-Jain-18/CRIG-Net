# CRIG-Net

CRIG-Net (Cross-modal Representation Integrated Generation Network) is a two-stage framework for RGB-to-NIR image translation.

This repository currently contains the implementation of:

* Stage 1: Cross-modal representation learning between RGB and NIR modalities
* Stage 2: RGB → NIR image generation framework

The framework is designed for generating synthetic Near-Infrared (NIR) images from RGB inputs for cross-modal and spectral-domain vision research.

---

# Repository Structure

```text
CRIG-Net/
├── stage_1/
├── stage_2/
└── README.md
```

---

# Installation

```bash
conda create -n crignet python=3.10
conda activate crignet
```

Install dependencies:

```bash
pip install -r stage_1/requirements.txt
pip install -r stage_2/working_requirements.txt
```

---

# Training

## Stage 1

```bash
python stage_1/train/train_stage1.py
```

## Stage 2

```bash
python stage_2/UNET/trainer/train.py
```

---

# Inference

Generate NIR images:

```bash
python stage_2/UNET/tester/test_unet.py
```

Generate COCO-style NIR images:

```bash
python stage_2/UNET/tester/generate_coco_nir.py
```

---

# Notes

* This repository currently contains the core implementation of CRIG-Net.
* Pretrained weights and datasets are not included.
* Additional documentation and updates will be released after review.

---

# License

Released for research purposes only.
