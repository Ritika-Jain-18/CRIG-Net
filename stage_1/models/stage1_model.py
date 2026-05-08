import torch
import torch.nn as nn

from models.encoders.vit_rgb import ViTRGBEncoder
from models.encoders.vit_nir import ViTNIREncoder

from models.cross_modal.r2n import R2N
from models.cross_modal.n2r import N2R

from models.decoders.mae_rgb import MAERGBDecoder
from models.decoders.mae_nir import MAENIRDecoder

from masking.complementary_mask import complementary_patch_mask


class Stage1AVFF(nn.Module):
    """
    AVFF Stage-1 for RGB–NIR representation learning.

    Pipeline:
        RGB / NIR
          → ViT encoders
          → complementary spatial-slice masking
          → R2N / N2R
          → token replacement (a', v')
          → MAE-style transformer decoders
          → masked reconstruction + GAN + contrastive
    """

    def __init__(self):
        super().__init__()

        # -----------------
        # Encoders
        # -----------------
        self.rgb_enc = ViTRGBEncoder()
        self.nir_enc = ViTNIREncoder()

        # -----------------
        # Cross-modal networks
        # -----------------
        self.r2n = R2N()
        self.n2r = N2R()

        # -----------------
        # Transformer decoders (VideoMAE-style)
        # -----------------
        self.rgb_dec = MAERGBDecoder(num_patches=196)
        self.nir_dec = MAENIRDecoder(num_patches=196)

    def forward(self, rgb, nir):
        """
        Args:
            rgb : (B, 3, 224, 224)
            nir : (B, 1, 224, 224)

        Returns:
            dict containing reconstructions, masks, and features
        """

        B = rgb.size(0)

        # -----------------
        # Encode
        # -----------------
        rgb_feat = self.rgb_enc(rgb)   # (B, N, D)
        nir_feat = self.nir_enc(nir)

        N = rgb_feat.size(1)

        # -----------------
        # Complementary spatial-slice masking (AVFF-style)
        # -----------------
        rgb_mask, nir_mask = complementary_patch_mask(
            batch_size=B,
            num_patches=N,
            device=rgb.device,
        )

        # -----------------
        # Visible tokens
        # -----------------
        rgb_vis = rgb_feat * rgb_mask.unsqueeze(-1)
        nir_vis = nir_feat * nir_mask.unsqueeze(-1)

        # -----------------
        # Cross-modal prediction
        # -----------------
        nir_from_rgb = self.r2n(rgb_vis)
        rgb_from_nir = self.n2r(nir_vis)

        # -----------------
        # Token replacement → a′ , v′
        # -----------------
        nir_fused = (
            nir_feat * nir_mask.unsqueeze(-1)
            + nir_from_rgb * (1.0 - nir_mask.unsqueeze(-1))
        )

        rgb_fused = (
            rgb_feat * rgb_mask.unsqueeze(-1)
            + rgb_from_nir * (1.0 - rgb_mask.unsqueeze(-1))
        )

        # -----------------
        # Decode (MAE / VideoMAE style)
        # -----------------
        rgb_rec = self.rgb_dec(rgb_fused)   # (B, N, 768)
        nir_rec = self.nir_dec(nir_fused)   # (B, N, 256)

        return {
            "rgb_rec": rgb_rec,
            "nir_rec": nir_rec,
            "rgb_mask": rgb_mask,
            "nir_mask": nir_mask,
            "rgb_feat": rgb_feat,
            "nir_feat": nir_feat,
            "rgb_fused": rgb_fused,
            "nir_fused": nir_fused,
        }
