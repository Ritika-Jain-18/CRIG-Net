import torch
import torch.nn as nn
import torch.nn.functional as F


class R2NBridge(nn.Module):
    """
    Converts Pix2Next encoder features into R2N tokens,
    applies R2N conversion, and converts back to feature maps.
    """

    def __init__(self, r2n_module):
        super().__init__()

        self.r2n = r2n_module

        # Spatial alignment: 32x32 → 14x14
        self.pool = nn.AdaptiveAvgPool2d((14, 14))

        # Channel projection: 512 → 768
        self.to_r2n_dim = nn.Linear(512, 768)

        # Back projection: 768 → 512
        self.from_r2n_dim = nn.Conv2d(768, 512, kernel_size=1)

    def forward(self, x):
        """
        x: (B, 512, 32, 32)
        """

        B, C, H, W = x.shape

        # -------------------------
        # 1. Spatial align to 14x14
        # -------------------------
        x = self.pool(x)  # (B, 512, 14, 14)

        # -------------------------
        # 2. Flatten to tokens
        # -------------------------
        x = x.flatten(2).transpose(1, 2)  # (B, 196, 512)

        # -------------------------
        # 3. Project to R2N dim
        # -------------------------
        x = self.to_r2n_dim(x)  # (B, 196, 768)

        # -------------------------
        # 4. Apply R2N conversion
        # -------------------------
        # x = self.r2n(x)  # (B, 196, 768)
        with torch.no_grad():
            x = self.r2n(x)

        # -------------------------
        # 5. Convert back to feature map
        # -------------------------
        x = x.transpose(1, 2).reshape(B, 768, 14, 14)

        # -------------------------
        # 6. Restore spatial size
        # -------------------------
        x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)

        # -------------------------
        # 7. Project back to 512 channels
        # -------------------------
        x = self.from_r2n_dim(x)  # (B, 512, 32, 32)

        return x
