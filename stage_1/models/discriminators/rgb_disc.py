import torch
import torch.nn as nn


class RGBPatchDiscriminator(nn.Module):
    def __init__(self, patch_dim=16*16*3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(patch_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, patches):
        # patches: (B*M, D)
        return self.net(patches).squeeze()
