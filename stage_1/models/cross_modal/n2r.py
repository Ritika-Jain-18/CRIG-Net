import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


class N2R(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.transformer = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True
        )

    def forward(self, nir_tokens):
        """
        nir_tokens: (B, N, D)
        """
        x = self.mlp(nir_tokens)
        x = self.transformer(x)
        return x
