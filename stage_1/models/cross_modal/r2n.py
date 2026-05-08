import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


class R2N(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()

        # Single-layer MLP (paper-faithful)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        # Single transformer block
        self.transformer = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True
        )

    def forward(self, rgb_tokens):
        """
        rgb_tokens: (B, N, D)
        """
        x = self.mlp(rgb_tokens)
        x = self.transformer(x)
        return x
