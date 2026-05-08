import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


class MAENIRDecoder(nn.Module):
    def __init__(
        self,
        num_patches=196,
        encoder_dim=768,
        decoder_dim=512,
        depth=4,
        num_heads=8,
        patch_size=16
    ):
        super().__init__()

        self.patch_size = patch_size
        self.decoder_dim = decoder_dim

        self.input_proj = nn.Linear(encoder_dim, decoder_dim)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_dim)
        )

        self.blocks = nn.ModuleList([
            Block(
                dim=decoder_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(decoder_dim)

        self.head = nn.Linear(
            decoder_dim,
            patch_size * patch_size * 1
        )

    def forward(self, tokens):
        """
        tokens: (B, N, encoder_dim)
        """

        x = self.input_proj(tokens)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.head(x)

        return x
