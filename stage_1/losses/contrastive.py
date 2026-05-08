import torch
import torch.nn.functional as F


def contrastive_loss(rgb_feat, nir_feat, temperature=0.07):
    """
    rgb_feat, nir_feat: (B, N, D)
    """

    rgb = rgb_feat.mean(dim=1)
    nir = nir_feat.mean(dim=1)

    rgb = F.normalize(rgb, dim=-1)
    nir = F.normalize(nir, dim=-1)

    logits = rgb @ nir.T / temperature
    labels = torch.arange(rgb.size(0)).to(rgb.device)

    loss_rgb = F.cross_entropy(logits, labels)
    loss_nir = F.cross_entropy(logits.T, labels)

    return 0.5 * (loss_rgb + loss_nir)
