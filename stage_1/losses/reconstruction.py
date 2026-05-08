import torch
import torch.nn.functional as F


def masked_reconstruction_loss(pred, target, mask):
    """
    pred: (B, N, D)
    target: (B, N, D)
    mask: (B, N) where 1 = visible, 0 = masked
    """
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # per-patch loss
    loss = loss * (1 - mask)  # only masked patches
    return loss.sum() / (1 - mask).sum()
