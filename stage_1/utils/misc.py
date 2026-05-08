import torch


def patchify(imgs, patch_size=16):
    """
    imgs: (B, C, H, W)
    returns: (B, N, patch_size*patch_size*C)
    """
    B, C, H, W = imgs.shape
    assert H % patch_size == 0 and W % patch_size == 0

    h = H // patch_size
    w = W // patch_size

    x = imgs.reshape(
        B, C, h, patch_size, w, patch_size
    )
    x = x.permute(0, 2, 4, 3, 5, 1)
    x = x.reshape(B, h * w, patch_size * patch_size * C)

    return x
def gather_masked_patches(patches, mask):
    """
    patches: (B, N, D)
    mask: (B, N)  (1=visible,0=masked)

    returns flattened masked patches: (B*M, D)
    """

    masked = patches[mask == 0]
    return masked

def unpatchify(patches, patch_size=16, channels=3):
    B, N, D = patches.shape
    h = w = int(N ** 0.5)

    patches = patches.reshape(
        B, h, w, patch_size, patch_size, channels
    )
    patches = patches.permute(0, 5, 1, 3, 2, 4)
    images = patches.reshape(
        B, channels, h * patch_size, w * patch_size
    )
    return images