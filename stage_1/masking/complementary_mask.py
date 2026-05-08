import torch


def complementary_patch_mask(
    batch_size,
    num_patches,
    grid_size=14,
    device="cpu",
):
    """
    AVFF-style structured masking.

    We divide the patch grid into 14 horizontal slices (rows).
    Mask 7 rows in RGB and complementary 7 in NIR.
    """

    assert num_patches == grid_size * grid_size

    rows = torch.arange(num_patches).view(grid_size, grid_size)

    rgb_masks = []
    nir_masks = []

    for _ in range(batch_size):

        perm = torch.randperm(grid_size)

        rgb_rows = perm[: grid_size // 2]
        nir_rows = perm[grid_size // 2 :]

        rgb_mask = torch.ones(num_patches)
        nir_mask = torch.ones(num_patches)

        for r in rgb_rows:
            rgb_mask[rows[r].flatten()] = 0

        for r in nir_rows:
            nir_mask[rows[r].flatten()] = 0

        rgb_masks.append(rgb_mask)
        nir_masks.append(nir_mask)

    return (
        torch.stack(rgb_masks).to(device),
        torch.stack(nir_masks).to(device),
    )
