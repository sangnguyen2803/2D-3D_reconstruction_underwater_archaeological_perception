"""Sparse BCE on OpenCV pixel centers, including empty/ignored batches."""

import torch.nn.functional as F


def sample_logits(logits, xy):
    """Nx2 or BxNx2 OpenCV coordinates; align_corners=False half-pixel grid.

    Coordinate zero is the center of the first pixel, not the image corner.
    The +0.5 here converts to the normalized sampling grid, never COLMAP.
    """
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("Expected Bx1xHxW logits")
    if xy.ndim == 2:
        xy = xy[None]
    h, w = logits.shape[-2:]
    grid = (xy.to(logits.dtype) + 0.5) * logits.new_tensor([2 / w, 2 / h]) - 1
    return F.grid_sample(
        logits, grid[:, None], mode="bilinear", padding_mode="border", align_corners=False
    )[:, 0, 0]


def sparse_bce(logits, xy, labels, pos_weight=1.0):
    sampled = sample_logits(logits, xy)
    target = labels.reshape_as(sampled).to(sampled.dtype)
    valid = target >= 0
    if not valid.any():
        return logits.sum() * 0
    return F.binary_cross_entropy_with_logits(
        sampled[valid].float(),
        target[valid].float(),
        pos_weight=logits.new_tensor(pos_weight).float(),
    )


def resize_coordinates(xy, old_hw, new_hw):
    """OpenCV resize preserves pixel centers: (x+.5)*scale-.5."""
    oh, ow = old_hw
    nh, nw = new_hw
    return (xy + 0.5) * xy.new_tensor([nw / ow, nh / oh]) - 0.5
