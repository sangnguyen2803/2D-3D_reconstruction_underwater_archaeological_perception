"""Conditional dynamic detector helpers; imported only if human Gate 0 allows it."""

import cv2
import numpy as np


def mask_boxes(mask, min_pixels=20):
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (np.asarray(mask) == 5).astype(np.uint8), 8
    )
    h, w = mask.shape
    boxes = []
    for i in range(1, count):
        x, y, bw, bh, pixels = stats[i]
        if pixels >= min_pixels:
            boxes.append(
                [0, float((x + bw / 2) / w), float((y + bh / 2) / h), float(bw / w), float(bh / h)]
            )
    return boxes
