"""Annotation-free local quality cues. All normalization uses fixed scales.

Per-image min/max normalization would incorrectly make uniformly poor images
look reliable. These parameters are tunable hypotheses, not physical visibility.
"""

import cv2
import numpy as np


def sample_map(values, xy):
    if len(xy) == 0:
        return np.empty(0, np.float32)
    xy = np.asarray(xy, np.float32)
    return cv2.remap(
        values.astype(np.float32),
        xy[:, 0].reshape(-1, 1),
        xy[:, 1].reshape(-1, 1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).ravel()


def estimate_visibility(rgb, window=15):
    if window < 3 or window % 2 == 0:
        raise ValueError("Quality window must be odd and >= 3")
    image = rgb.astype(np.float32) / 255
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def average(x):
        return cv2.boxFilter(x, -1, (window, window), normalize=True, borderType=cv2.BORDER_REFLECT)

    mean = average(gray)
    contrast = np.sqrt(np.maximum(average(gray * gray) - mean * mean, 0))
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness = np.sqrt(np.maximum(average(laplacian * laplacian) - average(laplacian) ** 2, 0))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8
    gradient = np.sqrt(average(gx * gx + gy * gy))
    # Quantized local entropy, without allocating an H x W x 256 tensor.
    bins = np.minimum((gray * 16).astype(int), 15)
    entropy = np.zeros_like(gray)
    for k in range(16):
        probability = average((bins == k).astype(np.float32))
        entropy -= probability * np.log2(np.maximum(probability, 1e-8)) / 4
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    exposure = np.clip(1 - np.abs(mean - 0.5) / 0.5, 0, 1)
    channel_mean = average(image)
    imbalance = (channel_mean.max(2) - channel_mean.min(2)) / np.maximum(channel_mean.max(2), 0.01)
    cues = {
        "contrast": 1 - np.exp(-contrast / 0.06),
        "sharpness": 1 - np.exp(-sharpness / 0.08),
        "texture": 1 - np.exp(-gradient / 0.035),
        "entropy": entropy,
        "exposure": exposure,
        "saturation": average(hsv[:, :, 1]),
        "color_balance": 1 - imbalance,
    }
    reliability = (
        0.25 * cues["contrast"]
        + 0.25 * cues["sharpness"]
        + 0.20 * cues["texture"]
        + 0.20 * cues["entropy"]
        + 0.08 * exposure
        + 0.02 * cues["color_balance"]
    )
    return np.clip(reliability, 0, 1).astype(np.float32), cues
