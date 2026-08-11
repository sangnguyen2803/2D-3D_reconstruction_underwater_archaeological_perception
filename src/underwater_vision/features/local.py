import cv2
import numpy as np


def extract_sift(image, max_features=4096, contrast_threshold=0.04):
    if not np.isfinite(contrast_threshold) or contrast_threshold <= 0:
        raise ValueError("SIFT contrast threshold must be finite and positive")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = cv2.SIFT_create(
        nfeatures=max_features, contrastThreshold=contrast_threshold
    ).detectAndCompute(gray, None)
    xy = np.array([k.pt for k in keypoints], dtype=np.float32).reshape(-1, 2)
    descriptors = np.zeros((0, 128), np.float32) if descriptors is None else descriptors
    # RootSIFT: nonnegative unit vectors and robust Hellinger matching.
    descriptors = np.sqrt(descriptors / (descriptors.sum(axis=1, keepdims=True) + 1e-12))
    return xy, descriptors.astype(np.float32)
