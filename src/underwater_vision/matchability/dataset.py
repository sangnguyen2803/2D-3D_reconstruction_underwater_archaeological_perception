"""Original-image labels, cropped coordinates and photometric-only augmentation."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from underwater_vision.matchability.protocol import read_json
from underwater_vision.utils.io import read_rgb


class MatchabilityDataset(Dataset):
    def __init__(
        self,
        root,
        names,
        protocol,
        graph="hybrid",
        variant="l1",
        crop_hw=None,
        seed=42,
        cache_rgb=True,
    ):
        self.root, self.names = Path(root), list(names)
        self.feature_root = Path(
            read_json(self.root / "manifest_v3.json").get("feature_source", str(self.root))
        )
        self.protocol, self.graph, self.variant = protocol, graph, variant
        self.crop_hw = crop_hw
        self.rng = np.random.default_rng(seed)
        self._cache = {}
        self.cache_rgb = cache_rgb and crop_hw is None

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        if name not in self._cache:
            image = (
                read_rgb(self.feature_root / "images" / (name + ".png")) if self.cache_rgb else None
            )
            with np.load(self.feature_root / "features" / (name + ".npz")) as z:
                xy = z["xy"]
            with np.load(self.root / "labels" / self.protocol / self.graph / (name + ".npz")) as z:
                labels = z[self.variant].astype(np.float32)
            self._cache[name] = (image if self.cache_rgb else None, xy, labels)
        original, coordinates, target = self._cache[name]
        image = (
            original
            if original is not None
            else read_rgb(self.feature_root / "images" / (name + ".png"))
        )
        xy, labels = coordinates.copy(), target.copy()
        if self.crop_hw:
            h, w = self.crop_hw
            if h > image.shape[0] or w > image.shape[1]:
                raise ValueError("Training crop larger than original RGB")
            left = int(self.rng.integers(image.shape[1] - w + 1))
            top = int(self.rng.integers(image.shape[0] - h + 1))
            xy -= [left, top]
            keep = (xy[:, 0] >= 0) & (xy[:, 0] <= w - 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= h - 1)
            xy, labels = xy[keep], labels[keep]
            image = image[top : top + h, left : left + w].astype(np.float32) / 255
            image = image * self.rng.uniform(0.8, 1.2) + self.rng.uniform(-0.06, 0.06)
            image *= self.rng.uniform(0.9, 1.1, (1, 1, 3))
            if self.rng.random() < 0.2:
                image = cv2.GaussianBlur(image, (3, 3), float(self.rng.uniform(0.3, 1.0)))
            if self.rng.random() < 0.2:
                image += self.rng.normal(0, 0.01, image.shape).astype(np.float32)
            image = np.clip(image, 0, 1)
        else:
            image = image.astype(np.float32) / 255
        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "xy": torch.from_numpy(xy),
            "labels": torch.from_numpy(labels),
            "name": name,
        }
