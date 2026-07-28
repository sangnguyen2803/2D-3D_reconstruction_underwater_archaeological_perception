"""Manifest-controlled image/mask training with underwater augmentations."""

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from underwater_vision.semantic.protocol import load_manifest
from underwater_vision.semantic.schema import merge_mask
from underwater_vision.utils.io import read_rgb


class SemanticDataset(Dataset):
    def __init__(
        self, manifest, role, size=(384, 512), augment=False, merge_map=None, entries=None
    ):
        self.entries = (
            entries
            if entries is not None
            else [
                e
                for e in load_manifest(manifest, require_frozen=True)["entries"]
                if e["role"] == role
            ]
        )
        self.role, self.size, self.augment, self.merge_map = role, tuple(size), augment, merge_map
        if not self.entries:
            raise ValueError(f"No {role} entries")
        if any(e["role"] != role or e["slab"] != role for e in self.entries):
            raise ValueError("Label role/slab leakage")
        if role != "train" and any(
            e["label_source"] != "hand" or not e.get("reviewed") for e in self.entries
        ):
            raise ValueError("Validation/test evaluation requires reviewed hand labels")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        e = self.entries[index]
        rgb = read_rgb(e["image"])
        mask = merge_mask(np.asarray(Image.open(e["mask"])), self.merge_map)
        if rgb.shape[:2] != mask.shape:
            # Annotation/pseudo masks are at 960px; explicitly align to original.
            rgb = cv2.resize(rgb, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
        if self.augment:
            if np.random.random() < 0.5:
                rgb, mask = rgb[:, ::-1].copy(), mask[:, ::-1].copy()
            h, w = mask.shape
            fraction = np.random.uniform(0.7, 1)
            ch, cw = max(1, int(h * fraction)), max(1, int(w * fraction))
            y, x = np.random.randint(h - ch + 1), np.random.randint(w - cw + 1)
            rgb, mask = rgb[y : y + ch, x : x + cw], mask[y : y + ch, x : x + cw]
            cast = np.random.uniform(0.85, 1.15, (1, 1, 3))
            rgb = np.clip(
                (rgb.astype(float) - 127) * np.random.uniform(0.8, 1.2)
                + 127
                + np.random.uniform(-18, 18),
                0,
                255,
            )
            rgb = np.clip(rgb * cast, 0, 255).astype(np.uint8)
            if np.random.random() < 0.2:
                rgb = cv2.GaussianBlur(rgb, (3, 3), np.random.uniform(0.3, 1.2))
        rgb = cv2.resize(rgb, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
        return (
            torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255,
            torch.from_numpy(mask.copy()).long(),
            e["image_id"],
        )
