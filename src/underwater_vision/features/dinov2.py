"""Frozen official DINOv2 with patch descriptors interpolated at SIFT points."""

import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F


class DinoExtractor:
    def __init__(self, backbone="dinov2_vitb14", device="auto", revision=None):
        if backbone not in {"dinov2_vits14", "dinov2_vitb14"}:
            raise ValueError("Consumer-GPU defaults support DINOv2 S/14 and B/14")
        os.environ.setdefault("XFORMERS_DISABLED", "1")
        torch.set_num_threads(4)
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.revision = revision or "main"
        self.model = (
            torch.hub.load(
                f"facebookresearch/dinov2:{self.revision}",
                backbone,
                pretrained=True,
                trust_repo=True,
                skip_validation=True,
            )
            .eval()
            .to(self.device)
        )
        self.model.requires_grad_(False)
        self.provenance = {
            "repository": "https://github.com/facebookresearch/dinov2",
            "revision": self.revision,
            "backbone": backbone,
            "torch": torch.__version__,
            "device": str(self.device),
        }

    @torch.inference_mode()
    def _forward(self, rgb):
        h, w = rgb.shape[:2]
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None] / 255
        tensor = tensor.to(self.device)
        # Pad bottom/right to multiples of 14, preserving all image coordinates.
        ph, pw = ((h + 13) // 14) * 14, ((w + 13) // 14) * 14
        tensor = F.pad(tensor, (0, pw - w, 0, ph - h), mode="replicate")
        mean = tensor.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = tensor.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        tensor = (tensor - mean) / std
        autocast = (
            torch.autocast("cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            tokens = self.model.forward_features(tensor)
        return tokens, ph, pw

    @torch.inference_mode()
    def extract_global(self, rgb):
        """Normalized CLS token for retrieval; no pose or match labels enter it."""
        tokens, _, _ = self._forward(rgb)
        return F.normalize(tokens["x_norm_clstoken"].float(), dim=1)[0].cpu().numpy()

    @torch.inference_mode()
    def extract(self, rgb, xy):
        tokens, ph, pw = self._forward(rgb)
        patches = tokens["x_norm_patchtokens"]
        channels = patches.shape[-1]
        grid = patches.reshape(1, ph // 14, pw // 14, channels).permute(0, 3, 1, 2).float()
        if len(xy) == 0:
            return np.empty((0, channels), np.float32)
        # OpenCV pixel centers x map to padded continuous centers x+0.5.
        coordinates = torch.as_tensor(xy, device=self.device, dtype=torch.float32).clone()
        coordinates[:, 0] = 2 * (coordinates[:, 0] + 0.5) / pw - 1
        coordinates[:, 1] = 2 * (coordinates[:, 1] + 0.5) / ph - 1
        descriptors = F.grid_sample(
            grid,
            coordinates[None, :, None],
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[0, :, :, 0].T
        return F.normalize(descriptors, dim=1).cpu().numpy().astype(np.float32)
