"""Standalone segmentation models: RGB [0,1] in, full-resolution logits out."""

import torch
import torch.nn.functional as F
from torch import nn


class SegmentationModel(nn.Module):
    def __init__(self, specification, pretrained=True):
        super().__init__()
        self.specification = dict(specification)
        self.kind = specification["kind"]
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        if self.kind == "segformer":
            from transformers import SegformerConfig, SegformerForSemanticSegmentation

            if pretrained:
                self.network = SegformerForSemanticSegmentation.from_pretrained(
                    specification["pretrained"],
                    revision=specification["revision"],
                    num_labels=6,
                    ignore_mismatched_sizes=True,
                )
            else:
                self.network = SegformerForSemanticSegmentation(
                    SegformerConfig(**specification.get("hf_config", {}), num_labels=6)
                )
        elif self.kind in {"dino_probe", "dino_decoder"}:
            from underwater_vision.features.dinov2 import DinoExtractor

            extractor = DinoExtractor(
                specification.get("backbone", "dinov2_vitb14"), "cpu", specification["revision"]
            )
            self.encoder = extractor.model
            channels = self.encoder.embed_dim
            self.decoder = (
                nn.Conv2d(channels, 6, 1)
                if self.kind == "dino_probe"
                else nn.Sequential(
                    nn.Conv2d(channels, 128, 1),
                    nn.GELU(),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(128, 6, 1),
                )
            )
        else:
            raise ValueError("Model kind must be segformer, dino_probe or dino_decoder")

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "encoder"):
            self.encoder.eval()
        return self

    def forward(self, rgb):
        image = (rgb - self.mean) / self.std
        if self.kind == "segformer":
            logits = self.network(pixel_values=image).logits
        else:
            h, w = image.shape[-2:]
            image = F.pad(image, (0, (-w) % 14, 0, (-h) % 14), mode="replicate")
            with torch.no_grad():
                tokens = self.encoder.forward_features(image)["x_norm_patchtokens"]
                grid = tokens.transpose(1, 2).reshape(
                    image.shape[0], tokens.shape[-1], image.shape[-2] // 14, image.shape[-1] // 14
                )
            logits = self.decoder(grid)
            logits = F.interpolate(
                logits, size=image.shape[-2:], mode="bilinear", align_corners=False
            )[:, :, :h, :w]
        return F.interpolate(logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False)


def load_model(checkpoint, device="cpu"):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("feature_spec") != "segmentation_six_v2":
        raise ValueError("Segmentation feature specification mismatch")
    model = SegmentationModel(
        saved["model_config"],
        pretrained=False if saved["model_config"]["kind"] == "segformer" else True,
    )
    model.load_state_dict(saved["state_dict"])
    return model.eval().to(device), saved
