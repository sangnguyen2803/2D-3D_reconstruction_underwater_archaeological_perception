"""Three pretrained encoders with a one-channel dense prediction head."""

import torch
import torch.nn.functional as F
from torch import nn


class Decode(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin + skip, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x, skip):
        return self.net(
            torch.cat(
                [
                    F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False),
                    skip,
                ],
                1,
            )
        )


class MatchabilityModel(nn.Module):
    def __init__(self, specification, pretrained=True):
        super().__init__()
        self.specification = dict(specification)
        self.kind = specification["kind"]
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        if self.kind == "resnet_unet":
            from torchvision.models import ResNet18_Weights, resnet18

            self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.encoder.fc = nn.Identity()
            self.decode = nn.ModuleList(
                [
                    Decode(512, 256, 128),
                    Decode(128, 128, 64),
                    Decode(64, 64, 32),
                    Decode(32, 64, 16),
                ]
            )
            self.head = nn.Conv2d(16, 1, 1)
        elif self.kind == "segformer_b0":
            from transformers import SegformerConfig, SegformerForSemanticSegmentation

            if pretrained:
                self.network = SegformerForSemanticSegmentation.from_pretrained(
                    specification["pretrained"],
                    revision=specification["revision"],
                    num_labels=1,
                    ignore_mismatched_sizes=True,
                )
            else:
                self.network = SegformerForSemanticSegmentation(
                    SegformerConfig(**specification.get("hf_config", {}), num_labels=1)
                )
        elif self.kind == "dino_head":
            from underwater_vision.features.dinov2 import DinoExtractor

            self.encoder = DinoExtractor(
                specification.get("backbone", "dinov2_vitb14"), "cpu", specification["revision"]
            ).model
            self.encoder.requires_grad_(False)
            self.head = nn.Sequential(
                nn.Conv2d(self.encoder.embed_dim, 64, 1),
                nn.GELU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 1, 1),
            )
        else:
            raise ValueError("Unknown matchability backbone")

    def train(self, mode=True):
        super().train(mode)
        if self.kind == "dino_head":
            self.encoder.eval()
        # Small-batch fine tuning must not corrupt pretrained BatchNorm moments.
        if self.kind == "resnet_unet":
            for layer in self.encoder.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
        return self

    def frozen_grid(self, rgb):
        if self.kind != "dino_head":
            raise ValueError("Frozen-grid caching is only valid for the DINO backbone")
        image = (rgb - self.mean) / self.std
        h, w = rgb.shape[-2:]
        image = F.pad(image, (0, (-w) % 14, 0, (-h) % 14), mode="replicate")
        with torch.no_grad():
            tokens = self.encoder.forward_features(image)["x_norm_patchtokens"]
            grid = tokens.transpose(1, 2).reshape(
                image.shape[0], tokens.shape[-1], image.shape[-2] // 14, image.shape[-1] // 14
            )
        return grid

    def decode_grid(self, grid, size):
        h, w = size
        padded = (grid.shape[-2] * 14, grid.shape[-1] * 14)
        out = F.interpolate(self.head(grid), size=padded, mode="bilinear", align_corners=False)[
            :, :, :h, :w
        ]
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)

    def forward(self, rgb):
        image = (rgb - self.mean) / self.std
        h, w = rgb.shape[-2:]
        if self.kind == "segformer_b0":
            out = self.network(pixel_values=image).logits
        elif self.kind == "resnet_unet":
            e = self.encoder
            x0 = e.relu(e.bn1(e.conv1(image)))
            x1 = e.layer1(e.maxpool(x0))
            x2 = e.layer2(x1)
            x3 = e.layer3(x2)
            out = e.layer4(x3)
            for block, skip in zip(self.decode, [x3, x2, x1, x0], strict=True):
                out = block(out, skip)
            out = self.head(out)
        else:
            return self.decode_grid(self.frozen_grid(rgb), (h, w))
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)


def load_model(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("feature_spec") != "dense_matchability_v3" or saved.get("status") != "trained":
        raise ValueError("A trained dense_matchability_v3 checkpoint is required")
    model = MatchabilityModel(
        saved["model_config"], pretrained=saved["model_config"]["kind"] == "dino_head"
    )
    state = saved["state_dict"]
    incompatible = model.load_state_dict(state, strict=False)
    allowed = saved["model_config"]["kind"] == "dino_head" and all(
        k.startswith("encoder.") for k in incompatible.missing_keys
    )
    if incompatible.unexpected_keys or (incompatible.missing_keys and not allowed):
        raise ValueError("Incomplete matchability checkpoint")
    return model.eval().to(device), saved
