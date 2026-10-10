"""Small reliability head; geometry-generated weak labels, frozen backbone."""

from pathlib import Path

import numpy as np
import torch
from torch import nn


class ReliabilityHead(nn.Module):
    def __init__(self, channels=768):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(channels, channels // 2), nn.GELU(), nn.Linear(channels // 2, 1)
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)

    def reliability(self, features):
        return torch.sigmoid(self(features))


def load_head(path):
    if not path or not Path(path).is_file():
        raise FileNotFoundError("Train the reliability head and set matching.learned_checkpoint")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    head = ReliabilityHead(checkpoint["channels"])
    head.load_state_dict(checkpoint["state_dict"])
    return head.eval(), checkpoint


@torch.inference_mode()
def predict(features, path, calibrated=False):
    head, metadata = load_head(path)
    if features.shape[1] != head.network[0].in_features:
        raise ValueError("Checkpoint and backbone feature dimensions do not agree")
    temperature = metadata.get("temperature", 1) if calibrated else 1
    return torch.sigmoid(head(torch.from_numpy(features)) / temperature).numpy().astype(np.float32)
