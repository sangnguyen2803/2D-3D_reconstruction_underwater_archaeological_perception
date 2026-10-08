"""Reference-supervised MLP and permutation-equivariant contextual match filter."""

import cv2
import numpy as np
import torch
from torch import nn

FEATURE_NAMES = [
    "distance",
    "ratio",
    "mutual",
    "dino_cosine",
    "quality_a",
    "quality_b",
    "ax",
    "ay",
    "bx",
    "by",
    "dx",
    "dy",
]


def candidates(a, b, width, height):
    """Retain top-1 candidates before ratio filtering, including hard negatives."""
    if len(a["local"]) < 2 or len(b["local"]) < 2:
        return np.empty((0, 2), np.uint32), np.empty((0, 12), np.float32)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(a["local"], b["local"], k=2)
    reverse = matcher.match(b["local"], a["local"])
    backward = {m.queryIdx: m.trainIdx for m in reverse}
    pairs = np.asarray([[m.queryIdx, m.trainIdx] for m, _ in forward], np.uint32)
    distance = np.asarray([m.distance for m, _ in forward])
    ratio = np.asarray([m.distance / max(n.distance, 1e-9) for m, n in forward])
    mutual = np.asarray([backward.get(j) == i for i, j in pairs], float)
    cosine = np.sum(a["semantic"][pairs[:, 0]] * b["semantic"][pairs[:, 1]], axis=1)
    xy_a = a["xy"][pairs[:, 0]] / [width, height]
    xy_b = b["xy"][pairs[:, 1]] / [width, height]
    qa = a["quality"][pairs[:, 0]]
    qb = b["quality"][pairs[:, 1]]
    x = np.c_[distance, ratio, mutual, cosine, qa, qb, xy_a, xy_b, xy_b - xy_a]
    return pairs, x.astype(np.float32)


class MatchFilter(nn.Module):
    def __init__(self, architecture="context", channels=12, hidden=64):
        super().__init__()
        if architecture not in {"mlp", "context"}:
            raise ValueError("architecture must be mlp or context")
        self.architecture = architecture
        self.embed = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )
        if architecture == "context":
            self.seeds = nn.Parameter(torch.randn(1, 8, hidden) * 0.02)
            self.pool = nn.MultiheadAttention(hidden, 4, batch_first=True)
            self.context = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * (3 if architecture == "context" else 1), hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        if not len(x):
            return x.new_empty(0)
        z = self.embed(x)[None]
        if self.architecture == "context":
            pooled, _ = self.pool(self.seeds, z, z, need_weights=False)
            context, _ = self.context(z, pooled, pooled, need_weights=False)
            mean = z.mean(1, keepdim=True).expand_as(z)
            z = torch.cat([z, context, mean], dim=-1)
        return self.head(z)[0, :, 0]


class MatchPredictor:
    def __init__(self, checkpoint):
        self.metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if self.metadata.get("feature_spec", "match_v1_12") != "match_v1_12":
            raise ValueError("Match feature specification mismatch")
        self.model = MatchFilter(
            self.metadata["architecture"], channels=len(self.metadata["mean"])
        ).eval()
        self.model.load_state_dict(self.metadata["state_dict"])
        self.mean = np.asarray(self.metadata["mean"], np.float32)
        self.std = np.asarray(self.metadata["std"], np.float32)

    @torch.inference_mode()
    def __call__(self, x):
        if x.ndim != 2 or x.shape[1] != len(self.mean):
            raise ValueError("Unexpected match feature width")
        data = torch.from_numpy(((x - self.mean) / self.std).astype(np.float32))
        return torch.sigmoid(self.model(data) / self.metadata["temperature"]).numpy()
