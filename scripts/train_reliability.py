"""Train a frozen-feature MLP on geometry labels from a separate trajectory block."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from underwater_vision.utils.io import write_json
from underwater_vision.visibility.learned import ReliabilityHead


def collect(run, cache):
    manifest = json.loads((run / "input_manifest.json").read_text(encoding="utf-8"))
    features = {}
    for entry in manifest:
        with np.load(cache / "features" / (entry["cache_key"] + ".npz")) as data:
            if "semantic" not in data:
                raise ValueError("Training run must use model.extractor=sift_dinov2")
            features[entry["name"]] = data["semantic"]
    totals = {n: np.zeros(len(f)) for n, f in features.items()}
    hits = {n: np.zeros(len(f)) for n, f in features.items()}
    for path in sorted((run / "matches").glob("*.npz")):
        with np.load(path) as pair:
            for side, name in enumerate([str(pair["a"]), str(pair["b"])]):
                np.add.at(totals[name], pair["pairs"][:, side], 1)
                np.add.at(hits[name], pair["pairs"][:, side], pair["inliers"].astype(float))
    x, y, groups = [], [], []
    for group, (name, feature) in enumerate(features.items()):
        observed = totals[name] >= 2
        x.append(feature[observed])
        y.append(hits[name][observed] / totals[name][observed])
        groups.extend([group] * int(observed.sum()))
    return np.concatenate(x), np.concatenate(y), np.array(groups), list(features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--cache", default="cache", type=Path)
    parser.add_argument("--output", default="outputs/models/reliability.pt", type=Path)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    X, y, groups, names = collect(args.run, args.cache)
    if len(X) < 50 or y.min() == y.max():
        raise ValueError("Insufficient varied weak supervision")
    # Contiguous frame-group validation, with two-frame boundary gap.
    boundary = int(len(names) * 0.7)
    train = groups < max(1, boundary - 2)
    validation = groups >= min(len(names) - 1, boundary + 2)
    if not train.any() or not validation.any():
        raise ValueError("Need more training views for group-separated validation")
    head = ReliabilityHead(X.shape[1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    x = torch.from_numpy(X)
    target = torch.from_numpy(y.astype(np.float32))
    history, best, state = [], float("inf"), None
    for epoch in range(args.epochs):
        head.train()
        generator = torch.Generator().manual_seed(args.seed + epoch)
        order = torch.where(torch.from_numpy(train))[0]
        order = order[torch.randperm(len(order), generator=generator)]
        losses = []
        for ids in order.split(512):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x[ids]), target[ids])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        head.eval()
        with torch.no_grad():
            val = torch.nn.functional.binary_cross_entropy_with_logits(
                head(x[validation]), target[validation]
            ).item()
            brier = torch.mean((head.reliability(x[validation]) - target[validation]) ** 2).item()
        history.append(
            {
                "epoch": epoch + 1,
                "train_bce": float(np.mean(losses)),
                "validation_bce": val,
                "validation_soft_label_brier": brier,
            }
        )
        if val < best:
            best = val
            state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.to_container(OmegaConf.load(args.run / "config.yaml"), resolve=True)
    torch.save(
        {
            "state_dict": state,
            "channels": X.shape[1],
            "seed": args.seed,
            "training_images": names,
            "backbone": cfg["model"]["backbone"],
            "revision": cfg["model"]["revision"],
            "supervision": "candidate_match_geometric_inlier_frequency",
        },
        args.output,
    )
    write_json(
        args.output.with_suffix(".json"),
        {
            "samples": len(X),
            "train_samples": int(train.sum()),
            "validation_samples": int(validation.sum()),
            "best_validation_bce": best,
            "training_images": names,
            "history": history,
        },
    )
    print(f"Saved {args.output}, {len(X)} weakly supervised observations")


if __name__ == "__main__":
    main()
