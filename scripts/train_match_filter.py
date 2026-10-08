"""Train both v1 filters on reference-F labels with a spatial validation block."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.evaluation.uncertainty import binary_metrics, choose_threshold
from underwater_vision.features.dinov2 import DinoExtractor
from underwater_vision.features.store import feature_record
from underwater_vision.geometry.reference import camera_parameters, reference_labels
from underwater_vision.matching.learned_filter import FEATURE_NAMES, MatchFilter
from underwater_vision.utils.io import atomic_npz, write_json


def select_training_pairs(names, lookup):
    centers = np.stack([lookup[n].reference_c2w[:3, 3] for n in names])
    distance = np.linalg.norm(centers[:, None] - centers[None], axis=2)
    np.fill_diagonal(distance, np.inf)
    pairs = set()
    for i in range(len(names)):
        for j in np.argsort(distance[i])[:2]:
            pairs.add(tuple(sorted((names[i], names[j]))))
        pairs.add(tuple(sorted((names[i], names[(i + len(names) // 2) % len(names)]))))
    return sorted(pairs)


def main():
    from underwater_vision.matching.learned_filter import candidates

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("outputs/match_filter_v1"))
    p.add_argument("--audit", type=Path, default=Path("outputs/reference_audit_v1"))
    p.add_argument("--train-views", type=int, default=64)
    p.add_argument("--validation-views", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    args = p.parse_args()
    if (
        json.loads((args.audit / "audit_v1.json").read_text())["status"]
        != "passed_proxy_sanity_check"
    ):
        raise ValueError("Reference geometry audit must pass before supervision")
    if (args.output / "training_v1.json").exists():
        raise FileExistsError("Existing training results are immutable")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    split = json.loads((args.audit / "spatial_split_v1.json").read_text())
    data = OmegaConf.to_container(OmegaConf.load("configs/data/mermaid.yaml"))
    lookup = {f.name: f for f in MermaidDataset(data["root"]).frames}
    selected = {}
    for group, count in [("train", args.train_views), ("validation", args.validation_views)]:
        names = split["groups"][group]
        selected[group] = [
            names[i] for i in np.linspace(0, len(names) - 1, min(count, len(names)), dtype=int)
        ]
    selected["test"] = Path("configs/evaluation_images.txt").read_text().splitlines()
    if not set(selected["test"]).issubset(set(split["groups"]["test"])):
        raise ValueError("Evaluation list is outside the locked spatial test region")
    write_json(args.output / "spatial_manifest_v1.json", {"selected": selected, "split": split})
    pair_root = args.output / "pairs"
    pair_root.mkdir(exist_ok=True)
    neural = None
    features = {}

    def feature(name):
        nonlocal neural
        if name not in features:
            if neural is None:
                neural = DinoExtractor(
                    "dinov2_vitb14", "auto", "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
                )
            features[name] = feature_record(lookup[name], neural)
        return features[name]

    records = {s: [] for s in selected}
    for group, names in selected.items():
        pairs = (
            select_training_pairs(names, lookup)
            if group != "test"
            else [
                (names[i], names[j])
                for i in range(len(names))
                for j in range(i + 1, min(i + 7, len(names)))
            ]
        )
        for i, (a, b) in enumerate(pairs):
            path = pair_root / f"{group}_{i:04d}.npz"
            if path.exists():
                with np.load(path) as z:
                    record = {k: z[k] for k in z.files}
            else:
                fa, fb = feature(a), feature(b)
                height, width = fa["shape"]
                ids, x = candidates(fa, fb, width, height)
                K, distortion = camera_parameters(int(width), int(height), data["calibration"])
                labels, errors, F = reference_labels(
                    fa["xy"][ids[:, 0]],
                    fb["xy"][ids[:, 1]],
                    lookup[a].reference_c2w,
                    lookup[b].reference_c2w,
                    K,
                    distortion,
                )
                record = {
                    "a": np.asarray(a),
                    "b": np.asarray(b),
                    "pairs": ids,
                    "x": x,
                    "labels": labels,
                    "reference_error": errors,
                    "reference_F": F,
                }
                atomic_npz(path, **record)
            records[group].append(record)
            if i % 16 == 0:
                print(f"Reference-labeled {group} pairs {i+1}/{len(pairs)}", flush=True)
    features.clear()
    neural = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    train_x = np.concatenate([r["x"][r["labels"] >= 0] for r in records["train"]])
    mean, std = train_x.mean(0), np.maximum(train_x.std(0), 1e-3)
    for group in records:
        for r in records[group]:
            r["tensor"] = torch.from_numpy((r["x"] - mean) / std)
    validation_y = np.concatenate([r["labels"] for r in records["validation"]])
    results = {}
    for architecture in ["mlp", "context"]:
        torch.manual_seed(42)
        model = MatchFilter(architecture)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        best, state, history = -1, None, []
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for idx in np.random.default_rng(42 + epoch).permutation(len(records["train"])):
                r = records["train"][idx]
                valid = r["labels"] >= 0
                if not valid.any():
                    continue
                logit = model(r["tensor"])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logit[valid], torch.from_numpy(r["labels"][valid].astype(np.float32))
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                losses.append(loss.item())
            model.eval()
            with torch.no_grad():
                logits = np.concatenate([model(r["tensor"]).numpy() for r in records["validation"]])
            probability = torch.sigmoid(torch.from_numpy(logits)).numpy()
            metric = binary_metrics(validation_y, probability)
            ap = metric.get("average_precision", 0)
            history.append(
                {"epoch": epoch + 1, "train_bce": float(np.mean(losses)), "validation_ap": ap}
            )
            if ap > best:
                best, state = ap, copy.deepcopy(model.state_dict())
            if epoch % 5 == 0:
                print(f"{architecture} epoch {epoch+1}: validation AP {ap:.4f}", flush=True)
        model.load_state_dict(state)
        with torch.no_grad():
            logits = np.concatenate([model(r["tensor"]).numpy() for r in records["validation"]])
        known = validation_y >= 0
        temperatures = np.linspace(0.5, 3, 51)
        nll = [
            torch.nn.functional.binary_cross_entropy_with_logits(
                torch.from_numpy(logits[known]) / t,
                torch.from_numpy(validation_y[known].astype(np.float32)),
            ).item()
            for t in temperatures
        ]
        temperature = float(temperatures[int(np.argmin(nll))])
        probability = torch.sigmoid(torch.from_numpy(logits) / temperature).numpy()
        threshold = choose_threshold(validation_y, probability)
        checkpoint = {
            "state_dict": state,
            "architecture": architecture,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "temperature": temperature,
            "threshold": threshold,
            "feature_names": FEATURE_NAMES,
            "training_images": selected["train"],
            "validation_images": selected["validation"],
            "spatial_split": split,
            "backbone": "dinov2_vitb14",
            "revision": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
            "labels": "published_reference_F_compatibility_with_ambiguous_band_excluded",
            "seed": 42,
            "image_size": 960,
            "max_features": 4096,
        }
        torch.save(checkpoint, args.output / f"{architecture}_v1.pt")
        metric = binary_metrics(validation_y, probability, threshold)
        metric.pop("threshold_curve", None)
        results[architecture] = {"validation": metric, "history": history}
    chosen = max(results, key=lambda k: results[k]["validation"]["average_precision"])
    report = {
        "models": results,
        "selected_architecture": chosen,
        "selection_rule": "highest spatial-validation AP; temperature and threshold fitted on validation only",
        "pair_counts": {s: len(r) for s, r in records.items()},
        "known_label_counts": {
            s: int(sum(np.sum(r["labels"] >= 0) for r in rows)) for s, rows in records.items()
        },
    }
    write_json(args.output / "training_v1.json", report)
    print(f"Selected {chosen} by spatial validation", flush=True)


if __name__ == "__main__":
    main()
