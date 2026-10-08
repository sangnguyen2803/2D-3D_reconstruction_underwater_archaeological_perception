"""Gate v1 reference supervision by checking calibration and pose alternatives."""

import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.data.spatial import spatial_split
from underwater_vision.features.local import extract_sift
from underwater_vision.geometry.reference import camera_parameters, reference_labels
from underwater_vision.matching.matcher import match_local
from underwater_vision.preprocessing.images import resize_rgb
from underwater_vision.utils.io import atomic_npz, fingerprint, read_rgb, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data/raw/mermaid")
    p.add_argument("--output", default="outputs/reference_audit_v1", type=Path)
    p.add_argument("--pairs", default=128, type=int)
    p.add_argument("--max-size", default=960, type=int)
    args = p.parse_args()
    if (args.output / "audit_v1.json").exists():
        raise FileExistsError("An existing audit report is immutable")
    args.output.mkdir(parents=True, exist_ok=True)
    frames = MermaidDataset(args.root).frames
    if any(f.reference_c2w is None for f in frames):
        raise ValueError("Audit requires reference poses for all frames")
    calibration = OmegaConf.to_container(OmegaConf.load("configs/data/mermaid.yaml"))["calibration"]
    names = [f.name for f in frames]
    split = spatial_split(
        names,
        [f.reference_c2w[:3, 3] for f in frames],
        Path("configs/evaluation_images.txt").read_text().splitlines(),
    )
    write_json(args.output / "spatial_split_v1.json", split)
    indices = np.linspace(0, len(frames) - 3, min(args.pairs, len(frames) - 2), dtype=int)
    errors = {
        n: []
        for n in [
            "brown_center_c2w",
            "brown_center_w2c",
            "no_distortion_c2w",
            "absolute_principal_c2w",
        ]
    }
    records = []
    cache = Path("cache/reference_features_v1")
    cache.mkdir(parents=True, exist_ok=True)

    def features(frame):
        key = fingerprint(
            frame.path, {"audit_v1": True, "max_size": args.max_size, "features": 2048}
        )
        path = cache / (key + ".npz")
        if path.exists():
            with np.load(path) as z:
                return z["xy"], z["local"], tuple(z["shape"])
        image, _ = resize_rgb(read_rgb(frame.path), args.max_size)
        xy, local = extract_sift(image, 2048)
        atomic_npz(path, xy=xy, local=local, shape=np.asarray(image.shape[:2]))
        return xy, local, image.shape[:2]

    for k, index in enumerate(indices):
        a, b = frames[index], frames[index + 2]
        xa, da, (height, width) = features(a)
        xb, db, _ = features(b)
        pairs, _ = match_local(da, db, 0.7)
        record = {"a": a.name, "b": b.name, "tight_ratio_matches": len(pairs)}
        for name in errors:
            convention, pose = name.rsplit("_", 1)
            K, distortion = camera_parameters(width, height, calibration, convention)
            ca, cb = a.reference_c2w, b.reference_c2w
            if pose == "w2c":
                ca, cb = np.linalg.inv(ca), np.linalg.inv(cb)
            _, error, _ = reference_labels(xa[pairs[:, 0]], xb[pairs[:, 1]], ca, cb, K, distortion)
            errors[name].extend(error.tolist())
            record[name] = float(np.median(error)) if len(error) else None
        records.append(record)
        if k % 16 == 0:
            print(f"Reference audit {k + 1}/{len(indices)}", flush=True)
    summary = {}
    for name, values in errors.items():
        e = np.asarray(values)
        summary[name] = {
            "matches": len(e),
            "median_px": float(np.median(e)) if len(e) else None,
            "fraction_below_1_5_px": float(np.mean(e <= 1.5)) if len(e) else 0,
            "p90_px": float(np.quantile(e, 0.9)) if len(e) else None,
        }
    nominal = summary["brown_center_c2w"]
    passed = (
        nominal["matches"] >= 1000
        and nominal["median_px"] <= 1.5
        and nominal["fraction_below_1_5_px"] >= 0.6
    )
    report = {
        "status": "passed_proxy_sanity_check" if passed else "failed",
        "reference_views": len(frames),
        "audited_pairs": len(indices),
        "alternatives": summary,
        "split_counts": {k: len(v) for k, v in split["groups"].items()},
        "thresholds": {"minimum_matches": 1000, "median_px_max": 1.5, "fraction_min": 0.6},
        "meaning": "RANSAC-independent reference compatibility; BA poses and epipolar labels are not independent surveyed GT",
    }
    write_json(args.output / "audit_v1.json", report)
    write_json(args.output / "pairs_v1.json", records)
    print(report, flush=True)
    if not passed:
        raise RuntimeError("Reference audit failed; training/evaluation must not proceed")


if __name__ == "__main__":
    main()
