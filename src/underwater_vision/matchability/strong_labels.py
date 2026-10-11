"""Immutable L2 labels certified by excluded-view reference reprojection."""

from pathlib import Path

import numpy as np
import yaml

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.geometry.reference import camera_parameters, undistort
from underwater_vision.matchability.build import load_features
from underwater_vision.matchability.confidence import Tracks, holdout_rows
from underwater_vision.matchability.labels import label_counts, stronger_labels
from underwater_vision.matchability.protocol import digest, new_run, read_json
from underwater_vision.utils.io import atomic_npz, write_json


def build_strong_labels(source, output):
    source, root = Path(source), new_run(output)
    manifest = read_json(source / "manifest_v3.json")
    pairs = read_json(source / "pairs_v3.json")
    lookup = {f.name: f for f in MermaidDataset("data/raw/mermaid").frames}
    calibration = yaml.safe_load(Path("configs/data/mermaid.yaml").read_text())["calibration"]
    xy = {}
    for item in manifest["images"]:
        f = load_features(source, item["name"])
        h, w = map(int, f["shape"])
        K, distortion = camera_parameters(w, h, calibration)
        xy[item["name"]] = undistort(f["xy"], K, distortion)
    report = {}
    for protocol in manifest["protocols"]:
        role = {n: s for s, names in protocol["groups"].items() for n in names}
        tracks = Tracks()
        for i, info in enumerate(pairs):
            a, b = info["a"], info["b"]
            if role[a] != role[b] or role[a] == "excluded":
                continue
            with np.load(source / "pairs" / f"{i:06d}.npz") as z:
                forward, reverse = z["forward"], z["reverse"]
                error, reverse_error = z["forward_error"], z["reverse_error"]
            if np.sum(error <= 1.5) + np.sum(reverse_error <= 1.5) < 15:
                continue
            mutual = reverse[forward[:, 1], 0] == forward[:, 0]
            for ai, bi in forward[mutual & (error <= 1.5)]:
                tracks.join((a, int(ai)), (b, int(bi)))
        held_error = {
            n: np.full(len(xy[n]), np.nan, np.float32) for n in role if role[n] != "excluded"
        }
        certified = total = 0
        for nodes in tracks.tracks():
            poses = [lookup[n].reference_c2w for n, _ in nodes]
            observations = [xy[n][i] for n, i in nodes]
            size = len(nodes)
            rows = holdout_rows(
                nodes, poses, observations, np.ones((size, 1)), np.ones(size), np.ones(size), K
            )
            if not rows:
                continue
            error = float(np.median([r["error"] for r in rows]))
            total += 1
            certified += error <= 2
            for n, idx in nodes:
                held_error[n][idx] = error
        counts = {
            s: {k: 0 for k in ["positive", "negative", "ignore"]}
            for s in ["train", "validation", "test"]
        }
        target = root / "labels" / protocol["name"] / "hybrid"
        target.mkdir(parents=True)
        for name in held_error:
            with np.load(source / "labels" / protocol["name"] / "hybrid" / (name + ".npz")) as z:
                record = {k: z[k] for k in z.files}
            l2 = stronger_labels(record["l1"], held_error[name])
            atomic_npz(target / (name + ".npz"), **record, l2=l2, heldout_error=held_error[name])
            for key, value in label_counts(l2).items():
                counts[role[name]][key] += value
        report[protocol["name"]] = {
            "counts": counts,
            "tracks_evaluated": total,
            "tracks_certified": certified,
        }
        print(f"L2 {protocol['name']}: {certified}/{total} tracks certified", flush=True)
    manifest["feature_source"] = str(source.resolve())
    manifest["parent_manifest_sha256"] = digest(source / "manifest_v3.json")
    manifest["label_spec"] = "rootsift_reference_matchability_L2_v3"
    write_json(root / "manifest_v3.json", manifest)
    write_json(
        root / "audit_v3.json",
        {
            "status": "complete",
            "protocols": report,
            "positive_rule": "L1 positive AND conflict-free mutual reference-F track; median of three excluded-view target errors <=2 px",
            "negative_rule": "Keep L1 negatives; uncertified positive is ignore, not negative",
            "common_evaluation": "Original L1 labels for validation and test; L2 affects training only",
        },
    )
    return root
