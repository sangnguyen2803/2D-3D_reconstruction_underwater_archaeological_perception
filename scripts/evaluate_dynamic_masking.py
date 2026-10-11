"""Q2 on held-out pairs with hand-confirmed dynamic regions, fixed proposals."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.evaluation.segmentation import paired_bootstrap
from underwater_vision.features.dinov2 import DinoExtractor
from underwater_vision.features.store import feature_record
from underwater_vision.matching.learned_filter import MatchPredictor
from underwater_vision.semantic.infer import SemanticPredictor, TeacherCache
from underwater_vision.semantic.masks import sample_probabilities
from underwater_vision.semantic.protocol import load_manifest, new_run, require_gate
from underwater_vision.utils.io import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matches", type=Path, default=Path("outputs/match_filter_v1"))
    p.add_argument("--baseline", type=Path, default=Path("outputs/match_filter_v1/context_v1.pt"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument(
        "--manifest", type=Path, default=Path("outputs/semantic_labels_v2/manifest.json")
    )
    p.add_argument("--output", type=Path, default=Path("outputs/dynamic_masking_v2"))
    a = p.parse_args()
    gate = require_gate("outputs/semantic_audit_v2/gate_review.json")
    m = load_manifest(a.manifest, require_frozen=True)
    hand = {
        e["image_id"]: np.asarray(Image.open(e["mask"]))
        for e in m["entries"]
        if e["role"] == "test"
    }
    root = new_run(a.output)
    student = SemanticPredictor(a.checkpoint)
    teacher = TeacherCache(a.teacher)
    baseline = MatchPredictor(a.baseline)
    lookup = {f.name: f for f in MermaidDataset("data/raw/mermaid").frames}
    neural = DinoExtractor("dinov2_vitb14", "auto", "7764ea0f912e53c92e82eb78a2a1631e92725fc8")
    cache = {}

    def image_data(name):
        if name not in cache:
            f = feature_record(lookup[name], neural)
            xy = f["xy"]
            h, w = f["shape"]
            mask = cv2.resize(hand[name], (w, h), interpolation=cv2.INTER_NEAREST)
            integer = np.round(xy).astype(int)
            integer[:, 0] = np.clip(integer[:, 0], 0, w - 1)
            integer[:, 1] = np.clip(integer[:, 1], 0, h - 1)
            truth = mask[integer[:, 1], integer[:, 0]]
            sp, _ = sample_probabilities(student.cached(lookup[name].path), xy)
            tp, _ = sample_probabilities(teacher.cached(lookup[name].path), xy)
            cache[name] = (truth, sp[:, 5], tp[:, 5])
        return cache[name]

    records = []
    for file in sorted((a.matches / "pairs").glob("test_*.npz")):
        with np.load(file) as z:
            first, second, pairs, x, y = str(z["a"]), str(z["b"]), z["pairs"], z["x"], z["labels"]
        if first not in hand or second not in hand:
            continue
        if {first, second} & set(
            baseline.metadata["training_images"] + baseline.metadata["validation_images"]
        ):
            raise ValueError("Q2 test leaks baseline match training/validation views")
        if (
            first in student.metadata["training_images"] + student.metadata["validation_images"]
            or second in student.metadata["training_images"] + student.metadata["validation_images"]
        ):
            raise ValueError("Q2 test leaks segmentation training")
        fa, fb = image_data(first), image_data(second)
        labeled = (fa[0][pairs[:, 0]] != 255) & (fb[0][pairs[:, 1]] != 255) & (y >= 0)
        dynamic = (fa[0][pairs[:, 0]] == 5) | (fb[0][pairs[:, 1]] == 5)
        if not (dynamic & labeled).any():
            continue
        accepted = baseline(x[:, :12]) >= baseline.metadata["threshold"]
        record = {"labels": y[labeled], "none": accepted[labeled]}
        for name, index, threshold in [
            ("student", 1, student.metadata["dynamic_threshold"]),
            ("teacher", 2, 0.5),
        ]:
            dynamic_score = np.maximum(fa[index][pairs[:, 0]], fb[index][pairs[:, 1]])
            record[name] = (accepted & (dynamic_score < threshold))[labeled]
        records.append(record)

    def score(rows, key):
        y = np.concatenate([r["labels"] for r in rows])
        selected = np.concatenate([r[key] for r in rows])
        return {
            "precision": float((y[selected] == 1).mean()) if selected.any() else None,
            "recall": float(np.sum((y == 1) & selected) / max((y == 1).sum(), 1)),
            "retained": int(selected.sum()),
        }

    result = {
        "status": "qualitative_only" if gate.get("q2") == "qualitative_only" else "eligible",
        "verified_dynamic_pairs": len(records),
        "label_source": "reference F on hand-labeled endpoints; dynamic pair membership hand-confirmed",
        "variants": {},
        "precision_differences": {},
    }
    if len(records) >= 2:
        result["variants"] = {name: score(records, name) for name in ["none", "teacher", "student"]}
        for name in ["teacher", "student"]:

            def difference(rows, key=name):
                original, new = score(rows, "none")["precision"], score(rows, key)["precision"]
                return new - original if original is not None and new is not None else np.nan

            result["precision_differences"][name] = paired_bootstrap(records, difference)
    else:
        result["status"] = "insufficient_verified_dynamic_pairs"
    write_json(root / "evaluation_v2.json", result)


if __name__ == "__main__":
    main()
