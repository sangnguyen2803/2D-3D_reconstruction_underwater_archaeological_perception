"""Evaluation on reviewed, frozen hand labels only; bootstrap whole images."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from underwater_vision.evaluation.segmentation import (
    confusion,
    paired_bootstrap,
    pixel_ece,
    segmentation_metrics,
)
from underwater_vision.semantic.infer import SemanticPredictor
from underwater_vision.semantic.protocol import digest_file, load_manifest, new_run
from underwater_vision.semantic.schema import merge_mask
from underwater_vision.utils.io import read_rgb, write_json


def evaluate(manifest_path, checkpoint, output, role="test", teacher=None):
    manifest = load_manifest(manifest_path, require_frozen=True)
    entries = [e for e in manifest["entries"] if e["role"] == role]
    if not entries or any(e["label_source"] != "hand" or not e.get("reviewed") for e in entries):
        raise ValueError("Final evaluation needs reviewed hand labels")
    predictor = SemanticPredictor(checkpoint) if checkpoint else None
    if predictor and role == "test":
        used = set(predictor.metadata["training_images"] + predictor.metadata["validation_images"])
        if used & {e["image_id"] for e in entries}:
            raise ValueError("Segmentation test/training overlap")
    root = new_run(output)
    matrices = []
    rows = []
    for e in entries:
        target = np.asarray(Image.open(e["mask"]))
        if predictor:
            target = merge_mask(target, predictor.metadata.get("merge_map"))
        logits = None
        if predictor:
            rgb = cv2.resize(read_rgb(e["image"]), (target.shape[1], target.shape[0]))
            logits = predictor.logits(rgb).transpose(1, 2, 0)
            predicted = logits.argmax(-1).astype(np.uint8)
        else:
            with np.load(Path(teacher) / (Path(e["image_id"]).stem + ".npz")) as z:
                predicted = cv2.resize(
                    z["mask"], (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST
                )
        matrix = confusion(target, predicted)
        matrices.append(matrix)
        row = {
            "image_id": e["image_id"],
            "test_block": e.get("test_block"),
            **segmentation_metrics(matrix),
        }
        if logits is not None:
            row.update(
                ece_before=pixel_ece(logits, target),
                ece_after=pixel_ece(logits, target, predictor.temperature),
            )
        rows.append(row)
    interval = paired_bootstrap(
        matrices, lambda ms: segmentation_metrics(np.sum(ms, axis=0))["miou"]
    )
    result = {
        **segmentation_metrics(np.sum(matrices, axis=0)),
        "miou_bootstrap": interval,
        "images": rows,
        "hand_images": len(rows),
        "role": role,
        "checkpoint_sha256": digest_file(checkpoint) if checkpoint else None,
        "teacher": str(teacher) if teacher else None,
        "manifest_sha256": digest_file(manifest_path),
        "independent_spatial_regions": (
            len({e.get("test_block") for e in entries}) if role == "test" else 1
        ),
        "caveat": "Descriptive sub-blocks of the v1 test slab are not three rotated retrained folds",
    }
    if predictor:
        weights = np.array([r["labeled_pixels"] for r in rows])
        result["ece_before_image_weighted"] = float(
            np.average([r["ece_before"] for r in rows], weights=weights)
        )
        result["ece_after_image_weighted"] = float(
            np.average([r["ece_after"] for r in rows], weights=weights)
        )
    write_json(root / "evaluation_v2.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest", type=Path, default=Path("outputs/semantic_labels_v2/manifest.json")
    )
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--teacher", type=Path)
    p.add_argument("--output", type=Path, default=Path("outputs/segmentation_evaluation_v2"))
    p.add_argument("--role", choices=["test", "validation"], default="test")
    a = p.parse_args()
    if bool(a.checkpoint) == bool(a.teacher):
        p.error("Choose checkpoint or teacher, exactly one")
    evaluate(a.manifest, a.checkpoint, a.output, a.role, a.teacher)


if __name__ == "__main__":
    main()
